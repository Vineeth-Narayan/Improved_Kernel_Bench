import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
from datasets import load_dataset
from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref, fetch_baseline_time
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template, prompt_fix_compile, prompt_fix_correctness, prompt_for_optimization
from src.utils import extract_first_code, extract_error_msg, query_server, set_gpu_arch, read_file, create_inference_server_from_presets

"""
Generate and evaluate a single sample with iterative compilation fixes
"""
# NOTE: gen_for_correctness function is the main part. working on gen_for_optimization
#       output must be redirected to config.output_dir/output_{problem_id}  
#       to fetch compilation error msg, I printed out "compilation_start" at the start of eval_kernel_against_ref
#       and "compilation_end" when it returns. extract_error_msg() function searches for the last appearance of "compilation_end"
#       and "compilation_start" in the output file, and return the text (which is the compilation error msg) in between.
#        
# IDEAS: CoT (make LLMs add comments and explain what it does step by step) [DONE]
#       Temperature tuning. (alternate between high and low temp?) --> 
#       [IMPORTANT] still need to give the last history to avoid stuck in loop!! a serious issue especially debugging wmma
#       start over if iterations don't help?  --> have a "for s in range (max_samples)" outer loop [TODO]
#       less cringy and more concise prompts [DONE]
#       add common mistake reminders [DONE but always IN PROGRESS]
#       sample high level rec and few shot example [IN PROGRESS, only wmma now]
#       hw info [DONE], doesn't seem too helpful

# TODO: send max diff as correctness feedback
#       right now, optimization loop only tries out wmma, not nearly finished
#       more flexible logging in correctness loop, specifically, seperate correctness logs and optimization logs
#       

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_printoptions(precision=4, threshold=10)

class EvalConfig(Config):
    def __init__(self):
        self.dataset_src = REQUIRED
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = REQUIRED
        self.problem_id = REQUIRED
        self.eval_mode = "local"
        # self.gpu_arch = ["Ampere"]
        self.gpu_arch = ["Turing"]
        self.gpu_name = "T4"
        # self.server_type = "openai"
        self.server_type ="deepseek"
        # self.model_name = "o1-mini-2024-09-12"
        self.model_name = "deepseek-coder"
        self.max_tokens = 8192
        # self.temperature = 0.0
        self.high_temp = 1.0
        self.low_temp = 0.0
        self.kernels_dir = os.path.join(REPO_TOP_DIR, "results/kernels")

        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")
        self.output_dir = os.path.join(REPO_TOP_DIR, "results/outputs")
        self.baseline_path = os.path.join(REPO_TOP_DIR, f"results/timing/{self.gpu_name}/baseline_time_torch.json")
        self.verbose = True
        self.log = True
        self.debug = False
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True
        self.max_compile_fix_iterations = 3
        self.max_correctness_fix_iterations = 10
        self.max_optimization_iterations = 10
        self.max_optimization_samples = 1
    def verbose_logging(self):
        self.log = True
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


def gen_for_correctness(config, high_temp_server, low_temp_server, ref_arch_src, problem_name, max_iterations=-1, optimization_prompt=None):
     # 3. Iterative Kernel Generation and Evaluation
    if max_iterations == -1: 
        max_iterations = config.max_correctness_fix_iterations 
    custom_cuda = None
    kernel_exec_result = None
    compilation_success = False
    correctness_success = False
    final_code = None
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}: Generating and evaluating kernel")

        # Generate Prompt
        if iteration == 0:
            if optimization_prompt != None: 
                custom_cuda_prompt = optimization_prompt
            else:
                custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
        else:
            metadata = kernel_exec_result if (isinstance(kernel_exec_result, dict)) else kernel_exec_result.metadata
            error_metadata = "NONE"
            if not compilation_success:
                # print(f"Compilation failed in iteration {iteration}. Attempting to fix...")
                path = os.path.join(config.output_dir, f"output_{config.problem_id}")
                error_metadata = extract_error_msg(path)
                custom_cuda_prompt = prompt_fix_compile(ref_arch_src, custom_cuda, error_metadata)

            # NOTE: to gen only for compilation, take out this elif and correctness_success variable
            elif not correctness_success:
                # print(f"correctness failed in iteration {iteration}. Attempting to fix...")
                error_metadata =  metadata.get("correctness_issue") if not metadata.get("runtime_error") else metadata.get("runtime_error")
                custom_cuda_prompt = prompt_fix_correctness(ref_arch_src, custom_cuda, error_metadata)

        if config.debug and iteration != 0:
            print("--------DEBUG PRINT---------")
            print("KERNEL EXEC RESULT:")
            print(kernel_exec_result)
            print("ERROR_METADATA:")
            print(error_metadata)
            print("------END DEBUG PRINT------")
            sys.stdout.flush()


        # Log Prompt
        if config.log_prompt:
            prompt_filename = f"prompt_level_{config.level}_problem_{config.problem_id}_iter_{iteration}.txt"
            with open(os.path.join(config.logdir, prompt_filename), "w") as f:
                f.write(custom_cuda_prompt)

        # Query Server
        if iteration == 0 or (iteration > 3 and iteration % 2 == 0): # avoid deterministically stuck in the same issue
            custom_cuda = high_temp_server(custom_cuda_prompt)
        else:
            custom_cuda = low_temp_server(custom_cuda_prompt)


        # Log response 
        if config.log:
            kernel_filename = f"generated_kernel_level_{config.level}_problem_{config.problem_id}_iter_{iteration}_raw.py"
            with open(os.path.join(config.logdir, kernel_filename), "w") as f:
                f.write(custom_cuda)

        # extract code       
        custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
        assert custom_cuda is not None, f"Iteration {iteration + 1}: Custom CUDA code generation failed"

        # Log Generated Kernel
        if config.log:
            kernel_filename = f"generated_kernel_level_{config.level}_problem_{config.problem_id}_iter_{iteration}.py"
            with open(os.path.join(config.logdir, kernel_filename), "w") as f:
                f.write(custom_cuda)

        # Evaluate Kernel
        sys.stdout.flush()
        print("compilation_start")
        sys.stdout.flush()
        try:
            kernel_exec_result = eval_kernel_against_ref(
                ref_arch_src,
                custom_cuda,
                verbose=False,
                measure_performance=False,
                num_correct_trials=5,
                num_perf_trials=100
            )  

            # Check compilation success
            compilation_success = kernel_exec_result.compiled
            correctness_success = kernel_exec_result.correctness
            if compilation_success and correctness_success:
                final_code =  custom_cuda
                break
        except Exception as e:
            kernel_exec_result = {"compiled": False, "compilation_error": str(e)}
            compilation_success = False
            print(f"Compilation failed with error: {str(e)}")
        sys.stdout.flush()
        print("compilation_end")
        sys.stdout.flush()
        if compilation_success and correctness_success:
            final_code =  custom_cuda
            break

        # Log Evaluation Result
        if config.log:
            eval_filename = f"eval_result_level_{config.level}_problem_{config.problem_id}.txt"
            mode = "w" if iteration == 0 else "a"
            with open(os.path.join(config.logdir, eval_filename), mode) as f:
                f.write(f"Problem Name: {problem_name}\n")
                f.write(f"Iteration: {iteration + 1}\n")
                f.write(f"Compilation Success: {compilation_success}\n")
                f.write(f"Correctness Success: {correctness_success}\n")
                f.write(str(kernel_exec_result) + "\n")

    # Log Final Result
    if config.log:
        with open(os.path.join(config.logdir, f"result_level_{config.level}_problem_{config.problem_id}.txt"), "w") as f:
            f.write(f"Problem Name: {problem_name}\n")
            f.write(f"Final Result after {iteration+1} iterations\n")
            f.write(f"Compilation Success: {compilation_success}\n")
            f.write(f"Correctness Success: {correctness_success}\n")
            f.write(str(kernel_exec_result))
    return final_code, kernel_exec_result



def gen_for_optimization(config, initial_cuda, high_temp_server, low_temp_server, ref_arch_src, problem_name): 
     # 3. Iterative Kernel Generation and Evaluation
   
    max_samples = config.max_optimization_samples
    baseline_stats = fetch_baseline_time(config.level, problem_name, config.baseline_path)
    recommendations = (
    ("tensorcore", "utilize tensorcore wmma instruction when appropriate"),
    )   
    final_code = initial_cuda
    try:
        initial_exec_result = eval_kernel_against_ref(
            ref_arch_src,
            initial_cuda,
            verbose=False,
            measure_performance=True,
            num_correct_trials=5,
            num_perf_trials=100
        )  
    except Exception as e:
        initial_exec_result = {"compiled": False, "compilation_error": str(e)}
    # IDEA: 
    # [IMPORTANT] definitely sample tensor core few-shot examples!!
    # work iteratively on several samples?
    # prompt with question of "what do you think could be optimized?"
    # TODO
    # a container in form of dict(code str, runtime), entries ordered by runtime
    # save the original correct code and initial runtime
    # initial prompt specifically for optimization, include hardware info
    # iterative prompt with previous attempts, ranked by runtime
    # iterative prompt samples optimization few shot examples and hardware info
    # output container
    """
    con(size = 3)
    working_code
    for (max iterations)
       if not compiled or not correct: fix
       else: prompt for performance (con, initial_code)

       eval
       if compiled and correct: add to container
       else: assign to working_code
    """  
    # runtime stats access: 
    # kernel_exec_result.runtime = runtime_stats["mean"]
    # kernel_exec_result.runtime_stats = runtime_stats
            # check kernel_exec_result.metadata["error_during_performance"] first
    optimized_code = None
    
    for iteration in range(max_samples):
        print(f"Sample {iteration + 1}/{max_samples}: Optimizing kernel")

        shots = [recommendations[iteration][0]]
        sampled_rec = [recommendations[iteration][1]]
        print(baseline_stats)
        custom_cuda_prompt = prompt_for_optimization(ref_arch_src=ref_arch_src,
                                                    custom_cuda=initial_cuda,
                                                    cuda_runtime=initial_exec_result.runtime, 
                                                    torch_runtime=baseline_stats["mean"],
                                                    gpu_name=config.gpu_name, 
                                                    shots=shots, 
                                                    recommendation=sampled_rec)
        optimized_code, exec_result = gen_for_correctness(config, high_temp_server, low_temp_server, ref_arch_src, problem_name, max_iterations=-1, 
                        optimization_prompt=custom_cuda_prompt)
        if exec_result.runtime < initial_exec_result.runtime:
            final_code = optimized_code
       
    if config.log:
        if optimized_code:
            with open(os.path.join(config.logdir, f"optimize_result_level_{config.level}_problem_{config.problem_id}.txt"), "w") as f:
                f.write(f"Problem Name: {problem_name}\n")
                f.write(f"Final Result after {iteration+1} iterations\n")
                f.write(str(exec_result))
                f.write(f"initial exec result: \n")
                f.write(str(initial_exec_result))
                f.write(f"baseline stats: \n")
                f.write(str(baseline_stats))
        else: 
            with open(os.path.join(config.logdir, f"optimize_result_level_{config.level}_problem_{config.problem_id}.txt"), "w") as f:
                f.write(f"Problem Name: {problem_name}\n")
                f.write(f"No optimization succeeded after {iteration+1} iterations\n")
                f.write(f"initial exec result: \n")
                f.write(str(initial_exec_result))
                f.write(f"baseline stats: \n")
                f.write(str(baseline_stats))
    return final_code


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    print(f"Starting Eval with config: {config}")

    # Configurations
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    if config.gpu_arch:
        set_gpu_arch(config.gpu_arch)

    if config.log:
        os.makedirs(config.logdir, exist_ok=True)

    # Problem Checks
    num_problems = len(curr_level_dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")
    print(f"Start Generation + Evaluation for Level {config.level} Problem {config.problem_id}")
    assert config.problem_id <= num_problems, f"Problem ID {config.problem_id} out of range for Level {config.level}"

    # 1. Fetch Problem
    if config.dataset_src == "huggingface":
        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]
    elif config.dataset_src == "local":
        problem_idx_in_dataset = config.problem_id - 1
        ref_arch_path = curr_level_dataset[problem_idx_in_dataset]
        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)

    problem_number = int(problem_name.split("_")[0])
    assert problem_number == config.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"


    config.logdir = os.path.join(config.logdir, f"{config.problem_id}")
    os.makedirs(config.logdir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.kernels_dir, exist_ok=True)

    # 2. Create Inference Server
    high_temp_server = create_inference_server_from_presets(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.high_temp,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        time_generation=True
    )
    low_temp_server = create_inference_server_from_presets(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.low_temp,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        time_generation=True
    )
    # for i in range(max_samples)
    # correct_cuda, exec_result = gen_for_correctness(config, high_temp_server, low_temp_server, ref_arch_src, problem_name)
    # # return 
    # if correct_cuda:
    #     with open(os.path.join(config.kernels_dir, f"correct_level_{config.level}_problem_{config.problem_id}.py"), "w") as f:
    #         f.write(correct_cuda)
    #     gen_for_optimization(config = config, 
    #                         initial_cuda = correct_cuda,
                            # initial_exec_result = exec_result, 
    #                         high_temp_server = high_temp_server,
    #                         low_temp_server = low_temp_server, 
    #                         ref_arch_src = ref_arch_src, 
    #                         problem_name = problem_name
    #                         )

    # or
    correct_cuda = read_file(os.path.join(config.kernels_dir, f"correct_level_{config.level}_problem_{config.problem_id}.py"))
    gen_for_optimization(config = config, 
                    initial_cuda = correct_cuda,
                    # initial_exec_result = exec_result, 
                    high_temp_server = high_temp_server,
                    low_temp_server = low_temp_server, 
                    ref_arch_src = ref_arch_src, 
                    problem_name = problem_name
                    )

    # gen_for_optimization(config, correct_cuda, high_temp_server, low_temp_server, ref_arch_src, problem_name)



if __name__ == "__main__":
    main()
