import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
from datasets import load_dataset
from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref, fetch_baseline_time
from src.prompt_constructor import gen_first_prompt_correctness, prompt_fix_compile, prompt_fix_correctness, prompt_for_optimization
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

# TODO:  right now, optimization loop only tries out wmma, not nearly finished
#       more flexible logging in correctness loop, specifically, seperate correctness logs and optimization logs
#       

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORRECTNESS_SAMPLES = 3
CORRECTNESS_ITERATIONS = 10
OPTIMIZATION_SAMPLES = 3
OPTIMIZATION_ITERATIONS = 20


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
        # self.max_compile_fix_iterations = 3
        # self.max_correctness_fix_iterations = 10
        # self.max_optimization_iterations = 10
        # self.max_optimization_samples = 1
    def verbose_logging(self):
        self.log = True
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"

def get_fix_prompts(config, kernel_exec_result, ref_arch_src, custom_cuda, compiled, correct):
    metadata = kernel_exec_result if (isinstance(kernel_exec_result, dict)) else kernel_exec_result.metadata
    error_metadata = "NONE"
    if not compiled:
        path = os.path.join(config.output_dir, f"output_{config.problem_id}")
        error_metadata = extract_error_msg(path) + "\n" + metadata["compilation_error"]
        custom_cuda_prompt = prompt_fix_compile(ref_arch_src, custom_cuda, error_metadata)

    if not correct:
        metadata = kernel_exec_result.metadata
        if not "runtime_error" in metadata:
            error_metadata =  f"{metadata.get("correctness_issue")}. Max differences of each trial: {metadata.get("max_difference")}. "
        else:
            error_metadata = metadata.get("runtime_error")
        error_metadata =  metadata.get("correctness_issue") if not metadata.get("runtime_error") else metadata.get("runtime_error")
        custom_cuda_prompt = prompt_fix_correctness(ref_arch_src, custom_cuda, error_metadata)

    if config.debug:
        print("--------DEBUG PRINT---------")
        print("KERNEL EXEC RESULT:")
        print(kernel_exec_result)
        print("ERROR_METADATA:")
        print(error_metadata)
        print("------END DEBUG PRINT------")
        sys.stdout.flush()
    return custom_cuda_prompt
        
def gen_for_correctness_single_sample(config, high_temp_server, low_temp_server, ref_arch_src, problem_name, max_iterations, log_dir, optimization_prompt=None):
     # 3. Iterative Kernel Generation and Evaluation
    custom_cuda = None
    kernel_exec_result = None
    compiled = False
    correct = False
    final_code = None
    final_runtime = -1
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}: Generating and evaluating kernel")

        # Generate Prompt
        if iteration == 0:
            if optimization_prompt != None: 
                custom_cuda_prompt = optimization_prompt
            else:
                custom_cuda_prompt = gen_first_prompt_correctness(ref_arch_src)
        else:
            custom_cuda_prompt = get_fix_prompts(config, kernel_exec_result, ref_arch_src, custom_cuda, compiled, correct)

        # Log Prompt
        if config.log_prompt:
            prompt_filename = f"prompt_level_{config.level}_problem_{config.problem_id}_iter_{iteration}.txt"
            with open(os.path.join(log_dir, prompt_filename), "w") as f:
                f.write(custom_cuda_prompt)

        # Query Server
        if iteration == 0 or (iteration > 3 and iteration % 2 == 0): # avoid deterministically stuck in the same issue
            custom_cuda = high_temp_server(custom_cuda_prompt)
        else:
            custom_cuda = low_temp_server(custom_cuda_prompt)


        # Log response 
        if config.log:
            kernel_filename = f"generated_kernel_level_{config.level}_problem_{config.problem_id}_iter_{iteration}_raw.py"
            with open(os.path.join(log_dir, kernel_filename), "w") as f:
                f.write(custom_cuda)

        # extract code       
        custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
        assert custom_cuda is not None, f"Iteration {iteration + 1}: Custom CUDA code generation failed"

        # Log Generated Kernel
        if config.log:
            kernel_filename = f"generated_kernel_level_{config.level}_problem_{config.problem_id}_iter_{iteration}.py"
            with open(os.path.join(log_dir, kernel_filename), "w") as f:
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
                measure_performance=True,
                num_correct_trials=5,
                num_perf_trials=100
            )  

            # Check compilation success
            compiled = kernel_exec_result.compiled
            correct = kernel_exec_result.correctness
        except Exception as e:
            kernel_exec_result = {"compiled": False, "compilation_error": str(e)}
            compiled = False
            currect = False
            print(f"Compilation failed with error: {str(e)}")
        sys.stdout.flush()
        print("compilation_end")
        sys.stdout.flush()

        # Log Evaluation Result
        if config.log:
            eval_filename = f"eval_result_level_{config.level}_problem_{config.problem_id}.txt"
            mode = "w" if iteration == 0 else "a"
            with open(os.path.join(log_dir, eval_filename), mode) as f:
                f.write(f"Problem Name: {problem_name}\n")
                f.write(f"Iteration: {iteration + 1}\n")
                f.write(f"Compilation Success: {compiled}\n")
                f.write(f"Correctness Success: {correct}\n")
                f.write(str(kernel_exec_result) + "\n")
        if compiled and correct:
            final_code =  custom_cuda
            if not "error_during_performance" in kernel_exec_result: 
                final_runtime = kernel_exec_result.runtime
            iteration += 1
            break

    # Log Final Result
    if config.log:
        with open(os.path.join(log_dir, f"result_level_{config.level}_problem_{config.problem_id}.txt"), "w") as f:
            f.write(f"Problem Name: {problem_name}\n")
            f.write(f"Final Result after {iteration+1} iterations\n")
            f.write(f"Compilation Success: {compiled}\n")
            f.write(f"Correctness Success: {correct}\n")
            f.write(str(kernel_exec_result))
    return final_code, iteration, final_runtime

def gen_for_correctness(config, high_temp_server, low_temp_server, ref_arch_src, problem_name):
    for sample in range(CORRECTNESS_SAMPLES):
        log_dir = os.path.join(config.logdir, f"correctness_sample_{sample}")
        os.makedirs(log_dir, exist_ok=True)
        correct_cuda, _, runtime = gen_for_correctness_single_sample(config=config, 
                                                       high_temp_server=high_temp_server, 
                                                       low_temp_server=low_temp_server, 
                                                       ref_arch_src=ref_arch_src, 
                                                       problem_name=problem_name, 
                                                       max_iterations=CORRECTNESS_ITERATIONS,
                                                       log_dir=log_dir)
        if correct_cuda:
            return correct_cuda, runtime

    
        

def gen_for_optimization(config, initial_cuda, initial_runtime, baseline_runtime, high_temp_server, low_temp_server, ref_arch_src, problem_name): 
     # 3. Iterative Kernel Generation and Evaluation
   
    max_samples = OPTIMIZATION_SAMPLES
    max_iterations = OPTIMIZATION_ITERATIONS
    
    recommendations = (
    (None, None),
    ("tensorcore", "utilize tensorcore wmma instruction when appropriate"),
    )   
    
    best_cuda = initial_cuda
    if initial_runtime == -1:
        try:
            kernel_exec_result = eval_kernel_against_ref(
                ref_arch_src,
                best_cuda,
                verbose=False,
                measure_performance=True,
                num_correct_trials=5,
                num_perf_trials=100
            )  
            best_runtime = kernel_exec_result.runtime

        except Exception as e:
            best_runtime = 10000
    else:
        best_runtime = initial_runtime


    for sample in range(max_samples):
        shot = recommendations[sample % len(recommendations)][0]
        rec = recommendations[sample % len(recommendations)][1]
        for iteration in range (max_iterations):
            prompt = prompt_for_optimization(ref_arch_src=ref_arch_src, 
                                custom_cuda = best_cuda,
                                cuda_runtime = best_runtime, 
                                torch_runtime = baseline_runtime, 
                                gpu_name=config.gpu_name, 
                                shots=shot, 
                                recommendation=rec)
            log_dir = os.path.join(config.logdir, f"optimize_sample_{sample}")
            os.makedirs(log_dir, exist_ok=True)

            new_cuda, iterations_used, new_runtime = gen_for_correctness_single_sample(config=config, 
                                                            high_temp_server = high_temp_server, 
                                                            low_temp_server = low_temp_server, 
                                                            ref_arch_src = ref_arch_src, 
                                                            problem_name = problem_name, 
                                                            max_iterations= max_iterations - iteration, 
                                                            optimization_prompt=prompt,
                                                            log_dir = log_dir)
            iteration += iterations_used

            if not new_cuda or new_runtime == -1: 
                continue
            if new_runtime < best_runtime:
                best_runtime = new_runtime
                best_cuda = new_cuda
                if config.log:
                    with open(os.path.join(log_dir, f"optimize_result_level_{config.level}_problem_{config.problem_id}.py"), "w") as f:
                        f.write(f"# Problem Name: {problem_name}\n")
                        f.write(f"# Final Result after {iteration+1} iterations\n")
                        f.write(f"# runtime: {best_runtime}\n")
                        f.write(f"# baseline: {baseline_runtime}\n")
                        f.write(best_cuda)
    return best_cuda, best_runtime


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
    
        
    baseline_stats = fetch_baseline_time(config.level, problem_name, config.baseline_path)
    baseline_runtime = baseline_stats["mean"]
    
    correct_cuda, correct_runtime = gen_for_correctness(config, high_temp_server, low_temp_server, ref_arch_src, problem_name)
    
    with open(os.path.join(config.logdir, f"correct_level_{config.level}_problem_{config.problem_id}.py"), "w") as f:
        f.write(f"# runtime: {correct_runtime}\n")        
        f.write(f"# basline: {baseline_runtime}\n")
        f.write(correct_cuda)
    # or
    # correct_cuda = read_file(os.path.join(config.kernels_dir, f"correct_level_{config.level}_problem_{config.problem_id}.py"))
    optimized_cuda, optimized_runtime = gen_for_optimization(config = config, 
                                                    initial_cuda = correct_cuda,
                                                    initial_runtime = correct_runtime, 
                                                    baseline_runtime = baseline_runtime,
                                                    high_temp_server = high_temp_server,
                                                    low_temp_server = low_temp_server, 
                                                    ref_arch_src = ref_arch_src, 
                                                    problem_name = problem_name
                                                    )
    with open(os.path.join(config.logdir, f"optimized_level_{config.level}_problem_{config.problem_id}.py"), "w") as f:
        f.write(f"# runtime: {optimized_runtime}\n")        
        f.write(f"# basline: {baseline_runtime}\n")
        f.write(optimized_cuda)


if __name__ == "__main__":
    main()
