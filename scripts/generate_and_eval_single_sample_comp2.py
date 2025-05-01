import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
from datasets import load_dataset
from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template, prompt_fix_compile, prompt_fix_correctness
from src.utils import extract_first_code, extract_error_msg, query_server, set_gpu_arch, read_file, create_inference_server_from_presets

"""
Generate and evaluate a single sample with iterative compilation fixes
"""
# NOTE: gen_for_correctness function is the main part
#       output must be redirected to config.output_dir/output{problem_id}  
#       for fetching compilation error msg, I printed out "compilation_start" at the start of eval_kernel_against_ref
#       and "compilation_end" when it returns. extract_error_msg() function searches for the last appearance of "compilation_end"
#       and "compilation_start" in the output file, and return the text (which is the compilation error msg) in between
#        
# IDEAS: CoT (make LLMs add comments and explain what it does step by step), [IN PROGRESS]
#       Temperature tuning. (preferably, high temp at iteration 1 and low temp in future iterations)  [DONE][though doesn't look useful]
#       start over if iterations don't help?
# TODO: less cringy and more concise prompts 
#       add common mistake reminders!

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
        # self.server_type = "openai"
        self.server_type ="deepseek"
        # self.model_name = "o1-mini-2024-09-12"
        self.model_name = "deepseek-coder"
        self.max_tokens = 8192
        # self.temperature = 0.0
        self.high_temp = 1.0
        self.low_temp = 0.0
        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")
        self.output_dir = os.path.join(REPO_TOP_DIR, "results/outputs")

        self.verbose = True
        self.log = True
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True
        self.max_compile_fix_iterations = 3
        self.max_correctness_fix_iterations = 10


    def verbose_logging(self):
        self.log = True
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


def gen_for_correctness(config, high_temp_server, low_temp_server, ref_arch_src, problem_name):
     # 3. Iterative Kernel Generation and Evaluation
    max_iterations = config.max_correctness_fix_iterations
    custom_cuda = None
    kernel_exec_result = None
    compilation_success = False
    correctness_success = False
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}: Generating and evaluating kernel")

        # Generate Prompt
        if iteration == 0:
            custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
        else:
            metadata = kernel_exec_result if (isinstance(kernel_exec_result, dict)) else kernel_exec_result.metadata
            error_metadata = "NONE"
            if not compilation_success:
                # print(f"Compilation failed in iteration {iteration}. Attempting to fix...")
                path = os.path.join(config.output_dir, f"output{config.problem_id}")
                error_metadata = extract_error_msg(path)
                custom_cuda_prompt = prompt_fix_compile(ref_arch_src, custom_cuda, error_metadata)

            # NOTE: to gen only for compilation, take out this elif and correctness_success variable
            elif not correctness_success:
                # print(f"correctness failed in iteration {iteration}. Attempting to fix...")
                error_metadata =  metadata.get("correctness_issue") if not metadata.get("runtime_error") else metadata.get("runtime_error")
                custom_cuda_prompt = prompt_fix_correctness(ref_arch_src, custom_cuda, error_metadata)

        if iteration != 0:
            print("--------DEBUG PRINT---------")
            print("KERNEL EXEC RESULT:")
            print(kernel_exec_result)
            print("ERROR_METADATA:")
            print(error_metadata)
            print("------END DEBUG PRINT------")

        # Log Prompt
        if config.log_prompt:
            prompt_filename = f"prompt_level_{config.level}_problem_{config.problem_id}_iter_{iteration}.txt"
            with open(os.path.join(config.logdir, prompt_filename), "w") as f:
                f.write(custom_cuda_prompt)

        # Query Server
        if iteration == 0:
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
                break
        except Exception as e:
            kernel_exec_result = {"compiled": False, "compilation_error": str(e)}
            compilation_success = False
            print(f"Compilation failed with error: {str(e)}")
        print("compilation_end")
        sys.stdout.flush()


        # Log Evaluation Result
        if config.log:
            eval_filename = f"eval_result_level_{config.level}_problem_{config.problem_id}_iter_{iteration}.txt"
            with open(os.path.join(config.logdir, eval_filename), "w") as f:
                f.write(f"Problem Name: {problem_name}\n")
                f.write(f"Iteration: {iteration + 1}\n")
                f.write(f"Compilation Success: {compilation_success}\n")
                f.write(str(kernel_exec_result) + "\n")

    # Final Result
    if compilation_success and correctness_success:
        print(f"Evaluation result for level {config.level} problem {config.problem_id}:\n{kernel_exec_result}")
    else:
        print(f"Failed to generate a correct kernel after {max_iterations} iterations")
        kernel_exec_result = kernel_exec_result or {"compiled": False, "error": f"Failed after {max_iterations} iterations"}

    # Log Final Result
    if config.log:
        with open(os.path.join(config.logdir, f"eval_result_level_{config.level}_problem_{config.problem_id}.txt"), "a") as f:
            f.write(f"Problem Name: {problem_name}\n")
            f.write(f"Final Result after {max_iterations} iterations\n")
            f.write(f"Compilation Success: {compilation_success}\n")
            f.write(f"Correctness Success: {compilation_success}\n")
            f.write(str(kernel_exec_result))


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

    gen_for_correctness(config, high_temp_server, low_temp_server, ref_arch_src, problem_name)

if __name__ == "__main__":
    main()
