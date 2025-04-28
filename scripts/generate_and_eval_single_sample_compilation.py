import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
from datasets import load_dataset
from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template, prompt_fix_compile
from src.utils import extract_first_code, query_server, set_gpu_arch, read_file, create_inference_server_from_presets

"""
Generate and evaluate a single sample
Easiest way to get started, to test a single problem for experimentation or debugging
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_printoptions(precision=4, threshold=10)

class EvalConfig(Config):
    def __init__(self):
        self.dataset_src = REQUIRED
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = REQUIRED
        self.problem_id = REQUIRED
        self.eval_mode = "local"
        self.gpu_arch = ["Ampere"]
        self.server_type = "openai"
        self.model_name = "o1-mini-2024-09-12"
        self.max_tokens = 4096
        self.temperature = 0.0
        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")
        self.verbose = True
        self.log = True
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True
        # New configuration for iterative fixing
        self.max_compile_fix_iterations = 3  # Maximum number of fix attempts

    def verbose_logging(self):
        self.log = True
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"

@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """
    Keep it simple: Generate and evaluate a single sample with iterative compilation fixes
    """
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
    inference_server = create_inference_server_from_presets(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        time_generation=True
    )

    # 3. Iterative Kernel Generation and Evaluation
    max_iterations = config.max_compile_fix_iterations
    custom_cuda = None
    kernel_exec_result = None
    compilation_success = False

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}: Generating and evaluating kernel")

        # Generate Prompt
        if iteration == 0:
            # Initial attempt: Use the standard prompt
            custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
        else:
            # Subsequent attempts: Use fix_compile prompt with error metadata
            print(f"Compilation failed in iteration {iteration}. Attempting to fix...")
            if isinstance(kernel_exec_result, dict) and "compilation_error" in kernel_exec_result:
                error_metadata = kernel_exec_result.get("compilation_error", "No error details provided")
            else:
                error_metadata = str(kernel_exec_result) if kernel_exec_result else "Unknown compilation error"
            custom_cuda_prompt = prompt_fix_compile(ref_arch_src, custom_cuda, error_metadata)

        # Log Prompt
        if config.log_prompt:
            prompt_filename = f"prompt_level_{config.level}_problem_{config.problem_id}_iter_{iteration}.txt"
            with open(os.path.join(config.logdir, prompt_filename), "w") as f:
                f.write(custom_cuda_prompt)

        # Query Server
        custom_cuda = inference_server(custom_cuda_prompt)
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
                verbose=config.verbose,
                measure_performance=True,
                num_correct_trials=5,
                num_perf_trials=100
            )
            # Assume kernel_exec_result indicates compilation success (adjust based on actual return value)
            compilation_success = kernel_exec_result.get("compilation_success", True) if isinstance(kernel_exec_result, dict) else True
            if compilation_success:
                print(f"Iteration {iteration + 1}: Kernel compiled successfully")
                break
        except Exception as e:
            kernel_exec_result = {"compilation_success": False, "compilation_error": str(e)}
            compilation_success = False
            print(f"Iteration {iteration + 1}: Compilation failed with error: {str(e)}")

        # Log Evaluation Result
        if config.log:
            eval_filename = f"eval_result_level_{config.level}_problem_{config.problem_id}_iter_{iteration}.txt"
            with open(os.path.join(config.logdir, eval_filename), "a") as f:
                f.write(f"Problem Name: {problem_name}\n")
                f.write(f"Iteration: {iteration + 1}\n")
                f.write(f"Compilation Success: {compilation_success}\n")
                f.write(str(kernel_exec_result) + "\n")

    # Final Result
    if compilation_success:
        print(f"Evaluation result for level {config.level} problem {config.problem_id}:\n{kernel_exec_result}")
    else:
        print(f"Failed to generate a compilable kernel after {max_iterations} iterations")
        kernel_exec_result = {"compilation_success": False, "error": f"Failed after {max_iterations} iterations"}

    # Log Final Result
    if config.log:
        with open(os.path.join(config.logdir, f"eval_result_level_{config.level}_problem_{config.problem_id}.txt"), "a") as f:
            f.write(f"Problem Name: {problem_name}\n")
            f.write(f"Final Result after {max_iterations} iterations\n")
            f.write(str(kernel_exec_result))

if __name__ == "__main__":
    main()
