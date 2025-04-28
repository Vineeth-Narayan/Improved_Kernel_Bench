import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
from datasets import load_dataset
from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template, prompt_fix_compile, prompt_generate_ex_with_CoT_template
from src.utils import extract_first_code, query_server, set_gpu_arch, read_file, create_inference_server_from_presets

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
        self.max_compile_fix_iterations = 5
        self.use_cot_prompt = True

    def verbose_logging(self):
        self.log = True
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"

@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    print(f"Starting Eval with config: {config}")

    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    if config.gpu_arch:
        set_gpu_arch(config.gpu_arch)

    if config.log:
        os.makedirs(config.logdir, exist_ok=True)

    num_problems = len(curr_level_dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")
    print(f"Start Generation + Evaluation for Level {config.level} Problem {config.problem_id}")
    assert config.problem_id <= num_problems, f"Problem ID {config.problem_id} out of range for Level {config.level}"

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

    inference_server = create_inference_server_from_presets(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        time_generation=True
    )

    max_iterations = config.max_compile_fix_iterations
    custom_cuda = None
    kernel_exec_result = None
    compilation_success = False

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}: Generating and evaluating kernel")

        if iteration == 0:
            if config.use_cot_prompt:
                custom_cuda_prompt = prompt_generate_ex_with_CoT_template(ref_arch_src, cot_example="ex_fuse_gelu")
            else:
                custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
        else:
            print(f"Compilation failed in iteration {iteration}. Attempting to fix...")
            error_metadata = kernel_exec_result.get("compilation_error", "No error details provided") if isinstance(kernel_exec_result, dict) else str(kernel_exec_result)
            error_metadata = error_metadata[:1000] if len(error_metadata) > 1000 else error_metadata
            custom_cuda_prompt = prompt_fix_compile(ref_arch_src, custom_cuda, error_metadata)

        if config.log_prompt:
            prompt_filename = f"prompt_level_{config.level}_problem_{config.problem_id}_iter_{iteration}.txt"
            with open(os.path.join(config.logdir, prompt_filename), "w") as f:
                f.write(custom_cuda_prompt)

        try:
            custom_cuda = inference_server(custom_cuda_prompt)
            print(f"Iteration {iteration + 1}: Raw LLM response length: {len(custom_cuda)}")
            if config.log:
                with open(os.path.join(config.logdir, f"raw_response_level_{config.level}_problem_{config.problem_id}_iter_{iteration}.txt"), "w") as f:
                    f.write(custom_cuda)
        except Exception as e:
            print(f"Iteration {iteration + 1}: Inference server failed: {str(e)}")
            kernel_exec_result = {"compiled": False, "compilation_error": f"Inference server error: {str(e)}"}
            break

        custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
        if custom_cuda is None:
            print(f"Iteration {iteration + 1}: Failed to extract code from response")
            kernel_exec_result = {"compiled": False, "compilation_error": "No valid code extracted"}
            continue

        if config.log:
            kernel_filename = f"generated_kernel_level_{config.level}_problem_{config.problem_id}_iter_{iteration}.py"
            with open(os.path.join(config.logdir, kernel_filename), "w") as f:
                f.write(custom_cuda)

        try:
            kernel_exec_result = eval_kernel_against_ref(
                ref_arch_src,
                custom_cuda,
                verbose=config.verbose,
                measure_performance=True,
                num_correct_trials=5,
                num_perf_trials=100
            )
            print(f"Iteration {iteration + 1}: eval_kernel_against_ref result: {kernel_exec_result}")
            compilation_success = kernel_exec_result.get("compiled", False) if isinstance(kernel_exec_result, dict) else False
            if compilation_success:
                print(f"Iteration {iteration + 1}: Kernel compiled successfully")
                break
            else:
                print(f"Iteration {iteration + 1}: Compilation failed (result indicates compiled=False)")
                if config.log and "compilation_error" in kernel_exec_result:
                    with open(os.path.join(config.logdir, f"compiler_output_level_{config.level}_problem_{config.problem_id}_iter_{iteration}.txt"), "w") as f:
                        f.write(str(kernel_exec_result["compilation_error"]))
        except Exception as e:
            kernel_exec_result = {"compiled": False, "compilation_error": str(e)}
            compilation_success = False
            print(f"Iteration {iteration + 1}: Compilation failed with error: {str(e)}")
            if config.log:
                with open(os.path.join(config.logdir, f"compiler_output_level_{config.level}_problem_{config.problem_id}_iter_{iteration}.txt"), "w") as f:
                    f.write(str(e))

        if config.log:
            eval_filename = f"eval_result_level_{config.level}_problem_{config.problem_id}_iter_{iteration}.txt"
            with open(os.path.join(config.logdir, eval_filename), "w") as f:
                f.write(f"Problem Name: {problem_name}\n")
                f.write(f"Iteration: {iteration + 1}\n")
                f.write(f"Compilation Success: {compilation_success}\n")
                f.write(str(kernel_exec_result) + "\n")

    if compilation_success:
        print(f"Evaluation result for level {config.level} problem {config.problem_id}:\n{kernel_exec_result}")
    else:
        print(f"Failed to generate a compilable kernel after {max_iterations} iterations")
        kernel_exec_result = kernel_exec_result or {"compiled": False, "error": f"Failed after {max_iterations} iterations"}

    if config.log:
        with open(os.path.join(config.logdir, f"eval_result_level_{config.level}_problem_{config.problem_id}.txt"), "a") as f:
            f.write(f"Problem Name: {problem_name}\n")
            f.write(f"Final Result after {max_iterations} iterations\n")
            f.write(f"Compilation Success: {compilation_success}\n")
            f.write(str(kernel_exec_result))

if __name__ == "__main__":
    main()
