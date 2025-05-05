import os
import torch
import importlib
import pandas as pd
from datasets import Dataset
import types
import json
from datasets import load_dataset
from torch.utils.cpp_extension import load
from torch.utils.cpp_extension import include_paths

NUM_SAMPLES = 3

NUM_TRIALS = 3
NUM_WARMUP = 10
NUM_PERF_TRIALS = 50
START_FROM = 0
cuda_dir = "cuda"


dataset = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive")  


# Create working directories
os.makedirs("cuda", exist_ok=True)

device = torch.device("cuda:0")
# os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5" # change this for different GPU
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion

def time_execution(
    fn: callable, args):

   # Warm ups
   for _ in range(NUM_WARMUP):
      fn(*args)
      torch.cuda.synchronize(device=device)

   # Actual trials
   avg_time = 0.0
   for trial in range(NUM_PERF_TRIALS):
      # create event marker default is not interprocess
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)

      start_event.record()
      fn(*args)
      end_event.record()

      # Synchronize to ensure the events have completed
      torch.cuda.synchronize(device=device)
      time = start_event.elapsed_time(end_event) 
      avg_time += time / NUM_PERF_TRIALS
   return avg_time

def test_level(l):
   def select_dataset(ld):
      selected = ld.filter(lambda x: x["CUDA_Speedup_Native"] > 1.0)
      df = pd.DataFrame(selected)
      df_sorted = df.sort_values(["Task_ID", "CUDA_Speedup_Native"], ascending=[True, False])
      top3_per_task = df_sorted.groupby("Task_ID").head(NUM_SAMPLES)
      top3_per_task.reset_index(drop=True, inplace=True)  

      return Dataset.from_pandas(top3_per_task)
   tested_count = 0
   failed_count = 0
   level_dataset= dataset[f"level_{l}"]
   selected = select_dataset(level_dataset)
   
   for idx, entry in enumerate(selected):

      problem_name = entry['Op_Name']
      kernel_name = entry['Kernel_Name']
      level = entry['Level_ID']
      problem = entry['Task_ID']
      reported_correct = entry['Correct']
      reported_speedup = entry['CUDA_Speedup_Native']

      if problem < START_FROM: continue
      if not reported_correct: 
         continue # skip incorrect ones
      
      cuda_code = entry['CUDA_Code']
      torch_ref_code = entry['PyTorch_Code_Module']
      torch_func_code = entry['PyTorch_Code_Functional']
      expected_max_diff = entry['Max_Diff']

      cuda_filename = f"{cuda_dir}/{kernel_name}.cu"
      with open(cuda_filename, 'w') as f:
         f.write(cuda_code)

      # load custom cuda kernel
      try:
         custom_kernel = load(
            name= f"custom_kernel_{idx}",
            sources=[cuda_filename],   
            extra_include_paths=include_paths() ,
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3'],            
            verbose=False,
         )
      except Exception as e:
         failed_count += 1
         print(f"Level {level:<2} problem {problem:<3} {kernel_name[:30]:<30} build error: {str(e)}")
         continue
      
      # load torch
      def load_module_from_code(code_string):
         module = types.ModuleType("dynamic_module")
         exec(code_string, module.__dict__)
         return module
      torch_func_module = load_module_from_code(torch_func_code)
      torch_module = load_module_from_code(torch_ref_code)

      init_inputs_func = torch_func_module.get_init_inputs()
      init_inputs_func = [x.to(device) if isinstance(x, torch.Tensor) else x for x in init_inputs_func]

      init_inputs = torch_module.get_init_inputs()
      init_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in init_inputs]

      torch_model = torch_module.Model(*init_inputs).to(device)
      torch_model.eval()  


      # check correctness
      correct = True
      skip = False

      for i in range(NUM_TRIALS):
         inputs = torch_module.get_inputs()
         inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

         with torch.no_grad():
            # get cuda output
            try:
               actual = custom_kernel.forward(*(inputs + init_inputs_func))
            except Exception as e:
               if "incompatible function arguments" in str(e):
                  print(f"Level {level:<2} problem {problem:<3} {kernel_name[:30]:<30} incompatible function arguments. Skipped")
                  skip = True
               else:
                  correct = False
                  print(f"Level {level:<2} problem {problem:<3} {kernel_name[:30]:<30} runtime error: {str(e)}")
               break

            # deep copy output
            actual_sealed = actual.detach().clone()
            expected = torch_model.forward(*inputs)
            if not torch.allclose(expected, actual_sealed, atol=1e-02, rtol=1e-02): 
               correct = False
               max_diff = torch.max(torch.abs(expected - actual_sealed)).item()
               print(f"Level {level:<2} problem {problem:<3} {kernel_name[:30]:<30} incorrect. expected max diff={expected_max_diff:.2e}, actual max diff={max_diff:.2e}")
               break
      if skip: continue
      # record stats
      tested_count += 1
      if not correct:
         failed_count += 1
      else:
         try:
            torch.cuda.synchronize(device=device)
            inputs = torch_module.get_inputs()
            inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
            torch.cuda.synchronize(device=device)

            cuda_time = time_execution(custom_kernel.forward, inputs + init_inputs_func)
            torch_time = time_execution(torch_model, inputs)

         except Exception as e:
            print(f"Level {level:<2} problem {problem:<3} {kernel_name[:30]:<30} correct. runtime error when measuing perf: {str(e)}")
            cuda_time = -1.0
         if (cuda_time > 0):
            print(f"Level {level:<2} problem {problem:<3} {kernel_name[:30]:<30} correct. reported speedup={reported_speedup:<6.3f} actual speedup={(torch_time / cuda_time):<6.3f} cuda_time={cuda_time:<6.3f} torch_native_time={torch_time:<7.3f}")
         elif (cuda_time == 0):
            print(f"Level {level:<2} problem {problem:<3} {kernel_name[:30]:<30} correct. reported speedup={reported_speedup:<6.3f} actual speedup=NaN    cuda_time={cuda_time:<6.3f} torch_native_time={torch_time:<7.3f}")
         
   print(f"Level {l} result: failed {failed_count}/{tested_count} top performing kernels")
   return tested_count, failed_count

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l", type=int, default=1)
args = parser.parse_args()
assert args.l >= 1 and  args.l <= 3, "Please enter a valid level 1-3"

test_level(args.l)


                   
       