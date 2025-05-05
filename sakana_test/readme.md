## usage
1. Setup
```sh
./setup.sh
export TORCH_CUDA_ARCH_LIST="<your_arch>"
```
2. Run test over a level
```sh
python test.py -l <level_id>
```

To configure number of samples from each problem, change `NUM_SAMPLES` at the top of `test.py`

Test result is printed on stdout. 

Expect level 2 test to take a long time. 
