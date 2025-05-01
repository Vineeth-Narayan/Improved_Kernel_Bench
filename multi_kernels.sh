#!/bin/bash

problem_ids=(1 21 24 40 51)

for i in "${problem_ids[@]}"; do
    echo "Running problem_id=$i..."
    python3 scripts/generate_and_eval_single_sample_comp2.py dataset_src="huggingface" level=1 problem_id=$i > results/outputs/output_$i
done
