#!/bin/bash

: ' 
leaderboard:
 1: 0.17, 0.13, 0.12*
21: 1.00, 0.71, 0.67
24: 0.03, 0.00*
40: 0.07, 0.01*
51: 0.90, 0.50
61: 
74:
'
problem_ids=(1 21 24 40 51 61 74)


for i in "${problem_ids[@]}"; do
    echo "Running problem_id=$i..."
    python3 scripts/generate_and_eval_single_sample_comp2.py dataset_src="huggingface" level=1 problem_id=$i > results/outputs/output_$i
done
