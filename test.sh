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

mkdir -p results/outputs/
for i in "${problem_ids[@]}"; do
    echo "Running problem_id=$i..."
    nohup python3 scripts/generate_single_problem.py dataset_src="huggingface" level=1 problem_id=$i > results/outputs/output_1_$i 2>&1 &
done

