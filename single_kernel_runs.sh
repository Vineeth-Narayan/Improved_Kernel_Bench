mkdir -p results/outputs/
python3 scripts/generate_single_problem.py dataset_src="huggingface" level=$1 problem_id=$2 > results/outputs/output_$1_$2
