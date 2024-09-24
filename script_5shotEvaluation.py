import os
from datetime import datetime
import csv
from common.args import parse_args  # Import function to access arguments
import subprocess
import numpy as np

# Parse the arguments
args = parse_args()

# List of seeds to use
seeds = range(10)

# List of datasets to use
datasets = ["income", "diabetes", "dna"]

# List of values for num_ways and query_shots to test
num_ways_list = [5, 10]
query_shots_list = [10, 20]

# Set shot_num to 5
shot_num = 5

# Current date for the log folder
current_date = datetime.now().strftime("%y%m%d")  # Note the format yymmdd

# Create the results directory
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

print("Starting 5-shot evaluation script...")

# Loop over all datasets
for dataset in datasets:
    # Loop over all num_ways
    for num_ways in num_ways_list:
        # Loop over all query_shots
        for query_shots in query_shots_list:
            # Create the name for the CSV file
            results_file_name = f"{current_date}_{dataset}_5shotEval_{num_ways}way_{shot_num}shot_{query_shots}query.csv"
            results_file = os.path.join(results_dir, results_file_name)

            # Write the header to the CSV file
            with open(results_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ["Seed"] + [f"Seed_{seed}" for seed in seeds] + ["Average", "Standard Deviation"]
                writer.writerow(header)

            results_row = ["Accuracy"]

            # Loop over all seeds
            accuracies = []
            for seed in seeds:
                # Set the log directory for the current seed
                log_dir = os.path.join("logs", f"{current_date}_{dataset}_mlp_protonet_{num_ways}way_{shot_num}shot_{query_shots}query_seed{seed}")

                # Set the training for 5-shot with the current values of num_way, shot_num, and query_shots
                train_command = (
                    f"python main.py --mode protonet --model mlp --dataset {dataset} --seed {seed} "
                    f"--num_ways {num_ways} --num_shots 5 --num_shots_test {query_shots} --outer_steps 10000"
                )

                print(f"Running training command: {train_command}")
                result = subprocess.run(train_command, shell=True, text=True)
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr)

                # Create the path to the saved model
                model_path = os.path.join(log_dir, "best.model")

                # Check if the model exists
                if not os.path.exists(model_path):
                    print(f"Model not found: {model_path}. Skipping evaluation for this configuration.")
                    results_row.append('N/A')
                    continue

                # Run the evaluation and capture the output
                eval_command = (
                    f"python eval.py --data_name {dataset} --shot_num {shot_num} --seed {seed} "
                    f"--load_path {model_path}"
                )
                print(f"Running evaluation command: {eval_command}")
                result = subprocess.run(eval_command, shell=True, capture_output=True, text=True)
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr)

                raw_result = result.stdout.strip()
                print(f"Raw result for {dataset} with seed {seed}: {raw_result}")  # Debug output

                # Extract the result from the output
                try:
                    result_value = float(raw_result.split(" ")[-1])
                    accuracies.append(result_value)
                    results_row.append(result_value)
                except (ValueError, IndexError):
                    results_row.append('N/A')
                    print(f"Failed to parse result for {dataset} with seed {seed}. Raw result was: {raw_result}")

            # Calculate the average and standard deviation
            if accuracies:
                average = np.mean(accuracies)
                std_dev = np.std(accuracies)
            else:
                average = 'N/A'
                std_dev = 'N/A'

            # Add the average and standard deviation to the row
            results_row.extend([average, std_dev])

            # Write the result to the CSV file
            with open(results_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(results_row)

            print(f"Results saved to {results_file}")
