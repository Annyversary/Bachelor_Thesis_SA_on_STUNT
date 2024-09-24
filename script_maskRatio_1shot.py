import os
from datetime import datetime
import csv
import numpy as np
from common.args import parse_args
from data.dataset import get_meta_dataset

def save_results_to_csv(results_file, seeds, accuracies):
    """Saves the results to the CSV file."""
    # Calculate mean and standard deviation
    if accuracies:
        average = np.mean([a for a in accuracies if a != 'N/A'])
        std_dev = np.std([a for a in accuracies if a != 'N/A'])
    else:
        average = 'N/A'
        std_dev = 'N/A'

    # Add the mean and standard deviation to the row
    results_row = accuracies + [average, std_dev]

    # Write the result to the CSV file
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(results_row)

def run_training_and_evaluation(dataset, r1, r2, num_shots, num_query, num_way, seeds, current_date):
    """Runs training and evaluation for a specific dataset, r1, r2, and seed."""
    # Create the name of the CSV file
    results_file_name = f"{current_date}_{dataset}_r1_{r1}_r2_{r2}_1shotEval.csv"
    results_file = os.path.join("./results", results_file_name)

    # Write the header to the CSV file
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = [f"Seed_{seed}" for seed in seeds] + ["Average", "Standard Deviation"]
        writer.writerow(header)

    # List to store all seed accuracies
    accuracies = []

    # Loop over all seeds
    for seed in seeds:
        log_dir = os.path.join("logs", f"{current_date}_{dataset}_mlp_protonet_{num_way}way_{num_shots}shot_{num_query}query_r1_{r1}_r2_{r2}_seed{seed}")

        # Train the model
        train_command = (
            f"python main.py --mode protonet --model mlp --dataset {dataset} --seed {seed} "
            f"--num_shots {num_shots} --num_shots_test {num_query} --num_ways {num_way} --r1 {r1} --r2 {r2} --outer_steps 50"
        )

        print(f"Running training command: {train_command}")
        os.system(train_command)

        # Path to the saved model
        model_path = os.path.join(log_dir, "best.model")

        # Check if the model exists
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}. Skipping evaluation for this configuration.")
            accuracies.append('N/A')
            continue

        # Run the evaluation
        eval_command = (
            f"python eval.py --data_name {dataset} --shot_num {num_shots} --seed {seed} "
            f"--load_path {model_path}"
        )
        print(f"Running evaluation command: {eval_command}")
        result = os.popen(eval_command).read().strip()
        print(f"Raw result for {dataset} with seed {seed}: {result}")

        # Extract the last value as the result
        try:
            result_value = float(result.split()[-1])
            accuracies.append(result_value)
        except (ValueError, IndexError):
            accuracies.append('N/A')
            print(f"Failed to parse result for {dataset} with seed {seed}. Raw result was: {result}")

    # Save the results to the CSV file
    save_results_to_csv(results_file, seeds, accuracies)
    print(f"Results saved to {results_file}")

def main():
    # Parse the arguments
    args = parse_args()

    # List of seeds to use
    seeds = range(10)

    # List of datasets to use
    datasets = ["income", "diabetes", "dna"]

    # r1 and r2 values
    r1_values = [0.3, 0.5, 0.7, 0.9]
    r2_values = [0.3, 0.5, 0.7, 0.9]

    # Current date for the log folder
    current_date = datetime.now().strftime("%y%m%d")

    # Create results directory
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    print("Starting r1/r2 evaluation script...")

    # Loop over all datasets
    for dataset in datasets:
        # Set parameters based on the dataset
        if dataset in ["income", "dna"]:
            num_shots = 1
            num_query = 15
            num_way = 10
        elif dataset == "diabetes":
            num_shots = 1
            num_query = 15
            num_way = 5

        for r1 in r1_values:
            for r2 in r2_values:
                # Change to "!=" if r1 == r2 is needed
                if r1 >= r2:
                    continue  # Skip this combination as r1 must be less than r2
                # Run the training and evaluation
                run_training_and_evaluation(dataset, r1, r2, num_shots, num_query, num_way, seeds, current_date)

if __name__ == "__main__":
    main()
