import os
from datetime import datetime
import csv
import numpy as np  # Import numpy for calculating average and standard deviation
from common.args import parse_args  # Import function to access arguments

# Parse the arguments
args = parse_args()

# List of seeds to use
seeds = range(10)

# List of datasets to use
datasets = ["income", "diabetes", "dna"]

# Training steps to evaluate and save the model
training_steps = [1250, 2500, 5000, 10000]

# Default command template for training
train_command_template = "python main.py --mode protonet --model mlp --dataset {dataset} --seed {seed} --outer_steps {steps} --num_shots 1 --num_shots_test {num_shots_test} --num_ways {num_ways}"

# Default command template for resuming training
resume_command_template = "python main.py --mode protonet --model mlp --dataset {dataset} --resume_path {resume_path} --seed {seed} --outer_steps {steps} --num_ways {num_ways}"

# Default command template for evaluation
eval_command_template = ("python eval.py --data_name {dataset} --shot_num 1 --seed {seed} "
                         "--load_path {load_path}")

# Current date for the log folder, excluding the first two digits of the year
current_date = datetime.now().strftime("%y%m%d")

# Create the results directory
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

# Loop over all datasets
for dataset in datasets:
    # Set num_ways based on the dataset
    if dataset == "diabetes":
        num_ways = 5
        num_shots_test = 15
    elif dataset in ["income", "dna"]:
        num_ways = 10
        num_shots_test = 15
    else:
        num_ways = args.num_ways  # Use default from arguments
    
    # Create the name for the CSV file
    results_file_name = f"{current_date}_{dataset}_baseline_TrainingSteps.csv"
    results_file = os.path.join(results_dir, results_file_name)

    # Write the header to the CSV file
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Step/Seed"] + [f"Seed_{seed}" for seed in seeds] + ["Average", "Standard Deviation"]
        writer.writerow(header)

    # Loop over all training steps for the current seed
    for step in training_steps:
        results_row = [f"Step_{step}"]  # Start the result row with the training step

        # List to store numeric results for this step (for average and standard deviation)
        numeric_results = []

        # Loop over all seeds
        for seed in seeds:
            # Set the log directory for the current seed
            log_dir = os.path.join("logs", f"{current_date}_{dataset}_mlp_protonet_{num_ways}way_{args.num_shots}shot_{args.num_shots_test}query_seed{seed}")

            if step == training_steps[0]:
                # Run the training command for the first step
                train_command = train_command_template.format(dataset=dataset, seed=seed, steps=step, num_shots_test=num_shots_test, num_ways=num_ways)
            else:
                # Resume training for subsequent steps
                resume_path = log_dir  # Directory with seed in the path structure
                train_command = resume_command_template.format(dataset=dataset, resume_path=resume_path, seed=seed, steps=step, num_ways=num_ways)

            print(f"Running training command: {train_command}")
            os.system(train_command)

            # Create the path to the saved model
            model_path = os.path.join(log_dir, "best.model")

            # Run the evaluation and capture the output
            eval_command = eval_command_template.format(dataset=dataset, seed=seed, load_path=model_path)
            print(f"Running evaluation command: {eval_command}")

            result = os.popen(eval_command).read().strip()
            print(f"Raw result for {dataset} with seed {seed} at step {step}: {result}")  # Debug output

            # Extract the result from the output
            try:
                result_value = float(result.split(" ")[-1])
            except (ValueError, IndexError):
                result_value = None  # Error handling if result cannot be extracted
                print(f"Failed to parse result for {dataset} with seed {seed} at step {step}. Raw result was: {result}")

            # Add the result for the current seed to the row
            results_row.append(result_value if result_value is not None else 'N/A')

            # If the result is numeric, add it to the list of numeric results
            if isinstance(result_value, float):
                numeric_results.append(result_value)

        # Calculate the average and standard deviation for this step
        if numeric_results:
            avg = np.mean(numeric_results)
            stddev = np.std(numeric_results)
        else:
            avg = 'N/A'
            stddev = 'N/A'

        # Add the average and standard deviation as columns
        results_row.append(avg)
        results_row.append(stddev)

        # Write the result row to the CSV file
        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(results_row)

    print(f"Results saved to {results_file}")
