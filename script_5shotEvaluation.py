import os
from datetime import datetime
import csv
from common.args import parse_args  # Importiere die Funktion, um auf die Argumente zuzugreifen
import subprocess
import numpy as np

# Parse die Argumente
args = parse_args()

# Liste der Seeds, die verwendet werden sollen
seeds = range(10)

# Liste der Datensätze, die verwendet werden sollen
datasets = ["dna"]

# Liste der Werte für num_ways und query_shots, die getestet werden sollen
num_ways_list = [10]
query_shots_list = [20]

# Festlegen des Wertes für shot_num auf 5
shot_num = 5

# Aktuelles Datum für den Log-Ordner
current_date = datetime.now().strftime("%y%m%d")  # Beachte das Format yymmdd

# Ergebnisordner erstellen
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

print("Starting 5-shot evaluation script...")

# Schleife über alle Datensätze
for dataset in datasets:
    # Schleife über alle num_ways
    for num_ways in num_ways_list:
        # Schleife über alle query_shots
        for query_shots in query_shots_list:
            # Erstelle den Namen der CSV-Datei
            results_file_name = f"{current_date}_{dataset}_5shotEval_{num_ways}way_{shot_num}shot_{query_shots}query.csv"
            results_file = os.path.join(results_dir, results_file_name)

            # Schreibe die Kopfzeile in die CSV-Datei
            with open(results_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ["Seed"] + [f"Seed_{seed}" for seed in seeds] + ["Average", "Standard Deviation"]
                writer.writerow(header)

            results_row = ["Accuracy"]

            # Schleife über alle Seeds
            accuracies = []
            for seed in seeds:
                # Setze das Log-Verzeichnis für den aktuellen Seed
                log_dir = os.path.join("logs", f"{current_date}_{dataset}_mlp_protonet_{num_ways}way_{shot_num}shot_{query_shots}query_seed{seed}")

                # Setze das Training für 5-shot mit den aktuellen Werten für num_way, shot_num und query_shots
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

                # Erstelle den Pfad zum gespeicherten Modell
                model_path = os.path.join(log_dir, "best.model")

                # Überprüfe, ob das Modell existiert
                if not os.path.exists(model_path):
                    print(f"Model not found: {model_path}. Skipping evaluation for this configuration.")
                    results_row.append('N/A')
                    continue

                # Führe die Evaluation durch und fange die Ausgabe ab
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
                print(f"Raw result for {dataset} with seed {seed}: {raw_result}")  # Debugging-Ausgabe

                # Extrahiere das Ergebnis aus der Ausgabe
                try:
                    result_value = float(raw_result.split(" ")[-1])
                    accuracies.append(result_value)
                    results_row.append(result_value)
                except (ValueError, IndexError):
                    results_row.append('N/A')
                    print(f"Failed to parse result for {dataset} with seed {seed}. Raw result was: {raw_result}")

            # Berechne den Durchschnitt und die Standardabweichung
            if accuracies:
                average = np.mean(accuracies)
                std_dev = np.std(accuracies)
            else:
                average = 'N/A'
                std_dev = 'N/A'

            # Füge den Durchschnitt und die Standardabweichung zur Zeile hinzu
            results_row.extend([average, std_dev])

            # Schreibe das Ergebnis in die CSV-Datei
            with open(results_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(results_row)

            print(f"Results saved to {results_file}")
