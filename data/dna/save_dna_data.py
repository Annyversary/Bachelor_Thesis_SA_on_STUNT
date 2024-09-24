import os
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder

# Pfad zur ARFF-Datei
arff_file = 'data\dna\dna.arff'

# Lade die ARFF-Datei
data, meta = arff.loadarff(arff_file)

# Konvertiere die ARFF-Daten in ein pandas DataFrame
df = pd.DataFrame(data)

# Optional: Konvertiere byte-Strings in reguläre Strings (falls erforderlich)
for column in df.select_dtypes(['object']).columns:
    if isinstance(df[column].iloc[0], bytes):  # Prüfe, ob die Spalteneinträge byte-Strings sind
        df[column] = df[column].str.decode('utf-8')

# Trenne die Features (X) und das Ziel (y)
X = df.drop('class', axis=1).values  # Features (alle Spalten außer 'class')
y = df['class'].values  # Zielvariable (class-Spalte)

# One-Hot-Encoding der Features (da sie nominal sind)
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X)

# Konvertiere die Zielvariable in Integer (falls nötig)
y = np.array(pd.Categorical(y).codes, dtype=np.int32)

# Aufteilung in 80% Training und 20% Test
train_size = int(0.8 * len(X_encoded))
X_train, X_test = X_encoded[:train_size], X_encoded[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Stelle sicher, dass das Verzeichnis existiert
output_dir = 'data\dna\data'
os.makedirs(output_dir, exist_ok=True)

# Speichere die aufgeteilten Datensätze als .npy-Dateien
np.save(os.path.join(output_dir, 'xtrain.npy'), X_train)
np.save(os.path.join(output_dir, 'ytrain.npy'), y_train)
np.save(os.path.join(output_dir, 'xtest.npy'), X_test)
np.save(os.path.join(output_dir, 'ytest.npy'), y_test)

print("Die Dateien wurden erfolgreich aufgeteilt und gespeichert.")
