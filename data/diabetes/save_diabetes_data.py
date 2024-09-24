import os
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler

# Path to the ARFF file
arff_file = 'data/diabetes/diabetes.arff'

# Load the ARFF file
data, meta = arff.loadarff(arff_file)

# Convert the ARFF data to a pandas DataFrame
df = pd.DataFrame(data)

# Optional: Convert byte-strings to regular strings (if necessary)
for column in df.select_dtypes(['object']).columns:
    if isinstance(df[column].iloc[0], bytes):  # Check if the column entries are byte-strings
        df[column] = df[column].str.decode('utf-8')

# Separate the features (X) and the target (y)
X = df.drop('Outcome', axis=1).values  # Features (all columns except 'Outcome')
y = df['Outcome'].values  # Target variable (Outcome column)

# Safely convert the arrays to float32 (or another appropriate numeric type)
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Apply Min-Max scaling to the features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Split into 80% training and 20% test
train_size = int(0.8 * len(X_normalized))
X_train, X_test = X_normalized[:train_size], X_normalized[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Ensure the output directory exists
output_dir = 'data/diabetes/data'
os.makedirs(output_dir, exist_ok=True)

# Save the split and normalized datasets as .npy files
np.save(os.path.join(output_dir, 'xtrain.npy'), X_train)
np.save(os.path.join(output_dir, 'ytrain.npy'), y_train)
np.save(os.path.join(output_dir, 'xtest.npy'), X_test)
np.save(os.path.join(output_dir, 'ytest.npy'), y_test)

print("The normalized files have been successfully split and saved.")
