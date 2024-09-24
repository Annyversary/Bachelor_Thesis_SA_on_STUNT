import os
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder

# Path to the ARFF file
arff_file = 'data/dna/dna.arff'

# Load the ARFF file
data, meta = arff.loadarff(arff_file)

# Convert the ARFF data to a pandas DataFrame
df = pd.DataFrame(data)

# Optional: Convert byte-strings to regular strings (if necessary)
for column in df.select_dtypes(['object']).columns:
    if isinstance(df[column].iloc[0], bytes):  # Check if the column entries are byte-strings
        df[column] = df[column].str.decode('utf-8')

# Separate the features (X) and the target (y)
X = df.drop('class', axis=1).values  # Features (all columns except 'class')
y = df['class'].values  # Target variable (class column)

# One-hot encoding of the features (since they are nominal)
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X)

# Convert the target variable to integers (if necessary)
y = np.array(pd.Categorical(y).codes, dtype=np.int32)

# Split into 80% training and 20% test
train_size = int(0.8 * len(X_encoded))
X_train, X_test = X_encoded[:train_size], X_encoded[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Ensure the directory exists
output_dir = 'data/dna/data'
os.makedirs(output_dir, exist_ok=True)

# Save the split datasets as .npy files
np.save(os.path.join(output_dir, 'xtrain.npy'), X_train)
np.save(os.path.join(output_dir, 'ytrain.npy'), y_train)
np.save(os.path.join(output_dir, 'xtest.npy'), X_test)
np.save(os.path.join(output_dir, 'ytest.npy'), y_test)

print("The files have been successfully split and saved.")
