import pandas as pd
from scipy.io import arff

# Path to the DNA ARFF file
arff_file = 'data/dna/dna.arff'

# Load the ARFF file
data, meta = arff.loadarff(arff_file)

# Convert the ARFF data to a pandas DataFrame
df = pd.DataFrame(data)

# Optional: Convert byte-strings to regular strings (if necessary)
for column in df.select_dtypes(['object']).columns:
    if isinstance(df[column].iloc[0], bytes):  # Check if the column entries are byte-strings
        df[column] = df[column].str.decode('utf-8')

# Print the number of rows (instances) in the DataFrame
print(f"Number of instances in the dataset: {df.shape[0]}")

# Print the first few rows of the DataFrame to get an overview of the data
print(df.head())
