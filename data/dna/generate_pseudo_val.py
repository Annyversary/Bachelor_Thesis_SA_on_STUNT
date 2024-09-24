import numpy as np
from sklearn.cluster import KMeans
import os

np.random.seed(0)

# Load the training data
x = np.load('data/dna/data/xtrain.npy')
y = np.load('data/dna/data/ytrain.npy')

# Determine the number of training examples (80% of the data)
num_train = int(len(x) * 0.8)

# Create a random permutation of the indices of the data
idx = np.random.permutation(len(x))

# Split the indices into training and validation indices
train_idx = idx[:num_train]
val_idx = idx[num_train:]

# Split the data based on the indices into training and validation data
train_x = x[train_idx]
val_x = x[val_idx]

# Save the training and validation data
output_dir = 'data/dna/data/'
np.save(os.path.join(output_dir, 'train_x.npy'), train_x)
np.save(os.path.join(output_dir, 'val_x.npy'), val_x)

# Create a KMeans model with the number of clusters (e.g., 3)
model = KMeans(n_clusters=3)

# Fit the model to the validation data
model.fit(val_x)

# Predict the cluster labels for the validation data
labels = model.predict(val_x)

# Save the predicted cluster labels as pseudo-labels for the validation data
np.save(os.path.join(output_dir, 'pseudo_val_y.npy'), labels)
