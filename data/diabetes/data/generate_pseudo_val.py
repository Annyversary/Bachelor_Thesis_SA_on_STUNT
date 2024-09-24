import numpy as np
from sklearn.cluster import KMeans

# Set the random seed for reproducibility
np.random.seed(0)

# Load the training data and labels
x = np.load('xtrain.npy')
y = np.load('ytrain.npy')

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
np.save('train_x.npy', train_x)
np.save('val_x.npy', val_x)

# Create a KMeans model with 2 clusters
model = KMeans(n_clusters=2)

# Fit the model to the validation data
model.fit(val_x)

# Predict the cluster labels for the validation data
labels = model.predict(val_x)

# Save the predicted cluster labels as pseudo-labels for the validation data
np.save('pseudo_val_y.npy', labels)
