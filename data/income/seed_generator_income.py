import numpy as np
import os

class SeedGenerator:
    def __init__(self, data_size, num_train, output_dir):
        """
        Initializes the SeedGenerator.
        
        :param data_size: The total number of data points in the dataset.
        :param num_train: The number of data points to be used for training.
        :param output_dir: The directory where the .npy files will be saved.
        """
        self.data_size = data_size
        self.num_train = num_train
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate(self, seed):
        """
        Generates a new train_idx_<seed>.npy file based on the provided seed.
        
        :param seed: The seed based on which the training indices are generated.
        """
        # Use Generator and PCG64 for a better random number generator
        rng = np.random.default_rng(seed)
        all_indices = np.arange(self.data_size)
        rng.shuffle(all_indices)  # Shuffle the indices randomly

        train_indices = all_indices[:self.num_train]  # Select the first num_train indices

        # Save the indices as a .npy file
        output_path = os.path.join(self.output_dir, f'train_idx_{seed}.npy')
        np.save(output_path, train_indices)
        print(f'Saved {output_path}')

# Example usage
if __name__ == "__main__":
    # Given the number of instances in the dataset
    data_size = 24129  # Total size of the training dataset
    num_train = 10  # Example: number of data points to be used for training
    output_dir = 'data/income/index1'

    generator = SeedGenerator(data_size, num_train, output_dir)

    # Generate seeds from 10 to 99
    for seed in range(10, 100):
        generator.generate(seed)
