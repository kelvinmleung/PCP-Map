import pickle
import numpy as np

input_path = 'ens/177.p'
output_path = 'ens/177.npy'

# Load the .p file
with open(input_path, 'rb') as file:
    data = pickle.load(file)
if not isinstance(data, np.ndarray):
    data = np.array(data)

# Save the data as a .npy file
np.save(output_path, data)

print(f"Data from {input_path} has been saved to {output_path}")