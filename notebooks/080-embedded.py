# %% TAKE DATA FROM UART, STANDARDIZE IT AN D PERFORM CLUSTERING
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
import serial

# %% global variables
timestamps = np.array([])
features_matrix = np.array([])
file_path = r"C:\Users\ariel\Documents\Courses\Tesi\Code\data\putty.log"  # Change this to your actual file path

# %% LOAD DATA

with open(file_path, 'r') as file:
    for line in file:
        data = line.strip()
        if "Timestamp:" in data:
            timestamp_value = np.array([data.split("Timestamp: ")[1].rstrip('\n')])
            timestamps=np.append(timestamps,timestamp_value,axis=0)
        elif data == "Features:":
            features_line        = file.readline().strip()
            while features_line != "End of features.":
                features_values  = np.array([float(value) for value in features_line.split('\t')]).reshape(-1,1)
                if features_matrix.size == 0:
                    features_matrix = features_values.reshape(-1,1)
                else:
                    features_matrix  = np.concatenate((features_matrix,features_values),axis=1)
                features_line    = file.readline().strip()

print("Data loaded successfully.")
print("Timestamps: ", timestamps)       # timestamps.shape = (n_samples,)
print("Features: ", features_matrix)    # features_matrix.shape = (n_features, n_samples)