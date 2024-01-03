# %% 
import os
from click import style
import serial
import time
import numpy as np
from rich import print
from rich.console import Console
import datetime
console = Console()

# configurations
n_train_snaps = 5           # number of snapshots to take for training
n_features = 67             # number of features
n_samples = 6000            # number of samples per snapshot

# Open the serial port
ser = serial.Serial('COM5', 115200)

# Open the csv file to save the snapshots
if os.path.exists("train_data.csv"):
    snap_file = open("train_data.csv", "a")
else:
    snap_file = open("train_data.csv", "w")
    snap_file.write("Timestamp\t")
    for i in range(n_features):
        snap_file.write(f"Feature {i+1}\t")
    snap_file.write("\n")

# Open the csv file to save the timeseries
if os.path.exists("timeseries_data.csv"):
    snap_file = open("timeseries_data.csv", "a")
else:
    snap_file = open("timeseries_data.csv", "w")
    snap_file.write("Timestamp\t")
    for i in range(n_samples):
        snap_file.write(f"t_{i+1}\t")
    snap_file.write("\n")
    
for i in range(n_train_snaps):
    # Send '1' to the serial port to start the acquisition and conversion of features
    ser.write(b'3')
    data = ''
    # Read data from the serial port until the keyword "Transmission finished"
    console.print("Received data from microcontroller:", style="magenta")
    while True:  
        line = ser.readline().decode("UTF-8")
        print(line)
        data += (line)
        if "Transmission finished" in line:
            break
    console.print("End of data from microcontroller:", style="magenta")

    # Extract the array of features between the keywords "features:" and "end features"
    start_keyword = "Features: \r\n"
    end_keyword = "\t \r\nEnd of features."
    start_index = data.find(start_keyword) + len(start_keyword)
    end_index = data.find(end_keyword)
    floats_str = data[start_index:end_index].strip()
    current_features = [float(num) for num in floats_str.split()]

    # Extract the array of features between the keywords "features:" and "end features"
    start_keyword = "the time-domain sampled signal is: \r\n"
    end_keyword = "\t \r\ntime-domain end"
    start_index = data.find(start_keyword) + len(start_keyword)
    end_index = data.find(end_keyword)
    floats_str = data[start_index:end_index].strip()
    current_timeserie = [float(num) for num in floats_str.split()]

    if np.size(current_features) != n_features:
        raise ValueError(f"The number of features received ({np.size(current_features)}) is not {n_features}.")
    if np.size(current_timeserie) != n_samples:
        raise ValueError(f"The number of samples received ({np.size(current_timeserie)}) is not {n_samples}.")

    # Use the current time as the timestamp because the micro reset every time it is powered on
    current_timestamp = str(datetime.datetime.now()).rsplit('.')[0]

    # Print the snapshot
    console.print(f"Timestamp: {current_timestamp}", style="magenta")
    console.print(f"Extracted features: {current_features}", style="magenta")

    # save snap to file .csv
    tab_sep_line = "\t".join([str(f) for f in current_features])
    snap_file.write(f"{current_timestamp}\t{tab_sep_line}\n")
    console.print(f"Snapshot {i+1} saved to file.", style="magenta")

    # save the time series to file .csv
    tab_sep_line = "\t".join([str(f) for f in current_timeserie])
    snap_file.write(f"{current_timestamp}\t{tab_sep_line}\n")
    console.print(f"Timeserie {i+1} saved to file.", style="magenta")


# Close the serial port
ser.close()
