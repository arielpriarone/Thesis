# %% 
from click import style
import serial
import time
import numpy as np
from rich import print
from rich.console import Console
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
console = Console()

# configurations
n_train_snaps = 10          # number of snapshots to take for training
n_features = 67             # number of features

# Open the serial port
ser = serial.Serial('COM5', 115200)

# variables to store the data
timestamps = []
features = []
novelties = []

# Open the csv file to save the snapshots
if os.path.exists("test_data.csv"):
    snap_file = open("test_data.csv", "a")
else:
    snap_file = open("test_data.csv", "w")
    snap_file.write("Timestamp\t")
    for i in range(n_features):
        snap_file.write(f"Feature {i+1}\t")
    snap_file.write("Novelty")
    snap_file.write("\n")

for i in range(n_train_snaps):
    # Send '2' to the serial port to start the acquisition evaluation of features
    ser.write(b'2')

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

    # Extract the novelty indicator between the keywords "novelty:" and "\n"
    start_keyword = "Novelty indicator: "
    end_keyword = "\n"
    start_index = data.find(start_keyword) + len(start_keyword)
    end_index = data.find(end_keyword, start_index)
    current_novelty = float(data[start_index:end_index].strip())

    # Check if the number of features is correct
    if np.size(current_features) != 67:
        raise ValueError(f"The number of features received ({np.size(current_features)}) is not 67.")   
    
    # Use the current time as the timestamp because the micro reset every time it is powered on
    current_timestamp = str(datetime.datetime.now()).rsplit('.')[0]

    # Print the snapshot
    console.print(f"Timestamp: {current_timestamp}", style="magenta")
    console.print(f"Extracted features: {current_features}", style="magenta")
    console.print(f"Novelty indicator: {current_novelty*100} %", style="magenta")

    # save snap to file .csv
    tab_sep_features = "\t".join([str(f) for f in current_features])
    snap_file.write(f"{current_timestamp}\t{tab_sep_features}\t{current_novelty}\n")

    # update variables
    timestamps.append(current_timestamp)
    features.append(current_features)
    novelties.append(current_novelty)

# Close the serial port
ser.close()

# plot the data
fig, axs = plt.subplots()


# Format timestamps as HH.MM.SS
formatted_timestamps = [datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").strftime("%H.%M.%S") for ts in timestamps]

axs.plot(formatted_timestamps,novelties)
axs.set_xlabel("Time")
axs.set_ylabel("Novelty indicator")
axs.set_xticklabels(formatted_timestamps, rotation=45)

plt.show()


