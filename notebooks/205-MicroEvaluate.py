# %% 
from click import style
import serial
import time
import numpy as np
from rich import print
from rich.console import Console
import datetime
console = Console()

# configurations
n_train_snaps = 10      # number of snapshots to take for training

# Open the serial port
ser = serial.Serial('COM5', 115200)

# Open the csv file to save the snapshots
snap_file = open("test_data.csv", "a")

for i in range(n_train_snaps):
    # Send '2' to the serial port to start the acquisition evaluation of features
    ser.write(b'2')

    # Wait 4 seconds for the microcontroller to send the data
    time.sleep(2.5)

    # Read the data from the serial port
    data = ser.read_all().decode('utf-8')

    # Print the data
    console.print("Received data from microcontroller:", style="magenta")
    print(data)
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
    novelty = float(data[start_index:end_index].strip())

    # Check if the number of features is correct
    if np.size(current_features) != 67:
        raise ValueError(f"The number of features received ({np.size(current_features)}) is not 67.")   
    
    # Use the current time as the timestamp because the micro reset every time it is powered on
    current_timestamp = str(datetime.datetime.now()).rsplit('.')[0]

    # Print the snapshot
    console.print(f"Timestamp: {current_timestamp}", style="magenta")
    console.print(f"Extracted features: {current_features}", style="magenta")
    console.print(f"Novelty indicator: {novelty*100} %", style="magenta")

    # save snap to file .csv
    tab_sep_features = "\t".join([str(f) for f in current_features])
    snap_file.write(f"{current_timestamp}\t{tab_sep_features}\t{novelty}\n")

# Close the serial port
ser.close()
