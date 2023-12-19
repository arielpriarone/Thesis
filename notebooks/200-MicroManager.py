# %% 
from click import style
import serial
import time
import numpy as np
from rich import print
from rich.console import Console
import datetime
console = Console()

# Open the serial port
ser = serial.Serial('COM5', 115200)

# Send '1' to the serial port to start the acquisition and conversion of features
ser.write(b'1')

# Wait for 5 seconds to have the features back
time.sleep(5)

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

# Use the current time as the timestamp because the micro reset every time it is powered on
current_timestamp = str(datetime.datetime.now()).rsplit('.')[0]

# Print the snapshot
console.print(f"Timestamp: {current_timestamp}", style="magenta")
console.print(f"Extracted features: {current_features}", style="magenta")

# Close the serial port
ser.close()
