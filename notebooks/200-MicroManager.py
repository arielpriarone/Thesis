import serial
import time

# Open the COM port for writing
com_port = serial.Serial('COM5', baudrate=115200)

# # Open a file for storing the received data
# file_path = 'readfrommicro.txt'
# file = open(file_path, 'w')

for i in range(10):
    # Send the number 1 on the COM port
    print('Sending 1 on the COM port')
    com_port.write(b'1')

    # Read the data from the COM port
    data = com_port.readline().decode().strip()
    print('Received data: {}'.format(data))

    # # Store the received data in the file
    # file.write(data + '\n')
    # file.flush()

    # Wait for 5 seconds
    time.sleep(5)

# Close the file and the COM port
# file.close()
# com_port.close()
