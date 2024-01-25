import os
import src
import framework.MASTER as MASTER
from src.data.data import DB_Manager
from rich import print
from rich.console import Console
import threading

## ##
# This script is used to initialise the framework, clears the database and loads the training data

src.visualization.set_matplotlib_params() # set matplotlib parameters to generate plots for the thesis
console = Console()
# %% script settings
configFilePath  = r"C:\Users\ariel\Documents\Courses\Tesi\Code\config.yaml" # configuration file path
rawDataDirPath  = r"C:\Users\ariel\Documents\Courses\Tesi\Code\data\raw\1st_test_IMSBearing" # folder path
N_train         = 300 # number of training samples

# %% initialise script
Config = src.data.DB_Manager.loadConfig(configFilePath) # load configuration file
console.print("Loaded configuration:", style="magenta")
print(Config)

# %% ARESE AND CREATE empty database
src.data.DB_Manager.eraseDB(configFilePath) #  clear current database
src.data.DB_Manager.createEmptyDB(configFilePath) # create empty database

# %% load training data into the database - ideally it is a sensor directly
filelists = os.listdir(rawDataDirPath) # list of files in the folder
filelists.sort() # sort the list of files
filelists = filelists[:N_train] # select only the first N_train files
console.print(f"Loading {len(filelists)} files into the database...", style="magenta")
MASTER._IMS_converter_withrange(Config['Database']['db'], # database name
                     Config['Database']['collection']['raw'], # collection name
                     rawDataDirPath, # folder path
                     test= 1, # test number
                     sensor= list(Config['Database']['sensors'].keys()), # list of sensors
                     URI=Config['Database']['URI'], # database URI
                     sleep = None,   # sleep time between files
                     startfile=filelists[0],# start file
                     endfile=filelists[-1]) # end file