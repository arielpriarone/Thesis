import os
import src
import importlib
import numpy as np
import typer
import inspect
from typing import List
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import print
import subprocess
import json
import time
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel

app = typer.Typer()

@app.command()
def IMS_converter(
    database    : str   = typer.Option(default='IMS',help='The name of the MongoDB database to write to'),
    collection  : str   = typer.Option(default='RAW',help='The name of the MongoDB collection to write to'),
    dirpath     : str   = typer.Option(default=r'C:\Users\ariel\Documents\Courses\Tesi\Code\data\raw\1st_test_IMSBearing',help='The path of the folder containing the files to read'),
    test        : int   = typer.Option(default=1,help='The code of the IMS test (www.imscenter.net) (admitted 1,2,3)',
                                       min=1,max=3),
    sensor      : List[str]  = typer.Option(default=['Bearing 1 x', 'Bearing 1 y'],help='The sensor names you want to read, can be repeated! (axcepted values:\nBearing 1 x\nBearing 1 y\nBearing 2 x\nBearing 2 y\nBearing 3 x\nBearing 3 y\nBearing 4 x\nBearing 4 y\nBearing 1\n Bearing 2\nBearing 3\nBearing 4)'),
    URI         : str   = typer.Option(default='mongodb://localhost:27017',help='The URI to connect to MongoDB')
):
    """
    Transfer the data from the IMS textual files into the MongoDB database in a suitable way.
    EXAMPLE:\n
        python master.py ims-converter --sensor 'Bearing 1 x' --sensor 'Bearing 1 y' --sensor 'Bearing 2 x' --sensor 'Bearing 2 y' --database outer --collection inner
        Enter start file [2003.10.22.12.06.24]:
        Enter end file [2003.11.25.23.39.56]: 2003.10.22.12.59.13
    """
    _fileList            = sorted(os.listdir(dirpath))                                               # al things in folder
    _fileList            = [x for x in _fileList if os.path.isfile(os.path.join(dirpath, x))]         # selsct only the files in the folders
    _filerange           = []
    _filerange.append(typer.prompt('Enter start file', default=_fileList[0]))                         # prompt start file
    _filerange.append(typer.prompt('Enter end file', default=_fileList[-1]))                          # prompt stop file
    _fileList            =   _fileList[_fileList.index(_filerange[0]):_fileList.index(_filerange[1])+1]    # dump all not needed items

    _table = Table(title="\n \n Current options of the command")
    _table.add_column('Option',style='bright_magenta')
    _table.add_column('Value',style='bright_cyan')
    _table.add_row('Database',database)
    _table.add_row('Collection',collection)
    _table.add_row('Folder path',dirpath)
    _table.add_row('Type of test (code)',str(test))
    _table.add_row('Sensor names',str(sensor))
    _table.add_row('URI to connect to MongoDB',URI)
    _table.add_row('Range of file to import',str(_filerange))
    _console = Console()
    _console.print(_table)  
    
    if not typer.confirm("do you want to proceed importing the data from files to MongoDB?", abort=True):
        return
    _slow = typer.prompt('Do you want to slow down the import? (y/n)', default='n')
    for _fileName in track(_fileList,description=f'Writing files to MongoDB',):
        path=os.path.join(dirpath, _fileName) # complete path including filename
        src.data.IMS_to_mongo(database=database,collection=collection,filePath=path,n_of_test=test,sensors=sensor,URI=URI,printout=False)
        if _slow == 'y':
            time.sleep(1)
    print('\n Finished: '+str(len(_fileList))+' files inserted in '+str(database)+'\n')

@app.command()
def create_Empty_DB(configPath:str = typer.Option(default='../config.yaml',help='The path of the configuration file')):
    """
    Create an empty database in MongoDB. The database should not exist already. It is configured according with "config.yaml"
    """
    configPath=os.path.abspath(configPath)
    config=src.data.DB_Manager.loadConfig(configPath)
    print("You are about to create an empty database in MongoDB with the following configuration:")
    print(config)
    typer.confirm("Do you want to proceed?", abort=True)
    src.data.DB_Manager.createEmptyDB(configPath)
    print("Empty database created!")
    
@app.command()
def plot_features():
    """
    Plot the features of the last snapshot in the UNCONSUMED collection
    """
    subprocess.run(["python", "RT_PlotFeatures.py"])

@app.command()
def run_feature_agent():
    """
    Run the Feature Agent - takes last snapshot from RAW collection, extract features and write them to UNCONSUMED collection
    """
    subprocess.run(["python", "FA.py"])

if __name__ == "__main__":
    # RUN THE CLI APP
    app()