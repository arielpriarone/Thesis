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
    fileList            = sorted(os.listdir(dirpath))                                               # al things in folder
    fileList            = [x for x in fileList if os.path.isfile(os.path.join(dirpath, x))]         # selsct only the files in the folders
    filerange           = []
    filerange.append(typer.prompt('Enter start file', default=fileList[0]))                         # prompt start file
    filerange.append(typer.prompt('Enter end file', default=fileList[-1]))                          # prompt stop file
    fileList            =   fileList[fileList.index(filerange[0]):fileList.index(filerange[-1])]    # dump all not needed items

    table = Table(title="\n \n Current options of the command")
    table.add_column('Option',style='bright_magenta')
    table.add_column('Value',style='bright_cyan')
    table.add_row('Database',database)
    table.add_row('Collection',collection)
    table.add_row('Folder path',dirpath)
    table.add_row('Type of test (code)',str(test))
    table.add_row('Sensor names',str(sensor))
    table.add_row('URI to connect to MongoDB',URI)
    table.add_row('Range of file to import',str(filerange))
    console = Console()
    console.print(table)  
    
    if not typer.confirm("do toy want to proceed importing the data from files to MongoDB?"):
        return
    for fileName in track(fileList,description=f'Writing files to MongoDB',):
        path=os.path.join(dirpath, fileName) # complete path including filename
        src.data.IMS_to_mongo(database=database,collection=collection,filePath=path,n_of_test=test,sensors=sensor,URI=URI,printout=False)
    print('Finished: '+str(len(fileList))+' files inserted in '+str(database))



@app.command()
def dummy_cmd():
    pass

if __name__ == "__main__":
    app()