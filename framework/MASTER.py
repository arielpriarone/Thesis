import os
import src
import importlib
import numpy as np
import typer
from typing_extensions import Annotated
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel


app = typer.Typer()

@app.command()
def IMS_converter(
    database: Annotated[str,typer.Argument(help='THE name of the database to write to')]='IMS'
):
    # dump some files the files in the folder to mondoDB
    defpath             = r'C:\Users\ariel\Documents\Courses\Tesi\Code\data\raw\1st_test_IMSBearing'
    dirpath             = typer.prompt('Enter the folder path',default=defpath)
    fileList            = sorted(os.listdir(dirpath))                                               # al things in folder
    fileList            = [x for x in fileList if os.path.isfile(os.path.join(dirpath, x))]         # selsct only the files in the folders
    filerange           = []
    filerange.append(typer.prompt('Enter start file', default=fileList[0]))
    filerange.append(typer.prompt('Enter end file', default=fileList[-1]))
    for fileName in fileList[fileList.index(filerange[0]):fileList.index(filerange[-1])]:
        # check if current path is a file
        path=os.path.join(dirpath, fileName) # complete path including filename
        src.data.IMS_to_mongo(database=database,collection='Raw',
                            filePath= path, n_of_test=1, # this folder os first test
                                sensors=['Bearing 1 x', 'Bearing 1 y'])
            
@app.command()
def dummy_cmd():
    pass

if __name__ == "__main__":
    app()