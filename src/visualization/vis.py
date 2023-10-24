import os
from socket import RCVALL_SOCKETLEVELONLY
import timeit
from turtle import color
from matplotlib import cm, lines
import matplotlib as mpl
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import src
from matplotlib.lines import Line2D
from datetime import datetime
from typing import List
import numpy as np
import matplotlib.dates as mdates
import timeit

def set_matplotlib_params():
    font = {'family' : 'serif',
            'size'   : 12,
            'serif':  'cmr10'
            }
    mpl.rc('font', **font)
    plt.rcParams["figure.figsize"] = (5.78851, 5.78851/16*9)
    plt.rcParams["axes.formatter.use_mathtext"] = True


def isNotebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
def custom_tick_locator(n_ticks,labels):
    # Function to select the subset of tick locations based on the width of the plot
    num_labels = len(labels)
    tick_step = max(num_labels // (n_ticks-1), 1)
    return range(0, num_labels, tick_step)

class Plotter:
    def __init__(self,confstr:str,type_predict:str) -> None:
        self.tab10_cmap = cm.get_cmap("Set1")
        self.DB=src.data.DB_Manager(confstr)
        self.type = type_predict
        self.err_dict = {'values': List[float], 'timestamp': List[datetime],
                         'assigned_cluster': List[int], 'anomaly': List[bool]} # dictionary of the error
    
    def load_indicator_data(self):
        try:
            self.Err_dict = self.DB.col_models.find({'_id': f'Kmeans cluster {self.type} indicator'})[0]
        except IndexError:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"No Kmeans error dictionary found in {self.DB.col_models.full_name}, waiting...")
            return False
        return True

    def plot_Kmeans_error_init(self,ax: plt.Axes):
        while not self.load_indicator_data():
            return None # wait for data to be available
        Err_dict= self.Err_dict
        range_clusters = range(max(Err_dict['assigned_cluster'])+1)
        color_legend = [self.tab10_cmap(x) for x in range_clusters]
        self.__legend_labels = [f"cluster {i}" for i in range_clusters]
        self.__legend_lines = [Line2D([0], [0],marker='o', color='w',markerfacecolor=color_legend[indx], lw=4, alpha=1) for indx in range_clusters] # type: ignore
        self.__legend_labels.append('novelty threshold')
        self.__legend_lines.append(Line2D([0], [0], linestyle='-.', color='k'))
        print(f"Plot Kmeans error initialized with {self.__legend_labels} and {self.__legend_lines}")
        return ax

    def plot_Kmeans_error(self,ax: plt.Axes):
        self.load_indicator_data()
        Err_dict= self.Err_dict
        ax.clear()  # Clear last data frame
        ax.set_title(f"Latest {self.DB.Config['kmeans']['error_plot_size']} distance error.")  # set title
        self.__colors = [self.tab10_cmap(x) for x in Err_dict['assigned_cluster']]

        xlocator=np.array([Err_dict['timestamp'][x].timestamp() for x in range(len(Err_dict['timestamp']))])
        ax.scatter(xlocator, Err_dict['values'],marker='.', c=self.__colors)  # type: ignore #plot <data
        ax.axhline(self.DB.Config['novelty']['threshold'],linestyle='-.',color='k')
        ax.set_xlim(min(xlocator),max(xlocator))
        ax.set_ylabel('Distance relative error [%]')
        ax.set_xlabel('Time [s]')
        ax.set_xticklabels([datetime.fromtimestamp(ts).strftime(r'%d/%m/%Y, %H:%M:%S') for ts in ax.get_xticks()])
        ax.legend(self.__legend_lines, self.__legend_labels,loc='upper left')

        # x locator and formatter
        plt.tight_layout()
        return ax


if __name__=='__main__': 
    # just for testin, not useful as package functionality
    # timeSerie=src.data.readSnapshot('IMS','RAW','mongodb://localhost:27017')['Bearing 1 x']['timeSerie']
    # coef, pows, nodes, _, _ = packTrasform(timeSerie, plot=True)
    # plt.show()
    set_matplotlib_params()
    fig, ax = plt.subplots()
    PLTR = Plotter(r"C:\Users\ariel\Documents\Courses\Tesi\Code\config.yaml",'novelty')
    while PLTR.plot_Kmeans_error_init(ax) is None:
        pass
    elapsed = timeit.timeit(lambda: PLTR.plot_Kmeans_error(ax), number=1)
    print(f"Elapsed time: {elapsed:.6f} seconds")
    plt.show()
