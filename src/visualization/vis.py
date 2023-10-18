from calendar import c
from matplotlib import cm, lines
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import src

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
    def __init__(self,confstr:str) -> None:
        self.tab10_cmap = cm.get_cmap("Set1")
        self.DB=src.data.DB_Manager(confstr)
    def plot_Kmeans_error(self,ax: plt.Axes):
        Err_dict = self.DB.col_models.find({'_id': 'Kmeand cluster error dictionary'})[0]
        ax.clear()  # Clear last data frame
        ax.set_title(f"Latest {self.DB.Config['kmeans']['error_plot_size']} distance error.")  # set title
        ax.scatter(Err_dict['timestamp'], Err_dict['value'], c=Err_dict['assigned_cluster'],Colormap=self.tab10_cmap)  # plot data
        ax.axhline(self.DB.Config['kmeans']['novelty_threshold'],linestyle='-.',color='r')
        return ax

    