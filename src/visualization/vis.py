from turtle import color
from matplotlib import cm, lines, markers
import matplotlib as mpl
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import src
from matplotlib.lines import Line2D

def set_matplotlib_params():
    font = {'family' : 'serif',
            'size'   : 12,
            'serif':  'cmr10'
            }
    mpl.rc('font', **font)
    plt.rcParams["figure.figsize"] = (5.78851, 5.78851/16*9)


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
    def __init__(self,confstr:str,type:str) -> None:
        self.tab10_cmap = cm.get_cmap("Set1")
        self.DB=src.data.DB_Manager(confstr)

    def plot_Kmeans_error_init(self,ax: plt.Axes):
        try:
            Err_dict = self.DB.col_models.find({'_id': 'Kmeand cluster {type} indicator'})[0]
        except IndexError:
            print(f"No Kmeans error dictionary found in {self.DB.col_models.full_name}, waiting...")
            return ax
        color_legend = [self.tab10_cmap(x) for x in range(max(Err_dict['assigned_cluster']))]
        self.__legend_labels = [f"cluster {i}" for i in range(max(Err_dict['assigned_cluster']))]
        self.__legend_lines = [Line2D([0], [0],marker='o', color='w',markerfacecolor=color_legend[indx], lw=4, alpha=1) for indx in range(max(Err_dict['assigned_cluster']))] # type: ignore
        self.__legend_labels.append('novelty threshold')
        self.__legend_lines.append(Line2D([0], [0], linestyle='-.', color='r'))

    def plot_Kmeans_error(self,ax: plt.Axes):
        try:
            Err_dict = self.DB.col_models.find({'_id': 'Kmeand cluster error dictionary'})[0]
        except IndexError:
            print(f"No Kmeans error dictionary found in {self.DB.col_models.full_name}, waiting...")
            return ax
        ax.clear()  # Clear last data frame
        ax.set_title(f"Latest {self.DB.Config['kmeans']['error_plot_size']} distance error.")  # set title
        self.__colors = [self.tab10_cmap(x) for x in Err_dict['assigned_cluster']]
        xlabels = [str(x) for x in Err_dict['timestamp']]
        ax.scatter(xlabels, Err_dict['values'],marker='.', c=self.__colors)  # type: ignore #plot <data
        ax.axhline(self.DB.Config['kmeans']['novelty_threshold'],linestyle='-.',color='k')
        ax.set_ylabel('Distance relative error [%]')
        ax.set_xlabel('Timestamp')
        ax.set_xticks(custom_tick_locator(18,xlabels))
        ax.legend(self.__legend_lines, self.__legend_labels,loc='upper left')
        plt.xticks(rotation=45, ha='right')
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
    PLTR.plot_Kmeans_error_init(ax)
    PLTR.plot_Kmeans_error(ax)
    plt.show()
