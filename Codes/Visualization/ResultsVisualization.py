from Codes.Configuration import Result_Path
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import os

class ResultsVisualization:
    
    def Results_plot(self, junction_id, result, method_name, x_label, y_label, name_folder='Results', step=5, vline_x=None):
        plt.figure(figsize=(25, 8))
        # plt.title(method_name[0])
        for i, res in enumerate(result):
            x = np.arange(step, len(res) + step, step)
            y = res[::step]
            plt.plot(x, y, label=method_name[i])
            plt.scatter(x, y, s=20)
        plt.legend(loc="best")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        ax = plt.gca()
        ax.locator_params(axis='y', nbins=10)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
        num_ticks = 10 
        ax.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
        ax.xaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False, useMathText=True))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        plt.xticks(rotation=45, fontsize='xx-small')
        plt.grid()
        if vline_x is not None:
            plt.axvline(x=vline_x, color='r', linestyle='--', label='Test')
        save_path = os.path.join(Result_Path, name_folder)
        os.makedirs(save_path, exist_ok=True)

        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Example adjustment for x-axis
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Example adjustment for y-axis

        plt.savefig(os.path.join(save_path, str(method_name[0]) + "_" + junction_id + '.png'))
        plt.savefig(os.path.join(save_path, str(method_name[0]) + "_" + junction_id + '.svg'), format='svg')
        # plt.show()  # Display the plot

    def box_plot(self, DataFram, name_folder='Results', polt_name=""):
        plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
        sns.boxplot(data=DataFram)
        plt.title('Box plot of Objectives Function')
        plt.xlabel('objectives')
        plt.ylabel('Values')
        save_path = os.path.join(Result_Path, name_folder)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, polt_name + '.png'))
        plt.savefig(os.path.join(save_path, polt_name + '.svg'), format='svg')
        # plt.show()