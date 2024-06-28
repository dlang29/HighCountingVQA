import matplotlib.pyplot as plt
import numpy as np

import os

def create_bar_plots(x, Y, labels=None, save_name="barplots.png"):
    """
    Create a figure with multiple bar plots. 
    
    Params:
    x: Shared x-values for the barplots, shape (num_data_points, )
    Y: Set of y-values for the barplots, shape (num_plots, num_datapoints)
    labels: Optional dictionary to set labels {'x_label': 'x-axis label', 'y_label': 'y-axis label', title': 'title name', 'subtitles': ['model1', 'model2',...]}
    save_name: Name of the created figure
    """

    num_plots = Y.shape[0]
    width = 2
    height = int(np.ceil(num_plots / width))
    fig, axes = plt.subplots(height, width, figsize=(10 * width, 5 * height), squeeze=False)
    
    if labels is not None:
        x_label = labels.get('x_label', 'x-axis')
        y_label = labels.get('y_label', 'y-axis')
        title = labels.get('title', 'Barplots')
        subtitles = labels.get('subtitles', [f'Model {i}' for i in range(num_plots)])

    for i in range(width):
        for j in range(height):
            if (i+1) * (j+1) <= num_plots:
                axes[j, i].bar(x, Y[i+j])
                axes[j, i].set_title(subtitles[i+j])

                axes[j, i].set_xlabel(x_label)
                axes[j, i].set_ylabel(y_label)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    plt.savefig(os.path.join("./data/plots", save_name))



def create_dual_axis_bar_plots(x, Y, labels=None, save_name="dual_barplots.png"):
    """
    Create a figure with 2 barplots per axis. 
    
    Params:
    x: Shared x-values for the barplots, shape (num_data_points, )
    Y: Set of y-values for the barplots, shape (num_plots, 2, num_datapoints)
    labels: Optional dictionary to set labels {'x_label': 'x-axis label', 'y_label': 'y-axis label', 'y_label2': '2. y-axis label', title': 'title name', 'subtitles': ['model1', 'model2',...]}
    save_name: Name of the created figure
    """

    num_plots = Y.shape[0]
    width = 2
    height = int(np.ceil(num_plots / width))
    fig, axes = plt.subplots(height, width, figsize=(10 * width, 5 * height), squeeze=False)
    
    if labels is not None:
        x_label = labels.get('x_label', 'x-axis')
        y_label = labels.get('y_label', 'y-axis')
        y_label2 = labels.get('y_label2', '2. y-axis')
        title = labels.get('title', 'Dual-Barplots')
        subtitles = labels.get('subtitles', [f'Model {i}' for i in range(num_plots)])

    bar_width = 0.4
    for i in range(width):
        for j in range(height):
            if (i+1) * (j+1) <= num_plots:
                ax1 = axes[j, i]
                ax1.bar(x - bar_width/2, Y[i+j, 0], bar_width, color='g', label=y_label)
                ax1.set_xlabel(x_label)
                ax1.set_ylabel(y_label, color='g')
                ax1.tick_params(axis='y', labelcolor='g')
                
                ax2 = ax1.twinx()
                ax2.bar(x + bar_width/2, Y[i+j, 1], bar_width, color='b', label=y_label2)
                ax2.set_ylabel(y_label2, color='b')
                ax2.tick_params(axis='y', labelcolor='b')
                
                ax1.set_title(subtitles[i+j])
                ax1.set_xticks(x)
                ax1.set_xticklabels(x)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    plt.savefig(os.path.join("./data/plots", save_name))
