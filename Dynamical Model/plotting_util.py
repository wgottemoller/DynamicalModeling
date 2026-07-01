## Plotting and basic operations
import matplotlib.pyplot as plt 
import numpy as np

## Axes on environment histograms
from matplotlib.ticker import MaxNLocator, FuncFormatter, MultipleLocator, LogLocator
import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

## Files 
import csv
import pandas as pd
import os 

### Set style for plots
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)



# =============================
# Function to set unified plotting style
# =============================


# Get rid of the duplicate zero

def no_redundant_zero(val, pos):
    return '' if val == 0 and pos != 0 else f'{val:g}'

def strip_zeros(x, pos):
    if x == int(x):
        return f"{int(x)}"
    else:
        return f"{x:.1f}"


def unified_plotting_style(xmin, xmax, ymin, ymax, figsize, x_major_locator, x_minor_locator, y_major_locator, xlabel, ylabel,
                        axfig = False, num_plots = (1, 1), bar = False, ylim2 = None, yticks2 = None, y_major_locator2 = None, Horizontal = False,
                        x_major_locator2 = None, xticks2 = None, ticksize2 = None):
    """
    unified plotting style for matplotlib figures for Gottemoller et al. paper.
    
    Parameters:
    - xmin (float): Minimum x-axis value
    - xmax (float): Maximum x-axis value
    - ymin (float): Minimum y-axis value
    - ymax (float): Maximum y-axis value
    - figsize (tuple): Size of the figure in inches, e.g., (16, 16)
    """

    # Set figure size
    plt.figure(figsize=figsize)

    # If ax is provided, use it; otherwise, create a new figure
    if axfig is True and bar is True and Horizontal is False:
        fig, ax = plt.subplots(num_plots[0], num_plots[1], figsize = figsize, gridspec_kw={"height_ratios": [3, 1]})

        axes_list = np.array(ax).flatten() if isinstance(ax, np.ndarray) else [ax]
        for axes in axes_list:
            if axes == axes_list[0]:
                axes.set_xlim(xmin, xmax)
                axes.set_ylim(ymin, ymax)
                axes.set_xlabel(xlabel, fontsize=60, color='black')
                axes.set_ylabel(ylabel, fontsize=60, color='black')
                axes.xaxis.set_major_formatter(FuncFormatter(strip_zeros))
                axes.yaxis.set_major_formatter(FuncFormatter(strip_zeros))
                axes.yaxis.set_major_formatter(FuncFormatter(no_redundant_zero))
            elif axes == axes_list[1]:
                if ylim2 is not None:
                    axes.set_ylim(ylim2[0], ylim2[1])
                axes.set_xlabel(xlabel, fontsize=60, color='black')
                axes.set_ylabel(ylabel, fontsize=60, color='black')
                axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:g}'.format(x)))
                axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:g}'.format(x)))
            if yticks2 is not None and axes == axes_list[1]:
                axes.yaxis.set_major_locator(MultipleLocator(yticks2))
            else:
                axes.yaxis.set_major_locator(MultipleLocator(y_major_locator))
            axes.tick_params(axis='both', which='both', colors='black', width=1.5, bottom=True, top=True, left=True, right=True, direction='in', length=50)
            axes.tick_params(axis='both', which='minor', colors='black', width=1.5, bottom=True, top=True, left=True, right=True, direction='in', length=25)
            axes.tick_params(axis='x', labelsize=50, colors='black')
            axes.tick_params(axis='y', labelsize=50, colors='black')
            axes.xaxis.set_major_locator(MultipleLocator(x_major_locator))
            axes.xaxis.set_minor_locator(MultipleLocator(x_minor_locator))
            axes.yaxis.set_major_locator(MultipleLocator(y_major_locator2))
            for spine in axes.spines.values():
                spine.set_color('black')
                spine.set_linewidth(1.5)
            axes.grid(False)
        plt.tight_layout()
        plt.grid(False)
        plt.minorticks_on()
    
    elif axfig is True and bar is True and Horizontal is True:
        fig, ax = plt.subplots(num_plots[0], num_plots[1], figsize = figsize, gridspec_kw={"width_ratios": [1.5, 1]})

        axes_list = np.array(ax).flatten() if isinstance(ax, np.ndarray) else [ax]
        for axes in axes_list:
            if axes == axes_list[0]:
                axes.set_xlim(xmin, xmax)
                axes.set_ylim(ymin, ymax)
                axes.set_xlabel(xlabel, fontsize=60, color='black')
                axes.set_ylabel(ylabel, fontsize=60, color='black')
                axes.tick_params(axis='both', which='both', colors = 'black', width = 1.5, bottom=True, top=True, left=True, right=True, direction='in', length=50)
                axes.tick_params(axis='both', which='minor', colors = 'black', width = 1.5, bottom=True, top=True, left=True, right=True, direction='in', length=25)
                axes.xaxis.set_major_formatter(FuncFormatter(strip_zeros))
                axes.yaxis.set_major_formatter(FuncFormatter(strip_zeros))
                axes.yaxis.set_major_formatter(FuncFormatter(no_redundant_zero))
                axes.xaxis.set_major_locator(MultipleLocator(x_major_locator))
            elif axes == axes_list[1]:
                if ylim2 is not None:
                    axes.set_ylim(ylim2[0], ylim2[1])
                axes.set_xlabel(xlabel, fontsize=60, color='black')
                axes.set_ylabel(ylabel, fontsize=60, color='black')
                axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:g}'.format(x)))
                axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:g}'.format(x)))
                axes.xaxis.set_major_locator(MultipleLocator(x_major_locator2))
                
                axes.tick_params(axis='both', which='both', colors='black', width=1.5, bottom=True, top=True, left=True, right=True, direction='in', length=ticksize2)
                axes.tick_params(axis='both', which='minor', colors='black', width=1.5, bottom=True, top=True, left=True, right=True, direction='in', length=ticksize2 / 2)
            axes.tick_params(axis='y', labelsize=50, colors='black')
            axes.tick_params(axis='x', labelsize=50, colors='black')
            for spine in axes.spines.values():
                spine.set_color('black')
                spine.set_linewidth(1.5)
            axes.grid(False)
        plt.tight_layout()
        plt.grid(False)
        plt.minorticks_on()

    elif axfig is True and bar is False:
        fig, ax = plt.subplots(num_plots[0], num_plots[1], figsize = figsize)

        axes_list = np.array(ax).flatten() if isinstance(ax, np.ndarray) else [ax]
        for axes in axes_list:
            axes.set_xlim(xmin, xmax)
            axes.set_ylim(ymin, ymax)
            axes.set_xlabel(xlabel, fontsize=60, color='black')
            axes.set_ylabel(ylabel, fontsize=60, color='black')
            if axes == axes_list[0]:
                axes.xaxis.set_major_formatter(FuncFormatter(strip_zeros))
                axes.yaxis.set_major_formatter(FuncFormatter(strip_zeros))
                axes.yaxis.set_major_formatter(FuncFormatter(no_redundant_zero))
            axes.tick_params(axis='both', which='both', colors='black', width=1.5, bottom=True, top=True, left=True, right=True, direction='in', length=50)
            axes.tick_params(axis='both', which='minor', colors='black', width=1.5, bottom=True, top=True, left=True, right=True, direction='in', length=25)
            axes.xaxis.set_major_locator(MultipleLocator(x_major_locator))
            axes.xaxis.set_minor_locator(MultipleLocator(x_minor_locator))
            axes.yaxis.set_major_locator(MultipleLocator(y_major_locator))
            axes.tick_params(axis='x', labelsize=50, colors='black')
            axes.tick_params(axis='y', labelsize=50, colors='black')
            for spine in axes.spines.values():
                spine.set_color('black')
                spine.set_linewidth(1.5)
        plt.tight_layout()
        plt.grid(False)
        plt.minorticks_on()

    else:

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        # Set figure labels
        plt.xlabel(xlabel, fontsize=60, color='black')
        plt.ylabel(ylabel, fontsize=60, color='black')

        # Set tick parameters
        axes = plt.gca()

        # Set tick label/ticks sizes
        axes.xaxis.set_major_formatter(FuncFormatter(strip_zeros))
        axes.yaxis.set_major_formatter(FuncFormatter(strip_zeros))
        axes.yaxis.set_major_formatter(FuncFormatter(no_redundant_zero))
        plt.tick_params(axis='both', which='both', colors = 'black', width = 1.5, bottom=True, top=True, left=True, right=True, direction='in', length=50)
        plt.tick_params(axis='both', which='minor', colors = 'black', width = 1.5, bottom=True, top=True, left=True, right=True, direction='in', length=25)
        plt.xticks(fontsize=50, color='black')
        plt.yticks(fontsize=50, color='black')

        # Set spine (i.e. edge axes) properties
        for spine in axes.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.5) # same line width as ticks

        # Set positions of the ticks and labels
        axes.xaxis.set_major_locator(MultipleLocator(x_major_locator))
        axes.xaxis.set_minor_locator(MultipleLocator(x_minor_locator))
        axes.yaxis.set_major_locator(MultipleLocator(y_major_locator))

        plt.tight_layout()
        plt.grid(False)
        plt.minorticks_on()

        # saving is left to the user, just in case there are additional modifications (i.e. Gaussian fits) beyond the scope of this function


    if axfig is True:
        return fig, ax
    else:
        return
    


def color_options(num):

    colors = ['#009E73', '#D55E00', '#0072B2', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00']

    if num < len(colors):
        return colors[num]
    else:
        raise ValueError(f"Color option index {num} is out of range. Available options are 0 to {len(colors)-1}.")
    
# from Knabel TDCOSMO XIX analysis, see Knabel et al. (2025)
def plot_map(
    ax,
    fig,
    x, 
    y, 
    val,
    title,
    cmap="viridis",
    vmax=None,
    vmin=None,
    unit="",
    pad = 0.25
        ):
    """
    Plot a 2D map with colorbar

    :param ax: matplotlib axis
    :type ax: matplotlib axis
    :param fig: matplotlib figure
    :type fig: matplotlib figure
    :param x: x-coordinates of the data points
    :type x: np.ndarray
    :param y: y-coordinates of the data points
    :type y: np.ndarray
    :param val: values at the data points
    :type val: np.ndarray
    :param title: Title of the plot
    :type title: str
    :param symmetrize_cmap: whether to symmetrize the colormap
    :type symmetrize_cmap: bool
    :return: None
    :rtype: None
    """

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(val)

    im = ax.scatter(
        x[finite],
        y[finite],
        c=val[finite],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        marker=(4, 0, -25 + 45),
        s=900
    )

    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)

    # Zoom in on the plotted region
    if np.any(finite):
        xmin = np.nanmin(x[finite]) - pad
        xmax = np.nanmax(x[finite]) + pad
        ymin = np.nanmin(y[finite]) - pad
        ymax = np.nanmax(y[finite]) + pad

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])