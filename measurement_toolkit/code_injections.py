import warnings
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import QuadMesh
from matplotlib.colors import LogNorm, SymLogNorm
from measurement_toolkit.tools.plot_tools import create_diverging_cmap


def get_lim(ax, above_zero=False):
    min_val = None
    max_val = None
    
    for child in ax.get_children():
        if isinstance(child, QuadMesh):
            arr = child.get_array()
            if above_zero:
                arr = arr.copy()
                arr[arr<=0] = np.nan

            if min_val is None or np.nanmin(arr) < min_val:
                min_val = np.nanmin(arr)
            if max_val is None or np.nanmax(arr) > max_val:
                max_val = np.nanmax(arr)
    return min_val, max_val


def inject_matplotlib_axis_lim():
    Axes.get_lim = get_lim


def inject_matplotlib_axis_clim():
    def set_clim(ax, vmin=None, vmax=None):
        if vmin is None:
            vmin = ax.get_lim()[0]
        if vmax is None:
            vmax = ax.get_lim()[1]
        for child in ax.get_children():
            if isinstance(child, QuadMesh):
                child.set_clim(vmin, vmax)

    def get_clim(ax):
        for child in ax.get_children():
            if isinstance(child, QuadMesh):
                return child.get_clim()

    Axes.set_clim = set_clim
    Axes.get_clim = get_clim


def inject_matplotlib_diverging_cmap():
    def set_diverging_cmap(ax, vmin=None, vmax=None, center=0, positive_cmap=None, negative_limit_color='red', negative_scale=0.5):
        if vmin is None:
            vmin = ax.get_lim()[0]
        if vmax is None:
            vmax = ax.get_lim()[1]

        diverging_cmap = create_diverging_cmap(
            vmin=vmin,
            vmax=vmax,
            center=center,
            positive_cmap=positive_cmap,
            negative_limit_color=negative_limit_color,
            negative_scale=negative_scale
        )

        for child in ax.get_children():
            if isinstance(child, QuadMesh):
                child.set_cmap(diverging_cmap)

    Axes.set_diverging_cmap = set_diverging_cmap

def inject_matplotlib_axis_logscale():
    def set_logscale(ax, vmin=None, vmax=None, force_positive=False):
        if vmin is None or vmax is None:
            min_vals = get_lim(ax, above_zero=True)
            if vmin is None:
                vmin = min_vals[0]
            if vmax is None:
                vmax = min_vals[1]

        for child in ax.get_children():
            if isinstance(child, QuadMesh):
                if force_positive:
                    arr = child.get_array()
                    arr[arr < vmin] = vmin
                child.set_norm(LogNorm(vmin, vmax))

    Axes.set_logscale = set_logscale

    def set_logscale_symmetric(ax, vmin, vmax, lin_threshold=1e-3, linscale=0.1, cmap='RdBu_r'):
        for child in ax.get_children():
            if isinstance(child, QuadMesh):
                child.set_norm(SymLogNorm(vmin=vmin, vmax=vmax, linscale=linscale, linthresh=lin_threshold))
                child.set_cmap(cmap=cmap)

    Axes.set_logscale = set_logscale
    Axes.set_logscale_symmetric = set_logscale_symmetric


def inject_matplotlib_figure_title_functions():
    def prepend_title(fig, title, separator='\n'):
        original_title = fig._suptitle.get_text()
        fig.suptitle(f'{title}{separator}{original_title}')

    def append_title(fig, title, separator='\n'):
        original_title = fig._suptitle.get_text()
        fig.suptitle(f'{original_title}{separator}{title}')

    Figure.prepend_title = prepend_title
    Figure.append_title = append_title


def inject_matplotlib_axis_title_functions():
    def prepend_title(ax, title, separator='\n'):
        original_title = ax.get_title()
        ax.set_title(f'{title}{separator}{original_title}')

    def append_title(ax, title, separator='\n'):
        original_title = ax.get_title()
        ax.set_title(f'{original_title}{separator}{title}')

    Axes.prepend_title = prepend_title
    Axes.append_title = append_title


def inject_matplotlib_axis_color_left_right():
    def color_right_axis(ax, color):
        ax.yaxis.label.set_color(color)
        ax.spines["right"].set_edgecolor(color)
        ax.tick_params(axis='y', colors=color)

    def color_left_axis(ax, color):
        ax.yaxis.label.set_color(color)
        ax.spines["left"].set_edgecolor(color)
        ax.tick_params(axis='y', colors=color)

    Axes.color_right_axis = color_right_axis
    Axes.color_left_axis = color_left_axis


def inject_matplotlib_axis_get_colorbar():
    def get_colorbar(ax):
        colorbars = []
        for elem in ax.collections:
            if getattr(elem, 'colorbar') is not None:
                colorbars.append(elem.colorbar)

        if len(colorbars) > 1:
            warnings.warn(f'Found {len(colorbars)} instead of 1')
            return colorbars
        elif not colorbars:
            return None
        else:
            return colorbars[0]
    Axes.get_colorbar = get_colorbar


def perform_code_injections():
    inject_matplotlib_axis_clim()
    inject_matplotlib_axis_lim()
    inject_matplotlib_diverging_cmap()
    inject_matplotlib_axis_logscale()
    inject_matplotlib_figure_title_functions()
    inject_matplotlib_axis_title_functions()
    inject_matplotlib_axis_color_left_right()
    inject_matplotlib_figure_title_functions()
    inject_matplotlib_axis_get_colorbar()