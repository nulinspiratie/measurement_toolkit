import warnings

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import QuadMesh


def inject_matplotlib_axis_clim():
    def attach_set_clim(ax, vmin, vmax):
        for child in ax.get_children():
            if isinstance(child, QuadMesh):
                child.set_clim(vmin, vmax)

    def attach_get_clim(ax):
        for child in ax.get_children():
            if isinstance(child, QuadMesh):
                return child.get_clim()

    Axes.set_clim = attach_set_clim
    Axes.get_clim = attach_get_clim


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

def perform_injections():
    inject_matplotlib_axis_clim()
    inject_matplotlib_figure_title_functions()
    inject_matplotlib_axis_title_functions()
    inject_matplotlib_axis_color_left_right()
    inject_matplotlib_figure_title_functions()
    inject_matplotlib_axis_get_colorbar()