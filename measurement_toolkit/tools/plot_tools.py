import numpy as np
from pathlib import Path
import warnings

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, EngFormatter
from matplotlib.axis import Axis
from matplotlib.axes import Subplot
from matplotlib.figure import Figure

from .data_tools import convert_to_dataset

__all__ = ['plot_data', 'plot_ds', 'plot_dual_axis', 'show_image']


def plot_data(
        dataset,
        axes_structure: str = None,
        figsize=None,
        clim=None,
        axes=None,
        xscale=True,
        yscale=True,
        transpose=False,
        set_title=True,
        negative_clim=False,
        print_info=False,
        arr_modifier=None,
        swap_dims=None,
        **plot_kwargs
):
    # Ensure we have an xarray
    dataset = convert_to_dataset(dataset, 'xarray')

    if swap_dims is not None:
        dataset = dataset.swap_dims(swap_dims)
        
    if axes_structure is None:
        axes_structure = f'[{"1" * len(dataset.data_vars)}]'

    # Count expected number of data arrays and ensure it matches
    expected_arrays = axes_structure.count('1') + axes_structure.count('0')
    num_arrays = len(dataset.data_vars)
    assert expected_arrays == num_arrays, f'{expected_arrays=} does not match {num_arrays=}'

    level = 0
    structure = []
    array_iterator = iter(dataset.data_vars)

    try:
        for character in axes_structure:
            # Change level
            if character in '[]':
                if character == '[':
                    if level == 0:
                        structure.append([])
                    elif level == 1:
                        structure[-1].append([])
                    else:
                        SyntaxError('Cannot have more than two brackets opening')
                    level += 1
            elif character == ']':
                if level <= 0:
                    raise SyntaxError('Cannot close more than all open brackets')
                level -= 1
            elif character == '1':
                array_label = next(array_iterator)
                if level == 0:
                    # Create new figure
                    structure.append([array_label])
                elif level == 1:
                    structure[-1].append([array_label])
                elif level == 2:
                    structure[-1][-1].append(array_label)
            elif character == '0':
                # Skip the upcoming array
                next(array_iterator)
    except StopIteration:
        raise SyntaxError(f'Not enough arrays: {len(dataset.data_vars)}')

    figure_list = []
    axes_list = []
    for figure_structure in structure:
        num_axes = len(figure_structure)
        if num_axes < 3:
            rows, columns = 1, num_axes
        elif not num_axes % 3:
            rows, columns = num_axes // 3, 3
        elif not (num_axes + num_axes % 2) % 3:
            rows, columns = (num_axes + 1) // 3, 3
        else:
            rows, columns = (num_axes + 1) % 2, 2

        if axes is None:
            if figsize is None:
                figsize = (2 + 4 * columns, 4)

            fig, fig_axes = plt.subplots(rows, columns, figsize=figsize)
            if num_axes == 1:
                fig_axes = [fig_axes]
        else:
            if isinstance(axes, (Axis, Subplot)):
                axes = [axes]
            fig_axes = axes
            fig = fig_axes[0].get_figure()


        figure_list.append(fig)
        axes_list.append(fig_axes) 

        for ax, array_labels in zip(fig_axes, figure_structure):
            for array_label in array_labels:
                array = dataset.data_vars[array_label]
                if array.ndim == 2 and transpose:
                    array = array.transpose()
                if np.iscomplexobj(array):
                    print('Array is complex, plotting real part')
                    array = array.real
                if arr_modifier is not None:
                    array = arr_modifier(array)
                        
                array.plot(ax=ax, **plot_kwargs)

            # Optionally modify clim
            if array.ndim == 2:
                if clim is not None:
                    ax.set_clim(*clim)
                elif not negative_clim:
                    _clim = ax.get_clim()
                    if _clim is not None:
                        ax.set_clim(max(_clim[0], 0), _clim[1])

            if xscale:
                ax.xaxis.set_major_formatter(EngFormatter(sep=''))
            if yscale:
                ax.yaxis.set_major_formatter(EngFormatter(sep=''))

        if len(figure_list) == 1:
            figure_list = figure_list[0]
            axes_list = axes_list[0]
            if num_axes == 1:
                axes_list = axes_list[0]

        if set_title and isinstance(figure_list, Figure):
            if set_title is True:
                initials = getattr(dataset, 'initials', 'SA')
                title = f'#{dataset.run_id} {initials} {dataset.run_timestamp}'

            if isinstance(axes_list, (Axis, Subplot)) and not axes_list.get_title():
                axes_list.set_title(title)
            else:
                fig.suptitle(title)

        fig.tight_layout()

        return figure_list, axes_list

# Deprecated
plot_ds = plot_data

import imageio


def create_animation(filename, figs, fps=1):
    filepath = Path(filename)
    filepath = filepath.with_suffix('.gif')
    # if not path.is_absolute():
    #     filepath =

    images = []
    for fig in figs:
        fig.dpi = 300
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
    imageio.mimsave(str(filepath), images, fps=1)


def plot_dual_axis(dataset, reverse=False, axes=None, figsize=None):
    dataset = convert_to_dataset(dataset, 'xarray')

    assert len(dataset.data_vars) == 2

    # Define figure and axes
    if axes is None:
        fig, ax1 = plt.subplots(figsize=figsize)
        axes = [ax1, ax1.twinx()]
    elif isinstance(axes, list):
        fig = axes[0].get_figure()
    else:  # Axes is a single axis
        fig = axes.get_figure()
        axes = [axes, axes.twinx()]


    iterable = dataset.data_vars.items()
    if reverse:
        iterable = reversed(iterable)
    for k, (key, array) in enumerate(iterable):
        ax = axes[k]
        color = f'C{k}'
        array.plot(ax=ax, color=color)

        if not k:
            ax.color_left_axis(color)
        else:
            ax.color_right_axis(color)

    # Set title
    initials = getattr(dataset, 'initials', 'SA')
    title = f'#{dataset.run_id} {initials} {dataset.run_timestamp}'
    axes[0].set_title(title)

    return fig, axes


def show_image(filepath, show=True):
    try:
        import io
        import fitz
        from PIL import Image
        from IPython.display import display
    except ModuleNotFoundError:
        warnings.warn('Module "fitz" not found, please run "pip install pymupdf"')
        return

    filepath = Path(filepath)

    if filepath.suffix == '.pdf':
        pdf_file = fitz.open(filepath)

        # in case there is a need to loop through multiple PDF pages
        images = []
        for page_number in range(len(pdf_file)):
            page = pdf_file[page_number]
            rgb = page.get_pixmap()
            pil_image = Image.open(io.BytesIO(rgb.tobytes()))
            images.append(pil_image)

        if show:
            display(pil_image)
        
        return images if len(images) > 1 else images[0]

    elif filepath.suffix == '.png':
        pil_image = Image.open(filepath)

        if show:
            display(pil_image)
        
        return pil_image