import builtins
import os
import sys
from pathlib import Path
from typing import Union
import logging
import nbformat
from IPython import get_ipython
from copy import deepcopy
import nbconvert
import nbformat

import qcodes as qc

__all__ = [
    'export_notebook',
    'get_last_input_cells',
    'using_ipython',
    'execute_file',
    'run_script',
    'create_new_notebook',
    'configure_device_folder',
    'update_namespace'
]

logger = logging.getLogger(__name__)


def get_notebook_path():
    from IPython import get_ipython
    from jupyter_server import serverapp
    from jupyter_server.utils import url_path_join
    from pathlib import Path
    import re
    import requests

    kernelIdRegex = re.compile(r"(?<=kernel-)[\w\d\-]+(?=\.json)")
    kernelId = kernelIdRegex.search(get_ipython().config["IPKernelApp"]["connection_file"])[0]

    for jupServ in serverapp.list_running_servers():
        for session in requests.get(url_path_join(jupServ["url"], "api/sessions"),
                                    params={"token": jupServ["token"]}).json():
            if kernelId == session["kernel"]["id"]:
                return Path(jupServ["root_dir"]) / session["notebook"]['path']


def prepare_notebook_for_conversion(notebook, input_code=True):
    new_notebook = deepcopy(notebook)
    
    # Convert data description cells, i.e. "#{idx}"
    for cell in new_notebook['cells']:
        if cell['cell_type'] != 'markdown':
            continue
        if not cell['source'].startswith('#'):
            continue
        if len(cell['source']) < 2:
            continue
        if not cell['source'][1].isdigit():
            continue
        cell['source'] = '✎' + cell['source']

    # Remove plot object outputs
    for cell in new_notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        if 'outputs' not in cell:
            continue

        remove_outputs = []
        for output in cell['outputs']:
            if 'data' not in output or 'text/plain' not in output['data']:
                continue
            if output.get('output_type') != 'execute_result':
                continue

            exclude_start_strings = [
                '<', '(<', '(array(', '[<', '([', 'array(', 'Text('
            ]
            text_data = output['data']['text/plain']
            is_excluded = any(text_data.startswith(elem) for elem in exclude_start_strings)
            if is_excluded:
                remove_outputs.append(output)
                
        for remove_output in remove_outputs:
            cell['outputs'].remove(remove_output)

    return new_notebook


def export_notebook(notebook_name=None, input_code=False, output='html'):
    from IPython import get_ipython
    ipython = get_ipython()

    if notebook_name is None:
        notebook_name = get_notebook_path().name

    notebook_path = Path(notebook_name).with_suffix('.ipynb')
    assert notebook_path.exists()
    print(f'Converting notebook "{notebook_path}"')

    if output != 'html':
        # compose command
        command_args = [
            'jupyter nbconvert',
            f'"{notebook_path}"',
            f'--to={output}'
        ]
        if not input_code:
            command_args.append('--no-input')
        command = ' '.join(command_args)
        # print(f'Running command: {command}\n')

        ipython.run_cell_magic('cmd', '', command)
    else:
        # Read notebook into dict
        notebook = nbformat.read(notebook_path, as_version=4)

        # Prepare notebook
        notebook = prepare_notebook_for_conversion(notebook)
        
        # Create HTML exporter
        html_exporter = nbconvert.HTMLExporter()

        # Remove input code
        if not input_code:
            html_exporter.exclude_input = True
            html_exporter.exclude_input_prompt = True
            html_exporter.exclude_output_prompt = True

        # Convert notebook to HTML string
        notebook_data, resources = html_exporter.from_notebook_node(notebook)

        # Save notebook to file
        notebook_html_path = notebook_path.with_suffix('.html')
        notebook_html_path.write_text(notebook_data, encoding='utf8')

    # Print notebook link
    from IPython.core.display import display, HTML
    notebook_HTML_path = notebook_path.with_suffix(".html")
    display(HTML(f'<a href="{notebook_HTML_path}">{notebook_HTML_path}</a>'))


def get_last_input_cells(cells=3):
    """
    Get last input cell. Note that get_last_input_cell.globals must be set to
    the ipython globals
    Returns:
        last cell input if successful, else None
    """
    global In
    if 'In' in globals() or hasattr(builtins, 'In'):
        return In[-cells:]
    else:
        logging.warning('No input cells found')


def using_ipython() -> bool:
    """Check if code is run from IPython (including jupyter notebook/lab)"""
    return hasattr(builtins, '__IPYTHON__')


def execute_file(filepath, mode=None, globals=None, locals=None):
    if isinstance(filepath, str):
        filepath = Path(filepath)

    assert filepath.suffix == '.py'

    try:
        if using_ipython():
            from IPython import get_ipython
            shell = get_ipython()
            shell.safe_execfile(filepath, shell.user_ns, raise_exceptions=True)
        else:
            if globals is None and locals is None:
                # Register globals and locals of the above frame for code execution
                globals = sys._getframe(1).f_globals
                locals = sys._getframe(1).f_locals

            execution_code = filepath.read_text()
            exec(execution_code, globals, locals)
    except Exception as e:
        e.args = (e.args[0], f"SilQ initialization error in {filepath}", *e.args[1:])
        tb = sys.exc_info()[2]
        raise e.with_traceback(tb.tb_next.tb_next.tb_next)


def run_script(
        script_name: str,
        folder: Union[Path, str] = ''  ,
        silent: bool = False,
        mode: str = None,
        globals: dict = None,
        locals: dict = None
):
    """Run a single script, by default from the experiment scripts folder

    Args:
        script_name: Name of script. Can contain slashes (/) for a script in a folder
        folder: Script folder. Can be either absolute or relative path.
            If relative, the folder is w.r.t. the experiment folder.
        silent: Whether to print the execution of the script
        mode: Whether a specific initialize mode is used.
            script files can start with '#ignore_mode: {mode}'.
            If 'ignore_mode' matches the specified mode, the script is not executed.
        globals: Optional global variables
        locals: Optional local variables

    Returns:
        None
    """
    if globals is None and locals is None:
        # Register globals and locals of the above frame (for code execution)
        globals = sys._getframe(1).f_globals
        locals = sys._getframe(1).f_locals

    if isinstance(folder, str):
        folder = Path(folder)

    if not folder.is_absolute():
        folder = Path(qc.config.user.mainfolder) / folder

    file = folder / script_name
    file = file.with_suffix('.py')
    if not file.exists():
        raise FileNotFoundError(f'Script file {file} not found')

    if mode is not None:
        execution_code = file.read_text() + "\n"
        for line in execution_code.split('\n'):
            if not line.startswith('# '):
                break
            else:
                line = line.lstrip('# ')
                if line.startswith('ignore_mode: '):
                    ignore_mode = line.split('ignore_mode: ')[1]
                    if mode == ignore_mode:
                        if not silent:
                            print(f'Ignoring script {file.stem}')
                        return

    if not silent:
        print(f'Running script {file.stem}')
    execute_file(file, globals=globals, locals=locals)


def create_new_notebook(path, open=False, create_dirs=False, cells=None):
    """Creates a new empty notebook"""
    path = Path(path)

    if path.exists():
        print(f'Notebook already exists {path}')
    else:
        # Create parent folder(s) if necessary
        if create_dirs and not path.parent.exists():
            os.makedirs(path.parent)

        # Create notebook in folder
        notebook = nbformat.v4.new_notebook()

        if cells is not None:
            for cell in cells:
                notebook['cells'].append(nbformat.v4.new_code_cell(cell))

        nbformat.write(notebook, path)

    if open:
        os.startfile(path)


def configure_device_folder(
    root_folder=None, 
    silent=True, 
    create_daily_measurement_notebook=True
):
    if root_folder is None:
        root_folder = qc.config.user.analysis_folder
    root_folder = Path(root_folder)

    folders = [
        root_folder, 
        root_folder / 'Measurement notebooks',
        root_folder / 'Analysis notebooks',
    ]

    for folder in folders:
        if not folder.exists():
            os.mkdir(folder)
            if not silent:
                print(f'Created folder {folder}')

    # Create daily notebook file
    if create_daily_measurement_notebook:
        import time
        today = time.strftime("%Y-%m-%d")
        notebook_file = root_folder / f'Measurement notebooks/{today}.ipynb'
        if not notebook_file.exists():
            namespace_code = update_namespace(replace=False)
            create_new_notebook(notebook_file, open=True, cells=[namespace_code, ""])
            if not silent:
                print(f'Created daily measurement notebook {notebook_file}')


def update_namespace(silent=True, replace=True, separator='; '):
    import builtins
    builtin_types= [t  for t in builtins.__dict__.values()  if isinstance(t, type)]
    builtin_types += [type(update_namespace)]  # function, is there a better way for this?

    # Collect all elements
    elems = []
    for key, val in list(globals().items()):
        var_type = val.__class__.__name__
        text = f'{key}:{var_type}'

        if type(val) in builtin_types or var_type in globals():
            elems.append(text)
        elif not silent:
            print(f'Skipped {text}')

    definition_code = '# Updating namespace types\n'
    definition_code += 'function = type(update_namespace)'+separator
    definition_code += separator.join(elems)

    # Update current cell
    if replace:
        shell = get_ipython()
        shell.set_next_input(definition_code, replace=True)
    else:
        return definition_code