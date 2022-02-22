import builtins
import sys
from pathlib import Path
from typing import Union
import logging

import qcodes as qc

__all__ = [
    'export_notebook',
    'get_last_input_cells',
    'using_ipython',
    'execute_file',
    'run_script'
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


def export_notebook(notebook_name=None, input_code=False, output='html'):
    from IPython import get_ipython
    ipython = get_ipython()

    if notebook_name is None:
        notebook_name = get_notebook_path().name

    notebook_path = Path(notebook_name).with_suffix('.ipynb')
    assert notebook_path.exists()
    print(f'Converting notebook "{notebook_path}"')

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


def execute_file(filepath: (str, Path), mode=None, globals=None, locals=None):
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
        e.args = (
            e.args[0] + f"\nSilQ initialization error in {filepath}", *e.args[1:]
        )
        raise e


def run_script(
        script_name: str,
        folder: Union[Path, str] = 'scripts',
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