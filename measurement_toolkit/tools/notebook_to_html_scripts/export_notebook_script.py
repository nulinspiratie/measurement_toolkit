import sys
from pathlib import Path
from measurement_toolkit.tools.notebook_tools import export_notebook
    

if __name__ == '__main__':
    notebook_path = Path(sys.argv[1])
    print(f'{notebook_path=}')
    assert notebook_path.exists()

    export_notebook(
        notebook_name=notebook_path.name
    )