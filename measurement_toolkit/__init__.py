# Disable annoying NoTagError message in qcodes
from versioningit.logging import log
log.disabled = True

from .analysis import *
from .parameters import *
from .tools import *
from .measurements import *
from .code_injections import *

import qcodes as qc


__all__ = ['qc']
from . import (
    analysis,
    parameters, 
    measurements, 
    tools, 
    code_injections
)
__all__ += analysis.__all__
__all__ += parameters.__all__
__all__ += measurements.__all__
__all__ += tools.__all__
__all__ += code_injections.__all__


### Perform default initialization routines when in IPython
# This can be skipped by setting SKIP_MEASUREMENT_TOOLKIT_INITIALIZATION to True
from IPython import get_ipython
shell = get_ipython()
if shell is not None:
    if not shell.user_ns.get('SKIP_MEASUREMENT_TOOLKIT_INITIALIZATION', False):
        # Perform code injections, mostly matplotlib functionality
        perform_code_injections()

        # Start logging all input / output
        from qcodes.logger import start_all_logging
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            start_all_logging()
        
        # Initialize station
        from pathlib import Path
        if qc.Station.default is not None:
            station = qc.Station.default
        elif 'station_file' in qc.config.user and Path(qc.config.user.station_file).exists():
            from qcodes.station import ValidationWarning
            import warnings
            warnings.simplefilter('ignore', category=ValidationWarning)
            station = qc.Station(config_file=qc.config.user.station_file)
        else:
            print('creating new station')
            station = qc.Station()
        station.instruments_loaded = getattr(station, 'instruments_loaded', False)
        shell.user_ns.setdefault('station', station)