from .parameters import *
from .measurements import *
from .tools import *
from .code_injections import *


__all__ = []
from . import parameters, measurements, tools, code_injections
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
        start_all_logging()
        
        # Initialize station
        import qcodes as qc
        station = qc.Station.default or qc.Station()
        station.instruments_loaded = getattr(station, 'instruments_loaded', False)
        shell.user_ns.setdefault('station', station)