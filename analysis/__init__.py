from ._filter import fvtool
from ._audio import plot, feature

# Find all the methods in the plot, and instantiate them
for method in dir(plot):
    if not method.startswith("__") and callable(getattr(plot, method)):
        globals()[method] = getattr(plot, method)

# Find all the methods in the plot, and instantiate them
for method in dir(feature):
    if not method.startswith("__") and callable(getattr(feature, method)):
        globals()[method] = getattr(feature, method)

__all__ = ["fvtool", "waveform", "plot", "peak", "RMS"]
