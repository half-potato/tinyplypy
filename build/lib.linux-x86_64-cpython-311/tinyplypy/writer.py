import numpy as np
from . import _tinyplypy_binding

def write_ply(filename, data, is_binary=True):
    """
    Ensure all arrays in 'data' are contiguous before passing to the C++ extension.
    data is expected to have the structure:
      data[element_name][property_name] = np.array(...)
    """
    # We'll create a new dictionary, new_data, that is guaranteed to have contiguous arrays
    new_data = {}

    for element_name, props in data.items():
        contiguous_props = {}
        for prop_name, arr in props.items():
            # Convert to a NumPy array if it's not already
            arr = np.asarray(arr)
            arr = np.ascontiguousarray(arr)
            contiguous_props[prop_name] = arr
        new_data[element_name] = contiguous_props

    # Call the pybind11-based function to write out the PLY
    _tinyplypy_binding.write_ply(filename, new_data, is_binary)
