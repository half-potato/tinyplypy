# tinyply-numpy-binding

A minimal Python library that provides a [pybind11](https://github.com/pybind/pybind11)-based wrapper around [tinyply](https://github.com/ddiakopoulos/tinyply), allowing you to read/write `.ply` files directly into/from NumPy arrays. This lets you seamlessly integrate PLY geometry data with Python libraries such as NumPy, SciPy, or scikit-learn.

Written entirely by ChatGPT O1 because `plyfile` is horrifically slow.

## Features

- **Read `.ply` files** into a Python nested dictionary:
  ```python
  data = read_ply("mesh.ply")
  ```
- Write .ply files from the same nested dictionary structure:
```python
write_ply("mesh_out.ply", data_dict, isBinary=True)
```
- Handles scalar properties as 1D NumPy arrays and fixed-size list properties (e.g., face indices) as 2D arrays.
- Supports both ASCII and binary PLY formats.
- DATA MUST BE CONTIGUOUS for each element. Not sure how to fix this right now while still handling ownership of the contiguous array correctly.
## Installation
You can install this package from source or via wheel:
Install pybind11 (required):
```bash
pip install pybind11
```
Build and install:
```bash
# If you have the source repo:
python setup.py install
```
Alternatively, build a wheel:
```bash
python setup.py bdist_wheel
# Then install from the generated .whl
```
