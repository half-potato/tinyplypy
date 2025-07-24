
# Re-export from our writer module
from .writer import write_ply

# If you want direct access to the C++ read_ply, you can do:
from ._tinyplypy_binding import read_ply

__all__ = [
    "write_ply",
    "read_ply",
]

