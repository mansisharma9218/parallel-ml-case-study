# Must run before NumPy is imported anywhere in this package.
from . import _pin_blas  # noqa: F401
