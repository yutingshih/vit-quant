from vit_quant._core import hello_from_bin

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

print(hello_from_bin())
