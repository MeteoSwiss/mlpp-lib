from importlib.metadata import PackageNotFoundError, version
import os

try:
    __version__ = version("mlpp-lib")
except PackageNotFoundError:
    __version__ = "0.0.0"

os.environ["TF_USE_LEGACY_KERAS"] = "1"
