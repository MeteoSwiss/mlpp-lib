__version__ = "1.0.0"

import os

# mlpp-lib relies on the torch backend of keras (and on torch.distributions).
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402

if keras.backend.backend() != "torch":
    raise ImportError(
        "mlpp-lib requires the torch backend of keras, but keras is using the "
        f"'{keras.backend.backend()}' backend. Set the environment variable "
        "KERAS_BACKEND=torch before importing keras (or mlpp_lib)."
    )

# import the modules defining serializable keras objects, so that saved models
# can be loaded after `import mlpp_lib`
from mlpp_lib import layers, losses, metrics, models, physical_layers  # noqa: E402,F401
from mlpp_lib import probabilistic_layers  # noqa: E402,F401
