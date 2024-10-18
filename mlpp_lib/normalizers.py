import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from abc import abstractmethod

import xarray as xr
from typing_extensions import Self

LOGGER = logging.getLogger(__name__)


def create_transformation_from_str(class_name: str, inputs: Optional[dict] = None):
    """"""
    cls = globals().get(class_name)
    if not cls or not issubclass(cls, DataTransformation):
        raise ValueError(f"{class_name} is not a valid subclass of DataTransformation")
    return cls(**(inputs or {}))


@dataclass
class DataTransformer:
    """
    Class to handle the transformation of data in a xarray.Dataset object with different techniques.
    """

    name: str = "DataTransformer"

    def __init__(
        self,
        method_vars_dict: Optional[dict[str, list[str]]] = None,
        default: str = "Standardizer",
        fillvalue: Optional[float] = -5,
    ):
        self.method_vars_dict = method_vars_dict
        self.default = default
        self.fillvalue = fillvalue
        self.all_vars = []
        self.transformers = {}

        # Initialize default transformation
        default_transformer = create_transformation_from_str(
            self.default, {"fillvalue": self.fillvalue}
        )
        self.transformers[self.default] = (default_transformer, [])

        # Initialize other transformations
        if method_vars_dict is None:
            method_vars_dict = {}
        for method, variables in method_vars_dict.items():
            variables = [var for var in variables if var not in self.all_vars]
            if not variables:
                continue
            method_cls = create_transformation_from_str(method)
            self.transformers[method] = (method_cls, variables)
            self.all_vars.extend(variables)

    def fit(self, dataset: xr.Dataset, dims: Optional[list] = None):
        """Fit transformations to the dataset."""
        datavars = list(dataset.data_vars)
        remaining_var = list(set(datavars) - set(self.all_vars))

        if remaining_var:
            # assign remaining variables to default transformation
            self.transformers[self.default][1].extend(remaining_var)
            self.all_vars.extend(remaining_var)

        for name, (transformer, variables) in self.transformers.items():
            try:
                transformer.fit(dataset=dataset[variables], dims=dims)
                LOGGER.info(f"Fitted {name} to variables: {variables}")
            except Exception as e:
                LOGGER.error(f"Failed to fit {name}: {e}")
                raise e

    def transform(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        """Apply transformations to the dataset(s)."""
        for transformation in self.transformers.values():
            transformer, variables = transformation
            # ensure that only variables that are in the dataset are transformed
            vars_in_data = list(set(variables) & set(datasets[0].data_vars))
            if not vars_in_data:
                continue
            datasets_ = transformer.transform(*[ds[vars_in_data] for ds in datasets])
            for i, dataset_ in enumerate(datasets_):
                datasets[i].update(dataset_)
        return tuple(datasets)

    def inverse_transform(self, *datasets: xr.Dataset) -> xr.Dataset:
        """Apply inverse transformations to the dataset(s)."""
        for transformer, variables in self.transformers.values():
            vars_in_data = list(set(variables) & set(datasets[0].data_vars))
            if not vars_in_data:
                continue
            datasets_ = transformer.inverse_transform(
                *[ds[vars_in_data] for ds in datasets]
            )
            for i, dataset_ in enumerate(datasets_):
                datasets[i].update(dataset_)
        return tuple(datasets)

    @classmethod
    def from_dict(cls, in_dict: dict) -> Self:
        """Create a DataTransformer instance from a dictionary representation."""
        init_dict, method_dicts = cls._parse_init_dicts(in_dict)
        datatransfomer = cls(**init_dict)
        for method_name, method_dict in method_dicts.items():
            subclass = create_transformation_from_str(method_name).from_dict(
                method_dict
            )
            datatransfomer.transformers[method_name] = (
                subclass,
                datatransfomer.transformers[method_name][1],
            )
        return datatransfomer

    @staticmethod
    def _parse_init_dicts(in_dict: dict) -> tuple[dict[str, list[str]], dict]:
        """Parse the input dictionary to the expected format for initialization."""
        first_key = next(iter(in_dict))

        if first_key not in [
            cls.__name__ for cls in DataTransformation.__subclasses__()
        ]:
            channels = list(in_dict["mean"]["data_vars"].keys())
            init_dict = {"method_vars_dict": {"Standardizer": channels}}
            method_dicts = {"Standardizer": in_dict}
        else:
            method_vars_dict = {}
            method_dicts = {}
            for method_name, inner_dict in in_dict.items():
                method_vars_dict[method_name] = inner_dict.pop("channels")
                method_dicts[method_name] = inner_dict
            init_dict = {"method_vars_dict": method_vars_dict}
            method_dicts = in_dict
        return init_dict, method_dicts

    def to_dict(self):
        """Convert the DataTransformer instance to a dictionary."""
        out_dict = {}
        for name, (transformer, variables) in self.transformers.items():
            out_dict_tmp = transformer.to_dict()
            out_dict_tmp["channels"] = variables
            out_dict[name] = out_dict_tmp
        return out_dict

    @classmethod
    def from_json(cls, in_fn: str) -> Self:
        """Create a DataTransformer instance from a JSON file."""
        with open(in_fn, "r") as f:
            in_dict = json.load(f)
        return cls.from_dict(in_dict)

    def save_json(self, out_fn: str) -> None:
        """Save the DataTransformer configuration to a JSON file."""
        out_dict = self.to_dict()
        if not out_dict or not out_dict[list(out_dict.keys())[0]]:
            raise ValueError(f"{self.name} wasn't fit to data")
        with open(out_fn, "w") as outfile:
            json.dump(out_dict, outfile, indent=4)


class DataTransformation:
    """
    Abstract class for nromalization techniques in a xarray.Dataset object.
    """

    @abstractmethod
    def fit(self, dataset: xr.Dataset, dims: Optional[list] = None) -> None:
        pass

    @abstractmethod
    def transform(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        pass

    @abstractmethod
    def inverse_transform(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        pass

    @abstractmethod
    def from_dict(self, in_dict) -> Self:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass


@dataclass
class Identity(DataTransformation):
    """
    Identity transformation, returns the input data without any transformation.
    """

    identity_vars: str = field(default=None)
    fillvalue: Optional[float] = field(default=None)
    name = "Identity"

    def fit(
        self,
        dataset: xr.Dataset,
        dims: Optional[list] = None,
    ):
        self.fillvalue = self.fillvalue
        self.identity_vars = list(dataset.data_vars)

    def transform(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        def f(ds: xr.Dataset) -> xr.Dataset:
            ds = ds[self.identity_vars].copy()
            if self.fillvalue:
                ds = ds.fillna(self.fillvalue)
            else:
                if ds.isnull().any():
                    raise ValueError("Missing values found in the data. Please provide a fill value.")
            return ds.astype("float32")

        return tuple(f(ds) for ds in datasets)

    def inverse_transform(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        def f(ds: xr.Dataset) -> xr.Dataset:
            ds = ds[self.identity_vars].copy()
            if self.fillvalue:
                ds = ds.where(ds != self.fillvalue)
            return ds.astype("float32")

        return tuple(f(ds) for ds in datasets)

    @classmethod
    def from_dict(cls, in_dict: dict) -> Self:
        identity_vars = in_dict["identity_vars"]
        fillvalue = in_dict["fillvalue"]
        return cls(identity_vars, fillvalue=fillvalue)

    def to_dict(self) -> dict:
        out_dict = {
            "identity_vars": self.identity_vars,
            "fillvalue": self.fillvalue,
        }
        return out_dict


@dataclass
class Standardizer(DataTransformation):
    """
    Transforms data using a z-normalization in a xarray.Dataset object.
    """

    mean: xr.Dataset = field(default=None)
    std: xr.Dataset = field(default=None)
    fillvalue: dict[str, float] = field(init=True, default=-5.0)
    name = "Standardizer"

    def fit(
        self,
        dataset: xr.Dataset,
        dims: Optional[list] = None,
    ):
        self.mean = dataset.mean(dims).compute().copy()
        self.std = dataset.std(dims).compute().copy()
        self.fillvalue = self.fillvalue
        # Check for near-zero standard deviations and set them equal to one
        self.std = xr.where(self.std < 1e-6, 1, self.std)

    def transform(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        if self.mean is None:
            raise ValueError("Standardizer wasn't fit to data")

        def f(ds: xr.Dataset):
            ds = ds.copy()
            ds = (ds - self.mean) / self.std
            if self.fillvalue:
                ds = ds.fillna(self.fillvalue)
            return ds.astype("float32")

        return tuple(f(ds) for ds in datasets)

    def inverse_transform(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        if self.mean is None:
            raise ValueError("Standardizer wasn't fit to data")

        def f(ds: xr.Dataset) -> xr.Dataset:
            ds = ds.copy()
            if self.fillvalue:
                ds = ds.where(ds > self.fillvalue)
            ds = ds * self.std + self.mean
            return ds.astype("float32")

        return tuple(f(ds) for ds in datasets)

    @classmethod
    def from_dict(cls, in_dict: dict) -> Self:
        mean = xr.Dataset.from_dict(in_dict["mean"])
        std = xr.Dataset.from_dict(in_dict["std"])
        fillvalue = in_dict["fillvalue"]
        return cls(mean, std, fillvalue=fillvalue)

    def to_dict(self):
        out_dict = {
            "mean": self.mean.to_dict(),
            "std": self.std.to_dict(),
            "fillvalue": self.fillvalue,
        }
        return out_dict


@dataclass
class MinMaxScaler(DataTransformation):
    """
    Transforms data to [0, 1] using a min/max scaling in a xarray.Dataset object.
    """

    min: xr.Dataset = field(default=None)
    max: xr.Dataset = field(default=None)
    fillvalue: dict[str, float] = field(init=True, default=-0.1)
    name = "MinMaxScaler"

    def fit(
        self,
        dataset: xr.Dataset,
        dims: Optional[list] = None,
    ):
        self.min = dataset.min(dims).compute().copy()
        self.max = dataset.max(dims).compute().copy()
        self.fillvalue = self.fillvalue

    def transform(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        if self.min is None:
            raise ValueError("MinMaxScaler wasn't fit to data")

        def f(ds: xr.Dataset) -> xr.Dataset:
            ds = ds.copy()
            ds = (ds - self.min) / (self.max - self.min)
            if self.fillvalue is not None:
                ds = ds.fillna(self.fillvalue)
            return ds.astype("float32")

        return tuple(f(ds) for ds in datasets)

    def inverse_transform(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        if self.min is None:
            raise ValueError("MinMaxScaler wasn't fit to data")

        def f(ds: xr.Dataset) -> xr.Dataset:
            ds = ds.copy()
            if self.fillvalue:
                ds = ds.where(ds > self.fillvalue)
            ds = ds * (self.max - self.min) + self.min
            return ds.astype("float32")

        return tuple(f(ds) for ds in datasets)

    @classmethod
    def from_dict(cls, in_dict: dict) -> Self:
        minimum = xr.Dataset.from_dict(in_dict["minimum"])
        maximum = xr.Dataset.from_dict(in_dict["maximum"])
        return cls(minimum, maximum)

    def to_dict(self):
        out_dict = {
            "minimum": self.min.to_dict(),
            "maximum": self.max.to_dict(),
            "fillvalue": self.fillvalue,
        }
        return out_dict


@dataclass
class MaxAbsScaler(DataTransformation):
    """
    Transforms data to [-1, 1] by scaling with the maximum of the absolute values in a xarray.Dataset object.
    """

    maxabs: xr.Dataset = field(default=None)
    fillvalue: dict[str, float] = field(init=True, default=-1.1)
    name = "MaxAbsScaler"

    def fit(
        self,
        dataset: xr.Dataset,
        dims: Optional[list] = None,
    ):
        self.maxabs = abs(dataset).max(dims).compute().copy()
        self.fillvalue = self.fillvalue

    def transform(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        if self.maxabs is None:
            raise ValueError("MaxAbsScaler wasn't fit to data")

        def f(ds: xr.Dataset) -> xr.Dataset:
            ds = ds.copy()
            ds = ds / self.maxabs
            if self.fillvalue:
                ds = ds.fillna(self.fillvalue)
            return ds.astype("float32")

        return tuple(f(ds) for ds in datasets)

    def inverse_transform(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        if self.maxabs is None:
            raise ValueError("MaxAbsScaler wasn't fit to data")

        def f(ds: xr.Dataset) -> xr.Dataset:
            ds = ds.copy()
            ds = ds * self.maxabs
            return ds.astype("float32")

        return tuple(f(ds) for ds in datasets)

    @classmethod
    def from_dict(cls, in_dict: dict) -> Self:
        maxabs = xr.Dataset.from_dict(in_dict["maxabs"])
        return cls(maxabs)

    def to_dict(self):
        out_dict = {
            "maxabs": self.maxabs.to_dict(),
            "fillvalue": self.fillvalue,
        }
        return out_dict
