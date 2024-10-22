import logging
from dataclasses import dataclass, field
from abc import abstractmethod

import numpy as np
import xarray as xr
from typing_extensions import Self

_LOGGER = logging.getLogger(__name__)


def create_imputation_from_str(class_name: str, inputs: Optional[dict] = None):
    """"""
    cls = globals().get(class_name)
    if not cls or not issubclass(cls, DataImputation):
        raise ValueError(f"{class_name} is not a valid subclass of DataImputation")
    return cls(**(inputs or {}))


@dataclass
class DataImputer:
    """
    Class for filling missing values in a dataset with different techniques.
    """
    #TODO: 
    #    - create subclasses to handle different types of imputing:
    #        - location-based (stations) / elevation
    #        - based on valid time -> may be more appropriate than lead time


    name: str = "DataImputer"

    def __init__(
        self,
        method_vars_dict: Optional[dict[str, list[str]]] = None,
        default: str = "ConstantImputer",
        fillvalue: Optional[float] = -5,
    ):
        self.method_vars_dict = method_vars_dict
        self.default = default
        self.fillvalue = fillvalue
        self.all_vars = []
        self.imputers = {}

        # Initialize default imputation
        default_imputer = create_imputation_from_str(
            self.default, {"fillvalue": self.fillvalue} #TODO: need to think about this, a priori most imputers won't have a fillvalue field
        )
        self.imputers[self.default] = (default_imputer, [])

        # Initialize other imputations
        if method_vars_dict is None:
            method_vars_dict = {}
        for method, variables in method_vars_dict.items():
            variables = [var for var in variables if var not in self.all_vars]
            if not variables:
                continue
            method_cls = create_imputation_from_str(method)
            self.imputer[method] = (method_cls, variables)
            self.all_vars.extend(variables)

    def fit(self, dataset: xr.Dataset, dims: Optional[list] = None):
        """Fit imputations to the dataset."""
        datavars = list(dataset.data_vars)
        remaining_var = list(set(datavars) - set(self.all_vars))

        if remaining_var:
            # assign remaining variables to default imputation
            self.imputers[self.default][1].extend(remaining_var)
            self.all_vars.extend(remaining_var)

        for name, (imputer, variables) in self.imputer.items():
            try:
                imputer.fit(dataset=dataset[variables], dims=dims)
                LOGGER.info(f"Imputed {name} to variables: {variables}")
            except Exception as e:
                LOGGER.error(f"Failed to fit {name}: {e}")
                raise e

    def fill(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        """Apply filling to the dataset(s)."""
        for imputation in self.imputers.values():
            imputer, variables = imputation
            # ensure that only variables that are in the dataset are imputed
            vars_in_data = list(set(variables) & set(datasets[0].data_vars))
            if not vars_in_data:
                continue
            datasets_ = imputer.fill(*[ds[vars_in_data] for ds in datasets])
            for i, dataset_ in enumerate(datasets_):
                datasets[i].update(dataset_)
        return tuple(datasets)

    def inverse_fill(self, *datasets: xr.Dataset) -> xr.Dataset:
        """Apply inverse filling to the dataset(s)."""
        for imputer, variables in self.imputers.values():
            vars_in_data = list(set(variables) & set(datasets[0].data_vars))
            if not vars_in_data:
                continue
            datasets_ = imputer.inverse_fill(
                *[ds[vars_in_data] for ds in datasets]
            )
            for i, dataset_ in enumerate(datasets_):
                datasets[i].update(dataset_)
        return tuple(datasets)

    @classmethod
    def from_dict(cls, in_dict: dict) -> Self:
        """Create a DataImputer instance from a dictionary representation."""
        init_dict, method_dicts = cls._parse_init_dicts(in_dict)
        dataimputer = cls(**init_dict)
        for method_name, method_dict in method_dicts.items():
            subclass = create_imputation_from_str(method_name).from_dict(
                method_dict
            )
            dataimputer.imputers[method_name] = (
                subclass,
                dataimputer.imputers[method_name][1],
            )
        return dataimputer

    @staticmethod
    def _parse_init_dicts(in_dict: dict) -> tuple[dict[str, list[str]], dict]:
        """Parse the input dictionary to the expected format for initialization."""
        first_key = next(iter(in_dict))

        #TODO: this part should ensure retro-compatibility and therefore access DataTransformer ...
        if first_key not in [
            cls.__name__ for cls in DataImputation.__subclasses__()
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
        """Convert the DataImputer instance to a dictionary."""
        out_dict = {}
        for name, (imputer, variables) in self.imputers.items():
            out_dict_tmp = imputer.to_dict()
            out_dict_tmp["channels"] = variables
            out_dict[name] = out_dict_tmp
        return out_dict

    @classmethod
    def from_json(cls, in_fn: str) -> Self:
        """Create a DataImputer instance from a JSON file."""
        with open(in_fn, "r") as f:
            in_dict = json.load(f)
        return cls.from_dict(in_dict)

    def save_json(self, out_fn: str) -> None:
        """Save the DataImputer configuration to a JSON file."""
        out_dict = self.to_dict()
        if not out_dict or not out_dict[list(out_dict.keys())[0]]:
            raise ValueError(f"{self.name} wasn't fit to data")
        with open(out_fn, "w") as outfile:
            json.dump(out_dict, outfile, indent=4)

class DataImputation:
    """
    Abstract class for nromalization techniques in a xarray.Dataset object.
    """
    
    @abstractmethod
    def fit(self, dataset: xr.Dataset, dims: Optional[list] = None) -> None:
        pass

    @abstractmethod
    def fill(self, dataset: xr.Dataset, dims: Optional[list] = None) -> tuple[xr.Dataset, ...]:
        pass

    @abstractmethod
    def inverse_fill(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        pass

    @abstractmethod
    def from_dict(self, in_dict) -> Self:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass


@dataclass
class ConstantImputer(DataImputation):
    """
    Fill missing values with a constant value.
    """
    #TODO: rethink a bit the logic, a priori fillvalue could have one distinct value for each variable

    fillvalue: dict[str, float] = field(init=True, default=-5)
    name = "ConstantImputer"

    def fit(self,
        dataset: xr.Dataset,
        dims: Optional[list] = None,
    ):
        self.fillvalue = self.fillvalue

    def fill(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        
        def f(ds: xr.Dataset):
            ds = ds.copy()
            ds = ds.fillna(self.fillvalue)
            return ds.astype("float32")

        return tuple(f(ds) for ds in datasets)

    def inverse_fill(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        def f(ds: xr.Dataset) -> xr.Dataset:
            ds = ds.copy()
            ds = ds.where(ds > self.fillvalue)
            return ds.astype("float32")

        return tuple(f(ds) for ds in datasets)

    @classmethod
    def from_dict(cls, in_dict: dict) -> Self:
        fillvalue = in_dict["fillvalue"]
        return cls(fillvalue=fillvalue)

    def to_dict(self):
        out_dict = {
            "fillvalue": self.fillvalue,
        }
        return out_dict

@dataclass
class PersitentImputer(DataImputation):
    """
    Fill missing value by using the latest lead time available (persistence)
    """
    persisted_vars: str = field(default=None)
    name = "PersistentImputer"

    def fit(self,
        dataset: xr.Dataset,
        dims: Optional[list] = None,
    ):
        self.persisted_vars = [var for var in dataset.data_vars if "lead_time" in dataset[var].dims]

    def fill(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        
        def f(ds: xr.Dataset):
            ds = ds.copy()
            for var in self.persisted_vars:
               ds[var] = ds[var].ffill("lead_time")

            return ds.astype("float32")

        return tuple(f(ds) for ds in datasets)

    def inverse_fill(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        # WIP, needs to be tested
        def f(ds: xr.Dataset) -> xr.Dataset:
            ds = ds.copy()
            for var in self.persisted_vars:
                shifted = ds[var].shift({"lead_time": -1})
                mask = (ds[var] == shifted)
                mask = mask.cumsum(dim="lead_time") > 0
                ds[var] = ds[var].where(~mask, np.nan)
                    
            return ds.astype("float32")

        return tuple(f(ds) for ds in datasets)

    @classmethod
    def from_dict(cls, in_dict: dict) -> Self:
        persisted_vars = in_dict["persisted_vars"]
        return cls(persisted_vars=persisted_vars)

    def to_dict(self):
        out_dict = {
            "persisted_vars": self.persisted_vars,
        }
        return out_dict
