import logging
from dataclasses import dataclass, field
from abc import abstractmethod

import numpy as np
import xarray as xr
from typing_extensions import Self

_LOGGER = logging.getLogger(__name__)


def create_transformation_from_str(class_name: str, inputs: Optional[dict] = None):
    """"""
    cls = globals().get(class_name)
    if not cls or not issubclass(cls, DataImputation):
        raise ValueError(f"{class_name} is not a valid subclass of DataImputation")
    return cls(**(inputs or {}))


@dataclass
class Imputer:
    """
    Abstract class for filling missing values in a dataset.
    """
    #TODO: 
    #    - create subclasses to handle different types of imputing:
    #        - location-based (stations) / elevation
    #        - based on valid time -> may be more appropriate than lead time

    @abstractmethod
    def fill(self, data: xr.Dataset) -> xr.Dataset:
        """
        Fill missing values in the dataset.
        """
        pass

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
class ConstantImputation(DataImputation):
    """
    Fill missing values with a constant value.
    """

    fillvalue: dict[str, float] = field(init=True, default=-5)
    name = "ConstantImputation"

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
class PersitentImputation(DataImputation):
    """
    Fill missing value by using the latest lead time available (persistence)
    """
    persisted_vars: str = field(default=None)
    name = "PersistentImputation"

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
