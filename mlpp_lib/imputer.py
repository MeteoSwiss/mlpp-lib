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
    #        - constant
    #        - location-based (stations) / elevation
    #        - based on lead time ? E.g., replace by the mean of the distribution for observed values at such lead time
    #        - based on valid time -> may be more appropriate than lead time

    @abstractmethod
    def fill(self, data: xr.Dataset) -> xr.Dataset:
        """
        Fill missing values in the dataset.
        """
        pass

class DataImputation:
    pass


@dataclass
class ConstantImputation(DataImputation):
    """
    Fill missing values with a constant value.
    """

    fillvalue: dict[str, float] = field(init=True, default=-5)
    name = "ConstantImputation"

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
