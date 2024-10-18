import logging
from dataclasses import dataclass, field
from abc import abstractmethod

import numpy as np
import xarray as xr
from typing_extensions import Self


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


@dataclass
class ConstantFiller(ValueFiller):
    """
    Fill missing values with a constant value.
    """

    fillvalue: dict[str, float] = field(init=True, default=-5)

    def fill(self, data: xr.Dataset) -> xr.Dataset:
        return data.fillna(self.fillvalue)
