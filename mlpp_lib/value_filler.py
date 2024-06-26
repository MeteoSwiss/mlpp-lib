import logging
from dataclasses import dataclass, field
from abc import abstractmethod

import numpy as np
import xarray as xr
from typing_extensions import Self


class ValueFiller:
    """
    Abstract class for filling missing values in a dataset.
    """

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