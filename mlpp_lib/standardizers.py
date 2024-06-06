import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from abc import abstractmethod

import numpy as np
import xarray as xr
from typing_extensions import Self

LOGGER = logging.getLogger(__name__)


class Normalizer():
    """
    Abstract class for normalizing data in a xarray.Dataset object.
    """

    def __init__(self, subclass_var_dict: dict[Normalizer, list[str]]):
        pass

    @abstractmethod
    def fit():
        pass
    
    @abstractmethod
    def transform():
        pass
    
    @abstractmethod
    def inverse_transform():
        pass
    
    @abstractmethod
    def from_dict():
        pass
    
    @abstractmethod
    def to_dict():
        pass
    
    @abstractmethod
    def save_json():
        pass

    @classmethod
    def normalise(cls, data: xr.Dataset, method_variable_dict: dict[Normalizer, list[str]]) -> xr.Dataset:
        normalized_vars = []
        for method, variables in method_variable_dict.items():
            vars_to_remove = [var for var in vars if var in normalized_vars]
            if len(vars_to_remove) > 0:
                LOGGER.info(f"Variable(s) {[var for var in vars_to_remove]} already normalized")
                variables = [var for var in variables if var not in vars_to_remove]
            
            method.fit(data, variables)
            data = method.transform(data, variables)
            normalized_vars.extend(variables)


@dataclass
class Standardizer(Normalizer):
    """
    Standardizes data in a xarray.Dataset object.
    """

    mean: xr.Dataset = field(default=None)
    std: xr.Dataset = field(default=None)
    fillvalue: dict[str, float] = field(init=True, default=-5)

    def fit(self, dataset: xr.Dataset, vars: Optional[list] = None, dims: Optional[list] = None):
        
        if vars is None:
            vars = dataset.data_vars
        if not all(var in dataset.data_vars for var in vars):
            raise KeyError(f"There are variables not in dataset: {[var for var in vars if var not in dataset.data_vars]}")

        self.mean = dataset[vars].mean(dims).compute().copy()
        self.std = dataset[vars].std(dims).compute().copy()
        self.fillvalue = self.fillvalue
        # Check for near-zero standard deviations and set them equal to one
        self.std = xr.where(self.std < 1e-6, 1, self.std)

    def transform(self, *datasets: xr.Dataset, vars: Optional[list] = None) -> tuple[xr.Dataset, ...]:
        if self.mean is None:
            raise ValueError("Standardizer wasn't fit to data")
        if vars is None:
            vars = dataset.data_vars

        def f(ds: xr.Dataset):
            for var in vars:
                assert var in self.mean.data_vars, f"{var} not in Standardizer"
            ds = ((ds - self.mean) / self.std).astype("float32")
            if self.fillvalue is not None:
                ds = ds.fillna(self.fillvalue)
            return ds

        return tuple(f(ds) for ds in datasets)

    def inverse_transform(self, *datasets: xr.Dataset, vars: Optional[list] = None) -> xr.Dataset:
        if self.mean is None:
            raise ValueError("Standardizer wasn't fit to data")
        if vars is None:
            vars = dataset.data_vars

        def f(ds: xr.Dataset) -> xr.Dataset:
            for var in vars:
                assert var in self.mean.data_vars, f"{var} not in Standardizer"
            ds = xr.where(ds > self.fillvalue, ds, np.nan)
            return (ds * self.std + self.mean).astype("float32")

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

    @classmethod
    def from_json(cls, in_fn: str) -> Self:
        with open(in_fn, "r") as f:
            in_dict = json.load(f)
        return cls.from_dict(in_dict)

    def save_json(self, out_fn: str) -> None:
        if self.mean is None:
            raise ValueError("Standardizer wasn't fit to data")
        out_dict = self.to_dict()
        with open(out_fn, "w") as outfile:
            json.dump(out_dict, outfile, indent=4)


def standardize_split_dataset(
    split_dataset: dict[str, xr.Dataset],
    save_to_json: Optional[Path] = None,
    return_standardizer: bool = False,
    fit_kwargs: Optional[dict] = None,
) -> dict[str, xr.Dataset]:
    """Fit standardizer to the train set and applies it to all
    the sets. Optionally exports the standardizer as a json."""
    if fit_kwargs is None:
        fit_kwargs = {}
    standardizer = Standardizer()

    # Subsample reftimes and stations to speed up fitting of standardizer
    reftimes = split_dataset["train"].forecast_reference_time.values
    stations = split_dataset["train"].station.values
    reftimes = reftimes[: min(365 * 2, len(reftimes)) : 7]
    stations = stations[: min(100, len(stations))]
    subset = split_dataset["train"].sel(
        forecast_reference_time=reftimes, station=stations
    )
    standardizer.fit(subset, **fit_kwargs)
    if save_to_json:
        standardizer.save_json(save_to_json)

    for split in split_dataset:
        split_dataset[split] = standardizer.transform(split_dataset[split])

    if return_standardizer:
        return split_dataset, standardizer
    else:
        return split_dataset
