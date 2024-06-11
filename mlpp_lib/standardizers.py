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



def create_instance_from_str(class_name: str):

    cls = globals()[class_name]

    if issubclass(cls, Normalizer):
        return cls()
    else:
        raise ValueError(f"{class_name} is not a subclass of Normalizer")

class Normalizer:
    """
    Abstract class for normalizing data in a xarray.Dataset object.
    In principle it should not be instantiated, it only adds an extra level of abstraction.
    """

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
    def from_json():
        pass

    @abstractmethod
    def save_json():
        pass


class MultiNormalizer(Normalizer):
    """
    Abstract class for normalizing data in a xarray.Dataset object with different normalizations.
    """

    name = "MultiNormalizer"

    def __init__(self, method_var_dict: dict[str, list[str]] = None):
        seen_vars = []
        self.method_vars_list = []

        if method_var_dict is not None:
            for method, variables in method_var_dict.items():
                vars_to_remove = [var for var in variables if var in seen_vars]
                method_cls = create_instance_from_str(method) # ensure the proper functionning in case it is not an str
                
                if len(vars_to_remove) > 0:
                    LOGGER.info(f"Variable(s) {[var for var in vars_to_remove]} are already assigned to another normalization method")
                    variables = [var for var in variables if var not in vars_to_remove]
                
                self.method_vars_list.append((method_cls, variables))


    def fit(self, dataset: xr.Dataset, dims: Optional[list] = None):
        
        for i in range(len(self.method_vars_list)):
            method, variables = self.method_vars_list[i]
            method.fit(dataset=dataset, variables = variables, dims = dims)
            self.method_vars_list[i] = (method, variables)

    
    def transform(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        
        for item in self.method_vars_list:
            method, variables = item
            datasets = method.transform(*datasets, variables=variables)

        return datasets
    
    
    def inverse_transform(self, *datasets: xr.Dataset) -> xr.Dataset:
        
        for item in self.method_vars_list:
            method, variables = item
            datasets = method.inverse_transform(*datasets, variables=variables)

        return datasets
    
    
    @classmethod
    def from_dict(cls, in_dict: dict) -> Self:
        method_var_dict = {}
        for method_name, inner_dict in in_dict.items():
            tmp_class = create_instance_from_str(method_name)
            subclass = tmp_class.from_dict(inner_dict)
            variables = inner_dict["channels"]
            method_var_dict[subclass.name] = variables
        return cls(method_var_dict)
    
    
    def to_dict(self):
        out_dict = {}
        for item in self.method_vars_list:
            method, variables = item
            out_dict_tmp = method.to_dict()
            out_dict_tmp["channels"] = variables
            out_dict[method.name] = out_dict_tmp
        return out_dict


    @classmethod
    def from_json(cls, in_fn: str) -> Self:
        with open(in_fn, "r") as f:
            in_dict = json.load(f)
        return cls.from_dict(in_dict)
    
    
    def save_json(self, out_fn) -> None:
        out_dict = self.to_dict()
        with open(out_fn, "w") as outfile:
            json.dump(out_dict, outfile, indent=4)


@dataclass
class Standardizer(Normalizer):
    """
    Standardizes data in a xarray.Dataset object.
    """

    mean: xr.Dataset = field(default=None)
    std: xr.Dataset = field(default=None)
    fillvalue: dict[str, float] = field(init=True, default=-5)
    name = "Standardizer"

    def fit(self, dataset: xr.Dataset, variables: Optional[list] = None, dims: Optional[list] = None):

        if variables is None:
            variables = list(dataset.data_vars)
        if not all(var in dataset.data_vars for var in variables):
            raise KeyError(f"There are variables not in dataset: {[var for var in variables if var not in dataset.data_vars]}")

        self.mean = dataset[variables].mean(dims).compute().copy()
        self.std = dataset[variables].std(dims).compute().copy()
        self.fillvalue = self.fillvalue
        # Check for near-zero standard deviations and set them equal to one
        self.std = xr.where(self.std < 1e-6, 1, self.std)


    def transform(self, *datasets: xr.Dataset, variables: Optional[list] = None) -> tuple[xr.Dataset, ...]:
        if self.mean is None:
            raise ValueError("Standardizer wasn't fit to data")

        def f(ds: xr.Dataset, variables: Optional[list] = None):
            
            if variables is None:
                variables = list(ds.data_vars)
            for var in variables:
                assert var in self.mean.data_vars, f"{var} not in Standardizer"

                standardized_var = ((ds[var] - self.mean[var]) / self.std[var]).astype("float32")
                if self.fillvalue is not None:
                    standardized_var = standardized_var.fillna(self.fillvalue)
                
                ds[var] = standardized_var

            return ds

        return tuple(f(ds, variables) for ds in datasets)
    

    def inverse_transform(self, *datasets: xr.Dataset, variables: Optional[list] = None) -> tuple[xr.Dataset, ...]:
        if self.mean is None:
            raise ValueError("Standardizer wasn't fit to data")

        def f(ds: xr.Dataset, variables: Optional[list] = None) -> xr.Dataset:
            if variables is None:
                variables = ds.data_vars

            ds = xr.where(ds > self.fillvalue, ds, np.nan)
            for var in variables:
                assert var in self.mean.data_vars, f"{var} not in Standardizer"
            
                unstandardized_var = (ds[var] * self.std[var] + self.mean[var]).astype("float32")
                ds[var] = unstandardized_var
            
            return ds

        return tuple(f(ds, variables) for ds in datasets)

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

@dataclass
class MinMaxScaler(Normalizer):
    """
    Normalize data using a min/max scaling in a xarray.Dataset object.
    """

    minimum: xr.Dataset = field(default=None)
    maximum: xr.Dataset = field(default=None)
    fillvalue: dict[str, float] = field(init=True, default=-5)
    name = "MinMaxScaler"

    def fit(self, dataset: xr.Dataset, variables: Optional[list] = None, dims: Optional[list] = None):
        if variables is None:
            variables = list(dataset.data_vars)
        if not all(var in dataset.data_vars for var in variables):
            raise KeyError(f"There are variables not in dataset: {[var for var in variables if var not in dataset.data_vars]}")

        self.minimum = dataset[variables].min(dims).compute().copy()
        self.maximum = dataset[variables].max(dims).compute().copy()
        self.fillvalue = self.fillvalue

    def transform(self, *datasets: xr.Dataset, variables: Optional[list] = None) -> tuple[xr.Dataset, ...]:
        if self.minimum is None:
            raise ValueError("MinMaxScaler wasn't fit to data")

        def f(ds: xr.Dataset, variables: Optional[list] = None) -> xr.Dataset:
            if variables is None:
                variables = ds.data_vars
            for var in variables:
                assert var in self.minimum.data_vars, f"{var} not in MinMaxScaler"

                scaled_var = ((ds[var] - self.minimum[var]) / (self.maximum[var] - self.minimum[var])).astype("float32")

                if self.fillvalue is not None:
                    scaled_var = scaled_var.fillna(self.fillvalue)

                ds[var] = scaled_var  
            return ds
        
        return tuple(f(ds, variables) for ds in datasets)
    

    def inverse_transform(self, *datasets: xr.Dataset, variables: Optional[list] = None) -> tuple[xr.Dataset, ...]:
        if self.minimum is None:
            raise ValueError("MinMaxScaler wasn't fit to data")

        def f(ds: xr.Dataset, variables: Optional[list] = None) -> xr.Dataset:
            if variables is None:
                variables = ds.data_vars
            
            ds = xr.where(ds > self.fillvalue, ds, np.nan)
            for var in variables:
                assert var in self.minimum.data_vars, f"{var} not in MinMaxScaler"

                unscaled_var = (ds[var] * (self.maximum[var] - self.minimum[var]) + self.minimum[var]).astype("float32")
                ds[var] = unscaled_var
            
            return ds
        
        return tuple(f(ds, variables) for ds in datasets)
    

    @classmethod
    def from_dict(cls, in_dict: dict) -> Self:
        minimum = xr.Dataset.from_dict(in_dict["minimum"])
        maximum = xr.Dataset.from_dict(in_dict["maximum"])
        return cls(minimum, maximum)
    

    def to_dict(self):
        out_dict = {
            "minimum": self.minimum.to_dict(),
            "maximum": self.maximum.to_dict(),
        }
        return out_dict
    

    @classmethod
    def from_json(cls, in_fn: str) -> Self:
        with open(in_fn, "r") as f:
            in_dict = json.load(f)
        return cls.from_dict(in_dict)
    

    def save_json(self, out_fn: str) -> None:
        if self.minimum is None:
            raise ValueError("MinMaxScaler wasn't fit to data")
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
