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



def create_normalization_from_str(class_name: str, inputs: Optional[dict] = None):

    cls = globals()[class_name]

    if issubclass(cls, Normalization):
        if inputs is None:
            return cls(fillvalue=-5)
        else:
            if "fillvalue" not in inputs.keys():
                inputs["fillvalue"] = -5
            return cls(**inputs)
    else:
        raise ValueError(f"{class_name} is not a subclass of Normalization")


@dataclass
class Normalizer:
    """
    Class to handle the normalization of data in a xarray.Dataset object with different techniques.
    """

    name = "Normalizer"

    def __init__(self, method_var_dict: dict[str, tuple[list[str], dict[str, float]]] = None, 
                 default_norma: str = "Standardizer", fillvalue: float = -5):

        self.all_vars = []
        self.parameters = []
        self.fillvalue = fillvalue
        self.default_norma = default_norma
        self.default_norma_index = None

        if method_var_dict is not None:
            for i, (method, params) in enumerate(method_var_dict.items()):
                variables, input_params = params
                # handle the case of user passing the default norma with some features and forgetting to assign all features to a norma
                if method == default_norma:
                    self.default_norma_index = i
                
                if input_params is None:
                    input_params = {}
                if input_params == {} or "fillvalue" not in input_params.keys():
                    input_params["fillvalue"] = self.fillvalue
                vars_to_remove = [var for var in variables if var in self.all_vars]
                
                method_cls = create_normalization_from_str(method, inputs=input_params)
                
                if len(vars_to_remove) > 0:
                    LOGGER.error(f"Variable(s) {[var for var in vars_to_remove]} are already assigned to another normalization method.\nRemoving thenm from this normalization.")
                    variables = [var for var in variables if var not in vars_to_remove]
                
                self.parameters.append((method_cls, variables, input_params))
                self.all_vars.extend(variables)


    def fit(self, dataset: xr.Dataset, dims: Optional[list] = None):
        
        datavars = list(dataset.data_vars)
        remaining_var = list(set(datavars) - set(self.all_vars))
        if len(remaining_var) > 0:
            LOGGER.info(f"Variables {[var for var in remaining_var]} are not assigned to any normalization method. They will be assigned to {self.default_norma}")
            if self.default_norma_index is not None:
                self.parameters[self.default_norma_index][1].extend(remaining_var)
            else:
                self.parameters.append((create_normalization_from_str(self.default_norma), remaining_var, {"fillvalue": self.fillvalue}))
            self.all_vars.extend(remaining_var)

        for i in range(len(self.parameters)):
            normalizer, variables, inputs = self.parameters[i]
            normalizer.fit(dataset=dataset, variables = variables, dims = dims)
            self.parameters[i] = (normalizer, variables, inputs)

    
    def transform(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        
        for parameter in self.parameters:
            method, variables, _ = parameter
            variables = list(set(variables) & set(datasets[0].data_vars)) # ensure that only variables that are in the dataset are transformed
            datasets = method.transform(*datasets, variables=variables)
        return datasets
    
    
    def inverse_transform(self, *datasets: xr.Dataset) -> xr.Dataset:
        
        for parameter in self.parameters:
            method, variables, _ = parameter
            variables = list(set(variables) & set(datasets[0].data_vars)) # ensure that only variables that are in the dataset are inv-transformed
            datasets = method.inverse_transform(*datasets, variables=variables)

        return datasets
    
    
    @classmethod
    def from_dict(cls, in_dict: dict) -> Self:
        method_var_dict = {}

        # check whether dict corresponds to old Standardizer format
        first_key = list(in_dict.keys())[0]
        list_of_classes = [cls.__name__ for cls in Normalization.__subclasses__()]
        if first_key not in list_of_classes:
            subclass = Standardizer().from_dict(in_dict)
            inputs = {"mean": subclass.mean, "std": subclass.std, "fillvalue": subclass.fillvalue}
            method_var_dict[subclass.name] = (list(subclass.mean.data_vars), inputs)
        else:
            for method_name, inner_dict in in_dict.items():
                tmp_class = create_normalization_from_str(method_name).from_dict(inner_dict)
                inputs = {key: getattr(tmp_class, key) for key in inner_dict if getattr(tmp_class, key, None) is not None}
                method_var_dict[tmp_class.name] = (inner_dict["channels"], inputs)
        return cls(method_var_dict)
    
    
    def to_dict(self):
        out_dict = {}
        for parameter in self.parameters:
            method, variables, _ = parameter
            out_dict_tmp = method.to_dict()
            out_dict_tmp["channels"] = variables
            out_dict[method.name] = out_dict_tmp
        return out_dict
    
    @classmethod
    def from_json(cls, in_fn: str) -> Self:
        with open(in_fn, "r") as f:
            in_dict = json.load(f)
        return cls.from_dict(in_dict)
    
    def save_json(self, out_fn: str) -> None:
        out_dict = self.to_dict()
        if len(out_dict) == 0 or out_dict[list(out_dict.keys())[0]] is None:
            raise ValueError(f"{self.name} wasn't fit to data")
        with open(out_fn, "w") as outfile:
            json.dump(out_dict, outfile, indent=4)

class Normalization:
    """
    Abstract class for nromalization techniques in a xarray.Dataset object.
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

@dataclass
class Identity(Normalization):
    """
    Identity normalizer, returns the input data without any transformation.
    """

    fillvalue: float = field(default=-5)
    name = "Identity"

    def fit(self, dataset: xr.Dataset, variables: Optional[list] = None, dims: Optional[list] = None):
        self.fillvalue = self.fillvalue

    def transform(self, *datasets: xr.Dataset, variables: Optional[list] = None) -> tuple[xr.Dataset, ...]:

        def f(ds: xr.Dataset, variables: Optional[list] = None) -> xr.Dataset:
            if variables is None:
                variables = list(ds.data_vars)
            for var in variables:
                identity_value = ds[var].astype("float32")
                if self.fillvalue is not None:
                    identity_value = identity_value.fillna(self.fillvalue)
                ds[var] = identity_value
            return ds
        
        return tuple(f(ds, variables) for ds in datasets)
    
    def inverse_transform(self, *datasets: xr.Dataset, variables: Optional[list] = None) -> tuple[xr.Dataset, ...]:
        return tuple(xr.where(ds > self.fillvalue, ds, np.nan) for ds in datasets)
    
    @classmethod
    def from_dict(cls, in_dict: dict) -> Self:
        fillvalue = in_dict["fillvalue"]
        return cls(fillvalue)
    
    def to_dict(self):
        out_dict = {
            "fillvalue": self.fillvalue,
        }
        return out_dict


@dataclass
class Standardizer(Normalization):
    """
    Standardizes data (z-normalization) in a xarray.Dataset object.
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
            
            ds_copy = ds.copy()
            if variables is None:
                variables = list(ds_copy.data_vars)
            for var in variables:
                assert var in self.mean.data_vars, f"{var} not in Standardizer"

                standardized_var = ((ds_copy[var] - self.mean[var]) / self.std[var]).astype("float32")
                if self.fillvalue is not None:
                    standardized_var = standardized_var.fillna(self.fillvalue)
                
                ds_copy[var] = standardized_var

            return ds_copy

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


@dataclass
class MinMaxScaler(Normalization):
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

            ds_copy = ds.copy()
            if variables is None:
                variables = ds_copy.data_vars
            for var in variables:
                assert var in self.minimum.data_vars, f"{var} not in MinMaxScaler"

                scaled_var = ((ds_copy[var] - self.minimum[var]) / (self.maximum[var] - self.minimum[var])).astype("float32")

                if self.fillvalue is not None:
                    scaled_var = scaled_var.fillna(self.fillvalue)

                ds_copy[var] = scaled_var
            return ds_copy
        
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
            "fillvalue": self.fillvalue,
        }
        return out_dict


@dataclass
class MaxAbsScaler(Normalization):
    """
    Normalize data using a max absolute scaling in a xarray.Dataset object.
    """

    absmax: xr.Dataset = field(default=None)
    fillvalue: dict[str, float] = field(init=True, default=-5)
    name = "MaxAbsScaler"

    def fit(self, dataset: xr.Dataset, variables: Optional[list] = None, dims: Optional[list] = None):
        if variables is None:
            variables = list(dataset.data_vars)
        if not all(var in dataset.data_vars for var in variables):
            raise KeyError(f"There are variables not in dataset: {[var for var in variables if var not in dataset.data_vars]}")

        self.absmax = abs(dataset[variables]).max(dims).compute().copy()
        self.fillvalue = self.fillvalue


    def transform(self, *datasets: xr.Dataset, variables: Optional[list] = None) -> tuple[xr.Dataset, ...]:
        if self.absmax is None:
            raise ValueError("MaxAbsScaler wasn't fit to data")

        def f(ds: xr.Dataset, variables: Optional[list] = None) -> xr.Dataset:

            ds_copy = ds.copy()
            if variables is None:
                variables = ds_copy.data_vars
            for var in variables:
                assert var in self.absmax.data_vars, f"{var} not in MaxAbsScaler"

                scaled_var = (ds_copy[var] / self.absmax[var]).astype("float32")

                if self.fillvalue is not None:
                    scaled_var = scaled_var.fillna(self.fillvalue)

                ds_copy[var] = scaled_var
            return ds_copy
        
        return tuple(f(ds, variables) for ds in datasets)
    
    def inverse_transform(self, *datasets: xr.Dataset, variables: Optional[list] = None) -> tuple[xr.Dataset, ...]:
        if self.absmax is None:
            raise ValueError("MaxAbsScaler wasn't fit to data")
        
        def f(ds: xr.Dataset, variables: Optional[list] = None) -> xr.Dataset:
            if variables is None:
                variables = ds.data_vars
            ds = xr.where(ds > self.fillvalue, ds, np.nan)
            for var in variables:
                assert var in self.absmax.data_vars, f"{var} not in MaxAbsScaler"

                unscaled_var = (ds[var] * self.absmax[var]).astype("float32")
                ds[var] = unscaled_var
            return ds
        
        return tuple(f(ds, variables) for ds in datasets)
    
    @classmethod
    def from_dict(cls, in_dict: dict) -> Self:
        absmax = xr.Dataset.from_dict(in_dict["absmax"])
        return cls(absmax)
    
    def to_dict(self):
        out_dict = {
            "absmax": self.absmax.to_dict(),
            "fillvalue": self.fillvalue,
        }
        return out_dict
    

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
