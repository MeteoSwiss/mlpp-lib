import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from abc import abstractmethod

import numpy as np
import xarray as xr
from typing_extensions import Self
#from mlpp_lib.utils import calculate_median

LOGGER = logging.getLogger(__name__)



def create_normalizer_from_str(class_name: str, inputs: Optional[dict] = None):

    cls = globals()[class_name]

    if issubclass(cls, Normalizer):
        if inputs is None:
            return cls(fillvalue=-999)
        else:
            if "fillvalue" not in inputs.keys():
                inputs["fillvalue"] = -999
            return cls(**inputs)
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

    # TODO: when provided with nothing use, Standardizer as the only method
    def __init__(self, method_var_dict: dict[str, tuple[list[str], dict[str, float]]] = None, fillvalue: float = -999):
        seen_vars = []
        self.parameters = []
        self.fillvalue = fillvalue

        if method_var_dict is not None:
            for method, params in method_var_dict.items():
                variables, input_params = params
                
                if input_params == {} or "fillvalue" not in input_params.keys():
                    input_params["fillvalue"] = self.fillvalue
                vars_to_remove = [var for var in variables if var in seen_vars]
                
                method_cls = create_normalizer_from_str(method, inputs=input_params) # ensure the proper functionning in case it is not an str
                
                if len(vars_to_remove) > 0:
                    LOGGER.info(f"Variable(s) {[var for var in vars_to_remove]} are already assigned to another normalization method")
                    variables = [var for var in variables if var not in vars_to_remove]
                
                self.parameters.append((method_cls, variables, input_params))
        LOGGER.info(f"Created MultiNormalizer with parameters {self.parameters}")


    def fit(self, dataset: xr.Dataset, dims: Optional[list] = None):
        
        for i in range(len(self.parameters)):
            normalizer, variables, inputs = self.parameters[i]
            normalizer.fit(dataset=dataset, variables = variables, dims = dims)
            self.parameters[i] = (normalizer, variables, inputs)

    
    def transform(self, *datasets: xr.Dataset) -> tuple[xr.Dataset, ...]:
        
        for parameter in self.parameters:
            method, variables, _ = parameter
            datasets = method.transform(*datasets, variables=variables)

        return datasets
    
    
    def inverse_transform(self, *datasets: xr.Dataset) -> xr.Dataset:
        
        for parameter in self.parameters:
            method, variables, _ = parameter
            datasets = method.inverse_transform(*datasets, variables=variables)

        return datasets
    
    
    @classmethod
    def from_dict(cls, in_dict: dict) -> Self:
        method_var_dict = {}
        for method_name, inner_dict in in_dict.items():
            tmp_class = create_normalizer_from_str(method_name)
            subclass = tmp_class.from_dict(inner_dict)
            variables = inner_dict["channels"]
            # TODO: the following line is a bit convoluted, maybe there is a better way to do it
            inputs = {key: getattr(subclass, key) for key in inner_dict if getattr(subclass, key, None) is not None}
            method_var_dict[subclass.name] = (variables, inputs)
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
    fillvalue: dict[str, float] = field(init=True, default=-999)
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
    fillvalue: dict[str, float] = field(init=True, default=-999)
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
            "fillvalue": self.fillvalue,
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

# I can't make it work with the calculation of the median
# TODO: find a way to do it
@dataclass
class RobustScaler(Normalizer):

    median: xr.Dataset = field(default=None)
    iqr: xr.Dataset = field(default=None)
    fillvalue: dict[str, float] = field(init=True, default=-999)
    name = "RobustScaler"

    def fit(self, dataset: xr.Dataset, variables: Optional[list] = None, dims: Optional[list] = None):
        if variables is None:
            variables = list(dataset.data_vars)
        if not all(var in dataset.data_vars for var in variables):
            raise KeyError(f"There are variables not in dataset: {[var for var in variables if var not in dataset.data_vars]}")

        self.median = dataset[variables].quantile(0.5, dims).compute().copy()
        self.iqr = (dataset[variables].quantile(0.75, dims).compute() - dataset[variables].quantile(0.25, dims).compute()).copy()
        self.fillvalue = self.fillvalue

    def transform(self, *datasets: xr.Dataset, variables: Optional[list] = None) -> tuple[xr.Dataset, ...]:
        if self.median is None:
            raise ValueError("RobustScaling wasn't fit to data")

        def f(ds: xr.Dataset, variables: Optional[list] = None) -> xr.Dataset:
            if variables is None:
                variables = ds.data_vars
            for var in variables:
                assert var in self.median.data_vars, f"{var} not in RobustScaling"

                scaled_var = ((ds[var] - self.median[var]) / self.iqr[var]).astype("float32")

                if self.fillvalue is not None:
                    scaled_var = scaled_var.fillna(self.fillvalue)

                ds[var] = scaled_var
                if "quantile" in ds.coords:
                    ds = ds.drop("quantile")
            return ds
        
        return tuple(f(ds, variables) for ds in datasets)
    
    def inverse_transform(self, *datasets: xr.Dataset, variables: Optional[list] = None) -> tuple[xr.Dataset, ...]:
        if self.median is None:
            raise ValueError("RobustScaling wasn't fit to data")
        
        def f(ds: xr.Dataset, variables: Optional[list] = None) -> xr.Dataset:
            if variables is None:
                variables = ds.data_vars
            ds = xr.where(ds > self.fillvalue, ds, np.nan)
            for var in variables:
                assert var in self.median.data_vars, f"{var} not in RobustScaling"

                unscaled_var = (ds[var] * self.iqr[var] + self.median[var]).astype("float32")
                ds[var] = unscaled_var

                
            if "quantile" in ds.coords:
                ds = ds.drop("quantile")
            return ds
        
        return tuple(f(ds, variables) for ds in datasets)
    
    @classmethod
    def from_dict(cls, in_dict: dict) -> Self:
        median = xr.Dataset.from_dict(in_dict["median"])
        iqr = xr.Dataset.from_dict(in_dict["iqr"])
        return cls(median, iqr)
    
    def to_dict(self):
        out_dict = {
            "median": self.median.to_dict(),
            "iqr": self.iqr.to_dict(),
            "fillvalue": self.fillvalue,
        }
        return out_dict
    
    @classmethod
    def from_json(cls, in_fn: str) -> Self:
        with open(in_fn, "r") as f:
            in_dict = json.load(f)
        return cls.from_dict(in_dict)
    
    def save_json(self, out_fn: str) -> None:
        if self.median is None:
            raise ValueError("RobustScaling wasn't fit to data")
        out_dict = self.to_dict()
        with open(out_fn, "w") as outfile:
            json.dump(out_dict, outfile, indent=4)


@dataclass
class MaxAbsScaler(Normalizer):

    absmax: xr.Dataset = field(default=None)
    fillvalue: dict[str, float] = field(init=True, default=-999)
    name = "MaxAbsScaler"

    def fit(self, dataset: xr.Dataset, variables: Optional[list] = None, dims: Optional[list] = None):
        if variables is None:
            variables = list(dataset.data_vars)
        if not all(var in dataset.data_vars for var in variables):
            raise KeyError(f"There are variables not in dataset: {[var for var in variables if var not in dataset.data_vars]}")

        self.absmax = dataset[variables].max(dims).compute().copy()
        self.fillvalue = self.fillvalue

    def transform(self, *datasets: xr.Dataset, variables: Optional[list] = None) -> tuple[xr.Dataset, ...]:
        if self.absmax is None:
            raise ValueError("MaxAbsScaler wasn't fit to data")

        def f(ds: xr.Dataset, variables: Optional[list] = None) -> xr.Dataset:
            if variables is None:
                variables = ds.data_vars
            for var in variables:
                assert var in self.absmax.data_vars, f"{var} not in MaxAbsScaler"

                scaled_var = (ds[var] / self.absmax[var]).astype("float32")

                if self.fillvalue is not None:
                    scaled_var = scaled_var.fillna(self.fillvalue)

                ds[var] = scaled_var
            return ds
        
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
    
    @classmethod
    def from_json(cls, in_fn: str) -> Self:
        with open(in_fn, "r") as f:
            in_dict = json.load(f)
        return cls.from_dict(in_dict)
    
    def save_json(self, out_fn: str) -> None:
        if self.absmax is None:
            raise ValueError("MaxAbsScaler wasn't fit to data")
        out_dict = self.to_dict()
        with open(out_fn, "w") as outfile:
            json.dump(out_dict, outfile, indent=4)


@dataclass
class BoxCoxScaler(Normalizer):

    lambda_: float = field(default=2)
    fillvalue: dict[str, float] = field(init=True, default=-999)
    name = "BoxCoxScaler"

    def fit(self, dataset: xr.Dataset, variables: Optional[list] = None, dims: Optional[list] = None):
        if variables is None:
            variables = list(dataset.data_vars)
        if not all(var in dataset.data_vars for var in variables):
            raise KeyError(f"There are variables not in dataset: {[var for var in variables if var not in dataset.data_vars]}")

        self.lambda_ = self.lambda_
        self.fillvalue = self.fillvalue

    def transform(self, *datasets: xr.Dataset, variables: Optional[list] = None) -> tuple[xr.Dataset, ...]:
        if self.lambda_ is None:
            raise ValueError("BoxCox wasn't fit to data")

        def f(ds: xr.Dataset, variables: Optional[list] = None) -> xr.Dataset:
            if variables is None:
                variables = ds.data_vars
            for var in variables:
                assert var in ds, f"{var} not in input dataset"

                scaled_var = (ds[var] ** self.lambda_ - 1) / self.lambda_ if self.lambda_ != 0 else np.log(ds[var])

                if self.fillvalue is not None:
                    scaled_var = scaled_var.fillna(self.fillvalue)

                ds[var] = scaled_var
            return ds
        
        return tuple(f(ds, variables) for ds in datasets)
    
    def inverse_transform(self, *datasets: xr.Dataset, variables: Optional[list] = None) -> tuple[xr.Dataset, ...]:
        if self.lambda_ is None:
            raise ValueError("BoxCox wasn't fit to data")
        
        def f(ds: xr.Dataset, variables: Optional[list] = None) -> xr.Dataset:
            if variables is None:
                variables = ds.data_vars
            ds = xr.where(ds > self.fillvalue, ds, np.nan)
            for var in variables:
                assert var in ds, f"{var} not in input dataset"

                unscaled_var = (ds[var] * self.lambda_ + 1) ** (1 / self.lambda_) if self.lambda_ != 0 else np.exp(ds[var])

                ds[var] = unscaled_var
            return ds

        return tuple(f(ds, variables) for ds in datasets)
    
    @classmethod
    def from_dict(cls, in_dict: dict) -> Self:
        lambda_ = in_dict["lambda_"]
        return cls(lambda_)
    
    def to_dict(self):
        out_dict = {
            "lambda_": self.lambda_,
            "fillvalue": self.fillvalue,
        }
        return out_dict
    
    @classmethod
    def from_json(cls, in_fn: str) -> Self:
        with open(in_fn, "r") as f:
            in_dict = json.load(f)
        return cls.from_dict(in_dict)
    
    def save_json(self, out_fn: str) -> None:
        if self.lambda_ is None:
            raise ValueError("BoxCox wasn't fit to data")
        out_dict = self.to_dict()
        with open(out_fn, "w") as outfile:
            json.dump(out_dict, outfile, indent=4)


@dataclass
class YeoJohnsonScaler(Normalizer):

    lambda_: float = field(default=3)
    fillvalue: dict[str, float] = field(init=True, default=-999)
    name = "YeoJohnsonScaler"

    def fit(self, dataset: xr.Dataset, variables: Optional[list] = None, dims: Optional[list] = None):
        if variables is None:
            variables = list(dataset.data_vars)
        if not all(var in dataset.data_vars for var in variables):
            raise KeyError(f"There are variables not in dataset: {[var for var in variables if var not in dataset.data_vars]}")

        self.lambda_ = self.lambda_
        self.fillvalue = self.fillvalue


    def yeo_johnson_transform(self, x: float, lmbda: float) -> np.ndarray:
        if lmbda == 0:
            return np.log1p(x)
        elif lmbda == 2:
            return -np.log1p(-x)
        elif x>=0:
            return ((x + 1) ** lmbda - 1) / lmbda
        else:
            return -(((-x + 1) ** (2 - lmbda)) - 1) / (2 - lmbda)
        
    def yeo_johnson_inverse_transform(self, x: float, lmbda: float) -> np.ndarray:
        if lmbda == 0:
            return np.expm1(x)
        elif lmbda == 2:
            return -np.expm1(-x)
        elif x>=0:
            return ((lmbda * x + 1) ** (1 / lmbda)) - 1
        else:
            return -((-lmbda * x + 1) ** (1 / (2 - lmbda))) + 1
        

    def transform(self, *datasets: xr.Dataset, variables: Optional[list] = None) -> tuple[xr.Dataset, ...]:
        if self.lambda_ is None:
            raise ValueError("YeoJohnson wasn't fit to data")

        def f(ds: xr.Dataset, variables: Optional[list] = None) -> xr.Dataset:
            if variables is None:
                variables = ds.data_vars
            for var in variables:
                assert var in ds, f"{var} not in input dataset"

                scaled_var = xr.apply_ufunc(
                    self.yeo_johnson_transform,
                    ds[var],
                    self.lambda_,
                    dask="parallelized",
                    output_dtypes=[np.float32],
                    vectorize=True,
                )

                if self.fillvalue is not None:
                    scaled_var = scaled_var.fillna(self.fillvalue)

                ds[var] = scaled_var
            return ds
        
        return tuple(f(ds, variables) for ds in datasets)
    
    def inverse_transform(self, *datasets: xr.Dataset, variables: Optional[list] = None) -> tuple[xr.Dataset, ...]:
        if self.lambda_ is None:
            raise ValueError("YeoJohnson wasn't fit to data")
        
        def f(ds: xr.Dataset, variables: Optional[list] = None) -> xr.Dataset:
            if variables is None:
                variables = ds.data_vars
            ds = xr.where(ds > self.fillvalue, ds, np.nan)
            for var in variables:
                assert var in ds, f"{var} not in input dataset"

                unscaled_var = xr.apply_ufunc(
                    self.yeo_johnson_inverse_transform,
                    ds[var],
                    self.lambda_,
                    dask="parallelized",
                    output_dtypes=[np.float32],
                    vectorize=True,
                )

                ds[var] = unscaled_var
            return ds
        
        return tuple(f(ds, variables) for ds in datasets)
                     
    @classmethod
    def from_dict(cls, in_dict: dict) -> Self:
        lambda_ = in_dict["lambda_"]
        return cls(lambda_)

    def to_dict(self):
        out_dict = {
            "lambda_": self.lambda_,
            "fillvalue": self.fillvalue,
        }
        return out_dict

    @classmethod
    def from_json(cls, in_fn: str) -> Self:
        with open(in_fn, "r") as f:
            in_dict = json.load(f)
        return cls.from_dict(in_dict)

    def save_json(self, out_fn: str) -> None:
        if self.lambda_ is None:
            raise ValueError("YeoJohnson wasn't fit to data")
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
