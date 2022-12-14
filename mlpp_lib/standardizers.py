import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

LOGGER = logging.getLogger(__name__)


@dataclass
class Standardizer:
    """
    Standardizes data in a xarray.Dataset object.
    """

    mean: dict[str, float] = field(init=False)
    std: dict[str, float] = field(init=False)
    fillvalue: dict[str, float] = field(init=False)

    def fit(self, dataset: xr.Dataset, dims: list = None):
        LOGGER.info("Calculating mean and standard deviation")
        self.mean = dataset.mean(dims).persist()
        self.std = dataset.std(dims).persist()
        self.fillvalue = -5
        # Check for near-zero standard deviations and set them equal to one
        self.std = xr.where(self.std < 1e-6, 1, self.std)

    def transform(self, dataset: xr.Dataset) -> xr.Dataset:
        if self.mean is None:
            raise ValueError("Standardizer wasn't fit to data")
        for var in dataset.data_vars:
            assert var in self.mean.data_vars, f"{var} not in Standardizer"
        return (
            ((dataset - self.mean) / self.std).astype("float32").fillna(self.fillvalue)
        )

    def inverse_transform(self, dataset: xr.Dataset) -> xr.Dataset:
        if self.mean is None:
            raise ValueError("Standardizer wasn't fit to data")
        for var in dataset.data_vars:
            assert var in self.mean.data_vars, f"{var} not in Standardizer"
        dataset = xr.where(dataset > self.fillvalue, dataset, np.nan)
        return (dataset * self.std + self.mean).astype("float32")

    def to_dict(self) -> dict:
        if self.mean is None:
            raise ValueError("Standardizer wasn't fit to data")
        out_dict = {
            "mean": self.mean.to_dict(),
            "std": self.std.to_dict(),
            "fillvalue": self.fillvalue,
        }
        return out_dict

    def from_dict(self, in_dict: dict) -> None:
        self.mean = xr.Dataset.from_dict(in_dict["mean"])
        self.std = xr.Dataset.from_dict(in_dict["std"])
        self.fillvalue = in_dict["fillvalue"]

    def save_json(self, out_fn: str) -> None:
        out_dict = self.to_dict()
        with open(out_fn, "w") as outfile:
            json.dump(out_dict, outfile, indent=4)

    def load_json(self, in_fn: str) -> None:
        with open(in_fn, "r") as f:
            in_dict = json.load(f)
        self.from_dict(in_dict)


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
