"""Unified battery dataset loaders."""

from pistgat_bench.datasets.base import BaseBatteryDataset, BatteryCycleRecord
from pistgat_bench.datasets.nasa import NASABatteryDataset
from pistgat_bench.datasets.calce import CALCEBatteryDataset
from pistgat_bench.datasets.oxford import OxfordBatteryDataset

__all__ = [
    "BaseBatteryDataset",
    "BatteryCycleRecord",
    "NASABatteryDataset",
    "CALCEBatteryDataset",
    "OxfordBatteryDataset",
    "get_dataset",
]


def get_dataset(name: str, **kwargs) -> BaseBatteryDataset:
    """Factory helper to retrieve a dataset instance by name."""
    name_lower = name.strip().lower()
    if "nasa" in name_lower:
        return NASABatteryDataset(**kwargs)
    elif "calce" in name_lower:
        return CALCEBatteryDataset(**kwargs)
    elif "oxford" in name_lower:
        return OxfordBatteryDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Available options: 'nasa', 'calce', 'oxford'")
