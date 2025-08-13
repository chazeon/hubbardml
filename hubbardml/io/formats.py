"""Simple data structures for HDF5 storage."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Site:
    """Atomic site data."""
    elements: np.ndarray = field(default_factory=lambda: np.array([]))
    occupancy: Optional[np.ndarray] = field(default=None)  # (N, 2, orb, orb) 
    orbital_dims: Optional[np.ndarray] = field(default=None)  # (N,) actual orbital count per entry


@dataclass
class Parameters:
    """Hubbard parameter values."""
    input: np.ndarray = field(default_factory=lambda: np.array([]))
    target: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass 
class UData:
    """U parameter dataset."""
    site: Site = field(default_factory=lambda: Site())
    params: Parameters = field(default_factory=lambda: Parameters())
    
    def __len__(self):
        return len(self.params.target)


@dataclass
class VData:
    """V parameter dataset.""" 
    site0: Site = field(default_factory=lambda: Site())
    site1: Site = field(default_factory=lambda: Site())
    distance: np.ndarray = field(default_factory=lambda: np.array([]))
    params: Parameters = field(default_factory=lambda: Parameters())
    
    def __len__(self):
        return len(self.params.target)


def save_hdf5(data, group):
    """Save data to HDF5 group."""
    for name, value in data.__dict__.items():
        if value is None:
            continue
        if hasattr(value, '__dict__'):  # nested dataclass
            subgroup = group.create_group(name)
            save_hdf5(value, subgroup) 
        elif isinstance(value, np.ndarray) and value.size > 0:
            if value.dtype.kind in ['U', 'S']:  # strings
                group.create_dataset(name, data=value)
            else:
                group.create_dataset(name, data=value, compression='gzip')
        elif isinstance(value, (int, float, str)):
            group.attrs[name] = value


def load_hdf5(data, group):
    """Load data from HDF5 group."""
    for name in data.__dict__.keys():
        if name in group:
            item = group[name]
            if hasattr(item, 'keys'):  # subgroup
                load_hdf5(getattr(data, name), item)
            else:  # dataset
                setattr(data, name, np.array(item))
        elif name in group.attrs:
            setattr(data, name, group.attrs[name])