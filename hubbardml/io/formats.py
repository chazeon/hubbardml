"""Simple data structures for HDF5 storage."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Site:
    """Atomic site data."""
    elems: np.ndarray = field(default_factory=lambda: np.array([]))
    occs: Optional[np.ndarray] = field(default=None)  # (N, 2, orb, orb) 
    orbs: Optional[np.ndarray] = field(default=None)  # (N,) orbital labels ["3d", "2p", ...]


@dataclass
class Edge:
    """Edge data between sites."""
    dist: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass 
class UData:
    """U parameter dataset."""
    site0: Site = field(default_factory=lambda: Site())
    input: np.ndarray = field(default_factory=lambda: np.array([]))
    target: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __len__(self):
        return len(self.target)


@dataclass
class VData:
    """V parameter dataset.""" 
    site0: Site = field(default_factory=lambda: Site())
    site1: Site = field(default_factory=lambda: Site())
    edge: Edge = field(default_factory=lambda: Edge())
    input: np.ndarray = field(default_factory=lambda: np.array([]))
    target: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __len__(self):
        return len(self.target)


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


def save_u_hdf5(u_data, group):
    """Save U data with flat HDF5 structure."""
    # Site data
    site_group = group.create_group("site0")
    save_hdf5(u_data.site0, site_group)
    
    # Hubbard values - flat in main group  
    group.create_dataset("target", data=u_data.target, compression='gzip')
    group.create_dataset("input", data=u_data.input, compression='gzip')


def save_v_hdf5(v_data, group):
    """Save V data with flat HDF5 structure."""
    # Site data
    site0_group = group.create_group("site0")
    save_hdf5(v_data.site0, site0_group)
    
    site1_group = group.create_group("site1")  
    save_hdf5(v_data.site1, site1_group)
    
    # Edge data
    edge_group = group.create_group("edge")
    save_hdf5(v_data.edge, edge_group)
    
    # Hubbard values - flat in main group
    group.create_dataset("target", data=v_data.target, compression='gzip')
    group.create_dataset("input", data=v_data.input, compression='gzip')


def _load_group_to_dataclass(data, group):
    """Load HDF5 group into dataclass (internal helper)."""
    # Name mapping for compatibility
    name_map = {
        'elems': ['elems', 'element'],  # Try both plural and singular
        'occs': ['occs', 'occupancy'],  # Try both names
        'orbs': ['orbs', 'orbital'],    # Try both names
    }
    
    for name in data.__dict__.keys():
        found = False
        # Try all possible names for this field
        possible_names = name_map.get(name, [name])
        
        for possible_name in possible_names:
            if possible_name in group:
                item = group[possible_name]
                if hasattr(item, 'keys'):  # subgroup
                    _load_group_to_dataclass(getattr(data, name), item)
                else:  # dataset
                    setattr(data, name, np.array(item))
                found = True
                break
            elif possible_name in group.attrs:
                setattr(data, name, group.attrs[possible_name])
                found = True
                break
        
        # If not found, try the original logic
        if not found:
            if name in group:
                item = group[name]
                if hasattr(item, 'keys'):  # subgroup
                    _load_group_to_dataclass(getattr(data, name), item)
                else:  # dataset
                    setattr(data, name, np.array(item))
            elif name in group.attrs:
                setattr(data, name, group.attrs[name])


def load_u_data(group):
    """Load U data from HDF5 group."""
    u = UData()
    
    # Load site data
    if 'site0' in group:
        _load_group_to_dataclass(u.site0, group['site0'])
    
    # Load Hubbard values
    if 'target' in group:
        u.target = np.array(group['target'])
    if 'input' in group:
        u.input = np.array(group['input'])
    
    return u


def load_v_data(group):
    """Load V data from HDF5 group."""
    v = VData()
    
    # Load site data
    if 'site0' in group:
        _load_group_to_dataclass(v.site0, group['site0'])
    if 'site1' in group:
        _load_group_to_dataclass(v.site1, group['site1'])
    
    # Load edge data
    if 'edge' in group:
        _load_group_to_dataclass(v.edge, group['edge'])
    
    # Load Hubbard values
    if 'target' in group:
        v.target = np.array(group['target'])
    if 'input' in group:
        v.input = np.array(group['input'])
    
    return v


def load_hdf5(file_path):
    """Load HDF5 file and return U/V data."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py not available. Install with: uv add h5py")
    
    u_data, v_data = None, None
    
    with h5py.File(file_path, 'r') as f:
        if 'hubbard' in f:
            hub = f['hubbard']
            if 'u' in hub:
                u_data = load_u_data(hub['u'])
            if 'v' in hub:
                v_data = load_v_data(hub['v'])
    
    return u_data, v_data