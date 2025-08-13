"""Convert Arrow to HDF5."""

import numpy as np
from pathlib import Path
from .formats import UData, VData, save_hdf5


def load_arrow(file_path):
    """Load Arrow file to dict."""
    import pyarrow as pa
    
    with pa.memory_map(file_path, "rb") as source:
        table = pa.ipc.RecordBatchFileReader(source).read_all()
    
    return {col: table[col].to_pylist() for col in table.column_names}


def fix_occs(occs_list):
    """Fix occupation matrices - some are flat, some are nested."""
    fixed = []
    for occs in occs_list:
        if occs is None:
            fixed.append(None)
            continue
            
        if isinstance(occs[0], list):
            fixed.append(occs)  # already matrix
        else:
            # flat array, need to reshape
            n = int(len(occs) ** 0.5)  # 3 for p, 5 for d
            matrix = [occs[i*n:(i+1)*n] for i in range(n)]
            fixed.append(matrix)
    
    return fixed


def make_occs_array(up_list, down_list):
    """Convert occupation lists to numpy array."""
    n = len(up_list)
    occs = np.zeros((n, 2, 5, 5))  # pad to 5x5
    dims = np.zeros(n, dtype=int)
    
    for i in range(n):
        for spin, data in enumerate([up_list[i], down_list[i]]):
            if data is not None:
                arr = np.array(data)
                size = arr.shape[0]
                occs[i, spin, :size, :size] = arr
                dims[i] = max(dims[i], size)
    
    return occs, dims


def convert_arrow_to_hdf5(input_path, output_path):
    """Convert Arrow to HDF5."""
    print(f"Loading {input_path}...")
    data = load_arrow(input_path)
    
    # Split U and V
    types = data["param_type"]
    u_idx = [i for i, t in enumerate(types) if t == "U"]
    v_idx = [i for i, t in enumerate(types) if t == "V"]
    
    print(f"Found {len(u_idx)} U, {len(v_idx)} V parameters")
    
    # U data
    u = UData()
    if u_idx:
        u.site.elements = np.array([data["atom_1_element"][i] for i in u_idx], dtype="S10")
        u.params.input = np.array([data.get("param_in", [0])[i] for i in u_idx])
        u.params.target = np.array([data["param_out"][i] for i in u_idx])
        
        up = fix_occs([data["atom_1_occs_1"][i] for i in u_idx])
        down = fix_occs([data["atom_1_occs_2"][i] for i in u_idx])
        u.site.occupancy, u.site.orbital_dims = make_occs_array(up, down)
    
    # V data  
    v = VData()
    if v_idx:
        v.site0.elements = np.array([data["atom_1_element"][i] for i in v_idx], dtype="S10")
        v.site1.elements = np.array([data["atom_2_element"][i] for i in v_idx], dtype="S10") 
        v.distance = np.array([data.get("dist_in", [0])[i] for i in v_idx])
        v.params.input = np.array([data.get("param_in", [0])[i] for i in v_idx])
        v.params.target = np.array([data["param_out"][i] for i in v_idx])
        
        up0 = fix_occs([data["atom_1_occs_1"][i] for i in v_idx])
        down0 = fix_occs([data["atom_1_occs_2"][i] for i in v_idx])
        up1 = fix_occs([data["atom_2_occs_1"][i] for i in v_idx])
        down1 = fix_occs([data["atom_2_occs_2"][i] for i in v_idx])
        
        v.site0.occupancy, v.site0.orbital_dims = make_occs_array(up0, down0)
        v.site1.occupancy, v.site1.orbital_dims = make_occs_array(up1, down1)
    
    # Save
    import h5py
    with h5py.File(output_path, 'w') as f:
        hub = f.create_group("hubbard")
        
        if len(u) > 0:
            save_hdf5(u, hub.create_group("u"))
        if len(v) > 0:
            save_hdf5(v, hub.create_group("v"))
    
    # Stats
    in_mb = Path(input_path).stat().st_size / 1024**2
    out_mb = Path(output_path).stat().st_size / 1024**2
    print(f"Done: {in_mb:.1f}MB -> {out_mb:.1f}MB ({in_mb/out_mb:.1f}x)")