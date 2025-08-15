from hubbardml.io import read_occupations_scf
from pw_parsers.parse_u import parse_hubbard


from pathlib import Path
import pyarrow as pa
import re
import pandas as pd

'''
Columns:
Index(['dir', 'material', 'is_vdw', 'uv_iter', 'formula', 'cell', 'n_atoms_uc',
       'person', 'structure_index', 'pw_time_unix', 'hp_time_unix', 'param_in',
       'param_out', 'param_type', 'dist_bohr_in', 'dist_bohr_out',
       'atom_1_idx', 'atom_1_idx_uc', 'atom_1_element', 'atom_1_mass',
       'atom_1_z_valence', 'atom_1_in_name', 'atom_1_in_type',
       'atom_1_out_name', 'atom_1_out_type', 'atom_1_occs_1', 'atom_1_occs_2',
       'atom_1_frac_coords', 'atom_1_starting_mag', 'atom_1_final_mag',
       'atom_2_idx', 'atom_2_idx_uc', 'atom_2_element', 'atom_2_mass',
       'atom_2_z_valence', 'atom_2_in_name', 'atom_2_in_type',
       'atom_2_out_name', 'atom_2_out_type', 'atom_2_occs_1', 'atom_2_occs_2',
       'atom_2_frac_coords', 'atom_2_starting_mag', 'atom_2_final_mag'],
      dtype='object')
'''

dataset = []

hp_results = {
    "/home/lc0965/PROJECTS/SCRATCH/20250218-MgFeO-B1-0.125",
    "/home/lc0965/PROJECTS/SCRATCH/20250323-FeMgO-B2-0.50-2",
    "/home/lc0965/PROJECTS/SCRATCH/20250327-FeMgO-B1-0.5-disordered",
    "/home/lc0965/PROJECTS/SCRATCH/20250320-FeMgO-B2-0.375",
    "/home/lc0965/PROJECTS/SCRATCH/20250403-MgFeO-train-hp",
}

for hp_dir in hp_results:
    for hub_file in Path(hp_dir).glob("**/*.Hubbard_parameters.dat"):

        try:
            scf_file = hub_file.parent / "scf.out"
            
            hubbard_parameters = parse_hubbard(hub_file.read_text().splitlines())
            occupation_data = read_occupations_scf(scf_file)
            hubbard_occupations = occupation_data.atoms

            print(scf_file)
            # print(len(hubbard_parameters), len(hubbard_occupations))

            for hp in hubbard_parameters:

                print(hp)

                element = re.sub(r'\d+$', '', hp.label)
                site = hp.site

                occupation = hubbard_occupations[site]

                row = {
                    "atom_1_element": element,
                    "atom_1_occs_1": occupation[1].occupations.tolist(),
                    "atom_1_occs_2": occupation[2].occupations.tolist(),

                    "atom_2_element": element,
                    "atom_2_occs_1": occupation[1].occupations.tolist(),
                    "atom_2_occs_2": occupation[2].occupations.tolist(),

                    "param_out": hp.u_value,
                    "param_type": "U",
                }

                dataset.append(row)

        except Exception as e:
            print(hub_file, "Error:", e)
            continue

print(len(dataset))
    
df = pd.DataFrame(dataset)
print(df.shape)

# Convert to Arrow table
table = pa.Table.from_pandas(df, preserve_index=False)

# Write to .arrow
with pa.OSFile('dataset.arrow', 'wb') as sink:
    with pa.ipc.new_file(sink, table.schema) as writer:
        writer.write(table)


    