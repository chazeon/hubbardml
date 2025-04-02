import numpy as np
from typing import NamedTuple, Dict, List


class Occupation(NamedTuple):
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    occupations: np.ndarray

    def __repr__(self):
        return f"Occupation(eigenvalues={self.eigenvalues})"


def parse_all_hubbard_occupations(filename: str) -> Dict[int, Dict[int, Occupation]]:
    """
    Parses the Hubbard occupations from the last SCF cycle in the file.
    Returns a nested dictionary of the form: {atom_index: {spin: Occupation}}
    """

    def parse_hubbard_block(lines: List[str]) -> List[str]:
        # Find the last SCF completion
        end_indices = [i for i, line in enumerate(lines) if "End of self-consistent calculation" in line]
        if not end_indices:
            return None
        last_end_idx = end_indices[-1]

        # Look for the last Hubbard occupation block after that
        occ_start = None
        for i in range(last_end_idx, len(lines)):
            if "=================== HUBBARD OCCUPATIONS ===================" in lines[i]:
                occ_start = i

        if occ_start is None:
            return None

        # Read until the next empty line
        hubbard_block = []
        for line in lines[occ_start:]:
            if line.strip() == "":
                break
            hubbard_block.append(line.rstrip())

        return hubbard_block

    def split_atom_blocks(hubbard_block: List[str]) -> List[List[str]]:
        return [hubbard_block[i:i+33] for i in range(1, len(hubbard_block), 33)]

    def split_spin_blocks(atom_block: List[str]) -> Dict[int, List[str]]:

        spin_blocks = {}
        current_spin = None
        current_lines = []

        for line in atom_block:
            if "SPIN  1" in line:
                if current_spin and current_lines:
                    spin_blocks[current_spin] = current_lines
                current_spin = 1
                current_lines = []
            elif "SPIN  2" in line:
                if current_spin and current_lines:
                    spin_blocks[current_spin] = current_lines
                current_spin = 2
                current_lines = []
            elif current_spin:
                current_lines.append(line)

        if current_spin and current_lines:
            spin_blocks[current_spin] = current_lines

        return spin_blocks

    def extract_occupations(spin_block: List[str]) -> Occupation:
        eigenvalues = np.fromstring(spin_block[1], sep=' ')
        eigenvectors = np.fromstring("\n".join(spin_block[3:8]), sep=' ').reshape(-1, 5)
        occupations = np.fromstring("\n".join(spin_block[9:14]), sep=' ').reshape(-1, 5)
        return Occupation(eigenvalues, eigenvectors, occupations)

    # === MAIN PROCESS ===
    with open(filename, 'r') as f:
        lines = f.readlines()

    hubbard_block = parse_hubbard_block(lines)
    if not hubbard_block:
        return {}  # Or raise an exception

    atom_blocks = split_atom_blocks(hubbard_block)
    all_data = {}

    for atom_idx, atom_block in enumerate(atom_blocks, 1):
        spin_blocks = split_spin_blocks(atom_block)
        atom_data = {}
        for spin, lines in spin_blocks.items():
            atom_data[spin] = extract_occupations(lines)
        all_data[atom_idx] = atom_data

    return all_data


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python parse_pw3.py <filename>")
        sys.exit(1)
    data = parse_all_hubbard_occupations(sys.argv[1])
    print(data)
