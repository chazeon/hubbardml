# HubbardML Jinja Templates

This directory contains Jinja2 templates for formatting Hubbard parameter predictions into various output formats, particularly for Quantum ESPRESSO input files.

## Template Usage

Templates are used by the predict CLI:

```bash
# Use built-in template
uv run python -m hubbardml predict models/ data.h5 --template qe_simple

# Use custom template
uv run python -m hubbardml predict models/ data.h5 --template my_custom.j2
```

## Available Templates

### Simple HUBBARD Format (`qe_simple.j2`)
Basic QE HUBBARD card format - most commonly used:
```
HUBBARD (ortho-atomic)
U Co-3d 7.70
V Co-3d O-2p 1 19 0.75
```

### V-only Format (`qe_v_format.j2`)  
Uses V notation for both U and V parameters:
```
HUBBARD (ortho-atomic)
V Co-3d Co-3d 1 1 7.70
V Co-3d O-2p 1 19 0.75
```

### Legacy Format (`qe_legacy.j2`)
Old Hubbard_V array notation:
```
&system
  lda_plus_u = .true.
  lda_plus_u_kind = 2
  U_projection_type = 'ortho-atomic'
  Hubbard_V(1,1,1) = 7.70
  Hubbard_V(1,19,1) = 0.75
/
```

### JSON Debug Format (`json.j2`)
For debugging and data inspection:
```json
{
  "model": "u_model",
  "predictions": [
    {"element": "Co", "orbital": "3d", "value": 7.70}
  ]
}
```

## Template Data Structure

Templates receive the following data:

```python
{
    'model_name': 'u_model_v1',
    'model_type': 'U',  # or 'V' or 'mixed'
    'timestamp': '2024-01-15T10:30:00',
    'system_name': 'CoO_supercell',
    'projection_type': 'ortho-atomic',  # QE projection type (customizable)
    'lda_plus_u_kind': 2,               # QE U+V formulation (legacy format)
    'u_predictions': [
        {
            'element': 'Co', 
            'orbital': '3d', 
            'value': 7.70,
            'confidence': 0.95,
            'site_data': {...}  # Full site information
        }
    ],
    'v_predictions': [
        {
            'site0_element': 'Co', 
            'site0_orbital': '3d',
            'site1_element': 'O', 
            'site1_orbital': '2p',
            'value': 0.75,
            'confidence': 0.92,
            'site0_data': {...},  # Full site 0 information
            'site1_data': {...}   # Full site 1 information
        }
    ]
}
```

## Custom Jinja Functions

Templates have access to custom functions for complex logic:

### `get_atom_index(site_data)`
Maps site data to QE atom indices. This handles the complex QE neighbor mapping logic.

### `get_interaction_type(v_prediction)`
Returns interaction type k value (1-4):
- k=1: standard-standard (most common)
- k=2: standard-background  
- k=3: background-background
- k=4: background-standard

### `format_orbital(element, orbital)`
Formats element-orbital pairs consistently.

### `group_by_couple(v_predictions)`
Groups V predictions by atom pairs for advanced QE formats.

## QE Format Complexity

QE supports multiple HUBBARD formats with increasing complexity:

### Basic U + V Format
```
HUBBARD (ortho-atomic)
U Co-3d 7.70
V Co-3d O-2p 1 19 0.75
```

### Advanced Multi-channel Format
```
HUBBARD (ortho-atomic)
V Co-3d Co-3d 1 1 7.70      # standard-standard (U equivalent)
V Co-3d Co-3p 1 1 1.00      # standard-background
V Co-3p Co-3p 1 1 2.00      # background-background  
V Co-3p Co-3d 1 1 1.00      # background-standard
V Co-3d O-2p 1 19 0.75      # inter-site interactions
```

### Legacy Array Format
```
&system
  lda_plus_u = .true.
  lda_plus_u_kind = 2
  U_projection_type = 'ortho-atomic'
  Hubbard_V(1,1,1) = 7.70
  Hubbard_V(1,19,1) = 0.75
/
```

## Creating Custom Templates

Users can create custom templates for specific needs:

```jinja2
# my_custom.j2
# Custom format for my specific QE setup
&control
  calculation = 'scf'
/
&system
  nat = {{ total_atoms }}
/
HUBBARD (ortho-atomic)
{%- for pred in u_predictions %}
U {{ pred.element }}-{{ pred.orbital }} {{ "%.3f"|format(pred.value) }}  # {{ pred.element }} atom
{%- endfor %}
{%- for pred in v_predictions %}
V {{ pred.site0_element }}-{{ pred.site0_orbital }} {{ pred.site1_element }}-{{ pred.site1_orbital }} {{ get_atom_index(pred.site0_data) }} {{ get_atom_index(pred.site1_data) }} {{ "%.3f"|format(pred.value) }}
{%- endfor %}
```

## QE Index Mapping

The most complex part is mapping HubbardML sites to QE atom indices. QE uses:
- Atom positions from ATOMIC_POSITIONS card
- Virtual 3×3×3 supercell for periodic neighbors  
- Complex neighbor detection via intersiteV.f90

The `get_atom_index()` function handles this mapping and can be customized for different structure types and QE setups.

## References

- QE HUBBARD documentation: [Quantum ESPRESSO User Guide](https://www.quantum-espresso.org/Doc/INPUT_PW.html#idm45922794565264)
- hp.x code for automatic U/V determination
- intersiteV.f90 for neighbor index generation

## QE Projection Types

The `projection_type` parameter controls QE's Hubbard projection method:

- **`ortho-atomic`** (default): Orthogonalized atomic orbitals - most common
- **`atomic`**: Raw atomic orbitals  
- **`ortho-atomic-wfc`**: Orthogonalized atomic wavefunctions
- **`norm-atomic`**: Normalized atomic orbitals

Templates use: `{{ projection_type|default('ortho-atomic') }}`

## CLI Usage Examples:
```bash
# Simple format with default ortho-atomic projection
uv run python -m hubbardml predict models/ data.h5 --template qe_simple

# Use different projection type
uv run python -m hubbardml predict models/ data.h5 --template qe_simple --projection-type atomic

# Legacy format with custom parameters  
uv run python -m hubbardml predict models/ data.h5 --template qe_legacy --lda-plus-u-kind 0
```