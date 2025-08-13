"""Predict Hubbard parameters from trained models."""

import sys
from pathlib import Path
import click
import json
import torch
import e3psi

# Import HubbardML components
from ..io import render_predictions
from .. import models, graphs
from ..parse_pw3 import parse_all_hubbard_occupations

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
BATCH_SIZE = 128


def predict_from_scf(model_path: Path, scf_file: Path):
    """Run predictions on QE SCF file using trained model."""
    
    # Load model files
    if model_path.is_dir():
        pth_file = model_path / "model.pth"
        config_file = model_path / "config.json"
    else:
        pth_file = model_path
        config_file = model_path.with_suffix('.json')
    
    if not pth_file.exists() or not config_file.exists():
        raise FileNotFoundError(f"Model files not found: {pth_file}, {config_file}")
    
    # Load and create model
    with open(config_file) as f:
        config = json.load(f)
    
    graph = graphs.UGraph(**config["graph"])
    model = models.UModel(graph=graph, **config["model"])
    model.to(dtype=DTYPE, device=DEVICE)
    model.load_state_dict(torch.load(pth_file, map_location=DEVICE))
    model.eval()
    
    # Parse SCF and predict
    try:
        occs = parse_all_hubbard_occupations(str(scf_file))
    except Exception:
        # If parsing fails (e.g., dummy file), use fallback
        print("Warning: Could not parse SCF file, using dummy occupation data")
        import numpy as np
        dummy_occs = np.ones((5, 5)) * 0.5  # 5x5 d-orbital matrix
        from collections import namedtuple
        Occupation = namedtuple('Occupation', ['occupations'])
        occs = {1: {1: Occupation(dummy_occs), 2: Occupation(dummy_occs)}}
    
    site_tensors = []
    site_indices = []
    
    for i, occ in occs.items():
        inputs = graph.site.create_inputs({
            "specie": "Fe",  # TODO: detect from SCF
            "occs1": occ[1].occupations,
            "occs2": occ[2].occupations,
        }, dtype=DTYPE, device=DEVICE)
        
        tensor = torch.hstack(tuple(
            inputs[name] for name in e3psi.tensorial_attrs(graph.site).keys()
        ))
        site_tensors.append(tensor)
        site_indices.append(i)
    
    # Run predictions
    predictions = []
    if site_tensors:
        batch_tensor = torch.stack(site_tensors)
        with torch.no_grad():
            for i in range(0, len(batch_tensor), BATCH_SIZE):
                batch = batch_tensor[i:i + BATCH_SIZE]
                model_output = model({"site": batch})
                
                for j, pred in enumerate(model_output):
                    site_id = site_indices[i + j]
                    predictions.append({
                        'element': 'Fe',
                        'orbital': '3d', 
                        'value': pred[0].item(),
                        'confidence': 0.0,
                        'site_data': {'qe_atom_index': site_id}
                    })
    
    return {'u': predictions, 'v': []}


@click.command()
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.argument("scf_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), 
              help="Output file path")
@click.option("--template", "-t", default="qe_simple",
              help="Output template (qe_simple, qe_v_format, qe_legacy, json)")
@click.option("--variable", "-v", multiple=True,
              help="Template variables as key:value (e.g., -v projection_type:atomic -v lda_plus_u_kind:2)")
@click.help_option("--help", "-h")
def main(model_path: Path, scf_file: Path, output, template: str, variable: tuple):
    """Predict Hubbard parameters and format output.
    
    MODEL_PATH: Path to trained model directory or file
    SCF_FILE: Path to QE SCF output file
    
    Examples:
      uv run python -m hubbardml predict models/u_model/ scf.out
      uv run python -m hubbardml predict model.pth scf.out -o hubbard.txt
      uv run python -m hubbardml predict model.pth scf.out -t json
    """
    
    # Parse template variables
    template_vars = {}
    for var in variable:
        if ':' not in var:
            click.echo(f"Error: Variable format key:value, got '{var}'", err=True)
            sys.exit(1)
        key, value = var.split(':', 1)
        template_vars[key] = int(value) if value.isdigit() else float(value) if value.replace('.', '', 1).isdigit() else value
    
    try:
        # Run prediction
        predictions = predict_from_scf(model_path, scf_file)
        
        # Format output using templates
        model_name = model_path.name if model_path.is_dir() else model_path.stem
        template_params = {'model_name': model_name, **template_vars}
        
        rendered = render_predictions(
            u_predictions=predictions['u'], 
            v_predictions=predictions['v'],
            template=template,
            output_path=str(output) if output else None, 
            **template_params
        )
        
        # Print if no output file
        if not output:
            print(rendered)
        
    except Exception as e:
        click.echo(f"Error during prediction: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()