import click
from . import models, graphs
from .parse_pw3 import parse_all_hubbard_occupations

import torch
import e3psi
import json


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
BATCH_SIZE = 128  # can be adjusted depending on GPU memory

@click.command()
@click.argument("scf_out", type=click.Path(exists=True))
@click.option("-m", "--model-path", help="Path to the saved model", type=click.Path(exists=True), default="trained_model.pth")
@click.option("-c", "--model-config", help="Path to the model configuration", type=click.Path(exists=True), default="trained_model.json")
def main(scf_out: str, model_path: str, model_config: str):

    # Load parsed occupations from SCF output
    occs = parse_all_hubbard_occupations(scf_out)

    # Load model config and instantiate model
    with open(model_config) as fp:
        config = json.load(fp)
    
    graph = graphs.UGraph(**config["graph"])

    model = models.UModel(
        graph=graph,
        **config["model"]
    )

    model.to(dtype=DTYPE, device=DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    site_tensors = []
    site_indices = []

    for i, occ in occs.items():
        inputs = graph.site.create_inputs({
            "specie": "Fe",
            "occs1": occ[1].occupations,
            "occs2": occ[2].occupations,
        }, dtype=DTYPE, device=DEVICE)

        tensor = torch.hstack(tuple(
            inputs[name]
            for name in e3psi.tensorial_attrs(graph.site).keys()
        ))

        site_tensors.append(tensor)
        site_indices.append(i)

    # Stack into a single batch tensor
    batch_tensor = torch.stack(site_tensors)

    # Split into batches if necessary
    with torch.no_grad():
        for i in range(0, len(batch_tensor), BATCH_SIZE):
            batch = batch_tensor[i:i + BATCH_SIZE]
            model_input = {"site": batch}
            predictions = model(model_input)
            for j, pred in enumerate(predictions):
                site_id = site_indices[i + j]
                u_param = pred[0].item()
                print(f"U Fe{site_id}-3d {u_param:.2f}")

if __name__ == "__main__":
    main()