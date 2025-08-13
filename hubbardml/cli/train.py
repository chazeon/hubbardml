"""Train Hubbard parameter prediction models."""

import sys
from pathlib import Path
import click
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import e3psi

# Import HubbardML components  
from .. import models, graphs
from ..io import load_hdf5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)


class HubbardDataset(Dataset):
    """PyTorch dataset for HubbardML training from HDF5 data."""
    
    def __init__(self, data, indices, graph, device=DEVICE):
        self.data = data
        self.indices = indices  
        self.graph = graph
        self.device = device
        
        # Pre-convert targets to tensors
        if data and hasattr(data, 'target'):
            self.targets = torch.from_numpy(data.target[indices]).to(dtype=DTYPE, device=device)
        else:
            self.targets = torch.zeros(len(indices), dtype=DTYPE, device=device)
        
        # Determine input structure
        if len(indices) > 0:
            sample_idx = indices[0]
            if data and hasattr(data, 'site0') and len(data.site0.elems) > sample_idx:
                self.input_keys = list(e3psi.tensorial_attrs(graph.site).keys())
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        # Get data for this sample
        element = self.data.site0.elems[actual_idx]
        if isinstance(element, bytes):
            element = element.decode()
        
        occs1 = self.data.site0.occs[actual_idx, 0]
        occs2 = self.data.site0.occs[actual_idx, 1]
        target = self.targets[idx]
        
        # Create model inputs
        inputs = self.graph.site.create_inputs({
            "specie": element,
            "occs1": occs1,
            "occs2": occs2,
        }, dtype=DTYPE, device=self.device)
        
        site_tensor = torch.hstack(tuple(inputs[name] for name in self.input_keys))
        return {"site": site_tensor}, target


def train_model(data, config, output_dir: Path):
    """Train HubbardML model on HDF5 data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Using device: {DEVICE}")
    print(f"Using dtype: {DTYPE}")
    
    # Create graph and model
    elements = set()
    if data and hasattr(data, 'site0'):
        for elem in data.site0.elems:
            elem_str = elem.decode() if isinstance(elem, bytes) else elem
            elements.add(elem_str)
    
    if not elements:
        raise ValueError("No elements found in data")
    
    print(f"Found elements: {sorted(elements)}")
    
    graph = graphs.UGraph(species=list(elements), **config["graph"])
    model = models.UModel(graph=graph, **config["model"]).to(dtype=DTYPE, device=DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data splitting  
    n_samples = len(data)
    val_split = config["training"].get("val_split", config["training"].get("validate_frac", 0.2))  # Backward compatibility
    val_size = int(n_samples * val_split)
    train_size = n_samples - val_size
    
    # Create full dataset then split
    full_indices = list(range(n_samples))
    full_dataset = HubbardDataset(data, full_indices, graph)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    
    print(f"Training batches: {len(train_loader)} (batch_size={batch_size})")
    print(f"Validation batches: {len(val_loader)} (batch_size={batch_size * 2})")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    
    # Training loop
    for epoch in range(config["training"]["max_epochs"]):
        # Training phase
        model.train()
        total_train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        # Logging
        log_freq = config["training"].get("log_every_n_epochs", config["training"].get("status_freq", 50))  # Backward compatibility
        if epoch % log_freq == 0:
            print(f"Epoch {epoch:04d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "model.pth")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        patience = config["training"].get("patience", config["training"].get("overfitting_patience", 100))  # Backward compatibility
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break
    
    # Save final config
    final_config = {
        "graph": {"species": list(elements), **config["graph"]},
        "model": config["model"],
        "model_type": "U"
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(final_config, f, indent=2)
    
    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    return model, history


@click.command()
@click.argument("data_file", type=click.Path(exists=True, path_type=Path))
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.help_option("--help", "-h")
def main(data_file: Path, config_file: Path, output_dir: Path):
    """Train HubbardML model on HDF5 data.
    
    DATA_FILE: Path to HDF5 training data
    CONFIG_FILE: Training configuration (YAML or JSON)
    OUTPUT_DIR: Directory to save trained model
    
    Examples:
      uv run python -m hubbardml train data.h5 train.yaml model_output/
      uv run python -m hubbardml train data.h5 config.json model_output/
    """
    
    # Validate input file
    if data_file.suffix.lower() not in ['.h5', '.hdf5']:
        click.echo(f"Error: Input must be HDF5 format, got {data_file.suffix}", err=True)
        sys.exit(1)
    
    # Load config
    with open(config_file) as f:
        if config_file.suffix in ['.yaml', '.yml']:
            train_config = yaml.safe_load(f)
        else:
            train_config = json.load(f)
    
    try:
        # Load data
        print(f"Loading data from: {data_file}")
        u_data, v_data = load_hdf5(str(data_file))
        
        if not u_data:
            click.echo("Error: No U data found for training", err=True)
            sys.exit(1)
        
        print(f"Loaded {len(u_data)} U parameter samples")
        
        # Train model
        train_model(u_data, train_config, output_dir)
        
        print(f"Model saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"Error during training: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()