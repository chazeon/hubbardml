"""Clean, symmetric training CLI for HubbardML U and V models."""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union
import click
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import e3psi

from .. import models, graphs
from ..io import load_hdf5, UData, VData

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


class Trainer:
    """Train U or V models."""

    def __init__(
        self,
        model_type: str,
        data: Union[UData, VData],
        config: Dict[str, dict],
        output_dir: Path,
    ) -> None:
        self.model_type = model_type
        self.data = data
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def elements(self) -> List[str]:
        """What elements are in the data?"""
        elements = set()
        if hasattr(self.data, "site0"):
            for elem in self.data.site0.elems:
                elem_str = elem.decode() if isinstance(elem, bytes) else elem
                elements.add(elem_str)
        if hasattr(self.data, "site1"):  # V models have site1
            for elem in self.data.site1.elems:
                elem_str = elem.decode() if isinstance(elem, bytes) else elem
                elements.add(elem_str)
        return sorted(elements)

    def make_model(self) -> Tuple[nn.Module, type]:
        """Build the model."""
        # Get species from config or auto-detect
        if "species" not in self.config["graph"]:
            # Auto-detect and warn
            elements = self.elements()
            print(f"Warning: No species in config, auto-detected: {elements}")
            self.config["graph"]["species"] = elements
        else:
            elements = self.config["graph"]["species"]
            print(f"Using config species: {elements}")

        # Check species exist in data
        data_elements = self.elements()
        missing = set(elements) - set(data_elements)
        if missing:
            raise ValueError(
                f"Config species {missing} not found in data {data_elements}"
            )

        # Build model
        if self.model_type == "u":
            graph = graphs.UGraph(**self.config["graph"])
            model = models.UModel(graph=graph, **self.config["model"])
            dataset_class = UDataset
        elif self.model_type == "v":
            graph = graphs.VGraph(**self.config["graph"])
            model = models.VModel(graph=graph, **self.config["model"])
            dataset_class = VDataset
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        model = model.to(dtype=DTYPE, device=DEVICE)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model, dataset_class

    def make_datasets(
        self, dataset_class: type, graph
    ) -> Tuple[DataLoader, DataLoader]:
        """Split data into train/validation."""
        dataset = dataset_class(self.data, graph, device=DEVICE)

        # Split 80/20
        val_split = self.config["training"].get("val_split", 0.2)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        train_data, val_data = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        # Make loaders
        batch_size = self.config["training"].get("batch_size", 32)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size * 2, shuffle=False)

        print(f"Training batches: {len(train_loader)} (batch_size={batch_size})")
        print(f"Validation batches: {len(val_loader)} (batch_size={batch_size * 2})")

        return train_loader, val_loader

    def train_loop(
        self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader
    ) -> Tuple[nn.Module, Dict[str, List[float]], float]:
        """Main training loop."""
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config["training"].get("learning_rate", 0.001)
        )
        loss_fn = nn.MSELoss()

        max_epochs = self.config["training"].get("max_epochs", 1000)
        patience = self.config["training"].get("patience", 100)
        log_freq = self.config["training"].get("log_every_n_epochs", 50)

        # Track progress
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            # Training pass
            model.train()
            total_train_loss = 0
            for batch_inputs, batch_targets in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = loss_fn(outputs.squeeze(), batch_targets)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            # Validation pass
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_inputs, batch_targets in val_loader:
                    outputs = model(batch_inputs)
                    loss = loss_fn(outputs.squeeze(), batch_targets)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            history["val_loss"].append(avg_val_loss)

            # Print progress
            if epoch % log_freq == 0:
                print(
                    f"Epoch {epoch:04d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}"
                )

            # Save if better
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), self.output_dir / "model.pth")
            else:
                epochs_no_improve += 1

            # Stop if stuck
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
        return model, history, best_val_loss

    def save_config(self) -> Path:
        """Save config."""
        # Species already in config from make_model()
        config_file = self.output_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2)
        return config_file

    def train(self) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """Train the model."""
        print(f"Using device: {DEVICE}")
        print(f"Using dtype: {DTYPE}")

        # Build model
        model, dataset_class = self.make_model()

        # Build graph for datasets (same as model)
        if self.model_type == "u":
            graph = graphs.UGraph(**self.config["graph"])
        else:  # v
            graph = graphs.VGraph(**self.config["graph"])

        # Split train/val
        train_loader, val_loader = self.make_datasets(dataset_class, graph)

        # Run training
        model, history, _best_val_loss = self.train_loop(
            model, train_loader, val_loader
        )

        # Save everything
        config_file = self.save_config()

        print(f"Model saved to: {self.output_dir}")
        print(f"Config saved to: {config_file}")

        return model, history


class UDataset(Dataset):
    """U model data."""

    def __init__(self, data, graph: graphs.UGraph, device=DEVICE):
        self.data = data
        self.graph = graph
        self.device = device
        self.targets = torch.from_numpy(data.target).to(dtype=DTYPE, device=device)

        # What tensors do we need?
        self.input_keys = list(e3psi.tensorial_attrs(graph.site).keys())

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        element = self.data.site0.elems[idx]
        if isinstance(element, bytes):
            element = element.decode()

        occs1 = self.data.site0.occs[idx, 0]
        occs2 = self.data.site0.occs[idx, 1]
        target = self.targets[idx]

        # Make tensors
        inputs = self.graph.site.create_inputs(
            {
                "specie": element,
                "occs1": occs1,
                "occs2": occs2,
            },
            dtype=DTYPE,
            device=self.device,
        )

        site_tensor = torch.hstack(tuple(inputs[name] for name in self.input_keys))
        return {"site": site_tensor}, target


class VDataset(Dataset):
    """V model data."""

    def __init__(self, data, graph: graphs.VGraph, device=DEVICE):
        self.data = data
        self.graph = graph
        self.device = device
        self.targets = torch.from_numpy(data.target).to(dtype=DTYPE, device=device)

        # What tensors for two sites?
        self.site0_keys = list(e3psi.tensorial_attrs(graph.site0).keys())
        self.site1_keys = list(e3psi.tensorial_attrs(graph.site1).keys())

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # First site data
        elem0 = self.data.site0.elems[idx]
        if isinstance(elem0, bytes):
            elem0 = elem0.decode()
        occs0_1 = self.data.site0.occs[idx, 0]
        occs0_2 = self.data.site0.occs[idx, 1]

        # Second site data
        elem1 = self.data.site1.elems[idx]
        if isinstance(elem1, bytes):
            elem1 = elem1.decode()
        occs1_1 = self.data.site1.occs[idx, 0]
        occs1_2 = self.data.site1.occs[idx, 1]

        target = self.targets[idx]

        # Make tensors for both sites
        site0_inputs = self.graph.site0.create_inputs(
            {
                "specie": elem0,
                "occs1": occs0_1,
                "occs2": occs0_2,
            },
            dtype=DTYPE,
            device=self.device,
        )

        site1_inputs = self.graph.site1.create_inputs(
            {
                "specie": elem1,
                "occs1": occs1_1,
                "occs2": occs1_2,
            },
            dtype=DTYPE,
            device=self.device,
        )

        # Combine tensors
        site0_tensor = torch.hstack(
            tuple(site0_inputs[name] for name in self.site0_keys)
        )
        site1_tensor = torch.hstack(
            tuple(site1_inputs[name] for name in self.site1_keys)
        )

        # Distance between sites
        edge_tensor = torch.tensor(
            [self.data.edge.dist[idx]], dtype=DTYPE, device=self.device
        )

        return {
            "site0": site0_tensor,
            "site1": site1_tensor,
            "edge": edge_tensor,
        }, target


def load_data(config: Dict[str, dict], config_file: Path) -> Tuple[object, str]:
    """Load the HDF5 data file."""
    # Get data file path
    data_file_str = config["data"]["file"]
    data_file = Path(data_file_str)

    # Handle relative paths
    if not data_file.is_absolute():
        data_file = config_file.parent / data_file

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Load data
    print(f"Loading data from: {data_file}")
    u_data, v_data = load_hdf5(str(data_file))

    # What model type?
    model_type = config.get("data", {}).get("model_type", "u")

    # Return the right data
    if model_type == "u":
        if not u_data or not hasattr(u_data, "target"):
            raise ValueError("No U data found")
        return u_data, model_type
    elif model_type == "v":
        if not v_data or not hasattr(v_data, "target"):
            raise ValueError("No V data found")
        return v_data, model_type
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@click.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.help_option("--help", "-h")
def main(config_file: Path, output_dir: Path):
    """Train HubbardML model using configuration file.

    CONFIG_FILE: Path to YAML configuration file
    OUTPUT_DIR: Directory to save trained model
    """

    try:
        # Load config
        with open(config_file) as f:
            if config_file.suffix.lower() in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

        # Load data
        data, model_type = load_data(config, config_file)
        print(f"Loaded {len(data.target)} {model_type.upper()} parameter samples")

        # Train model
        trainer = Trainer(model_type, data, config, output_dir)
        trainer.train()

    except Exception as e:
        click.echo(f"Error during training: {e}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
