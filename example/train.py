import torch
import hubbardml
from hubbardml import keys
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import e3psi # Needed for graph.site.create_inputs structure
import gc # Garbage collector

# --- Configuration (remains the same) ---
DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(DTYPE)

# --- Data Loading and Preparation (No DataFrame) ---

def load_data_from_arrow(arrow_files, required_cols, target_col):
    """Loads data from Arrow files directly into Python lists/arrays."""
    tables = []
    print(f"Loading data from: {arrow_files}")
    for file in arrow_files:
        try:
            # Use memory mapping for potentially large files
            with pa.memory_map(file, "rb") as source:
                # Read all record batches into a single table
                reader = pa.ipc.RecordBatchFileReader(source)
                table = reader.read_all()
            print(f"Loaded {table.num_rows} records from {file}")
            tables.append(table)
        except pa.lib.ArrowIOError as e:
            print(f"Warning: Could not read file {file}. Error: {e}. Skipping.")
        except FileNotFoundError:
            print(f"Warning: File not found {file}. Skipping.")

    if not tables:
        raise ValueError("No datasets were loaded. Check file paths and integrity.")

    # Concatenate tables if multiple files were loaded
    if len(tables) > 1:
        full_table = pa.concat_tables(tables)
        del tables # Free memory
        gc.collect()
        print(f"Concatenated tables. Total records: {full_table.num_rows}")
    elif len(tables) == 1:
        full_table = tables[0]
        del tables
        gc.collect()
    else:
         raise ValueError("No data loaded.")


    data_dict = {}
    all_needed_cols = list(set(required_cols + [target_col])) # Combine and unique

    print("Extracting columns and converting to NumPy/lists...")
    for col_name in all_needed_cols:
        if col_name not in full_table.column_names:
            raise KeyError(f"Required column '{col_name}' not found in Arrow table(s).")

        column = full_table.column(col_name)

        # --- Handle Occupation Columns Specifically ---
        # These keys need to match your intended occupation columns
        if col_name in [keys.ATOM_1_OCCS_1, keys.ATOM_1_OCCS_2,
                        keys.ATOM_2_OCCS_1, keys.ATOM_2_OCCS_2]:
            print(f"Processing occupation column: {col_name}")
            try:
                # Convert Arrow list<list<...>> to numpy array of arrays (object dtype)
                # then stack them into a single numpy array [n_samples, n_orbitals, n_channels]
                # This assumes the inner lists represent channels/spin and outer list orbitals
                list_of_lists = column.to_pylist() # More flexible than to_numpy for complex nests
                # Convert each item in the list explicitly to a numpy array
                # This handles cases where pyarrow might return lists instead of arrays directly
                np_arrays = [np.array(item, dtype=np.float32) for item in list_of_lists]
                # Stack the list of numpy arrays into a single large numpy array
                data_dict[col_name] = np.stack(np_arrays)
                print(f" -> Converted {col_name} to NumPy array with shape: {data_dict[col_name].shape}")
                del list_of_lists, np_arrays # Free memory
            except Exception as e:
                print(f"Error processing occupation column {col_name}: {e}")
                print("Ensure the Arrow column contains list-like structures suitable for np.array/np.stack.")
                raise
        elif col_name == keys.ATOM_1_ELEMENT or col_name == keys.ATOM_2_ELEMENT:
            # Assuming elements are strings, convert to list
             data_dict[col_name] = column.to_pylist()
             print(f" -> Extracted {col_name} as list (length {len(data_dict[col_name])})")
        elif col_name == target_col:
             # Assuming target is numeric, convert directly to NumPy array
             data_dict[col_name] = column.to_numpy(zero_copy_only=False)
             # Ensure target is float32
             if data_dict[col_name].dtype != np.float32:
                 data_dict[col_name] = data_dict[col_name].astype(np.float32)
             print(f" -> Converted {col_name} (target) to NumPy array with shape: {data_dict[col_name].shape}")
        else:
            # Default: convert other required columns to numpy arrays if numeric, lists otherwise
            try:
                data_dict[col_name] = column.to_numpy(zero_copy_only=False)
                print(f" -> Converted {col_name} to NumPy array with shape: {data_dict[col_name].shape}")
            except (TypeError, pa.lib.ArrowNotImplementedError):
                 data_dict[col_name] = column.to_pylist()
                 print(f" -> Extracted {col_name} as list (length {len(data_dict[col_name])})")

        del column # Free memory
        gc.collect()


    print("Data loading and conversion complete.")
    # Basic validation
    lengths = {k: len(v) for k, v in data_dict.items()}
    if len(set(lengths.values())) > 1:
        print(f"Warning: Columns have different lengths: {lengths}")
        # Potentially raise error or try to align, but for now just warn
    return data_dict

# --- PyTorch Dataset (Modified for data_dict and indices) ---

class HubbardDataset(Dataset):
    """
    Custom PyTorch Dataset reading from a dictionary of lists/arrays.
    """
    def __init__(self, data_dict, indices, graph, feature_keys, target_key, device):
        self.data = data_dict
        self.indices = indices # List of indices this dataset instance covers
        self.graph = graph
        self.feature_keys = feature_keys # Store keys needed for input creation
        self.target_key = target_key
        self.device = device

        # Pre-extract target values for the specified indices and convert to tensor
        # Use NumPy fancy indexing for efficiency if target is already a NumPy array
        if isinstance(self.data[target_key], np.ndarray):
             self.targets = torch.from_numpy(self.data[target_key][self.indices]).to(dtype=DTYPE)
        else: # Fallback for lists
             self.targets = torch.tensor(
                 [self.data[target_key][i] for i in self.indices],
                 dtype=DTYPE
            )

        # Determine the structure of inputs created by graph.site.create_inputs
        sample_idx = self.indices[0]
        # Ensure all necessary feature keys exist for the sample lookup
        if not all(k in self.data for k in [keys.ATOM_1_ELEMENT, keys.ATOM_1_OCCS_1, keys.ATOM_1_OCCS_2]):
             raise KeyError(f"Missing one or more required keys for sample input generation: "
                           f"{keys.ATOM_1_ELEMENT}, {keys.ATOM_1_OCCS_1}, {keys.ATOM_1_OCCS_2}")

        try:
             sample_input_dict = self.graph.site.create_inputs({
                 "specie": self.data[keys.ATOM_1_ELEMENT][sample_idx],
                 "occs1": self.data[keys.ATOM_1_OCCS_1][sample_idx],
                 "occs2": self.data[keys.ATOM_1_OCCS_2][sample_idx],
             }, dtype=DTYPE, device='cpu') # Create on CPU first
             self.input_keys = list(e3psi.tensorial_attrs(self.graph.site).keys())
             print(f"Dataset using input keys derived from graph.site: {self.input_keys}")
        except IndexError:
            print(f"Error: Could not access data at sample index {sample_idx}. Dataset might be empty or indices incorrect.")
            raise
        except Exception as e:
            print(f"Error determining input keys from graph.site: {e}")
             # Fallback or re-raise depending on desired robustness
            raise


    def __len__(self):
        return len(self.indices) # Length is the number of indices assigned

    def __getitem__(self, idx):
        # Map the sequential dataset index 'idx' to the actual data index
        actual_idx = self.indices[idx]

        # Retrieve data using the actual index from the full data dictionary
        # Use try-except for better error reporting if an index fails
        try:
            element = self.data[keys.ATOM_1_ELEMENT][actual_idx]
            # Assuming occs are NumPy arrays: slicing is efficient
            occs1 = self.data[keys.ATOM_1_OCCS_1][actual_idx]
            occs2 = self.data[keys.ATOM_1_OCCS_2][actual_idx]
            target = self.targets[idx] # Get the pre-converted target for this dataset item
        except IndexError:
             print(f"Error: Index {actual_idx} (from dataset index {idx}) out of bounds for stored data.")
             raise
        except KeyError as e:
             print(f"Error: Key {e} not found in data dictionary while accessing index {actual_idx}.")
             raise


        # Create input features (same logic as before)
        try:
            inputs = self.graph.site.create_inputs({
                "specie": element,
                "occs1": occs1, # Pass the NumPy array directly
                "occs2": occs2, # Pass the NumPy array directly
            }, dtype=DTYPE, device=self.device)

            # Combine the feature tensors into a single tensor per site
            site_tensor = torch.hstack(tuple(inputs[name] for name in self.input_keys))

        except Exception as e:
            print(f"Error creating inputs for actual index {actual_idx}, element {element}: {e}")
            # Include details about occs shapes if possible
            print(f"  Occs1 shape: {occs1.shape if isinstance(occs1, np.ndarray) else type(occs1)}")
            print(f"  Occs2 shape: {occs2.shape if isinstance(occs2, np.ndarray) else type(occs2)}")
            raise RuntimeError(f"Failed to create model inputs for dataset index {idx} (actual index {actual_idx})") from e

        # The UModel expects a dictionary with a key like "site"
        model_input = {"site": site_tensor.unsqueeze(0)} # Add batch dimension

        # Return model input dictionary and target tensor
        return model_input, target.unsqueeze(0) # Add dimension to target


# --- Training Loop (mostly unchanged, interacts with Dataset/DataLoader) ---

def train_model_pytorch(config, data_dict, graph):
    """Trains the HubbardML model using PyTorch loop and data_dict."""

    print(f"Using device: {DEVICE}")
    train_cfg = config["training"]
    model_cfg = config["model"]
    graph_cfg = config["graph"] # Graph config used for model init

    # --- 1. Model Initialization ---
    model = hubbardml.models.UModel(
        graph, # Use the graph passed in
        **model_cfg
    ).to(dtype=DTYPE, device=DEVICE)
    print("Model initialized.")
    # print(model) # Optional: print model structure

    # --- 2. Data Splitting (Indices) ---
    n_samples = len(data_dict[config["data"]["target_col"]]) # Get length from any column
    all_indices = list(range(n_samples))

    # TODO: replace with pytorch's random_split

    train_indices, validate_indices = train_test_split(
        all_indices,
        test_size=train_cfg["validate_frac"],
        random_state=42 # for reproducibility
    )
    print(f"Data split: {len(train_indices)} training indices, {len(validate_indices)} validation indices.")

    # --- 3. Datasets and DataLoaders ---
    # Pass the full data_dict and the respective indices list to each dataset
    train_dataset = HubbardDataset(
        data_dict, train_indices, graph,
        config["data"]["required_feature_cols"], # Pass needed keys
        config["data"]["target_col"], DEVICE
    )
    validate_dataset = HubbardDataset(
         data_dict, validate_indices, graph,
         config["data"]["required_feature_cols"], # Pass needed keys
         config["data"]["target_col"], DEVICE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        # num_workers=4, # Consider uncommenting if data loading is slow
        # pin_memory=True if DEVICE == 'cuda' else False
    )
    validate_loader = DataLoader(
        validate_dataset,
        batch_size=train_cfg["batch_size"] * 2,
        shuffle=False,
    )

    # --- 4. Optimizer and Loss ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])
    loss_fn = torch.nn.MSELoss()

    # --- 5. Training Loop (Identical to previous version) ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved in: {checkpoint_dir}")

    for epoch in range(train_cfg["max_epochs"]):
        # Training Phase
        model.train()
        total_train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in validate_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(validate_loader)
        history['val_loss'].append(avg_val_loss)

        # Logging
        if epoch % train_cfg["status_freq"] == 0 or epoch == train_cfg["max_epochs"] - 1:
             print(f"Epoch {epoch:04d}/{train_cfg['max_epochs']} | "
                   f"Train Loss: {avg_train_loss:.6f} | "
                   f"Val Loss: {avg_val_loss:.6f}")

        # Checkpointing & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_path = checkpoint_dir / "best_model.pth"
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1

        if train_cfg["checkpoint_freq"] > 0 and epoch % train_cfg["checkpoint_freq"] == 0 and epoch > 0:
             ckpt_path = checkpoint_dir / f"model_epoch_{epoch}.pth"
             if ckpt_path != best_model_path: # Avoid saving duplicate if best was this epoch
                torch.save(model.state_dict(), ckpt_path)
                print(f"Checkpoint saved to {ckpt_path}")

        if epochs_no_improve >= train_cfg["overfitting_patience"]:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # --- 6. Final Steps ---
    print("Training finished.")
    best_model_path = checkpoint_dir / "best_model.pth"
    if best_model_path.exists():
        print(f"Loading best model from {best_model_path} (Val Loss: {best_val_loss:.6f})")
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    else:
        print("Warning: No best model checkpoint found. Using the model from the final epoch.")

    print(f"Saving final model to: {train_cfg['final_model_path']}")
    torch.save(model.state_dict(), train_cfg['final_model_path'])

    # Ensure the config reflects the actual species used
    config['graph']['species'] = list(graph.species)
    print(f"Saving final config to: {train_cfg['final_config_path']}")
    with open(train_cfg['final_config_path'], 'w') as f:
        json.dump(config, f, indent=4)

    # Clean up large data structure if no longer needed
    del data_dict
    gc.collect()

    return model, history


# --- Main Execution ---
if __name__ == "__main__":

    # Load configuration

    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Load Data (No DataFrame)
    data_dict = load_data_from_arrow(
        config["data"]["arrow_files"],
        config["data"]["required_feature_cols"],
        config["data"]["target_col"]
    )

    # --- Optional: Add dummy columns if strictly needed by some internal part ---
    # Example: if graph.prepare_dataset or model expects them, though unlikely now
    # if 'is_vdw' not in data_dict:
    #     n_samples = len(data_dict[config['data']['target_col']])
    #     data_dict['is_vdw'] = [False] * n_samples
    #     data_dict['dir'] = [""] * n_samples
    #     data_dict['param_in'] = np.zeros(n_samples, dtype=np.float32)
    #     print("Added dummy columns ('is_vdw', 'dir', 'param_in') to data_dict")


    # 2. Determine Species and Initialize Graph
    # Use atom_1 and atom_2 elements if both exist, otherwise just atom_1
    species1 = set(data_dict.get(keys.ATOM_1_ELEMENT, []))
    species2 = set(data_dict.get(keys.ATOM_2_ELEMENT, []))
    actual_species = list(species1 | species2)
    if not actual_species:
         raise ValueError("Could not determine species from the loaded data. Ensure ATOM_1_ELEMENT exists.")

    print(f"Species found in data: {actual_species}")
    config["graph"]["species"] = actual_species # Update config for model init

    graph = hubbardml.UGraph(
        species=config["graph"]["species"],
        with_param_in=config["graph"]["with_param_in"]
    )
    print("Graph initialized.")

    # 3. Train the Model
    # Pass data_dict instead of DataFrame
    trained_model, training_history = train_model_pytorch(config, data_dict, graph)

    print("Script finished successfully.")

    # Optional: Plotting (same as before)