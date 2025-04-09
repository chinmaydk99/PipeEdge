""" Evaluate accuracy on ImageNet dataset using models pruned with DSnoT."""
import os
import argparse
import time
import torch
import numpy as np
import io
from contextlib import redirect_stdout
from typing import List, Dict, Any
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, ImageNet
from torchvision import transforms
from transformers import DeiTFeatureExtractor, ViTFeatureExtractor
from runtime import forward_hook_quant_encode, forward_pre_hook_quant_decode
from utils.data import ViTFeatureExtractorTransforms
# Import model_cfg_dsnot instead of model_cfg_wanda
import model_cfg_dsnot
from evaluation_tools.evaluation_quant_test import *

# Add the missing function if not available in the imports
def load_layers_partition(partition, layer_num):
    """Parse the partition string and return a list of tuples (start_layer, end_layer) for each stage.
    
    Args:
        partition (str): Comma-separated string of partition sizes or 'auto'
        layer_num (int): Total number of layers in the model
        
    Returns:
        List of tuples (start_layer, end_layer) for each stage
    """
    if ',' in partition:
        partitions = [int(p) for p in partition.split(',')]
        assert sum(partitions) == layer_num, f"Sum of partitions {sum(partitions)} must equal total layers {layer_num}"
        
        stages = []
        start_layer = 1
        for p in partitions:
            end_layer = start_layer + p - 1
            stages.append((start_layer, end_layer))
            start_layer = end_layer + 1
        return stages
    else:
        # Assume it's a single number representing equal-sized partitions
        n_partitions = int(partition)
        layers_per_partition = layer_num // n_partitions
        
        stages = []
        for i in range(n_partitions):
            start_layer = i * layers_per_partition + 1
            end_layer = (i + 1) * layers_per_partition if i < n_partitions - 1 else layer_num
            stages.append((start_layer, end_layer))
        return stages

# Keep EnhancedReportAccuracy class, update file naming slightly for clarity
class EnhancedReportAccuracy():
    def __init__(self, batch_size, output_dir, model_name, partition, quant) -> None:
        self.current_acc = 0.0
        self.total_acc = 0.0
        self.correct = 0
        self.tested_batch = 0
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.partition = partition
        self.quant = quant
        # Use the full model name for directory/file naming to distinguish variants
        self.model_name_full = model_name
        self.model_name_short = model_name.split('/')[1] if '/' in model_name else model_name

        self.pruning_method = "DSnoT" # Specify pruning method
        self.pruning_keep_ratio = None

        # Create directory using the full model name
        self.file_path = os.path.join(self.output_dir, self.model_name_full.replace('/', '_')) # Replace slashes
        os.makedirs(self.file_path, exist_ok=True)

        # Update filenames
        self.acc_file = os.path.join(self.file_path, f"result_{self.partition}_{str(self.quant)}_{self.pruning_method}.txt")
        self.sparsity_file = os.path.join(self.file_path, f"sparsity_{self.partition}_{str(self.quant)}_{self.pruning_method}.txt")
        self.final_file = os.path.join(self.file_path, f"final_{self.partition}_{str(self.quant)}_{self.pruning_method}.txt")

        self.sparsity_info = {} # Store sparsity per original layer name
        self.logical_layers = {} # Store mapped logical layer sparsity

    def set_pruning_keep_ratio(self, keep_ratio):
        self.pruning_keep_ratio = keep_ratio

    def update(self, pred, target):
        # Handle potential Nones or empty tensors
        if pred is None or target is None or pred.numel() == 0 or target.numel() == 0:
            print("Warning: Received None or empty tensor in accuracy update. Skipping batch.")
            return
        # Ensure target has compatible shape
        target = target.view_as(pred) if pred.shape != target.shape else target
        # Ensure tensors are on the same device (pred is likely on GPU, target might be CPU)
        if pred.device != target.device:
             target = target.to(pred.device)

        self.correct = pred.eq(target).float().sum().item()
        # Handle batch size potentially varying (e.g., last batch)
        current_batch_size = target.size(0)
        if current_batch_size > 0:
             self.current_acc = self.correct / current_batch_size
             # Weighted average for total accuracy
             total_samples_so_far = self.tested_batch * self.batch_size # Approximate, assumes constant batch size before
             self.total_acc = (self.total_acc * total_samples_so_far + self.correct) / (total_samples_so_far + current_batch_size)
             self.tested_batch += 1 # Increment tested batch count
        else:
             print("Warning: Batch size is zero in accuracy update.")

    def report(self):
        print(f"The accuracy so far is: {100*self.total_acc:.2f}")
        # Use updated acc_file path
        os.makedirs(os.path.dirname(self.acc_file), exist_ok=True)
        with open(self.acc_file, 'a') as f:
            f.write(f"{100*self.total_acc:.2f}\n")

    def capture_sparsity(self, layer_info):
        # Keep existing logic, ensure it uses self.sparsity_file
        if "Layer:" in layer_info and "Density:" in layer_info:
            try:
                layer_name = layer_info.split("Layer:")[1].split("=>")[0].strip()
                density = float(layer_info.split("Density:")[1].strip())
                sparsity = 1.0 - density
                if layer_name not in self.sparsity_info:
                    self.sparsity_info[layer_name] = sparsity
                    with open(self.sparsity_file, 'a') as f:
                        f.write(f"{layer_name}: {sparsity:.6f}\n")
            except (ValueError, IndexError) as e:
                print(f"Error parsing layer info: {e} from '{layer_info}'")

    def map_to_logical_layers(self):
        # Keep existing logic, maps based on layer name patterns for ViT
        # ... (existing map_to_logical_layers implementation) ...
        if not self.sparsity_info:
            return {}
            
        # For ViT: 6 linear layers per transformer block map to 4 logical layers
        logical_layers = {}
        
        # Group layers by their position in the transformer block
        layer_groups = {}
        
        # Process each linear layer and map to logical layers
        # Sort keys to ensure consistent processing order
        sorted_layer_names = sorted(self.sparsity_info.keys())
        
        # --- Logic to derive linear layer index (assumes vit.layers.X...) --- 
        temp_indexed_sparsity = {}
        for name in sorted_layer_names:
             parts = name.split('.')
             try:
                  # Find the index 'X' in 'vit.layers.X. ...'
                  layer_idx_part = next((p for p in parts if p.isdigit()), None)
                  if layer_idx_part is not None:
                       original_block_idx = int(layer_idx_part) # 0-indexed block
                       # Try to determine the specific linear layer within the block
                       linear_layer_name = parts[-2] # e.g., 'query', 'key', 'value', 'dense'
                       sub_component = parts[-3] # e.g., 'attention', 'output', 'intermediate'

                       # Heuristic mapping based on common ViT structure
                       if sub_component == 'self_attention':
                            if linear_layer_name == 'query': component_offset = 0
                            elif linear_layer_name == 'key': component_offset = 1
                            elif linear_layer_name == 'value': component_offset = 2
                            else: component_offset = -1 # Unknown
                       elif sub_component == 'self_output' and linear_layer_name == 'dense':
                            component_offset = 3
                       elif sub_component == 'intermediate' and linear_layer_name == 'dense':
                            component_offset = 4
                       elif sub_component == 'output' and linear_layer_name == 'dense':
                            component_offset = 5
                       else:
                            component_offset = -1 # Unknown

                       if component_offset != -1:
                            # Calculate a pseudo-global index (1-based)
                            global_linear_idx = original_block_idx * 6 + component_offset + 1
                            temp_indexed_sparsity[f"{global_linear_idx}_{name}"] = self.sparsity_info[name]
                       else:
                           print(f"Warning: Could not determine component offset for '{name}', skipping logical mapping.")
                  else:
                       print(f"Warning: Could not parse block index from layer name '{name}', skipping logical mapping.")
             except Exception as e:
                  print(f"Warning: Error parsing layer name '{name}' for logical mapping: {e}")

        # --- Proceed with grouping based on pseudo-global index ---
        for key, sparsity in temp_indexed_sparsity.items():
            try:
                idx_part = key.split('_')[0]
                idx = int(idx_part) # 1-based global linear index
            except (ValueError, IndexError):
                print(f"Warning: Could not parse index from key '{key}', skipping layer in logical mapping.")
                continue

            block_idx = (idx - 1) // 6 # 0-indexed block
            component_idx = (idx - 1) % 6 # 0-5 component
            
            # Map to logical layer (1-indexed, 4 per block)
            if component_idx < 3:  # Q, K, V layers 
                logical_idx = block_idx * 4 + 1
                layer_type = "Attention QKV"
            elif component_idx == 3:  # Attention output projection
                logical_idx = block_idx * 4 + 2
                layer_type = "Attention Output"
            elif component_idx == 4:  # First MLP layer
                logical_idx = block_idx * 4 + 3
                layer_type = "MLP1"
            else:  # Second MLP layer (component_idx == 5)
                logical_idx = block_idx * 4 + 4
                layer_type = "MLP2"
            
            if logical_idx not in layer_groups:
                layer_groups[logical_idx] = {
                    'sparsities': [],
                    'layer_type': layer_type,
                    'block': block_idx + 1  # 1-indexed block
                }
            layer_groups[logical_idx]['sparsities'].append(sparsity)
        
        # Calculate average sparsity for each logical layer
        for logical_idx, data in layer_groups.items():
            valid_sparsities = [s for s in data['sparsities'] if isinstance(s, (float, int))]
            if valid_sparsities:
                 logical_layers[logical_idx] = {
                     'sparsity': sum(valid_sparsities) / len(valid_sparsities),
                     'layer_type': data['layer_type'],
                     'block': data['block']
                 }
            else:
                 print(f"Warning: No valid sparsities found for logical layer {logical_idx}")
                 logical_layers[logical_idx] = {
                     'sparsity': 0.0, # Default to 0 if calculation fails
                     'layer_type': data['layer_type'],
                     'block': data['block']
                 }
            
        self.logical_layers = logical_layers
        return logical_layers


    def save_final_stats(self):
        # Keep existing logic, ensure it uses self.final_file
        self.map_to_logical_layers()
        with open(self.final_file, 'w') as f:
            f.write(f"Model: {self.model_name_full}\n") # Use full name
            f.write(f"Partition: {self.partition}\n")
            f.write(f"Pruning Method: {self.pruning_method}\n") # Add pruning method

            if self.pruning_keep_ratio is not None:
                f.write(f"Pruning Keep Ratio: {self.pruning_keep_ratio}\n")
                f.write(f"Target Sparsity: {1.0 - self.pruning_keep_ratio:.6f}\n")

            f.write(f"Quantization Bits: {self.quant}\n") # Add quant bits
            f.write(f"Final Accuracy: {100*self.total_acc:.6f}%\n")
            f.write(f"Total Batches Tested: {self.tested_batch}\n\n")

            # Write original layer-wise sparsity
            if self.sparsity_info:
                valid_sparsities = [s for s in self.sparsity_info.values() if isinstance(s, (float, int))]
                if valid_sparsities:
                    avg_sparsity = sum(valid_sparsities) / len(valid_sparsities)
                    f.write(f"Average Raw Sparsity (unique layers): {avg_sparsity:.6f}\n")
                else:
                    f.write("Average Raw Sparsity (unique layers): N/A\n")
                f.write("Raw Layer Sparsity:\n")
                for layer_name, sparsity in sorted(self.sparsity_info.items()):
                     f.write(f"  {layer_name}: {sparsity:.6f}\n")
                f.write("\n")

            # Write logical layer sparsity
            if self.logical_layers:
                 avg_logical_sparsity = sum(data['sparsity'] for data in self.logical_layers.values()) / len(self.logical_layers) if self.logical_layers else 0.0
                 f.write(f"Average Logical Layer Sparsity: {avg_logical_sparsity:.6f}\n")
                 # Assuming 48 logical layers for base/large ViT structure
                 num_logical_layers = 48 # Adjust if necessary for Huge model
                 f.write(f"Logical Layer Sparsity (Mapped to {num_logical_layers} Layers):\n")
                 # Write embedding layer (assumed not pruned)
                 f.write(f"  Layer 0 (Embedding): 0.000000\n")
                 for idx in range(1, num_logical_layers + 1):
                     if idx in self.logical_layers:
                         data = self.logical_layers[idx]
                         f.write(f"  Layer {idx} (Block {data['block']}, {data['layer_type']}): {data['sparsity']:.6f}\n")
                     else:
                         # Try to infer type based on index pattern
                         block = ((idx-1) // 4) + 1
                         comp = (idx-1) % 4
                         type_map = {0: "Attn QKV", 1: "Attn Out", 2: "MLP1", 3:"MLP2"}
                         inferred_type = type_map.get(comp, "Unknown")
                         f.write(f"  Layer {idx} (Block {block}, {inferred_type} - Missing Data): 0.000000\n")
                 # Write classification layer (assumed not pruned)
                 f.write(f"  Layer {num_logical_layers + 1} (Classification): 0.000000\n")

# Modify _make_shard to remove the prune argument
def _make_shard(model_name, model_file, stage_layers, stage, q_bits):
    # Use model_cfg_dsnot factory
    shard = model_cfg_dsnot.module_shard_factory(model_name, model_file, stage_layers[stage][0],
                                               stage_layers[stage][1], stage)
    # Quantization bits logic remains the same
    shard.register_buffer('quant_bits', torch.tensor([q_bits]))
    shard.eval()
    return shard

# Forward model remains the same structurally
def _forward_model(input_tensor, model_shards):
    # ... (keep existing implementation) ...
    num_shards = len(model_shards)
    temp_tensor = input_tensor
    for idx in range(num_shards):
        shard = model_shards[idx]

        # decoder
        if idx != 0:
            # Ensure input is correctly formatted (potentially tuple)
            if isinstance(temp_tensor, tuple):
                 temp_tensor = forward_pre_hook_quant_decode(shard, temp_tensor[0])
            else:
                 # Should not happen if previous stage encoded correctly
                 print("Warning: Input to decode hook is not a tuple.")
                 temp_tensor = forward_pre_hook_quant_decode(shard, temp_tensor)

        # forward
        # Handle tuples potentially returned by previous shards or initial input
        if isinstance(temp_tensor, tuple):
             # Check if it's the (encoded_tensor,) format from previous hook
             if len(temp_tensor) == 1 and isinstance(temp_tensor[0], torch.Tensor):
                  temp_tensor = temp_tensor[0]
             # Add more checks if other tuple formats are expected

        if not isinstance(temp_tensor, torch.Tensor):
             print(f"Error: Input to shard {idx} is not a tensor ({type(temp_tensor)}). Skipping forward.")
             # Decide how to handle: return None, raise error?
             return None

        temp_tensor = shard(temp_tensor)
        if temp_tensor is None: # Check if shard forward failed
             print(f"Error: Shard {idx} forward pass returned None. Aborting.")
             return None

        # encoder
        if idx != num_shards-1:
            # Output of encode hook is always a tuple (encoded_tensor,)
            temp_tensor = (forward_hook_quant_encode(shard, None, temp_tensor),)

    return temp_tensor


def evaluation(args, dataset_cfg):
    """ Evaluation main func for DSnoT pruning """
    # localize parameters
    dataset_path = args.dataset_root
    dataset_split = args.dataset_split
    batch_size = args.batch_size
    ubatch_size = args.ubatch_size # Keep microbatch size for potential use
    num_workers = args.num_workers
    partition = args.partition
    quant = args.quant
    output_dir = args.output_dir
    model_name = args.model_name # Should be the DSnoT variant name
    prune_level = args.prune_level # Sparsity level (1 - keep_ratio)
    keep_ratio = 1.0 - prune_level

    # --- DSnoT Specific Args --- 
    dsnot_max_cycles = args.dsnot_max_cycles
    dsnot_error_threshold = args.dsnot_error_threshold
    dsnot_percdamp = args.dsnot_percdamp
    dsnot_pow_var = args.dsnot_pow_var
    dsnot_skip_first_last = not args.dsnot_prune_first_last # Invert flag logic

    # Get model config using the dsnot config module
    model_config = model_cfg_dsnot.get_model_config(model_name)
    model_file = model_cfg_dsnot.get_model_default_weights_file(model_name)
    # Check if the original weights file exists, download if needed
    if not os.path.exists(model_file):
        print(f"Original weights file {model_file} not found. Attempting download...")
        model_cfg_dsnot.save_model_weights_file(model_name, model_file)
        if not os.path.exists(model_file):
            print(f"Error: Failed to download weights file {model_file}. Exiting.")
            return

    # Data loading setup (remains largely the same)
    img_size = (model_config.image_size, model_config.image_size)
    if model_name.startswith('facebook/deit'):
        feature_extractor = DeiTFeatureExtractor.from_pretrained(model_name)
    else:
        # Assume ViTFeatureExtractor for google/vit* models
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name.replace('-dsnot', '')) # Use base name

    transforms = ViTFeatureExtractorTransforms(feature_extractor)
    input_dataset = ImageNet(root=dataset_path, split=dataset_split, transform=transforms)
    input_loader = DataLoader(input_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load calibration data (required for DSnoT)
    # Use a subset of the evaluation dataset or a separate calibration set
    # For simplicity, let's use a subset of the ImageNet val split used for eval
    calibration_size = 256 # Number of samples for calibration
    if len(input_dataset) < calibration_size:
        print(f"Warning: Dataset size ({len(input_dataset)}) is smaller than calibration size ({calibration_size}). Using full dataset.")
        calibration_size = len(input_dataset)
    
    # Create a sampler for the calibration subset
    indices = torch.randperm(len(input_dataset))[:calibration_size]
    calibration_subset = torch.utils.data.Subset(input_dataset, indices)
    # Use a reasonable batch size for calibration to fit stats in memory
    calibration_loader = DataLoader(calibration_subset, batch_size=min(batch_size, 32), shuffle=False, num_workers=num_workers)
    
    # Concatenate calibration data into a single tensor (ubatch)
    print("Loading calibration data...")
    ubatch_list = []
    labels_list = [] # Keep labels if needed for _quick_eval during pruning
    for inputs, labels in calibration_loader:
        ubatch_list.append(inputs)
        labels_list.append(labels)
    ubatch_calib = torch.cat(ubatch_list, dim=0)
    labels_calib = torch.cat(labels_list, dim=0)
    # Create a tuple for potential use in _quick_eval
    calib_batch_for_eval = (ubatch_calib, labels_calib)
    print(f"Calibration data loaded: {ubatch_calib.shape}")


    # Load Partition
    layer_num = model_cfg_dsnot.get_model_layers(model_name)
    if partition == 'auto':
        partition = str(layer_num // 4)
    stage_layers = load_layers_partition(partition, layer_num)
    n_stages = len(stage_layers)
    print(f'Number of stages: {n_stages}')

    # Quantization Setup (remains the same)
    q_bits = _get_default_quant(n_stages) if quant == 'auto' else [int(quant)]*n_stages

    # Model Loading (Load Dense First)
    print("Loading dense model shards...")
    model_shards = [_make_shard(model_name, model_file, stage_layers, i, q_bits[i]) for i in range(n_stages)]

    # --- Apply DSnoT Pruning --- 
    print(f"Applying SparseGPT+DSnoT pruning with keep_ratio={keep_ratio:.4f}...")
    # Need the full model instance to run pruning. We assume pruning is done
    # on the entire model before splitting/evaluation, or applied shard-by-shard if feasible.
    # For now, let's assume we prune the *first* shard which contains the ViT model logic.
    # THIS IS A SIMPLIFICATION. Proper implementation would prune a full model instance.
    
    pruning_start_time = time.time()
    pruned_state_dict = None
    
    # Redirect stdout to capture sparsity printouts
    stdout_capture = io.StringIO()
    with redirect_stdout(stdout_capture):
        try:
            # We need a full model instance or apply pruning logic carefully across shards.
            # Let's create a temporary full model instance for pruning (requires enough memory).
            print("Creating temporary full model instance for pruning...")
            # Use the factory to create a single shard covering all layers
            full_model_shard = model_cfg_dsnot.module_shard_factory(model_name, model_file, 1, layer_num, 0)
            full_model_shard.to(devices.DEVICE) # Ensure it's on the right device
            
            # Perform pruning on the full model instance
            pruned_state_dict = full_model_shard.prune_sparsegpt_dsnot(
                ubatch=ubatch_calib.to(devices.DEVICE), 
                keep_ratio=keep_ratio,
                max_cycles=dsnot_max_cycles,
                error_threshold=dsnot_error_threshold,
                percdamp=dsnot_percdamp,
                pow_of_var_regrowing=dsnot_pow_var,
                skip_first_last=dsnot_skip_first_last
            )
            
            # Pruning finished, load the pruned state dict back into the shards
            print("Loading pruned state dict into model shards...")
            # Create a new set of shards and load the pruned weights
            # This requires careful state dict key mapping if sharded.
            # Simpler approach: Load the full pruned state dict into the temp model,
            # then extract shard state dicts.
            full_model_shard.load_state_dict(pruned_state_dict)
            
            # Now, reload the evaluation shards using the pruned full model's state
            # This is inefficient but demonstrates the flow.
            # A better way would be to directly load the relevant parts of pruned_state_dict.
            temp_full_model_state = full_model_shard.state_dict()
            del full_model_shard # Free memory of temp model
            if devices.DEVICE.type == 'cuda': torch.cuda.empty_cache()
            
            for i, shard in enumerate(model_shards):
                shard_state_dict = shard.state_dict()
                # Filter the full state dict to get only keys relevant to this shard
                relevant_keys = {k for k in temp_full_model_state if k in shard_state_dict}
                filtered_state_dict = {k: temp_full_model_state[k] for k in relevant_keys}
                try:
                     shard.load_state_dict(filtered_state_dict, strict=False)
                     print(f"Loaded pruned weights into shard {i}")
                except RuntimeError as e:
                     print(f"Error loading pruned state dict into shard {i}: {e}")
                     # Option: fall back to dense weights or stop
            
            del temp_full_model_state # Free memory
            if devices.DEVICE.type == 'cuda': torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error during pruning process: {e}")
            # Handle error, maybe proceed with dense model or exit
            print("Proceeding with dense model due to pruning error.")
            pruned_state_dict = None # Ensure we don't try to save pruned weights

    pruning_duration = time.time() - pruning_start_time
    print(f"Pruning finished in {pruning_duration:.2f} seconds.")

    # Capture sparsity info from stdout
    sparsity_output = stdout_capture.getvalue()
    print("--- Pruning Output ---")
    print(sparsity_output)
    print("----------------------")

    # --- Save Pruned Weights --- 
    if pruned_state_dict is not None:
        pruned_weights_file = model_cfg_dsnot.get_model_pruned_weights_file(model_name)
        if pruned_weights_file:
            print(f"Saving pruned weights to {pruned_weights_file}...")
            # Save the state dict (which is already on CPU)
            try:
                # Convert to numpy arrays for saving in .npz format if needed, or save directly as .pt
                # Assuming saving as .pt for simplicity
                torch.save(pruned_state_dict, pruned_weights_file)
                print("Pruned weights saved.")
            except Exception as e:
                print(f"Error saving pruned weights to {pruned_weights_file}: {e}")
        else:
            print("No pruned weights file specified in config. Skipping save.")


    # Initialize Accuracy Reporting
    report = EnhancedReportAccuracy(batch_size, output_dir, model_name, partition, quant)
    report.set_pruning_keep_ratio(keep_ratio) # Pass keep ratio

    # Process sparsity output captured from stdout
    for line in sparsity_output.splitlines():
        report.capture_sparsity(line)

    # Evaluation Loop
    print("Starting evaluation...")
    t_start = time.time()
    with torch.no_grad():
        for i, (images, target) in enumerate(input_loader):
            images = images.to(devices.DEVICE)
            target = target.to(devices.DEVICE)

            # Forward pass through the potentially pruned model shards
            output = _forward_model(images, model_shards)

            # Check if forward pass failed
            if output is None:
                print(f"Skipping batch {i} due to forward pass error.")
                continue

            # Post-processing output (ensure it's logits)
            if isinstance(output, tuple):
                # Assuming the first element is logits, adjust if needed
                logits = output[0]
            else:
                logits = output

            if not isinstance(logits, torch.Tensor):
                 print(f"Warning: Output from model is not a tensor ({type(logits)}). Skipping batch {i}.")
                 continue

            # Calculate accuracy
            maxk = max((1,))
            batch_size_actual = target.size(0)
            if batch_size_actual == 0:
                 continue # Skip empty batches

            _, pred = logits.topk(maxk, 1, True, True)
            pred = pred.t()

            # Update accuracy report
            report.update(pred, target)

            if (i+1) % 10 == 0:
                print(f"Finished batch {i+1}, The accuracy up to now is: {report.total_acc * 100:.2f}% ")

    t_end = time.time()
    print(f"Total accuracy: {report.total_acc * 100:.2f}%")
    print(f'Evaluation finished in {t_end - t_start:.2f}s')

    # Save final statistics
    report.save_final_stats()
    print(f"Final statistics saved to: {report.final_file}")


if __name__ == '__main__':
    # Argument parsing setup
    parser = argparse.ArgumentParser(description='PipeEdge Evaluation with DSnoT Pruning')
    parser.add_argument('--model-name', type=str, # required=True, # Make optional
                        choices=model_cfg_dsnot.get_model_names(),
                        default='google/vit-base-patch16-224-dsnot', # Add default
                        help='Model name (must be a DSnoT variant, e.g., google/vit-base-patch16-224-dsnot)')
    parser.add_argument('--dataset-root', type=str, default='/project/jpwalter_148/hnwang/datasets/ImageNet/', # Add default
                        help='Dataset root directory (default: /project/jpwalter_148/hnwang/datasets/ImageNet/)')
    parser.add_argument('--dataset-split', type=str, default='val', choices=['val', 'train'],
                        help='Dataset split (default: val)')
    parser.add_argument('--partition', type=str, default='1',
                        help='Partition file suffix (e.g., 1, 2, 4, auto; default: 1)')
    parser.add_argument('--quant', type=str, default='0',
                        help='Quantization bits (e.g., 8, 16, auto, 0=FP32; default: 0)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Input batch size (default: 16)')
    parser.add_argument('--ubatch-size', type=int, default=1,
                        help='Input micro-batch size (default: 1)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers (default: 4)')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory (default: ./results)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='Gpu id list (default: 0)')
    # Pruning arguments
    parser.add_argument('--prune-level', type=float, default=0.0,
                        help='Target sparsity level (0.0 to 1.0, default: 0.0 - no pruning)')
    parser.add_argument('--dsnot-max-cycles', type=int, default=50,
                        help='Max cycles for DSnoT refinement (default: 50)')
    parser.add_argument('--dsnot-error-threshold', type=float, default=0.1,
                        help='Error threshold for DSnoT convergence (default: 0.1)')
    parser.add_argument('--dsnot-percdamp', type=float, default=0.01,
                        help='Percdamp for SparseGPT Hessian inversion (default: 0.01)')
    parser.add_argument('--dsnot-pow-var', type=float, default=1.0,
                        help='Power of variance for DSnoT growing score (default: 1.0)')
    parser.add_argument('--dsnot-prune-first-last', action='store_true',
                        help='If set, prune the first and last linear layers (DSnoT default is to skip)')

    # Parse arguments
    args = parser.parse_args()

    # Select device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    import devices
    devices.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {devices.DEVICE}")

    # Run evaluation
    evaluation(args, None) 