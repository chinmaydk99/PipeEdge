#!/usr/bin/env python3
"""Evaluation script for comparing WANDA vs DSnoT pruned models."""

import argparse
import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

import model_cfg_dsnot
import model_cfg_wanda
from pipeedge.models import ModuleShardConfig
import devices
from runtime import forward_hook_quant_encode, forward_pre_hook_quant_decode

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
# DEFAULT_IMAGENET_DIR = os.path.join(DEFAULT_DATA_DIR, 'imagenet') # Incorrect default
DEFAULT_IMAGENET_DIR = "/project/jpwalter_148/hnwang/datasets/ImageNet/" # Correct default from evaluation_wanda.py
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')

def count_parameters(model):
    """Count the total parameters and non-zero parameters in a model."""
    total_params = 0
    nonzero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only count weights, not biases
            param_count = param.numel()
            total_params += param_count
            nonzero_params += (param != 0).sum().item()
    
    return total_params, nonzero_params

def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy on the given dataloader."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Calculate accuracy
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    accuracy = 100.0 * correct / total
    return accuracy

def load_model(model_name, weights_file, device, partition=None, prune=False):
    """Load a model with the specified weights."""
    # Get model configuration
    config = model_cfg_dsnot.get_model_config(model_name)
    
    # Create full model (one shard containing all layers)
    layers = model_cfg_dsnot.get_model_layers(model_name)
    
    # If partition is specified, use it to create sharded model
    if partition:
        parts = [int(i) for i in partition.split(',')]
        assert len(parts) % 2 == 0, "Partition must have an even number of elements"
        stage_layers = [(parts[i], parts[i+1]) for i in range(0, len(parts), 2)]
        model_shards = []
        
        for i, (layer_start, layer_end) in enumerate(stage_layers):
            is_first = i == 0
            is_last = i == len(stage_layers) - 1
            shard_config = ModuleShardConfig(layer_start=layer_start, layer_end=layer_end,
                                          is_first=is_first, is_last=is_last)
            
            # Get the appropriate model class - use WANDA for original weights, DSnoT for pruned weights
            if prune:
                model_class = model_cfg_dsnot.get_model_dict(model_name)['shard_module']
            else:
                model_class = model_cfg_wanda.get_model_dict(model_name)['shard_module']
            
            # Load weights
            if isinstance(weights_file, str) and os.path.isfile(weights_file):
                if weights_file.endswith('.pt'):
                    print(f"Loading state dict from .pt file: {weights_file}")
                    loaded_weights = torch.load(weights_file, map_location='cpu')
                elif weights_file.endswith('.npz'):
                    print(f"Loading from .npz file: {weights_file}")
                    loaded_weights = weights_file
                else:
                    raise ValueError(f"Unsupported weights file format: {weights_file}")
            elif isinstance(weights_file, dict):
                print("Loading from provided state dict mapping.")
                loaded_weights = weights_file
            else:
                raise TypeError(f"Invalid weights_file type: {type(weights_file)}")
            
            # Instantiate shard, pass the prune flag
            shard = model_class(config, shard_config, loaded_weights, prune=prune)
            shard.to(device)
            shard.eval()
            model_shards.append(shard)
        
        # Return model shards for partitioned execution
        return model_shards
    else:
        # Create single shard containing all layers (original behavior)
        shard_config = ModuleShardConfig(layer_start=1, layer_end=layers, 
                                      is_first=True, is_last=True)
        
        # Get the appropriate model class - use WANDA for original weights, DSnoT for pruned weights
        if prune:
            model_class = model_cfg_dsnot.get_model_dict(model_name)['shard_module']
        else:
            model_class = model_cfg_wanda.get_model_dict(model_name)['shard_module']
        
        # Load weights correctly depending on file type
        if isinstance(weights_file, str) and os.path.isfile(weights_file):
            if weights_file.endswith('.pt'):
                print(f"Loading state dict from .pt file: {weights_file}")
                loaded_weights = torch.load(weights_file, map_location='cpu') 
            elif weights_file.endswith('.npz'):
                 print(f"Loading from .npz file: {weights_file}")
                 loaded_weights = weights_file 
            else:
                 raise ValueError(f"Unsupported weights file format: {weights_file}")
        elif isinstance(weights_file, dict):
             print("Loading from provided state dict mapping.")
             loaded_weights = weights_file
        else:
             raise TypeError(f"Invalid weights_file type: {type(weights_file)}")
    
        # Instantiate the model, pass the prune flag
        model = model_class(config, shard_config, loaded_weights, prune=prune) 
        model.to(device)
        model.eval()
        
        return model

def _forward_model(input_tensor, model_shards, quant=None):
    """Forward pass through partitioned model."""
    num_shards = len(model_shards)
    if not isinstance(model_shards, list):
        # Single model case
        return model_shards(input_tensor)
    
    # Handle partitioned model case
    temp_tensor = input_tensor
    for idx in range(num_shards):
        shard = model_shards[idx]

        # decoder (if using quantization)
        if quant and idx != 0:
            temp_tensor = forward_pre_hook_quant_decode(shard, temp_tensor)

        # forward
        if isinstance(temp_tensor, tuple) and len(temp_tensor) > 0:
            if isinstance(temp_tensor[0], tuple) and len(temp_tensor[0]) == 2:
                temp_tensor = temp_tensor[0]
            elif isinstance(temp_tensor[0], torch.Tensor):
                temp_tensor = temp_tensor[0]
        temp_tensor = shard(temp_tensor)

        # encoder (if using quantization)
        if quant and idx != num_shards-1:
            temp_tensor = (forward_hook_quant_encode(shard, None, temp_tensor),)
            
    return temp_tensor

def evaluate_partitioned_model(model_shards, dataloader, device, quant=None):
    """Evaluate accuracy of a partitioned model."""
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass through partitioned model
            outputs = _forward_model(inputs, model_shards, quant)
            
            # Get predictions
            _, predicted = outputs.max(1)
            
            # Calculate accuracy
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    accuracy = 100.0 * correct / total
    return accuracy

def prepare_imagenet_dataloaders(data_dir, batch_size=64, workers=4):
    """Prepare ImageNet validation dataloader."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    
    return val_loader

def prepare_calibration_batch(val_loader, num_batches=1, device=None):
    """Prepare a batch of data for calibration."""
    calibration_data = []
    for i, (inputs, _) in enumerate(val_loader):
        if i >= num_batches:
            break
        if device is not None:
            inputs = inputs.to(device)
        calibration_data.append(inputs)
    
    return torch.cat(calibration_data, dim=0)

def fix_dsnot_bug():
    """
    Monkey patch the refine_dsnot method in vit_dsnot to fix the cycle variable bug.
    """
    try:
        from pipeedge.models.transformers.vit_dsnot import ViTShardForImageClassification

        # Store the original method
        original_refine_dsnot = ViTShardForImageClassification.refine_dsnot

        # Define a fixed version with proper cycle initialization
        def fixed_refine_dsnot(self, net, original_weights, initial_masks, layer_stats, dsnot_args):
            """
            Fixed version of refine_dsnot that properly initializes the cycle variable.
            """
            print("--- Starting DSnoT Refinement ---")
            max_cycle_time = dsnot_args.get('dsnot_cycles', 50) 
            update_threshold = dsnot_args.get('dsnot_threshold', 0.1)
            pow_of_var_regrowing = dsnot_args.get('pow_of_var_regrowing', 1.0)
            without_same_sign = dsnot_args.get('without_same_sign', True)

            for name, module in net.named_modules():
                if isinstance(module, torch.nn.Linear) and name in original_weights and name in initial_masks and name in layer_stats:
                    print(f"Refining layer: {name}")
                    
                    W_dense = original_weights[name]
                    M_initial = initial_masks[name]
                    stats = layer_stats[name]
                    
                    dev = W_dense.device
                    M_current = M_initial.clone().to(dev) 
                    
                    # Various setup code...
                    # Directly copied from the original implementation
                    W_dense_f32 = W_dense.float()
                    sum_metric_row = stats['sum'].to(dev).float()
                    variance = stats['var'].to(dev).float()
                    scaler_row = stats['scaler_row'].to(dev).float()
                    count = stats['count']

                    dsnot_metric = W_dense_f32 * sum_metric_row.unsqueeze(0) 

                    metric_for_regrowing = dsnot_metric.clone()
                    metric_for_regrowing[M_current] = 0 
                    reconstruction_error = torch.sum(metric_for_regrowing, dim=1, keepdim=True) 
                    initialize_error_sign = torch.sign(reconstruction_error).float()

                    # Prepare candidate scoring metrics
                    grow_metric = dsnot_metric.clone()
                    grow_metric[M_current] = -float('inf')
                    
                    if pow_of_var_regrowing > 0:
                         var_pow = torch.pow(variance.unsqueeze(0), pow_of_var_regrowing)
                         grow_metric = grow_metric / (var_pow + 1e-9) 
                    
                    prune_metric_base = torch.abs(W_dense_f32) * torch.sqrt(scaler_row.unsqueeze(0) / count)
                    prune_metric_base[~M_current] = float('inf')

                    # Initialize cycle counter BEFORE the loop
                    cycle = 0
                    
                    # Iterative Loop
                    update_mask = torch.ones(W_dense.shape[0], 1, dtype=torch.bool, device=dev) 
                    
                    for cycle in range(max_cycle_time):
                        if not torch.any(update_mask):
                            print(f"  Converged after {cycle} cycles.")
                            break
                            
                        rows_to_update_indices = update_mask.squeeze().nonzero().squeeze(dim=-1)
                        if rows_to_update_indices.numel() == 0:
                            print(f"  Converged after {cycle} cycles (no rows left).")
                            break
                        if rows_to_update_indices.dim() == 0:
                            rows_to_update_indices = rows_to_update_indices.unsqueeze(0)

                        current_error_sign = torch.sign(reconstruction_error[rows_to_update_indices]).float()

                        # Rest of the implementation follows
                        grow_candidates = grow_metric[rows_to_update_indices]
                        grow_indices_col = torch.where(current_error_sign > 0, 
                                                        torch.argmax(grow_candidates, dim=1), 
                                                        torch.argmin(grow_candidates, dim=1))
                        
                        prune_candidates = prune_metric_base[rows_to_update_indices]
                        dsnot_metric_kept = dsnot_metric[rows_to_update_indices]
                        
                        sign_condition_met = torch.where(current_error_sign > 0, 
                                                          dsnot_metric_kept < 0, 
                                                          dsnot_metric_kept > 0)
                        
                        prune_candidates_filtered = torch.where(sign_condition_met, prune_candidates, float('inf'))
                        prune_indices_col = torch.argmin(prune_candidates_filtered, dim=1)
                        
                        valid_prune_selection = prune_candidates_filtered[torch.arange(rows_to_update_indices.size(0)), prune_indices_col] != float('inf')
                        
                        valid_rows = rows_to_update_indices[valid_prune_selection]
                        valid_grow_cols = grow_indices_col[valid_prune_selection]
                        valid_prune_cols = prune_indices_col[valid_prune_selection]
                        valid_error_sign = current_error_sign[valid_prune_selection]

                        if valid_rows.numel() == 0:
                             print(f"  No valid prune/grow swaps found in cycle {cycle+1}.")
                             break

                        grow_metric_selected = dsnot_metric[valid_rows, valid_grow_cols]
                        prune_metric_selected = dsnot_metric[valid_rows, valid_prune_cols]

                        error_after_swap = reconstruction_error[valid_rows] + prune_metric_selected.unsqueeze(1) - grow_metric_selected.unsqueeze(1)

                        error_magnitude_check = torch.abs(reconstruction_error[valid_rows]) > update_threshold
                        
                        if without_same_sign:
                             should_update_row = error_magnitude_check
                        else:
                             sign_check = (initialize_error_sign[valid_rows] == torch.sign(error_after_swap).float()) | (torch.sign(error_after_swap) == 0)
                             should_update_row = error_magnitude_check & sign_check 

                        # Get actual indices to update in M_current
                        final_update_rows = valid_rows[should_update_row]
                        final_grow_cols = valid_grow_cols[should_update_row]
                        final_prune_cols = valid_prune_cols[should_update_row]

                        # Update Mask and Error
                        if final_update_rows.numel() > 0:
                            M_current[final_update_rows, final_prune_cols] = False # Prune
                            M_current[final_update_rows, final_grow_cols] = True  # Grow
                            
                            grow_contribution = dsnot_metric[final_update_rows, final_grow_cols]
                            prune_contribution = dsnot_metric[final_update_rows, final_prune_cols]
                            reconstruction_error[final_update_rows] += prune_contribution.unsqueeze(1) - grow_contribution.unsqueeze(1)
                            
                            grow_metric[final_update_rows, final_grow_cols] = -float('inf')
                            prune_metric_base[final_update_rows, final_prune_cols] = float('inf')

                        # Update the overall update_mask for the next iteration
                        update_mask.fill_(False)
                        update_mask[final_update_rows] = True

                    # Important: cycle is now defined even if we never enter the loop
                    print(f"  Finished DSnoT refinement for {name} after {cycle+1} cycles.")
                    
                    # Apply final mask to original dense weights and update the module IN-PLACE
                    final_mask_float = M_current.float().to(W_dense.dtype)
                    refined_weight = W_dense * final_mask_float
                    module.weight.data.copy_(refined_weight)
                    
                    density = final_mask_float.sum().item() / final_mask_float.numel()
                    print(f"Layer:{name} => Density (DSnoT): {density:.4f}")

            print("--- DSnoT Refinement Complete ---")
            return net.state_dict()
        
        # Replace the original method with our fixed version
        ViTShardForImageClassification.refine_dsnot = fixed_refine_dsnot
        print("Successfully patched DSnoT implementation to fix the cycle variable bug.")
    
    except Exception as e:
        print(f"Warning: Failed to patch DSnoT implementation: {e}")
        print("Will use fallback error handling instead.")

def run_pruning_comparison(model_name, keep_ratio, dsnot_args, calibration_batch, device):
    """Run and compare WANDA and DSnoT pruning methods."""
    # Fix the DSnoT bug before running any pruning
    fix_dsnot_bug()
    
    # Get the original model - Use vit_wanda implementation to load original model
    original_weights_file = model_cfg_dsnot.get_model_default_weights_file(model_name)
    
    # Get model class for WANDA models
    wanda_model_class = model_cfg_wanda.get_model_dict(model_name)['shard_module']
    
    # Create config and get layers
    config = model_cfg_dsnot.get_model_config(model_name)
    layers = model_cfg_dsnot.get_model_layers(model_name)
    shard_config = ModuleShardConfig(layer_start=1, layer_end=layers, 
                                    is_first=True, is_last=True)
    
    # Load original model with WANDA implementation that handles transposition correctly
    original_model = wanda_model_class(config, shard_config, original_weights_file, prune=False)
    original_model.to(device)
    original_model.eval()
    
    # Get model class for DSnoT
    dsnot_model_class = model_cfg_dsnot.get_model_dict(model_name)['shard_module']
    
    # Apply WANDA pruning using the WANDA class
    # Instantiate model using WANDA class
    wanda_pruning_instance = wanda_model_class(config, shard_config, original_weights_file, prune=False) 
    wanda_pruning_instance.to(device)
    wanda_pruning_instance.eval()
    
    print("--- Applying WANDA Pruning ---")
    start_time = time.time()
    # Call prune_wanda from the WANDA instance
    wanda_weights = wanda_pruning_instance.prune_wanda(calibration_batch, keep_ratio=keep_ratio) 
    wanda_time = time.time() - start_time
    del wanda_pruning_instance # Free memory
    torch.cuda.empty_cache() 
    
    # Create a model instance to load WANDA weights - Load with prune=True
    wanda_pruned_model = dsnot_model_class(config, shard_config, wanda_weights, prune=True) 
    wanda_pruned_model.to(device)
    wanda_pruned_model.eval()
    
    # Apply DSnoT pruning - BUT use the WANDA pruned weights as starting point!
    print("--- Applying DSnoT Refinement to WANDA Weights ---")
    start_time = time.time()
    
    # Instantiate DSnoT with WANDA pruned weights
    dsnot_pruning_instance = dsnot_model_class(config, shard_config, wanda_weights, prune=True)
    dsnot_pruning_instance.to(device)
    dsnot_pruning_instance.eval()
    
    # Call prune_wanda_dsnot from the DSnoT instance
    print("Starting DSnoT refinement using provided hyperparameters...")
    # Directly use the ubatch that was passed to WANDA pruning
    try:
        dsnot_weights = dsnot_pruning_instance.prune_wanda_dsnot(calibration_batch, dsnot_args, keep_ratio=keep_ratio) 
        dsnot_time = time.time() - start_time
    except UnboundLocalError as e:
        # This is the specific bug we patched, but keep this as a fallback
        print(f"Warning: Error during DSnoT refinement: {e}")
        print("Falling back to using WANDA weights directly. The monkey patch may not have worked.")
        print("Specific bug: 'cycle' variable not defined - likely happens if no refinement iterations occur")
        dsnot_weights = wanda_weights  # Use WANDA weights as fallback
        dsnot_time = 0.0
    except Exception as e:
        # Handle any other unexpected errors
        print(f"Warning: Unexpected error during DSnoT refinement: {e}")
        print("Falling back to using WANDA weights directly.")
        dsnot_weights = wanda_weights  # Use WANDA weights as fallback
        dsnot_time = 0.0
    
    del dsnot_pruning_instance # Free memory
    torch.cuda.empty_cache()
    
    # Create a model instance with DSnoT weights - Load with prune=True
    dsnot_pruned_model = dsnot_model_class(config, shard_config, dsnot_weights, prune=True)
    dsnot_pruned_model.to(device)
    dsnot_pruned_model.eval()
    
    # Count parameters
    original_total, original_nonzero = count_parameters(original_model)
    wanda_total, wanda_nonzero = count_parameters(wanda_pruned_model)
    dsnot_total, dsnot_nonzero = count_parameters(dsnot_pruned_model)
    
    results = {
        'original': {
            'total_params': original_total,
            'nonzero_params': original_nonzero,
            'sparsity': 1.0 - (original_nonzero / original_total),
            'model': original_model,
            'weights': original_weights_file
        },
        'wanda': {
            'total_params': wanda_total,
            'nonzero_params': wanda_nonzero,
            'sparsity': 1.0 - (wanda_nonzero / wanda_total),
            'pruning_time': wanda_time,
            'model': wanda_pruned_model,
            'weights': wanda_weights
        },
        'dsnot': {
            'total_params': dsnot_total,
            'nonzero_params': dsnot_nonzero,
            'sparsity': 1.0 - (dsnot_nonzero / dsnot_total),
            'pruning_time': dsnot_time,
            'model': dsnot_pruned_model,
            'weights': dsnot_weights
        }
    }
    
    return results

def save_pruned_weights(model_name, weights, method):
    """Save pruned weights to file."""
    if method == 'wanda':
        output_file = model_cfg_wanda.get_model_pruned_weights_file(model_name)
    else:  # dsnot
        output_file = model_cfg_dsnot.get_model_pruned_weights_file(model_name)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save weights
    torch.save(weights, output_file)
    return output_file

def _get_default_quant(n_stages: int) -> List[int]:
    """Get default quantization settings (0 = no quantization)."""
    return [0] * n_stages

def main():
    parser = argparse.ArgumentParser(description='Compare WANDA and DSnoT pruning approaches.')
    # Model options
    parser.add_argument('--model', type=str, default='google/vit-base-patch16-224',
                      help='Model name (default: google/vit-base-patch16-224)')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_IMAGENET_DIR,
                      help=f'Path to ImageNet data (default: {DEFAULT_IMAGENET_DIR})')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                      help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})')
    
    # Batch size options
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Evaluation batch size (default: 64)')
    parser.add_argument('-tb', '--train-batch-size', type=int, default=64,
                      help='Training batch size for pruning (default: 64)')
    parser.add_argument('--calib-batches', type=int, default=1,
                      help='Number of batches to use for calibration (default: 1)')
    
    # Pruning options
    parser.add_argument('--prune', type=bool, nargs='?', const=True, default=False,
                      help='Whether to perform pruning (default: False)')
    parser.add_argument('--keep-ratio', type=float, default=0.5,
                      help='Ratio of weights to keep (default: 0.5)')
    
    # DSnoT options
    parser.add_argument('--dsnot-cycles', type=int, default=50,
                      help='Maximum number of DSnoT refinement cycles (default: 50)')
    parser.add_argument('--dsnot-threshold', type=float, default=0.1,
                      help='DSnoT update threshold (default: 0.1)')
    
    # Partitioning options
    parser.add_argument('--partition', type=str, default=None,
                      help='Comma-delimited list of start/end layer pairs, e.g.: "1,24,25,48"')
    parser.add_argument('-q', '--quant', type=str, default=None,
                      help='Comma-delimited list of quantization bits to use after each stage')
    
    # Other options
    parser.add_argument('--evaluate-only', action='store_true',
                      help='Only evaluate pre-pruned models, skip pruning')
    parser.add_argument('--save-weights', action='store_true',
                      help='Save pruned weights to disk')
    parser.add_argument('--stop-at-batch', type=int, default=None,
                      help='The number of batches to stop evaluation (default: None)')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of dataloader workers (default: 4)')
                      
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Prepare data loaders
    val_loader = prepare_imagenet_dataloaders(args.data_dir, args.batch_size, args.num_workers)
    
    # Prepare calibration batch
    calib_batch = prepare_calibration_batch(val_loader, args.calib_batches, device)
    
    # Setup quantization if specified
    quant_values = None
    if args.partition and args.quant:
        parts = [int(i) for i in args.partition.split(',')]
        num_shards = len(parts)//2
        quant_values = [int(i) for i in args.quant.split(',')] if args.quant else _get_default_quant(num_shards)
    
    # DSnoT hyperparameters
    dsnot_args = {
        'dsnot_cycles': args.dsnot_cycles,
        'dsnot_threshold': args.dsnot_threshold,
        'pow_of_var_regrowing': 1.0,
        'without_same_sign': True
    }
    
    if args.prune and not args.evaluate_only:
        # Run pruning comparison
        results = run_pruning_comparison(
            args.model, args.keep_ratio, dsnot_args, calib_batch, device)
        
        # Display initial results
        print("\n===== Pruning Results =====")
        print(f"Model: {args.model}")
        print(f"Keep ratio: {args.keep_ratio}")
        
        print("\nOriginal model:")
        print(f"  Total parameters: {results['original']['total_params']:,}")
        print(f"  Non-zero parameters: {results['original']['nonzero_params']:,}")
        print(f"  Sparsity: {results['original']['sparsity']:.4f}")
        
        print("\nWANDA pruned model:")
        print(f"  Total parameters: {results['wanda']['total_params']:,}")
        print(f"  Non-zero parameters: {results['wanda']['nonzero_params']:,}")
        print(f"  Sparsity: {results['wanda']['sparsity']:.4f}")
        print(f"  Pruning time: {results['wanda']['pruning_time']:.2f} seconds")
        
        print("\nDSnoT pruned model:")
        print(f"  Total parameters: {results['dsnot']['total_params']:,}")
        print(f"  Non-zero parameters: {results['dsnot']['nonzero_params']:,}")
        print(f"  Sparsity: {results['dsnot']['sparsity']:.4f}")
        print(f"  Pruning time: {results['dsnot']['pruning_time']:.2f} seconds")
        
        # Save weights if requested
        if args.save_weights:
            if isinstance(results['wanda']['weights'], dict):
                wanda_file = save_pruned_weights(args.model, results['wanda']['weights'], 'wanda')
                print(f"\nSaved WANDA weights to: {wanda_file}")
            
            if isinstance(results['dsnot']['weights'], dict):
                dsnot_file = save_pruned_weights(args.model, results['dsnot']['weights'], 'dsnot')
                print(f"Saved DSnoT weights to: {dsnot_file}")
        
        # Partition models if partitioning is specified
        if args.partition:
            print("\n===== Partitioning Models =====")
            # Load models with partitioning - pass correct prune flag
            original_model = load_model(args.model, results['original']['weights'], device, args.partition, prune=False)
            wanda_model = load_model(args.model, results['wanda']['weights'], device, args.partition, prune=True)
            dsnot_model = load_model(args.model, results['dsnot']['weights'], device, args.partition, prune=True)
            
            # Evaluate partitioned models
            print("\n===== Evaluating Partitioned Models =====")
            print("Evaluating original partitioned model...")
            original_acc = evaluate_partitioned_model(original_model, val_loader, device, quant_values)
            
            print("Evaluating WANDA pruned partitioned model...")
            wanda_acc = evaluate_partitioned_model(wanda_model, val_loader, device, quant_values)
            
            print("Evaluating DSnoT pruned partitioned model...")
            dsnot_acc = evaluate_partitioned_model(dsnot_model, val_loader, device, quant_values)
        else:
            # Evaluate non-partitioned models
            print("\n===== Evaluating Models =====")
            print("Evaluating original model...")
            original_acc = evaluate_model(results['original']['model'], val_loader, device)
            
            print("Evaluating WANDA pruned model...")
            wanda_acc = evaluate_model(results['wanda']['model'], val_loader, device)
            
            print("Evaluating DSnoT pruned model...")
            dsnot_acc = evaluate_model(results['dsnot']['model'], val_loader, device)
        
        # Display evaluation results
        print("\n===== Evaluation Results =====")
        print(f"Original model accuracy: {original_acc:.2f}%")
        print(f"WANDA pruned model accuracy: {wanda_acc:.2f}% (delta: {wanda_acc - original_acc:.2f}%)")
        print(f"DSnoT pruned model accuracy: {dsnot_acc:.2f}% (delta: {dsnot_acc - original_acc:.2f}%)")
        
    else:
        # Evaluate pre-pruned models only
        print("\n===== Evaluating Pre-Pruned Models =====")
        
        # Get weight file paths
        original_weights_file = model_cfg_dsnot.get_model_default_weights_file(args.model)
        wanda_weights_file = model_cfg_wanda.get_model_pruned_weights_file(args.model)
        dsnot_weights_file = model_cfg_dsnot.get_model_pruned_weights_file(args.model)
        
        if args.partition:
            # Load models with partitioning - pass correct prune flag
            print(f"Loading original partitioned model from: {original_weights_file}")
            # Use WANDA implementation for original model (handles transposition correctly)
            original_model = load_model(args.model, original_weights_file, device, args.partition, prune=False)
            
            print(f"Loading WANDA pruned partitioned model from: {wanda_weights_file}")
            wanda_model = load_model(args.model, wanda_weights_file, device, args.partition, prune=True)
            
            print(f"Loading DSnoT pruned partitioned model from: {dsnot_weights_file}")
            dsnot_model = load_model(args.model, dsnot_weights_file, device, args.partition, prune=True)
            
            # Evaluate partitioned models
            print("\n===== Evaluating Partitioned Models =====")
            
            print("Evaluating original partitioned model...")
            original_acc = evaluate_partitioned_model(original_model, val_loader, device, quant_values)
            
            print("Evaluating WANDA pruned partitioned model...")
            wanda_acc = evaluate_partitioned_model(wanda_model, val_loader, device, quant_values)
            
            print("Evaluating DSnoT pruned partitioned model...")
            dsnot_acc = evaluate_partitioned_model(dsnot_model, val_loader, device, quant_values)
        else:
            # Load models without partitioning - pass correct prune flag
            print(f"Loading original model from: {original_weights_file}")
            # Use WANDA implementation for original model (handles transposition correctly)
            original_model = load_model(args.model, original_weights_file, device, prune=False)
            
            print(f"Loading WANDA pruned model from: {wanda_weights_file}")
            wanda_model = load_model(args.model, wanda_weights_file, device, prune=True)
            
            print(f"Loading DSnoT pruned model from: {dsnot_weights_file}")
            dsnot_model = load_model(args.model, dsnot_weights_file, device, prune=True)
            
            # Count parameters (only works properly on non-partitioned models)
            original_total, original_nonzero = count_parameters(original_model)
            wanda_total, wanda_nonzero = count_parameters(wanda_model)
            dsnot_total, dsnot_nonzero = count_parameters(dsnot_model)
            
            # Display parameter counts
            print("\n===== Parameter Counts =====")
            print("Original model:")
            print(f"  Total parameters: {original_total:,}")
            print(f"  Non-zero parameters: {original_nonzero:,}")
            print(f"  Sparsity: {1.0 - (original_nonzero / original_total):.4f}")
            
            print("\nWANDA pruned model:")
            print(f"  Total parameters: {wanda_total:,}")
            print(f"  Non-zero parameters: {wanda_nonzero:,}")
            print(f"  Sparsity: {1.0 - (wanda_nonzero / wanda_total):.4f}")
            
            print("\nDSnoT pruned model:")
            print(f"  Total parameters: {dsnot_total:,}")
            print(f"  Non-zero parameters: {dsnot_nonzero:,}")
            print(f"  Sparsity: {1.0 - (dsnot_nonzero / dsnot_total):.4f}")
            
            # Evaluate models
            print("\n===== Evaluating Models =====")
            
            print("Evaluating original model...")
            original_acc = evaluate_model(original_model, val_loader, device)
            
            print("Evaluating WANDA pruned model...")
            wanda_acc = evaluate_model(wanda_model, val_loader, device)
            
            print("Evaluating DSnoT pruned model...")
            dsnot_acc = evaluate_model(dsnot_model, val_loader, device)
        
        # Display evaluation results
        print("\n===== Evaluation Results =====")
        print(f"Original model accuracy: {original_acc:.2f}%")
        print(f"WANDA pruned model accuracy: {wanda_acc:.2f}% (delta: {wanda_acc - original_acc:.2f}%)")
        print(f"DSnoT pruned model accuracy: {dsnot_acc:.2f}% (delta: {dsnot_acc - original_acc:.2f}%)")

if __name__ == "__main__":
    main() 