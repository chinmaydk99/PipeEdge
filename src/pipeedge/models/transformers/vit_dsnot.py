"""ViT Transformers with SparseGPT + DSnoT Pruning.

This module implements the SparseGPT + DSnoT pruning approach
for Vision Transformers. SparseGPT provides initial pruning based on second-order
information (Optimal Brain Surgeon), and DSnoT refines the mask iteratively
without training, based on the paper "Dynamic Sparse No Training: Training-Free
Fine-tuning for Sparse LLMs" (Zhang et al., ICLR 2024).
"""
from collections.abc import Mapping
import logging
import math
import os
from typing import Optional, Union
import numpy as np
import requests
import torch
from torch import nn
from transformers import ViTConfig
from transformers.models.vit.modeling_vit import (
    ViTEmbeddings, ViTIntermediate, ViTOutput, ViTSelfAttention, ViTSelfOutput
)
from .. import ModuleShard, ModuleShardConfig
from . import TransformerShardData
import torch.nn.functional as F
import types
import copy

logger = logging.getLogger(__name__)

_WEIGHTS_URLS = {
    'google/vit-base-patch16-224': 'https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16-224.npz',
    'google/vit-large-patch16-224': 'https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16-224.npz',
    'google/vit-huge-patch14-224-in21k': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
}

# Helper function needed for reference DSnoT pruning logic (remains unchanged)
def return_reorder_indice(input_tensor):
    # ... (keep existing implementation) ...
    positive_tensor = input_tensor.clone()
    negative_tensor = input_tensor.clone()

    positive_mask = positive_tensor > 0
    negative_mask = negative_tensor < 0

    positive_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )
    negative_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )

    positive_indices[~positive_mask] = float("inf")
    negative_indices[~negative_mask] = float("inf")

    positive_value, _ = torch.sort(positive_indices, dim=1)
    negative_value, _ = torch.sort(negative_indices, dim=1)

    positive_value = torch.flip(positive_value, dims=[1])

    negative_value[negative_value == float("inf")] = 0
    positive_value[positive_value == float("inf")] = 0

    reorder_indice = (positive_value + negative_value).to(torch.int64)

    return reorder_indice


class ViTLayerShard(ModuleShard):
    """Module shard based on `ViTLayer`."""

    def __init__(self, config: ViTConfig, shard_config: ModuleShardConfig):
        super().__init__(config, shard_config)
        self.layernorm_before = None
        self.self_attention = None
        self.self_output = None
        self.layernorm_after = None
        self.intermediate = None
        self.output = None
        self._build_shard()

    def _build_shard(self):
        # ... (keep existing implementation) ...
        if self.has_layer(0):
            self.layernorm_before = nn.LayerNorm(self.config.hidden_size,
                                                 eps=self.config.layer_norm_eps)
            self.self_attention = ViTSelfAttention(self.config)
        if self.has_layer(1):
            self.self_output = ViTSelfOutput(self.config)
        if self.has_layer(2):
            self.layernorm_after = nn.LayerNorm(self.config.hidden_size,
                                                eps=self.config.layer_norm_eps)
            self.intermediate = ViTIntermediate(self.config)
        if self.has_layer(3):
            self.output = ViTOutput(self.config)


    def forward(self, data: TransformerShardData) -> TransformerShardData:
        # ... (keep existing implementation) ...
        if self.has_layer(0):
            data_norm = self.layernorm_before(data)
            data = (self.self_attention(data_norm)[0], data)
        if self.has_layer(1):
            skip = data[1]
            data = self.self_output(data[0], skip)
            data += skip
        if self.has_layer(2):
            data_norm = self.layernorm_after(data)
            data = (self.intermediate(data_norm), data)
        if self.has_layer(3):
            data = self.output(data[0], data[1])
        return data


class ViTModelShard(ModuleShard):
    """Module shard based on `ViTModel` (no pooling layer)."""

    # Remove 'prune' argument, always load dense weights
    def __init__(self, config: ViTConfig, shard_config: ModuleShardConfig,
                 model_weights: Union[str, Mapping]):
        super().__init__(config, shard_config)
        self.embeddings = None
        self.layers = nn.ModuleList()
        self.layernorm = None
        logger.debug(">>>> Model name: %s", self.config.name_or_path)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", model_weights)
            try:
                # Attempt to load weights, handle potential file not found or corruption
                with np.load(model_weights) as weights:
                    self._build_shard(weights)
            except FileNotFoundError:
                logger.error(f"Weight file not found: {model_weights}")
                raise
            except Exception as e:
                logger.error(f"Error loading weight file {model_weights}: {e}")
                raise
        else:
            self._build_shard(model_weights)

    # Remove 'prune' argument
    def _build_shard(self, weights):
        if self.shard_config.is_first:
            logger.debug(">>>> Load embeddings layer for the first shard")
            self.embeddings = ViTEmbeddings(self.config)
            self._load_weights_first(weights) # Pass only weights

        layer_curr = self.shard_config.layer_start
        while layer_curr <= self.shard_config.layer_end:
            layer_id = math.ceil(layer_curr / 4) - 1
            sublayer_start = (layer_curr - 1) % 4
            if layer_id == math.ceil(self.shard_config.layer_end / 4) - 1:
                sublayer_end = (self.shard_config.layer_end - 1) % 4
            else:
                sublayer_end = 3
            logger.debug(">>>> Load layer %d, sublayers %d-%d",
                         layer_id, sublayer_start, sublayer_end)
            layer_config = ModuleShardConfig(layer_start=sublayer_start, layer_end=sublayer_end)
            layer = ViTLayerShard(self.config, layer_config)
            self._load_weights_layer(weights, layer_id, layer) # Pass only weights, layer_id, layer
            self.layers.append(layer)
            layer_curr += sublayer_end - sublayer_start + 1

        if self.shard_config.is_last:
            logger.debug(">>>> Load layernorm for the last shard")
            self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            self._load_weights_last(weights) # Pass only weights

    # Remove 'prune' argument and logic
    @torch.no_grad()
    def _load_weights_first(self, weights):
        # Always load dense weights
        try:
            self.embeddings.cls_token.copy_(torch.from_numpy(weights["cls"]))
            self.embeddings.position_embeddings.copy_(torch.from_numpy((weights["Transformer/posembed_input/pos_embedding"])))
            conv_weight = weights["embedding/kernel"]
            conv_weight = conv_weight.transpose([3, 2, 0, 1])
            self.embeddings.patch_embeddings.projection.weight.copy_(torch.from_numpy(conv_weight))
            self.embeddings.patch_embeddings.projection.bias.copy_(torch.from_numpy(weights["embedding/bias"]))
        except KeyError as e:
            logger.error(f"Missing key in weights during first layer load: {e}")
            raise

    # Remove 'prune' argument and logic
    @torch.no_grad()
    def _load_weights_last(self, weights):
        # Always load dense weights
        try:
            self.layernorm.weight.copy_(torch.from_numpy(weights["Transformer/encoder_norm/scale"]))
            self.layernorm.bias.copy_(torch.from_numpy(weights["Transformer/encoder_norm/bias"]))
        except KeyError as e:
            logger.error(f"Missing key in weights during last layer load: {e}")
            raise

    # Remove 'prune' argument and logic
    @torch.no_grad()
    def _load_weights_layer(self, weights, layer_id, layer):
        # Always load dense weights
        root = f"Transformer/encoderblock_{layer_id}/"
        hidden_size = self.config.hidden_size
        try:
            if layer.has_layer(0):
                layer.layernorm_before.weight.copy_(torch.from_numpy(weights[root + "LayerNorm_0/scale"]))
                layer.layernorm_before.bias.copy_(torch.from_numpy(weights[root + "LayerNorm_0/bias"]))
                layer.self_attention.query.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/query/kernel"]).view(hidden_size, hidden_size).t())
                layer.self_attention.key.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/key/kernel"]).view(hidden_size, hidden_size).t())
                layer.self_attention.value.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/value/kernel"]).view(hidden_size, hidden_size).t())
                layer.self_attention.query.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/query/bias"]).view(-1))
                layer.self_attention.key.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/key/bias"]).view(-1))
                layer.self_attention.value.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/value/bias"]).view(-1))
            if layer.has_layer(1):
                layer.self_output.dense.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/out/kernel"]).view(hidden_size, hidden_size).t())
                layer.self_output.dense.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/out/bias"]).view(-1))
            if layer.has_layer(2):
                layer.layernorm_after.weight.copy_(torch.from_numpy(weights[root + "LayerNorm_2/scale"]))
                layer.layernorm_after.bias.copy_(torch.from_numpy(weights[root + "LayerNorm_2/bias"]))
                layer.intermediate.dense.weight.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_0/kernel"]).t())
                layer.intermediate.dense.bias.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_0/bias"]).t())
            if layer.has_layer(3):
                layer.output.dense.weight.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_1/kernel"]).t())
                layer.output.dense.bias.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_1/bias"]).t())
        except KeyError as e:
            logger.error(f"Missing key in weights during layer {layer_id} load: {e}")
            raise

    def forward(self, data: TransformerShardData) -> TransformerShardData:
        # ... (keep existing implementation) ...
        if self.shard_config.is_first:
            data = self.embeddings(data)
        for layer in self.layers:
            data = layer(data)
        if self.shard_config.is_last:
            data = self.layernorm(data)
        return data


    @staticmethod
    def save_weights(model_name: str, model_file: str, url: Optional[str]=None,
                     timeout_sec: Optional[float]=None) -> None:
        # ... (keep existing implementation) ...
        if url is None:
            url = _WEIGHTS_URLS[model_name]
        logger.info('Downloading model: %s: %s', model_name, url)
        try:
            req = requests.get(url, stream=True, timeout=timeout_sec)
            req.raise_for_status()
            with open(model_file, 'wb') as file:
                for chunk in req.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        file.flush()
                        os.fsync(file.fileno())
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading weights for {model_name} from {url}: {e}")
            raise
        except IOError as e:
            logger.error(f"Error writing weights file {model_file}: {e}")
            raise


class ViTShardForImageClassification(ModuleShard):
    """Module shard based on `ViTForImageClassification`."""
    # Remove 'prune' argument
    def __init__(self, config: ViTConfig, shard_config: ModuleShardConfig,
                 model_weights: Union[str, Mapping]):
        super().__init__(config, shard_config)
        self.vit = None
        self.classifier = None
        self.dtype = torch.float32 # Store original dtype, will be set in _collect_stats

        logger.debug(">>>> Model name: %s", self.config.name_or_path)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", model_weights)
            try:
                with np.load(model_weights) as weights:
                    self._build_shard(weights)
            except FileNotFoundError:
                logger.error(f"Weight file not found: {model_weights}")
                raise
            except Exception as e:
                logger.error(f"Error loading weight file {model_weights}: {e}")
                raise
        else:
            self._build_shard(model_weights)

    # Remove 'prune' argument
    def _build_shard(self, weights):
        ## all shards use the inner ViT model - pass weights directly
        self.vit = ViTModelShard(self.config, self.shard_config, weights)

        if self.shard_config.is_last:
            logger.debug(">>>> Load classifier for the last shard")
            # Handle potential missing num_labels in config
            num_labels = getattr(self.config, 'num_labels', 0)
            if num_labels > 0:
                self.classifier = nn.Linear(self.config.hidden_size, num_labels)
                self._load_weights_last(weights) # Pass only weights
            else:
                logger.warning("num_labels not found or is 0, creating nn.Identity classifier.")
                self.classifier = nn.Identity()


    # Remove 'prune' argument and logic
    @torch.no_grad()
    def _load_weights_last(self, weights):
        # Always load dense weights
        # Check if classifier weights exist before loading
        if 'head/kernel' in weights and 'head/bias' in weights:
             try:
                 self.classifier.weight.copy_(torch.from_numpy(np.transpose(weights["head/kernel"])))
                 self.classifier.bias.copy_(torch.from_numpy(weights["head/bias"]))
             except KeyError as e:
                  logger.error(f"Missing key in weights during last layer (classifier) load: {e}")
                  # Depending on desired behavior, either raise or allow model without loaded classifier weights
                  # raise
             except AttributeError:
                  logger.error("Classifier layer not properly initialized before loading weights.")
                  # This might happen if num_labels was 0
                  # raise
        else:
             logger.warning("Classifier weights ('head/kernel', 'head/bias') not found in weights file.")

    def forward(self, data: TransformerShardData) -> TransformerShardData:
        # ... (keep existing implementation) ...
        data = self.vit(data)
        if self.shard_config.is_last:
            # Ensure data has the expected shape for the classifier
            if isinstance(data, torch.Tensor) and data.dim() > 2:
                 # Take the CLS token representation [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
                 data = data[:, 0, :]
            elif isinstance(data, torch.Tensor) and data.dim() < 2:
                 logger.warning(f"Unexpected data dimension entering classifier: {data.dim()}")
                 # Attempt to handle or raise error
            elif not isinstance(data, torch.Tensor):
                 logger.error(f"Data entering classifier is not a Tensor: {type(data)}")
                 return None # Or raise error

            # Apply classifier only if it exists and is not Identity
            if self.classifier is not None and not isinstance(self.classifier, nn.Identity):
                 try:
                      data = self.classifier(data)
                 except Exception as e:
                      logger.error(f"Error during classifier forward pass: {e}", exc_info=True)
                      return None # Or raise error
            elif isinstance(self.classifier, nn.Identity):
                 logger.debug("Skipping Identity classifier.")
            else: # self.classifier is None
                 logger.warning("Classifier is None in the last shard, cannot apply.")
                 return None # Or raise error
        return data

    @staticmethod
    def save_weights(model_name: str, model_file: str, url: Optional[str]=None,
                     timeout_sec: Optional[float]=None) -> None:
        """Save the model weights file."""
        ViTModelShard.save_weights(model_name, model_file, url=url, timeout_sec=timeout_sec)


    # --- Start of SparseGPT + DSnoT Implementation ---

    def _collect_layer_stats(self, ubatch):
        """
        Helper to collect activations and other stats needed for pruning.
        Calculates 'mean_act', 'var_act', 'scaler_row', 'H' for each linear layer.
        """
        hooks = []
        # Determine device from model parameters
        try:
             device = next(self.parameters()).device
        except StopIteration:
             logger.warning("Model seems to have no parameters. Assuming CPU for stats collection.")
             device = torch.device('cpu')

        net_copy = copy.deepcopy(self).to(device)
        net_copy.eval()
        self.dtype = next(net_copy.parameters()).dtype if list(net_copy.parameters()) else torch.float32 # Store original dtype

        # Use a dictionary to store stats per layer, initialized on first hook encounter
        layer_stats_accum = {}

        def hook_fn(name, module): # Pass module to get in_features
            def _hook(mod, input_tensor, output):
                # Ensure input_tensor is the actual tensor, not a tuple
                if isinstance(input_tensor, tuple):
                    if len(input_tensor) > 0 and isinstance(input_tensor[0], torch.Tensor):
                        input_tensor = input_tensor[0]
                    else:
                         # Cannot determine input tensor from tuple
                         logger.warning(f"Could not extract tensor from input tuple for layer {name}. Skipping stats.")
                         return
                elif not isinstance(input_tensor, torch.Tensor):
                    logger.warning(f"Input to layer {name} is not a tensor ({type(input_tensor)}). Skipping stats.")
                    return

                # Initialize stats for the layer if first time seen
                if name not in layer_stats_accum:
                    # Handle potential missing in_features attribute
                    in_features = getattr(module, 'in_features', None)
                    if in_features is None:
                         logger.warning(f"Layer {name} missing 'in_features'. Cannot initialize Hessian. Skipping H calc.")
                         layer_stats_accum[name] = {
                             'inputs': [],
                             'H': None # Mark Hessian as unavailable
                         }
                    else:
                         layer_stats_accum[name] = {
                             'inputs': [],
                             'H': torch.zeros((in_features, in_features), dtype=torch.float32, device='cpu') # Accumulate H on CPU
                         }

                # --- Store Inputs (CPU) ---
                layer_stats_accum[name]['inputs'].append(input_tensor.detach().cpu())

                # --- Accumulate Hessian (GPU chunk -> CPU) --- 
                # Only if H was initialized successfully
                if layer_stats_accum[name]['H'] is not None:
                     inp_float = input_tensor.float().to(device) # Move input to GPU for matmul
                     if inp_float.dim() >= 3:
                         # Correctly handle sequence dimension: [B, SeqLen, Features] -> [B*SeqLen, Features]
                         act_reshaped = inp_float.reshape(-1, inp_float.size(-1))
                     elif inp_float.dim() == 2: # [B, Features]
                         act_reshaped = inp_float
                     else:
                         logger.warning(f"Unexpected input tensor dim {inp_float.dim()} for Hessian calc in layer {name}. Skipping H update.")
                         return

                     # Add contribution to H on CPU
                     with torch.no_grad():
                          hessian_contribution = (act_reshaped.t() @ act_reshaped).cpu()
                          layer_stats_accum[name]['H'] += hessian_contribution

            return _hook

        # Register hooks for Linear layers only
        for name, module in net_copy.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name, module)))

        logger.info("Collecting stats using calibration data...")
        total_samples_processed_forward = 0
        with torch.no_grad():
            ubatch_device = ubatch.to(device)
            # Process in reasonable chunks to avoid OOM during forward pass
            process_batch_size = min(len(ubatch_device), 32) # Adjust based on typical ViT batch sizes

            for i in range(0, len(ubatch_device), process_batch_size):
                batch_input = ubatch_device[i : i + process_batch_size]
                total_samples_processed_forward += len(batch_input)
                try:
                    # Forward pass - handle different input shapes
                    if len(batch_input.shape) == 4: # Image input
                        _ = net_copy(batch_input)
                    elif len(batch_input.shape) == 3 and hasattr(net_copy, 'vit'): # Embedded input
                         # Need to manually forward through ViTModelShard components
                         if hasattr(net_copy.vit, 'embeddings'):
                              embedded_tensor = batch_input # Assuming input is already embedded?
                         else:
                              # This case shouldn't happen if called on ViTShardForImageClassification
                              logger.warning("Input is 3D but vit.embeddings not found?")
                              embedded_tensor = batch_input # Proceed cautiously

                         for layer in net_copy.vit.layers:
                             embedded_tensor = layer(embedded_tensor)
                         if net_copy.vit.layernorm is not None:
                             embedded_tensor = net_copy.vit.layernorm(embedded_tensor)
                         if hasattr(net_copy, 'classifier') and net_copy.classifier is not None:
                              cls_token_output = embedded_tensor[:, 0, :] # Use CLS token
                              _ = net_copy.classifier(cls_token_output)
                    else: # Fallback/Unknown
                         _ = net_copy(batch_input)
                except Exception as e:
                    logger.error(f"Error during stat collection forward pass (batch {i // process_batch_size}): {e}", exc_info=True)
                    # Option: stop, or continue and hope enough data was collected
                    # raise e

        for hook in hooks:
            hook.remove()
        del net_copy
        if device.type == 'cuda': torch.cuda.empty_cache()

        # --- Post-process collected stats ---
        logger.info("Post-processing stats...")
        processed_stats = {}
        total_tokens = 0

        # Estimate total tokens more robustly from collected inputs
        for name, accum_data in layer_stats_accum.items():
            if accum_data['inputs']:
                 first_input_batch_cpu = accum_data['inputs'][0]
                 if first_input_batch_cpu.dim() >= 3:
                     tokens_per_sample = first_input_batch_cpu.size(1) # Sequence length
                     features_dim = first_input_batch_cpu.size(2)
                     total_tokens = total_samples_processed_forward * tokens_per_sample
                 elif first_input_batch_cpu.dim() == 2:
                     total_tokens = total_samples_processed_forward # Batch size is token count
                     features_dim = first_input_batch_cpu.size(1)
                 else:
                     continue # Skip layers with unexpected input dims
                 logger.info(f"Estimated total tokens = {total_tokens} (from layer {name}, {total_samples_processed_forward} samples)")
                 break # Estimate from first valid layer

        if total_tokens == 0:
             logger.error("Could not estimate total tokens. Aborting stats calculation.")
             return {}

        # --- Calculate Mean, Var, Scaler_Row, Normalize H --- 
        for name, accum_data in layer_stats_accum.items():
            if not accum_data['inputs']:
                 logger.warning(f"No inputs collected for layer {name}. Skipping stats.")
                 continue

            try:
                # Concatenate inputs on CPU
                all_inputs_cpu = torch.cat(accum_data['inputs'], dim=0)
                if all_inputs_cpu.dim() >= 3:
                    act_reshaped_cpu = all_inputs_cpu.reshape(-1, all_inputs_cpu.size(-1)).to(torch.float32)
                elif all_inputs_cpu.dim() == 2:
                    act_reshaped_cpu = all_inputs_cpu.to(torch.float32)
                else:
                    logger.warning(f"Skipping stats for layer {name} due to unexpected input dim {all_inputs_cpu.dim()}")
                    continue

                current_tokens = act_reshaped_cpu.size(0)
                if abs(current_tokens - total_tokens) > total_tokens * 0.1: # Allow 10% deviation
                     logger.warning(f"Token count mismatch for layer {name}. Expected ~{total_tokens}, got {current_tokens}. Using actual count {current_tokens} for normalization.")
                     normalization_factor = max(current_tokens, 1)
                else:
                     normalization_factor = max(total_tokens, 1)

                # Calculate Mean Activation (E[A])
                mean_act = torch.mean(act_reshaped_cpu, dim=0)

                # Calculate Variance (Var[A])
                var_act = torch.var(act_reshaped_cpu, dim=0, unbiased=False) + 1e-8 # Add epsilon

                # Calculate Scaler Row (Mean Squared Norm) - safer on CPU
                # Use float64 for sum to avoid overflow with large norms/counts
                sum_sq_norms_d64 = torch.tensor(0.0, dtype=torch.float64)
                chunk_size = 1024
                for i in range(0, current_tokens, chunk_size):
                    chunk = act_reshaped_cpu[i:i+chunk_size]
                    # Calculate norm per element in chunk, square, then sum
                    norms_sq = torch.norm(chunk.to(torch.float64), p=2, dim=1)**2
                    sum_sq_norms_d64 += torch.sum(norms_sq)
                # Average and broadcast to correct shape, convert back to float32
                scaler_row_scalar = (sum_sq_norms_d64 / normalization_factor).to(torch.float32)
                scaler_row = scaler_row_scalar * torch.ones_like(mean_act)

                # Normalize Hessian H (if available)
                H = accum_data['H']
                if H is not None:
                     H = H / normalization_factor

                processed_stats[name] = {
                     'mean_act': mean_act.cpu(),
                     'var_act': var_act.cpu(),
                     'scaler_row': scaler_row.cpu(),
                     'H': H.cpu() if H is not None else None,
                     'input_shape': all_inputs_cpu.shape
                }
                del all_inputs_cpu, act_reshaped_cpu, H, mean_act, var_act, scaler_row, accum_data # Free memory

            except Exception as e:
                logger.error(f"Error processing stats for layer {name}: {e}", exc_info=True)
                continue # Skip this layer if processing fails

        print("Stats collection finished.")
        return processed_stats


    def prune_sparsegpt_dsnot(self, ubatch, keep_ratio=0.5,
                               max_cycles=50, error_threshold=0.1,
                               percdamp=0.01, # blocksize=128, # Blocksize obsolete
                               pow_of_var_regrowing=1.0,
                               skip_first_last=True):
        """
        Prunes the model using SparseGPT initialization followed by DSnoT refinement.
        Follows logic from reference implementation (lib/prune.py).

        Args:
            ubatch: Calibration data.
            keep_ratio: Target sparsity (fraction of weights to keep).
            max_cycles: Max iterations for DSnoT refinement per row/neuron.
            error_threshold: Convergence threshold for DSnoT error reduction.
            percdamp: Damping factor for SparseGPT Hessian inversion.
            pow_of_var_regrowing: Power for variance scaling in DSnoT growing score.
            skip_first_last: Whether to skip pruning the first and last linear layers.

        Returns:
            Dictionary of pruned weights.
        """
        print(f"Starting SparseGPT+DSnoT pruning. Target keep_ratio: {keep_ratio}")
        device = next(self.parameters()).device

        # --- Step 0: Collect Stats ---
        layer_stats = self._collect_layer_stats(ubatch)
        if not layer_stats:
            print("Error: Failed to collect layer statistics. Aborting pruning.")
            return {k: v.cpu() for k, v in self.state_dict().items()}

        # --- Step 1: Initial Pruning with SparseGPT ---
        print("Applying Initial SparseGPT Pruning...")
        sparse_model = copy.deepcopy(self) # Work on a copy
        initial_masks = {} # Store KEEP masks (True=keep)
        linear_layers = {name: mod for name, mod in sparse_model.named_modules() if isinstance(mod, nn.Linear)}
        layer_names = list(linear_layers.keys())

        for i, (name, module) in enumerate(linear_layers.items()):
            is_first_or_last = (i == 0 or i == len(layer_names) - 1)
            if skip_first_last and is_first_or_last:
                 print(f"  Skipping SparseGPT for layer: {name} (First/Last)")
                 initial_masks[name] = torch.ones_like(module.weight.data, dtype=torch.bool)
                 continue

            if name not in layer_stats or layer_stats[name]['H'] is None:
                print(f"  Skipping SparseGPT for layer: {name} (No stats or Hessian found)")
                initial_masks[name] = torch.ones_like(module.weight.data, dtype=torch.bool)
                continue

            print(f"  SparseGPT Pruning layer: {name}")
            W = module.weight.data.clone().float()
            H = layer_stats[name]['H'].to(device, dtype=torch.float32)
            rows, cols = W.shape

            mask = torch.zeros_like(W, dtype=torch.bool) # Prune mask (True=prune)

            try:
                # Damp the Hessian
                damp = percdamp * torch.mean(torch.diag(H))
                diag_indices = torch.arange(cols, device=device)
                H = H.contiguous()
                H[diag_indices, diag_indices] += damp

                # Invert Hessian
                try:
                    H_chol = torch.linalg.cholesky(H)
                    Hinv = torch.cholesky_inverse(H_chol)
                    Hinv = (Hinv + Hinv.T) / 2.0
                except Exception as e_chol:
                    print(f"    Warning: Cholesky failed for {name}: {e_chol}. Using pseudo-inverse.")
                    Hinv = torch.linalg.pinv(H.cpu()).to(device)

                # Calculate OBS scores
                diag_hinv = torch.diag(Hinv)
                obs_scores = (W.to(device)**2) / (torch.abs(diag_hinv.reshape(1, -1)) + 1e-12)

                # Determine threshold
                num_elements_to_keep = int(W.numel() * keep_ratio)
                num_elements_to_prune = W.numel() - num_elements_to_keep

                if num_elements_to_prune > 0:
                    obs_scores_flat = obs_scores.flatten()
                    finite_scores = obs_scores_flat[torch.isfinite(obs_scores_flat)]
                    if len(finite_scores) == 0:
                         print(f"    Warning: All OBS scores non-finite in {name}. Skipping pruning.")
                         threshold = float('inf')
                    elif len(finite_scores) < num_elements_to_prune:
                         print(f"    Warning: Not enough finite OBS scores ({len(finite_scores)}) to prune {num_elements_to_prune} in {name}. Pruning available finite scores.")
                         threshold = torch.kthvalue(finite_scores, len(finite_scores)).values # Prune all finite
                    else:
                         threshold = torch.kthvalue(finite_scores, num_elements_to_prune).values

                    # Apply threshold, keep non-finite scores
                    mask = (~torch.isfinite(obs_scores)) | (obs_scores <= threshold)
                    mask = mask.cpu()
                else:
                    mask = torch.zeros_like(W, dtype=torch.bool)

            except Exception as e_sparsegpt:
                 print(f"  Error during SparseGPT calculation for layer {name}: {e_sparsegpt}. Skipping pruning.")
                 mask = torch.zeros_like(W, dtype=torch.bool) # Don't prune if error occurs

            initial_masks[name] = ~mask # Store KEEP mask
            module.weight.data[mask] = 0.0 # Apply initial mask

            del H, Hinv, obs_scores # Free GPU memory
            if 'diag_hinv' in locals(): del diag_hinv
            if device.type == 'cuda': torch.cuda.empty_cache()

        print("Initial SparseGPT pruning finished.")

        # --- Step 2: DSnoT Refinement ---
        print("Applying DSnoT Refinement...")
        dense_model = copy.deepcopy(self) # Keep original dense weights

        for i, (name, module) in enumerate(linear_layers.items()):
            is_first_or_last = (i == 0 or i == len(layer_names) - 1)

            if name not in layer_stats or name not in initial_masks:
                 print(f"  Skipping DSnoT for layer: {name} (Missing stats/mask)")
                 continue

            if skip_first_last and is_first_or_last:
                 print(f"  Skipping DSnoT for layer: {name} (First/Last)")
                 dense_module = dict(dense_model.named_modules())[name]
                 module.weight.data = dense_module.weight.data.clone().to(self.dtype)
                 continue

            print(f"  DSnoT Refining layer: {name}")

            try:
                dense_module = dict(dense_model.named_modules())[name]
                W_dense = dense_module.weight.data.float().to(device)
                current_mask = initial_masks[name].to(device)

                # Get stats, ensure they are tensors
                mean_act = layer_stats[name]['mean_act'].to(device)
                var_act = layer_stats[name]['var_act'].to(device)
                scaler_row_stat = layer_stats[name]['scaler_row'] # This might be scalar or tensor
                # Ensure scaler_row is a tensor with correct shape
                if not isinstance(scaler_row_stat, torch.Tensor):
                     scaler_row_stat = torch.tensor(scaler_row_stat, device=device)
                scaler_row = scaler_row_stat.reshape(1, -1).to(device) # Shape [1, C]
                if scaler_row.shape[1] != W_dense.shape[1]: # Check if broadcast from scalar worked
                    if scaler_row.numel() == 1:
                        scaler_row = scaler_row.expand(1, W_dense.shape[1])
                    else:
                        raise ValueError(f"Scaler row shape mismatch in {name}: {scaler_row.shape} vs weight cols {W_dense.shape[1]}")


                # Precompute Metrics
                dsnot_metric = W_dense * mean_act.unsqueeze(0)
                metric_for_growing = dsnot_metric / (var_act.unsqueeze(0)**pow_of_var_regrowing + 1e-12)
                metric_for_pruning = torch.abs(W_dense) * torch.sqrt(scaler_row + 1e-12)

                # DSnoT Iterative Loop
                rows, cols = W_dense.shape
                initial_error_proxy = torch.sum(dsnot_metric * (~current_mask), dim=1, keepdim=True)
                reconstruction_error = initial_error_proxy.clone()
                initialize_error_sign = torch.sign(reconstruction_error)

                # Handle potential NaN/inf in metrics - replace with safe values
                metric_for_growing.nan_to_num_(nan=0.0, posinf=1e10, neginf=-1e10)
                metric_for_pruning.nan_to_num_(nan=float('inf'), posinf=float('inf'))
                reconstruction_error.nan_to_num_(nan=0.0)
                initialize_error_sign = torch.sign(reconstruction_error) # Recalculate after nan_to_num


                # Prepare sorted indices
                grow_scores_sorted, grow_indices_sorted = torch.sort(metric_for_growing, dim=1, stable=True)
                prune_scores_masked = metric_for_pruning.clone()
                prune_scores_masked[~current_mask] = float('inf')
                prune_scores_sorted, prune_indices_sorted = torch.sort(prune_scores_masked, dim=1, stable=True)

                # Pointers
                grow_idx_pointers = torch.zeros((rows, 2), device=device, dtype=torch.long)
                grow_idx_pointers[:, 1] = cols - 1
                grow_direction = torch.tensor([-1, 1], device=device, dtype=torch.long)
                prune_idx_pointer = torch.zeros(rows, device=device, dtype=torch.long)

                update_mask = torch.ones((rows, 1), device=device, dtype=torch.bool)

                # ---- DSnoT Cycle Loop ----
                for cycle in range(max_cycles):
                    if not update_mask.any():
                        # print(f"    Converged early at cycle {cycle}") # Optional log
                        break
                    active_rows_mask = update_mask.squeeze()
                    active_rows_indices = torch.where(active_rows_mask)[0]
                    if len(active_rows_indices) == 0: break

                    current_error_sign = torch.sign(reconstruction_error[active_rows_mask])
                    grow_pointer_idx = (current_error_sign > 0).long()
                    current_grow_pointers = grow_idx_pointers[active_rows_mask, grow_pointer_idx]
                    grow_idx = torch.full_like(active_rows_indices, -1, dtype=torch.long)

                    # Find valid grow candidates
                    for k, row_idx in enumerate(active_rows_indices):
                        ptr = current_grow_pointers[k]
                        direction = grow_direction[grow_pointer_idx[k]]
                        found = False
                        for search_offset in range(cols):
                            candidate_ptr = ptr + direction * search_offset
                            if 0 <= candidate_ptr < cols:
                                candidate_idx = grow_indices_sorted[row_idx, candidate_ptr]
                                # Fix boolean ambiguity: use item() to convert tensor to scalar
                                if not current_mask[row_idx, candidate_idx].item():
                                    grow_idx[k] = candidate_idx
                                    grow_idx_pointers[row_idx, grow_pointer_idx[k]] = candidate_ptr + direction
                                    found = True
                                    break
                            else: break
                        if not found: update_mask[row_idx] = False

                    # Find valid prune candidates
                    current_prune_pointers = prune_idx_pointer[active_rows_mask]
                    prune_idx = torch.full_like(active_rows_indices, -1, dtype=torch.long)
                    for k, row_idx in enumerate(active_rows_indices):
                        if grow_idx[k] == -1: continue # Skip if no grow cand found
                        ptr = current_prune_pointers[k]
                        found = False
                        for search_offset in range(cols):
                            candidate_ptr = ptr + search_offset
                            if candidate_ptr < cols:
                                candidate_idx = prune_indices_sorted[row_idx, candidate_ptr]
                                # Fix boolean ambiguity: use item() to convert tensor to scalar
                                if current_mask[row_idx, candidate_idx].item() and prune_scores_masked[row_idx, candidate_idx] != float('inf'):
                                    prune_idx[k] = candidate_idx
                                    prune_idx_pointer[row_idx] = candidate_ptr + 1
                                    found = True
                                    break
                            else: break
                        if not found: update_mask[row_idx] = False

                    # Filter for Valid Swaps
                    valid_swap_mask = (grow_idx != -1) & (prune_idx != -1) & (grow_idx != prune_idx)
                    if not valid_swap_mask.any(): break
                    active_rows_final = active_rows_indices[valid_swap_mask]
                    grow_idx_final = grow_idx[valid_swap_mask]
                    prune_idx_final = prune_idx[valid_swap_mask]

                    # Get metrics for swap
                    # Need to handle potential OOB if indices became invalid
                    grow_metric = dsnot_metric[active_rows_final, grow_idx_final]
                    prune_metric = dsnot_metric[active_rows_final, prune_idx_final]

                    # Convergence Check
                    error_after_swap = reconstruction_error[active_rows_final] + prune_metric.unsqueeze(1) - grow_metric.unsqueeze(1)
                    error_after_swap.nan_to_num_(nan=0.0) # Handle potential NaNs from metrics
                    sign_check = (initialize_error_sign[active_rows_final] == torch.sign(error_after_swap))
                    threshold_check = (torch.abs(reconstruction_error[active_rows_final]) > error_threshold)
                    rows_to_update_mask = sign_check & threshold_check
                    if not rows_to_update_mask.any(): break

                    # Perform Swap and Update Error
                    active_update_rows = active_rows_final[rows_to_update_mask]
                    active_grow_idx = grow_idx_final[rows_to_update_mask]
                    active_prune_idx = prune_idx_final[rows_to_update_mask]
                    current_mask[active_update_rows, active_grow_idx] = True
                    current_mask[active_update_rows, active_prune_idx] = False
                    update_error_delta = (prune_metric[rows_to_update_mask] - grow_metric[rows_to_update_mask]).unsqueeze(1)
                    reconstruction_error[active_update_rows] += update_error_delta

                    # Deactivate rows that didn't update
                    temp_update_mask = torch.zeros_like(update_mask, dtype=torch.bool)
                    temp_update_mask[active_update_rows] = True
                    update_mask = update_mask & temp_update_mask

                # ---- End of DSnoT Cycle Loop ----

                # Calculate and print final density for this layer AFTER DSnoT
                final_mask_cpu = current_mask.cpu()
                # Density = proportion of non-zero elements (where mask is True)
                final_density = final_mask_cpu.float().sum() / final_mask_cpu.numel()
                print(f"Layer:{name} => Final Density after DSnoT: {final_density:.4f}") # This will be captured

                # Apply final refined mask
                module.weight.data = dense_module.weight.data.cpu() * final_mask_cpu
                module.weight.data = module.weight.data.to(self.dtype)

            except Exception as e_dsnot:
                 print(f"  Error during DSnoT refinement for layer {name}: {e_dsnot}. Applying initial SparseGPT mask only.")
                 # Keep the initial mask applied in sparse_model
                 initial_mask_cpu = initial_masks[name].cpu()
                 module.weight.data = dense_module.weight.data.cpu() * initial_mask_cpu
                 module.weight.data = module.weight.data.to(self.dtype)

            finally:
                 # Cleanup tensors for the current layer regardless of success/error
                 del W_dense, current_mask, mean_act, var_act, scaler_row, dsnot_metric
                 if 'metric_for_growing' in locals(): del metric_for_growing
                 if 'metric_for_pruning' in locals(): del metric_for_pruning
                 if 'reconstruction_error' in locals(): del reconstruction_error
                 if 'grow_scores_sorted' in locals(): del grow_scores_sorted, grow_indices_sorted
                 if 'prune_scores_sorted' in locals(): del prune_scores_sorted, prune_indices_sorted
                 if device.type == 'cuda': torch.cuda.empty_cache()

        print("DSnoT refinement finished.")

        # Return the state dict of the refined sparse model
        final_weights = {}
        for key, val in sparse_model.state_dict().items():
            final_weights[key] = val.cpu()

        del sparse_model, dense_model, layer_stats # Cleanup
        if device.type == 'cuda': torch.cuda.empty_cache()

        return final_weights


    def _quick_eval(self, test_batch):
        """ Quick evaluation helper """
        if test_batch is None:
            logger.warning("_quick_eval received None for test_batch.")
            return 0.0
        inputs, labels = test_batch
        # Ensure labels are on the correct device if inputs are moved
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

        inputs = inputs.to(device)
        labels = labels.to(device)

        original_mode = self.training
        self.eval()
        accuracy = 0.0
        try:
            with torch.no_grad():
                outputs = self(inputs)
                if outputs is None:
                    logger.error("_quick_eval: Forward pass returned None.")
                    return 0.0

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                if not isinstance(logits, torch.Tensor) or logits.dim() < 2:
                    logger.error(f"_quick_eval: Unexpected output type/dim {type(logits)} / {logits.dim() if isinstance(logits, torch.Tensor) else 'N/A'}")
                    return 0.0

                _, predicted = logits.max(1)
                correct = predicted.eq(labels).sum().item()
                accuracy = correct / labels.size(0) if labels.size(0) > 0 else 0.0
        except Exception as e:
             logger.error(f"Error during _quick_eval forward pass: {e}", exc_info=True)
             accuracy = 0.0 # Return 0 accuracy if evaluation fails
        finally:
            self.train(original_mode)

        return accuracy

    # --- End of SparseGPT + DSnoT Implementation ---
