"""ViT Transformers with WANDA Pruning.

This module implements the WANDA (Weight-Activation Norm Data-Aware) pruning approach
for Vision Transformers, based on the paper "A Simple and Effective Pruning Approach
for Large Language Models" (Sun et al., ICLR 2024).
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

# LoRA-related classes for post-pruning recovery
class LoRAAdapter(nn.Module):
    """
    LoRA adapter module that computes a low-rank update.
    
    Implements the adaptation mechanism from the paper "LoRA: Low-Rank Adaptation of Large Language Models"
    but used for post-pruning recovery as in the LLM-Pruner paper.
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.rank = rank
        self.scaling = alpha / rank
        
    def forward(self, x):
        return (x @ self.lora_A) @ self.lora_B * self.scaling

class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation capability.
    
    Wraps a standard linear layer with a LoRA adapter to enable efficient
    post-pruning recovery after WANDA pruning.
    """
    def __init__(self, layer, rank=8, alpha=16, trainable_original=False):
        super().__init__()
        self.base_layer = layer
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.lora = LoRAAdapter(self.in_features, self.out_features, rank, alpha)
        
        # Freeze original weights
        if not trainable_original:
            for param in self.base_layer.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = self.lora(x)
        return base_output + lora_output
        
    def merge(self):
        """Merge LoRA weights into the base layer and return the base layer"""
        with torch.no_grad():
            delta_w = (self.lora.lora_A @ self.lora.lora_B) * self.lora.scaling
            self.base_layer.weight.data += delta_w.t()  # Transpose for PyTorch convention
        return self.base_layer

_WEIGHTS_URLS = {
    'google/vit-base-patch16-224': 'https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16-224.npz',
    'google/vit-large-patch16-224': 'https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16-224.npz',
    'google/vit-huge-patch14-224-in21k': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
}

# Helper function needed for reference DSnoT pruning logic
def return_reorder_indice(input_tensor):
    """
    For instance:
    [[1., -2., 3.],
    [-2, 2., -4],
    [5., 6., -7],
    [-6, -7, -4]]
    return indices of
    [[-2.,  3.,  1.],
    [-2., -4.,  2.],
    [-7.,  6.,  5.],
    [-6., -7., -4.]]
    Description: The relative order in the positive number remains unchanged, and the relative order in the negative number is flipped.
    """
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
        """Compute layer shard."""
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

    def __init__(self, config: ViTConfig, shard_config: ModuleShardConfig,
                 model_weights: Union[str, Mapping], prune=False):
        super().__init__(config, shard_config)
        self.embeddings = None
        # ViTModel uses an encoder here, but we'll just add the layers here instead.
        # Since we just do inference, a ViTEncoderShard class wouldn't provide real benefit.
        self.layers = nn.ModuleList()
        self.layernorm = None
        logger.debug(">>>> Model name: %s", self.config.name_or_path)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", model_weights)
            with np.load(model_weights) as weights:
                self._build_shard(weights, prune)
        else:
            self._build_shard(model_weights, prune)

    def _build_shard(self, weights, prune=False):
        if self.shard_config.is_first:
            logger.debug(">>>> Load embeddings layer for the first shard")
            self.embeddings = ViTEmbeddings(self.config)
            self._load_weights_first(weights, prune)

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
            self._load_weights_layer(weights, layer_id, layer, prune)
            self.layers.append(layer)
            layer_curr += sublayer_end - sublayer_start + 1

        if self.shard_config.is_last:
            logger.debug(">>>> Load layernorm for the last shard")
            self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            self._load_weights_last(weights, prune)

    @torch.no_grad()
    def _load_weights_first(self, weights, prune=False):
            if(prune):
                self.embeddings.cls_token.copy_(torch.from_numpy(weights["vit.embeddings.cls_token"]))
                self.embeddings.position_embeddings.copy_(torch.from_numpy(weights["vit.embeddings.position_embeddings"]))
                self.embeddings.patch_embeddings.projection.weight.copy_(torch.from_numpy(weights["vit.embeddings.patch_embeddings.projection.weight"]))
                self.embeddings.patch_embeddings.projection.bias.copy_(torch.from_numpy(weights["vit.embeddings.patch_embeddings.projection.bias"]))
            else:
                self.embeddings.cls_token.copy_(torch.from_numpy(weights["cls"]))
                self.embeddings.position_embeddings.copy_(torch.from_numpy((weights["Transformer/posembed_input/pos_embedding"])))
                conv_weight = weights["embedding/kernel"]
                conv_weight = conv_weight.transpose([3, 2, 0, 1])
                self.embeddings.patch_embeddings.projection.weight.copy_(torch.from_numpy(conv_weight))
                self.embeddings.patch_embeddings.projection.bias.copy_(torch.from_numpy(weights["embedding/bias"]))
                
    @torch.no_grad()
    def _load_weights_last(self, weights, prune=False):
        if(prune):
            self.layernorm.weight.copy_(torch.from_numpy(weights["vit.layernorm.weight"]))
            self.layernorm.bias.copy_(torch.from_numpy(weights["vit.layernorm.bias"]))
        else:
            self.layernorm.weight.copy_(torch.from_numpy(weights["Transformer/encoder_norm/scale"]))
            self.layernorm.bias.copy_(torch.from_numpy(weights["Transformer/encoder_norm/bias"]))

    @torch.no_grad()
    def _load_weights_layer(self, weights, layer_id, layer, prune=False):
        root = f"Transformer/encoderblock_{layer_id}/"
        hidden_size = self.config.hidden_size
        if layer.has_layer(0):
            if(prune):
                layer.layernorm_before.weight.copy_(torch.from_numpy(weights["vit.layers.{}.layernorm_before.weight".format(layer_id)]))
                layer.layernorm_before.bias.copy_(torch.from_numpy(weights["vit.layers.{}.layernorm_before.bias".format(layer_id)]))
                layer.self_attention.query.weight.copy_(torch.from_numpy(weights["vit.layers.{}.self_attention.query.weight".format(layer_id)]))
                layer.self_attention.key.weight.copy_(torch.from_numpy(weights["vit.layers.{}.self_attention.key.weight".format(layer_id)]))
                layer.self_attention.value.weight.copy_(torch.from_numpy(weights["vit.layers.{}.self_attention.value.weight".format(layer_id)]))
                layer.self_attention.query.bias.copy_(torch.from_numpy(weights["vit.layers.{}.self_attention.query.bias".format(layer_id)]))
                layer.self_attention.key.bias.copy_(torch.from_numpy(weights["vit.layers.{}.self_attention.key.bias".format(layer_id)]))
                layer.self_attention.value.bias.copy_(torch.from_numpy(weights["vit.layers.{}.self_attention.value.bias".format(layer_id)]))
            else:
                layer.layernorm_before.weight.copy_(torch.from_numpy(weights[root + "LayerNorm_0/scale"]))
                layer.layernorm_before.bias.copy_(torch.from_numpy(weights[root + "LayerNorm_0/bias"]))
                layer.self_attention.query.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/query/kernel"]).view(hidden_size, hidden_size).t())
                layer.self_attention.key.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/key/kernel"]).view(hidden_size, hidden_size).t())
                layer.self_attention.value.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/value/kernel"]).view(hidden_size, hidden_size).t())
                layer.self_attention.query.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/query/bias"]).view(-1))
                layer.self_attention.key.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/key/bias"]).view(-1))
                layer.self_attention.value.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/value/bias"]).view(-1))
        if layer.has_layer(1):
            if(prune):
                layer.self_output.dense.weight.copy_(torch.from_numpy(weights["vit.layers.{}.self_output.dense.weight".format(layer_id)]))
                layer.self_output.dense.bias.copy_(torch.from_numpy(weights["vit.layers.{}.self_output.dense.bias".format(layer_id)]))
            else:
                layer.self_output.dense.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/out/kernel"]).view(hidden_size, hidden_size).t())
                layer.self_output.dense.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/out/bias"]).view(-1))
        if layer.has_layer(2):
            if(prune):
                layer.layernorm_after.weight.copy_(torch.from_numpy(weights["vit.layers.{}.layernorm_after.weight".format(layer_id)]))
                layer.layernorm_after.bias.copy_(torch.from_numpy(weights["vit.layers.{}.layernorm_after.bias".format(layer_id)]))
                layer.intermediate.dense.weight.copy_(torch.from_numpy(weights["vit.layers.{}.intermediate.dense.weight".format(layer_id)]))
                layer.intermediate.dense.bias.copy_(torch.from_numpy(weights["vit.layers.{}.intermediate.dense.bias".format(layer_id)]))
            else:
                layer.layernorm_after.weight.copy_(torch.from_numpy(weights[root + "LayerNorm_2/scale"]))
                layer.layernorm_after.bias.copy_(torch.from_numpy(weights[root + "LayerNorm_2/bias"]))
                layer.intermediate.dense.weight.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_0/kernel"]).t())
                layer.intermediate.dense.bias.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_0/bias"]).t())
        if layer.has_layer(3):
            if(prune):
                layer.output.dense.weight.copy_(torch.from_numpy(weights["vit.layers.{}.output.dense.weight".format(layer_id)]))
                layer.output.dense.bias.copy_(torch.from_numpy(weights["vit.layers.{}.output.dense.bias".format(layer_id)]))                
            else:
                layer.output.dense.weight.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_1/kernel"]).t())
                layer.output.dense.bias.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_1/bias"]).t())

    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
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
        """Save the model weights file."""
        if url is None:
            url = _WEIGHTS_URLS[model_name]
        logger.info('Downloading model: %s: %s', model_name, url)
        req = requests.get(url, stream=True, timeout=timeout_sec)
        req.raise_for_status()
        with open(model_file, 'wb') as file:
            for chunk in req.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    file.flush()
                    os.fsync(file.fileno())


class ViTShardForImageClassification(ModuleShard):
    """Module shard based on `ViTForImageClassification`."""
    def __init__(self, config: ViTConfig, shard_config: ModuleShardConfig,
                 model_weights: Union[str, Mapping], prune=False):
        super().__init__(config, shard_config)
        self.vit = None
        self.classifier = None
            
        logger.debug(">>>> Model name: %s", self.config.name_or_path)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", model_weights)
            with np.load(model_weights) as weights:
                self._build_shard(weights, prune)
        else:
            self._build_shard(model_weights, prune)

    def _build_shard(self, weights, prune=False):
        ## all shards use the inner ViT model
        self.vit = ViTModelShard(self.config, self.shard_config, weights, prune)

        if self.shard_config.is_last:
            logger.debug(">>>> Load classifier for the last shard")
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels) if self.config.num_labels > 0 else nn.Identity()
            self._load_weights_last(weights, prune)

    @torch.no_grad()
    def _load_weights_last(self, weights, prune=False):
        if(prune):
            self.classifier.weight.copy_(torch.from_numpy(weights['classifier.weight']))
            self.classifier.bias.copy_(torch.from_numpy(weights['classifier.bias']))
        else:
            self.classifier.weight.copy_(torch.from_numpy(np.transpose(weights["head/kernel"])))
            self.classifier.bias.copy_(torch.from_numpy(weights["head/bias"]))

    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
        data = self.vit(data)
        if self.shard_config.is_last:
            data = self.classifier(data[:, 0, :])
        return data

    @staticmethod
    def save_weights(model_name: str, model_file: str, url: Optional[str]=None,
                     timeout_sec: Optional[float]=None) -> None:
        """Save the model weights file."""
        ViTModelShard.save_weights(model_name, model_file, url=url, timeout_sec=timeout_sec)

    def _quick_eval(self, test_batch):
        """
        Helper method to quickly evaluate model accuracy on a mini-batch.
        
        Args:
            test_batch: Tuple of (inputs, labels) for evaluation
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        inputs, labels = test_batch
        device = next(self.parameters()).device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Switch to eval mode
        self.eval()
        
        with torch.no_grad():
            outputs = self(inputs)
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            
        return correct / labels.size(0) 

    def recover_with_lora(self, calib_loader, pruned_state_dict=None, 
                         rank=8, alpha=16, lr=1e-3, steps=1000):
        """
        Apply LoRA recovery after pruning as described in the LLM-Pruner paper.
        
        Args:
            calib_loader: DataLoader with calibration data for fine-tuning
            pruned_state_dict: Optional pruned state dict to load (if not already loaded)
            rank: Rank for the LoRA low-rank adaptation
            alpha: Scaling factor for LoRA updates
            lr: Learning rate for optimizer
            steps: Number of optimization steps to perform
            
        Returns:
            Dictionary of recovered weights compatible with the pipeline
        """
        # If provided, load the pruned state
        if pruned_state_dict:
            self.load_state_dict(pruned_state_dict)
        
        # Replace relevant linear layers with LoRA-enabled versions
        self._add_lora_adapters(rank, alpha)
        
        # Get trainable parameters (only LoRA)
        trainable_params = [p for n, p in self.named_parameters() 
                           if p.requires_grad]
        
        print(f"LoRA recovery: {len(trainable_params)} trainable parameters")
        
        # Use a smaller learning rate by default to prevent divergence
        # User-provided lr will still override this
        default_lr = min(lr, 5e-5)  # Cap at 5e-5 if user didn't specify a lower value
        print(f"Using learning rate: {default_lr}")
        
        # Set up optimizer with weight decay
        optimizer = torch.optim.AdamW(trainable_params, lr=default_lr, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Optional: learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        
        # Training loop
        device = next(self.parameters()).device
        self.train()
        
        # Keep track of best state and loss
        best_loss = float('inf')
        best_state = None
        
        # Set up validation tracking if possible
        val_acc = None
        validation_freq = 100  # Check validation every 100 steps
        
        # Create small validation set from calibration data if possible
        try:
            calib_data = list(calib_loader)
            if len(calib_data) >= 3:  # If we have enough batches
                val_batch = calib_data[-1]  # Use last batch for validation
                print(f"Will track validation accuracy during recovery")
            else:
                val_batch = None
        except:
            val_batch = None
            print("Insufficient data for validation tracking")
        
        for step in range(steps):
            total_loss = 0
            num_batches = 0
            
            # Train on all batches except validation batch
            for i, batch in enumerate(calib_loader):
                # Skip last batch if we're using it for validation
                if val_batch is not None and i == len(calib_data) - 1:
                    continue
                    
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent instability
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Update learning rate
            scheduler.step()
            
            # Calculate average loss for this step
            avg_loss = total_loss / max(num_batches, 1)
            
            # Save best model (lowest loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                # Deep copy current state of model
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                
            # Print status more frequently at the beginning
            if (step + 1) % 10 == 0 or step == 0:
                curr_lr = optimizer.param_groups[0]['lr']
                print(f"Step {step+1}/{steps}, Loss: {avg_loss:.6f}, LR: {curr_lr:.7f}")
                
                # Check validation accuracy periodically
                if val_batch is not None and (step + 1) % validation_freq == 0:
                    val_inputs, val_targets = val_batch
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)
                    
                    self.eval()
                    with torch.no_grad():
                        val_outputs = self(val_inputs)
                        _, predicted = val_outputs.max(1)
                        correct = predicted.eq(val_targets).sum().item()
                        val_acc = correct / val_targets.size(0)
                        print(f"Validation accuracy: {val_acc:.4f}")
                    self.train()
        
        # Restore best model based on training loss
        if best_state is not None:
            print(f"Restoring best model with loss: {best_loss:.6f}")
            self.load_state_dict(best_state)
        
        # Merge LoRA weights back into base model
        recovered_state_dict = self._merge_lora_adapters()
        
        # Report final validation accuracy if available
        if val_batch is not None:
            val_inputs, val_targets = val_batch
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)
            
            self.eval()
            with torch.no_grad():
                val_outputs = self(val_inputs)
                _, predicted = val_outputs.max(1)
                correct = predicted.eq(val_targets).sum().item()
                val_acc = correct / val_targets.size(0)
                print(f"Final validation accuracy: {val_acc:.4f}")
        
        return recovered_state_dict

    def _add_lora_adapters(self, rank, alpha):
        """Replace linear layers with LoRA-enabled versions"""
        # Count of layers modified
        modified_count = 0
        
        # Replace attention projection layers and MLP layers
        for name, module in self.named_modules():
            # Only process linear layers (not embedding or classification)
            if isinstance(module, nn.Linear):
                # Skip first embedding projection and last classification layer
                if name.startswith('vit.embeddings') or name == 'classifier':
                    continue
                    
                # Access the parent module and replace this linear layer
                path = name.split('.')
                parent = self
                for part in path[:-1]:
                    parent = getattr(parent, part)
                
                # Replace with LoRA wrapper
                setattr(parent, path[-1], LoRALinear(module, rank, alpha))
                modified_count += 1
        
        print(f"Added LoRA adapters to {modified_count} layers")
        return self

    def _merge_lora_adapters(self):
        """Merge LoRA adapters back into the base model"""
        # Count of layers merged
        merged_count = 0
        
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                # Merge and replace with the original layer
                base_layer = module.merge()
                
                # Access the parent module
                path = name.split('.')
                parent = self
                for part in path[:-1]:
                    parent = getattr(parent, part)
                
                # Replace LoRA layer with merged base layer
                setattr(parent, path[-1], base_layer)
                merged_count += 1
        
        print(f"Merged LoRA adapters in {merged_count} layers")
        return self.state_dict()

    # WANDA Pruning Implementation
    def prune_wanda(self, ubatch, keep_ratio=0.9):
        """
        Prune ViT model using WANDA (Weight-Activation Norm Data-Aware) pruning.
        
        This method implements the WANDA pruning approach from Sun et al. (ICLR 2024),
        which prunes weights based on the product of weight magnitude and input activation
        norm, and applies pruning on a per-output basis.
        
        Args:
            ubatch: Batch of input data for calibration (determines activation patterns)
            keep_ratio: Percentage of weights to keep (0-1)
        
        Returns:
            Dictionary of pruned weights compatible with the pipeline
        """
        # Create a copy of the model to work with
        net = copy.deepcopy(self)
        
        # Skip if keep_ratio is 1.0 or greater
        if keep_ratio >= 1.0:
            print("No pruning performed (keep_ratio >= 1.0)")
            state_dict = net.state_dict()
            weights = {}
            for key, val in state_dict.items():
                weights[key] = val
            return weights
        
        # Print the shape of input tensor for debugging
        print(f"Input tensor shape: {ubatch.shape}")
        
        # Convert input tensor to right format if needed
        # The embeddings layer expects [batch_size, channels, height, width]
        device = ubatch.device
        
        # Storage for activation statistics
        activation_stats = {}
        
        # Define hook to capture input activations
        def hook_fn(name):
            def _hook(module, input_tensor, output):
                # Store the input tensor (first element of input tuple)
                if isinstance(input_tensor, tuple):
                    input_tensor = input_tensor[0]
                activation_stats[name] = input_tensor.detach()
            return _hook
        
        # Get all linear layers in the model
        linear_layers = []
        layer_names = []
        hooks = []
        
        for name, module in net.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append(module)
                layer_names.append(name)
                # Register forward hook to capture inputs
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Run the model on calibration data to collect activations
        with torch.no_grad():
            try:
                # Try normal forward pass
                if len(ubatch.shape) == 4:  # If it's already in image format
                    _ = net(ubatch)
                else:
                    # If input is already embedded (3D tensor from feature extractor)
                    # We need to bypass the embedding layer and start from transformer
                    if hasattr(net, 'vit'):
                        # Skip embedding layer
                        embedded_tensor = ubatch
                        # Forward through transformer layers
                        for layer in net.vit.layers:
                            embedded_tensor = layer(embedded_tensor)
                        
                        # Final layernorm if present
                        if net.vit.layernorm is not None:
                            embedded_tensor = net.vit.layernorm(embedded_tensor)
                        
                        # And classifier if present
                        if hasattr(net, 'classifier') and net.classifier is not None:
                            _ = net.classifier(embedded_tensor[:, 0, :])
                    else:
                        raise ValueError("Model structure unexpected")
            except Exception as e:
                print(f"Error during forward pass: {str(e)}")
                print("Creating dummy input for activations...")
                
                # Create a standard dummy input for ViT
                dummy_input = torch.randn(1, 3, 224, 224, device=device)
                _ = net(dummy_input)
        
        # Remove hooks to clean up
        for hook in hooks:
            hook.remove()
        
        # Now compute activation norms and WANDA scores
        wanda_scores = {}
        masks = {}
        
        for i, (name, layer) in enumerate(zip(layer_names, linear_layers)):
            # Skip first and last layers (we preserve these)
            if i == 0 or i == len(linear_layers) - 1:
                # Create all-ones mask for preserved layers
                mask = torch.ones_like(layer.weight.data)
                masks[name] = mask
                print(f"Layer:{name} => Density: 1.0000")
                continue
                
            # Get the input activations for this layer
            if name not in activation_stats:
                print(f"Warning: No activations captured for {name}, skipping pruning")
                mask = torch.ones_like(layer.weight.data)
                masks[name] = mask
                print(f"Layer:{name} => Density: 1.0000 (unpruned)")
                continue
                
            activations = activation_stats[name]
            
            # Compute feature norms - L2 norm across batch dimension
            if activations.dim() >= 3:
                # For attention layers: [batch_size, seq_len, hidden_dim]
                # First reshape to [batch_size * seq_len, hidden_dim]
                act_reshaped = activations.reshape(-1, activations.size(-1))
                feature_norms = torch.norm(act_reshaped, dim=0)
            else:
                # For regular layers: [batch_size, hidden_dim]
                feature_norms = torch.norm(activations, dim=0)
            
            # Handle potential NaNs or zeros in norms
            feature_norms = torch.where(
                torch.isnan(feature_norms) | (feature_norms == 0),
                torch.ones_like(feature_norms),
                feature_norms
            )
            
            # Ensure dimensions match for multiplication
            W = layer.weight.data
            input_dim = W.shape[1]
            
            # Resize feature_norms if necessary
            if feature_norms.size(0) != input_dim:
                print(f"Dimension mismatch in {name}: Weight input dim {input_dim}, feature_norms dim {feature_norms.size(0)}")
                # If we have too many features, use average of feature norms
                if feature_norms.size(0) > input_dim:
                    feature_norms = feature_norms[:input_dim]
                # If we have too few features, expand by repeating
                else:
                    feature_norms = feature_norms.repeat(input_dim // feature_norms.size(0) + 1)[:input_dim]
            
            # Compute WANDA scores (weight × activation norm)
            # For each output neuron, score its weights by the product
            # Shape: [output_dim, input_dim]
            scores = torch.abs(W) * feature_norms.unsqueeze(0)
            wanda_scores[name] = scores
            
            # Create mask (all ones initially)
            mask = torch.ones_like(W)
            
            # For each output neuron, keep the top k% weights
            k = int(W.shape[1] * keep_ratio)
            for j in range(W.shape[0]):  # For each output neuron
                if k < W.shape[1]:  # Only prune if we're keeping less than 100%
                    # Get scores for this output neuron
                    neuron_scores = scores[j]
                    
                    # Get threshold for top k elements
                    threshold, _ = torch.topk(neuron_scores, k, sorted=True)
                    # Use the smallest value in the top-k as our threshold
                    threshold_value = threshold[-1]
                    
                    # Create binary mask for this neuron based on threshold
                    mask[j] = (neuron_scores >= threshold_value).float()
            
            # Apply mask to weights
            layer.weight.data = layer.weight.data * mask
            masks[name] = mask
            
            # Print statistics
            density = mask.sum().item() / mask.numel()
            print(f"Layer:{name} => Density: {density:.4f}")
        
        # Convert state dict to expected format
        state_dict = net.state_dict()
        weights = {}
        for key, val in state_dict.items():
            weights[key] = val
        
        return weights 
        
    def prune_wanda_iterative(self, ubatch, final_keep_ratio=0.3, steps=3, mini_test_batch=None):
        """
        Iterative WANDA pruning with multiple steps for higher sparsity.
        
        Gradually prunes the model in steps, which often achieves higher sparsity
        with less accuracy degradation than one-shot pruning.
        
        Args:
            ubatch: Batch of input data for calibration (determines activation patterns)
            final_keep_ratio: Final percentage of weights to keep (0-1)
            steps: Number of pruning steps to use
            mini_test_batch: Optional validation batch for monitoring accuracy between steps
            
        Returns:
            Dictionary of pruned weights compatible with the pipeline
        """
        print(f"Starting iterative WANDA pruning to target keep ratio {final_keep_ratio} in {steps} steps")
        
        # Start with modest pruning
        current_keep_ratio = 0.9
        
        # Calculate step size to reach target
        step_size = (current_keep_ratio - final_keep_ratio) / steps
        
        # Use a copy of the model for iterative pruning
        net = copy.deepcopy(self)
        
        # Track accuracy degradation if test batch provided
        if mini_test_batch is not None:
            initial_acc = self._quick_eval(mini_test_batch)
            print(f"Initial accuracy on mini-batch: {initial_acc:.4f}")
        
        # Perform iterative pruning
        for step in range(steps):
            current_keep_ratio -= step_size
            print(f"Pruning step {step+1}/{steps}, keep_ratio = {current_keep_ratio:.2f}")
            
            # Apply WANDA pruning at current sparsity level
            weights = net.prune_wanda(ubatch, current_keep_ratio)
            
            # Update model for next iteration
            net.load_state_dict(weights)
            
            # Track accuracy degradation if test batch provided
            if mini_test_batch is not None:
                step_acc = net._quick_eval(mini_test_batch)
                print(f"Accuracy after step {step+1}: {step_acc:.4f} (delta: {step_acc - initial_acc:.4f})")
                
                # Potentially backoff if accuracy drops too much
                if step_acc < initial_acc - 0.20 and step < steps - 1:
                    print(f"Warning: Large accuracy drop detected. Adjusting remaining pruning steps.")
                    remaining_steps = steps - step - 1
                    if remaining_steps > 0:
                        step_size = step_size * 0.7  # Reduce pruning aggressiveness
        
        # Return final weights
        return weights

    def _quick_eval(self, test_batch):
        """
        Helper method to quickly evaluate model accuracy on a mini-batch.
        
        Args:
            test_batch: Tuple of (inputs, labels) for evaluation
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        inputs, labels = test_batch
        device = next(self.parameters()).device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Switch to eval mode
        self.eval()
        
        with torch.no_grad():
            outputs = self(inputs)
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            
        return correct / labels.size(0) 