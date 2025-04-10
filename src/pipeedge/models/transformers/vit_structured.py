"""ViT Transformers with Structured Pruning and LoRA.

This module implements the LLM-Pruner structured pruning and LoRA fine-tuning approach
for Vision Transformers, based on the paper "LLM-Pruner: On the Structural Pruning of Large Language Models"
(Ma et al., 2023).
"""
from collections.abc import Mapping
import logging
import math
import os
from typing import Optional, Union, Dict, List, Tuple, Set
import numpy as np
import requests
import torch
from torch import nn
from transformers import ViTConfig
from transformers.models.vit.modeling_vit import (
    ViTEmbeddings, ViTIntermediate, ViTOutput, ViTSelfAttention, ViTSelfOutput
)
from peft import LoraConfig, get_peft_model
from .. import ModuleShard, ModuleShardConfig
from . import TransformerShardData
import torch.nn.functional as F
import types
import copy
import networkx as nx
import time
from functools import partial

logger = logging.getLogger(__name__)

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


# LLM-Pruner Components

class StructuralGroup:
    """Represents a group of structurally dependent components that must be pruned together."""
    
    def __init__(self, group_id: int, group_type: str):
        self.group_id = group_id
        self.group_type = group_type  # 'attention', 'mlp', or 'channel'
        self.modules = {}  # Dict mapping module names to modules
        self.parameters = {}  # Dict mapping parameter names to parameters
        self.importance = 0.0  # Will be set during importance estimation
        
    def add_module(self, name: str, module: nn.Module):
        """Add a module to this structural group."""
        self.modules[name] = module
        # Also track its parameters
        for param_name, param in module.named_parameters():
            full_name = f"{name}.{param_name}"
            self.parameters[full_name] = param
            
    def __repr__(self):
        return f"StructuralGroup(id={self.group_id}, type={self.group_type}, " \
               f"modules={list(self.modules.keys())}, importance={self.importance:.4f})"


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapters for efficient fine-tuning."""
    
    def __init__(self, 
                 base_layer: nn.Linear, 
                 rank: int = 8, 
                 alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        
        # Store the original layer
        self.base_layer = base_layer
        
        # LoRA hyperparameters
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Input and output dimensions
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # Regular forward pass
        base_output = self.base_layer(x)
        
        # LoRA adaptation
        lora_output = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t() * self.scaling
        
        # Combined output
        return base_output + lora_output


def apply_lora_to_model(model, rank=8, alpha=16.0, target_modules=None):
    """Apply LoRA adapters to specific modules in a model."""
    
    if target_modules is None:
        # Default: apply to all attention QKV and output, and MLP layers
        target_modules = ["query", "key", "value", "dense"]
        
    for name, module in list(model.named_modules()):
        # Check if module is a target for LoRA adaptation
        if any(target_name in name for target_name in target_modules) and isinstance(module, nn.Linear):
            parent_name = name.rsplit(".", 1)[0]
            child_name = name.split(".")[-1]
            
            # Get parent module
            parent = model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)
            
            # Replace with LoRA version
            lora_module = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_module)
            
    return model


def find_structural_groups(model, group_type='block'):
    """
    Identify structural groups in the model based on connectivity patterns.
    
    Args:
        model: The ViT model
        group_type: 'block' (attention/MLP blocks) or 'channel' (cross-layer channels)
        
    Returns:
        List of StructuralGroup objects
    """
    groups = []
    group_id = 0
    
    # Build a connectivity graph
    G = nx.DiGraph()
    
    # Map all named modules and parameters
    module_map = {}
    param_map = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module_map[name] = module
            
            # Add nodes for input and output of this layer
            in_node = f"{name}_in"
            out_node = f"{name}_out"
            G.add_node(in_node, type="feature", size=module.in_features)
            G.add_node(out_node, type="feature", size=module.out_features)
            G.add_node(name, type="module")
            
            # Connect input -> module -> output
            G.add_edge(in_node, name)
            G.add_edge(name, out_node)
            
            # Track parameters
            for param_name, param in module.named_parameters():
                full_param_name = f"{name}.{param_name}"
                param_map[full_param_name] = param
                G.add_node(full_param_name, type="parameter")
                G.add_edge(name, full_param_name)
    
    # Analyze the model structure to find connections between layers
    # based on ViT architecture
    
    if group_type == 'block':
        # For ViT, identify attention head groups and MLP groups
        
        # First, identify all transformer blocks
        transformer_blocks = {}
        for i in range(12):  # Assuming ViT-Base with 12 blocks
            block_prefix = f"vit.layers.{i}"
            
            # Find all modules in this block
            block_modules = {name: module for name, module in module_map.items() 
                            if name.startswith(block_prefix)}
            
            # Group 1: Attention head components (QKV + output projection)
            attn_group = StructuralGroup(group_id, 'attention')
            group_id += 1
            
            # Add query, key, value projections
            for comp in ['query', 'key', 'value']:
                module_name = f"{block_prefix}.self_attention.{comp}"
                if module_name in module_map:
                    attn_group.add_module(module_name, module_map[module_name])
            
            # Add output projection
            output_name = f"{block_prefix}.self_output.dense"
            if output_name in module_map:
                attn_group.add_module(output_name, module_map[output_name])
            
            groups.append(attn_group)
            
            # Group 2: MLP components (intermediate + output)
            mlp_group = StructuralGroup(group_id, 'mlp')
            group_id += 1
            
            # Add intermediate and output layers
            intermediate_name = f"{block_prefix}.intermediate.dense"
            if intermediate_name in module_map:
                mlp_group.add_module(intermediate_name, module_map[intermediate_name])
                
            mlp_output_name = f"{block_prefix}.output.dense"
            if mlp_output_name in module_map:
                mlp_group.add_module(mlp_output_name, module_map[mlp_output_name])
            
            groups.append(mlp_group)
    
    elif group_type == 'channel':
        # For channel-wise pruning, we group parameters across layers
        # This is more complex and less effective according to the paper
        pass
    
    return groups


def compute_group_importance(model, groups, data, loss_fn, method='vector'):
    """
    Compute importance scores for structural groups using gradient information.
    
    Args:
        model: The model to analyze
        groups: List of structural groups
        data: Calibration data batch (inputs, labels)
        loss_fn: Loss function to use
        method: Importance estimation method ('vector', 'element1', or 'element2')
        
    Returns:
        Updated groups with importance scores
    """
    # Ensure model is in training mode for gradient computation
    model.train()
    
    # Prepare inputs and targets
    inputs, targets = data
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Get loss and compute gradients
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    
    # Now compute importance scores for each group
    for group in groups:
        group_score = 0.0
        
        for param_name, param in group.parameters.items():
            if param.grad is None:
                continue
                
            if method == 'vector':
                # Use weight * gradient (first-order approximation)
                param_importance = torch.sum(torch.abs(param.data * param.grad)).item()
                
            elif method == 'element1':
                # Element-wise first-order approximation
                param_importance = torch.sum(torch.abs(param.data * param.grad)).item()
                
            elif method == 'element2':
                # Element-wise second-order approximation (using squared gradients)
                param_importance = torch.sum(torch.abs(param.data) * torch.abs(param.grad)**2).item()
            
            # Accumulate importance (paper found sum works best)
            group_score += param_importance
            
        # Store the importance score
        group.importance = group_score
    
    # Reset gradients
    model.zero_grad()
    
    # Return groups sorted by importance (ascending - least important first)
    return sorted(groups, key=lambda g: g.importance)


def prune_groups(model, groups, keep_ratio):
    """
    Prune the model by removing the least important structural groups.
    
    Args:
        model: The model to prune
        groups: List of structural groups sorted by importance
        keep_ratio: Fraction of parameters to keep
        
    Returns:
        Pruned model and a mask dictionary
    """
    # Count total parameters in pruneable groups
    total_params = sum(sum(p.numel() for p in group.parameters.values()) for group in groups)
    
    # Calculate how many parameters to keep
    params_to_keep = int(total_params * keep_ratio)
    
    # Start removing groups from the least important
    kept_params = total_params
    kept_groups = []
    pruned_groups = []
    
    # Create masks dictionary
    masks = {}
    
    # Initialize all masks to ones (keep everything)
    for group in groups:
        for param_name, param in group.parameters.items():
            masks[param_name] = torch.ones_like(param.data)
    
    # Remove least important groups until we reach desired sparsity
    for group in groups:
        group_params = sum(p.numel() for p in group.parameters.values())
        
        if kept_params - group_params >= params_to_keep:
            # We can prune this group
            kept_params -= group_params
            pruned_groups.append(group)
            
            # Update masks for this group's parameters
            for param_name, param in group.parameters.items():
                masks[param_name] = torch.zeros_like(param.data)
                
            print(f"Pruned: {group}")
        else:
            kept_groups.append(group)
    
    # Apply the masks to the model
    for name, param in model.named_parameters():
        if name in masks:
            param.data *= masks[name]
    
    return model, masks, pruned_groups, kept_groups


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

    # LLM-Pruner Structural Pruning Implementation
    def prune_structured(self, calib_batch, keep_ratio=0.7, group_type='block', importance_method='vector'):
        """
        Prune ViT model using LLM-Pruner structural pruning approach.
        
        This method implements the structural pruning approach from Ma et al. (2023),
        which removes entire structural components (attention heads or MLP blocks) based
        on gradient-based importance estimation.
        
        Args:
            calib_batch: Tuple of (inputs, labels) for calibration
            keep_ratio: Percentage of parameters to keep (0-1)
            group_type: Type of structural grouping to use ('block' or 'channel')
            importance_method: Method for computing importance ('vector', 'element1', or 'element2')
        
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
        
        print(f"Starting structured pruning with group_type={group_type}, importance_method={importance_method}")
        
        # Step 1: Discovery - Find structural groups
        print("Step 1: Discovering structural groups...")
        groups = find_structural_groups(net, group_type=group_type)
        print(f"Found {len(groups)} structural groups")
        
        # Step 2: Estimation - Compute importance scores
        print("Step 2: Estimating group importance...")
        # Define loss function for importance estimation
        loss_fn = nn.CrossEntropyLoss()
        
        # Compute importance for each group
        sorted_groups = compute_group_importance(net, groups, calib_batch, loss_fn, method=importance_method)
        
        # Print importance scores
        print("Group importance scores (least to most important):")
        for i, group in enumerate(sorted_groups):
            print(f"  {i+1}. {group}")
        
        # Step 3: Pruning - Remove least important groups
        print(f"Step 3: Pruning to {keep_ratio:.2f} keep ratio...")
        pruned_model, masks, pruned_groups, kept_groups = prune_groups(net, sorted_groups, keep_ratio)
        
        print(f"Pruned {len(pruned_groups)} groups, kept {len(kept_groups)} groups")
        
        # Log pruning stats
        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        pruned_params = sum(sum(p.numel() for p in group.parameters.values()) for group in pruned_groups)
        actual_sparsity = pruned_params / total_params
        print(f"Actual sparsity achieved: {actual_sparsity:.4f} (pruned {pruned_params} of {total_params} parameters)")
        
        # Convert state dict to expected format
        state_dict = net.state_dict()
        weights = {}
        for key, val in state_dict.items():
            weights[key] = val
        
        return weights
    
    def lora_recovery(self, pruned_weights, train_loader, num_epochs=2, rank=8, alpha=16.0, lr=2e-4, eval_batch=None):
        """
        Apply LoRA fine-tuning to recover accuracy after pruning.
        
        Args:
            pruned_weights: Pruned model weights dictionary
            train_loader: DataLoader with training samples
            num_epochs: Number of training epochs
            rank: Rank for LoRA adapters
            alpha: Scaling factor for LoRA
            lr: Learning rate
            eval_batch: Optional evaluation batch for progress tracking
            
        Returns:
            Dictionary of recovered weights with merged LoRA parameters
        """
        # Load pruned weights
        net = copy.deepcopy(self)
        net.load_state_dict(pruned_weights)
        
        # Track initial accuracy if eval batch provided
        if eval_batch is not None:
            initial_acc = net._quick_eval(eval_batch)
            print(f"Initial accuracy (before LoRA): {initial_acc:.4f}")
        
        # Step 1: Apply LoRA adapters
        print(f"Applying LoRA adapters (rank={rank}, alpha={alpha})...")
        lora_model = apply_lora_to_model(net, rank=rank, alpha=alpha)
        
        # Step 2: Set up fine-tuning
        # Freeze base model weights
        for name, param in lora_model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
                
        # Count trainable parameters
        lora_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in lora_model.parameters())
        print(f"Training {lora_params} LoRA parameters ({lora_params/total_params:.2%} of total)")
        
        # Set up optimizer and loss function
        optimizer = torch.optim.AdamW(
            [p for p in lora_model.parameters() if p.requires_grad],
            lr=lr
        )
        loss_fn = nn.CrossEntropyLoss()
        
        # Step 3: Train LoRA adapters
        print(f"Training LoRA adapters for {num_epochs} epochs...")
        device = next(lora_model.parameters()).device
        
        # Training loop
        lora_model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = lora_model(inputs)
                loss = loss_fn(outputs, targets)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track statistics
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Print progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, "
                          f"Loss: {epoch_loss/(batch_idx+1):.4f}, "
                          f"Acc: {correct/total:.4f}")
                    
            # Evaluate at end of epoch if eval batch provided
            if eval_batch is not None:
                lora_model.eval()
                recovery_acc = lora_model._quick_eval(eval_batch)
                lora_model.train()
                print(f"Epoch {epoch+1}/{num_epochs} eval accuracy: {recovery_acc:.4f} "
                      f"(delta: {recovery_acc - initial_acc:+.4f})")
        
        # Step 4: Merge LoRA weights into base model
        print("Merging LoRA weights into base model...")
        for name, module in lora_model.named_modules():
            if isinstance(module, LoRALinear):
                # Compute LoRA weight update: ∆W = BA
                delta_w = module.lora_B @ module.lora_A * module.scaling
                
                # Add to base weights: W' = W + ∆W
                module.base_layer.weight.data += delta_w
                
                # Update the module with merged weights
                parent_name = name.rsplit(".", 1)[0]
                child_name = name.split(".")[-1]
                
                # Get parent module
                parent = lora_model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)
                
                # Replace with original linear with merged weights
                setattr(parent, child_name, module.base_layer)
        
        # Evaluate final model if eval batch provided
        if eval_batch is not None:
            lora_model.eval()
            final_acc = lora_model._quick_eval(eval_batch)
            print(f"Final accuracy after LoRA merging: {final_acc:.4f} "
                  f"(delta: {final_acc - initial_acc:+.4f})")
        
        # Convert state dict to expected format
        state_dict = lora_model.state_dict()
        weights = {}
        for key, val in state_dict.items():
            weights[key] = val
        
        return weights
    
    def prune_structured_with_lora(self, calib_batch, train_loader, keep_ratio=0.7, 
                                   group_type='block', importance_method='vector',
                                   lora_rank=8, lora_alpha=16.0, lora_epochs=2, lora_lr=2e-4,
                                   eval_batch=None):
        """
        Combined structured pruning with LoRA recovery in one step.
        
        Args:
            calib_batch: Tuple of (inputs, labels) for calibration
            train_loader: DataLoader with training samples
            keep_ratio: Percentage of parameters to keep (0-1)
            group_type: Type of structural grouping to use ('block' or 'channel')
            importance_method: Method for computing importance ('vector', 'element1', or 'element2')
            lora_rank: Rank for LoRA adapters
            lora_alpha: Scaling factor for LoRA
            lora_epochs: Number of training epochs
            lora_lr: Learning rate
            eval_batch: Optional evaluation batch for progress tracking
            
        Returns:
            Dictionary of pruned and recovered weights
        """
        print("=== LLM-Pruner: Structured Pruning with LoRA Recovery ===")
        
        # Step 1: Structural Pruning
        print("\n[Phase 1: Structural Pruning]")
        pruned_weights = self.prune_structured(
            calib_batch, 
            keep_ratio=keep_ratio,
            group_type=group_type,
            importance_method=importance_method
        )
        
        # Step 2: LoRA Recovery
        print("\n[Phase 2: LoRA Recovery]")
        recovered_weights = self.lora_recovery(
            pruned_weights,
            train_loader,
            num_epochs=lora_epochs,
            rank=lora_rank, 
            alpha=lora_alpha,
            lr=lora_lr,
            eval_batch=eval_batch
        )
        
        return recovered_weights
    
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