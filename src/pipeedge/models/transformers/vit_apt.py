"""ViT Transformers with APT Pruning.

This module implements the APT (Accurate Post-Training Pruning) approach
for Vision Transformers, based on the paper "Pruning Foundation Models for High 
Accuracy without Retraining" (Zhao et al., 2024).
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

    # Helper functions for APT pruning
    def _calculate_hessian_inverse(self, activations, dampening=0.01):
        """
        Calculate the dampened Hessian inverse for APT pruning.
        
        For linear layers, Hessian approximation is 2xxᵀ.
        
        Args:
            activations: Input activations
            dampening: Dampening factor for numerical stability
            
        Returns:
            The Hessian inverse matrix
        """
        # For linear layers with a quadratic loss L(w) = ||wx||², 
        # the Hessian is H = 2xxᵀ where x is the input activation
        
        # Reshape activations if needed (e.g., for attention layers)
        if activations.dim() >= 3:
            # For attention layers: [batch_size, seq_len, hidden_dim]
            # Reshape to [batch_size * seq_len, hidden_dim]
            act_reshaped = activations.reshape(-1, activations.size(-1))
        else:
            # For regular layers: [batch_size, hidden_dim]
            act_reshaped = activations
        
        # Calculate 2xxᵀ
        xx_t = 2 * torch.matmul(act_reshaped.t(), act_reshaped)
        
        # Add dampening for stability
        # (2xxᵀ + γI) where γ is the dampening factor
        xx_t_damped = xx_t + dampening * torch.eye(xx_t.shape[0], device=xx_t.device)
        
        # Calculate inverse
        try:
            xx_t_inv = torch.inverse(xx_t_damped)
            return xx_t_inv
        except Exception as e:
            print(f"Error inverting Hessian: {str(e)}. Using pseudo-inverse instead.")
            # Fallback to pseudo-inverse if regular inverse fails
            return torch.pinverse(xx_t_damped)
        
    def _select_mask_solution_s(self, weights, activations, hessian_inverse, keep_ratio):
        """
        Select pruning mask using Solution S (simplified scoring).
        
        This implementation matches Equation 14 from the paper, ignoring interactions
        between multiple pruned weights.
        
        Args:
            weights: Weight matrix [output_dim, input_dim]
            activations: Input activations
            hessian_inverse: Pre-computed Hessian inverse
            keep_ratio: Percentage of weights to keep (0-1)
            
        Returns:
            Binary mask with the same shape as weights
        """
        # Extract diagonal elements from the Hessian inverse
        hessian_inv_diag = torch.diag(hessian_inverse)
        
        # Calculate pruning scores according to Equation 14
        # L* = [w]²ᵢⱼ / 2[(2xxᵀ)⁻¹]ⱼⱼ
        scores = torch.zeros_like(weights)
        for i in range(weights.shape[0]):  # For each output dimension
            denom = 2 * hessian_inv_diag
            # Handle potential division by zero
            denom = torch.where(denom > 0, denom, torch.ones_like(denom))
            scores[i] = weights[i]**2 / denom
        
        # Create mask (all ones initially)
        mask = torch.ones_like(weights)
        
        # Calculate number of weights to keep per output dimension
        k = int(weights.shape[1] * keep_ratio)
        
        # For each output dimension, keep the weights with highest scores
        for i in range(weights.shape[0]):
            if k < weights.shape[1]:  # Only prune if keep_ratio < 1.0
                # Find threshold for top-k weights
                row_scores = scores[i]
                threshold, _ = torch.topk(row_scores, k, sorted=True)
                threshold_value = threshold[-1]
                
                # Create binary mask based on threshold
                mask[i] = (row_scores >= threshold_value).float()
        
        return mask
    
    def _select_mask_solution_m(self, weights, activations, hessian_inverse, keep_ratio):
        """
        Select pruning mask using Solution M (full MRP).
        
        This implementation matches Equation 12 from the paper, considering interactions
        between the multiple pruned weights.
        
        Args:
            weights: Weight matrix [output_dim, input_dim]
            activations: Input activations 
            hessian_inverse: Pre-computed Hessian inverse
            keep_ratio: Percentage of weights to keep (0-1)
            
        Returns:
            Binary mask with the same shape as weights
        """
        # This is only implemented for N:M semi-structured sparsity where it's computationally feasible
        # to calculate the loss for each possible pruning pattern within a small block
        
        # Create mask (all ones initially)
        mask = torch.ones_like(weights)
        
        # Calculate number of weights to keep per row
        M = 4  # Assuming 2:4 pattern
        N = M - int(M * keep_ratio)  # Number to prune
        
        # Process weights in groups of M columns
        for i in range(weights.shape[0]):  # For each output dimension
            for j in range(0, weights.shape[1], M):
                # Handle edge case for last group (may be smaller than M)
                end_idx = min(j + M, weights.shape[1])
                group_size = end_idx - j
                
                if group_size < M:
                    # Just keep all weights in the partial group
                    continue
                
                if N == 0 or N >= group_size:
                    # Either keep all or prune all weights in the group
                    continue
                
                # Get the weights for this group
                w_group = weights[i, j:end_idx]
                
                # Calculate all possible combinations to prune N out of M weights
                import itertools
                best_loss = float('inf')
                best_mask = torch.ones(group_size, device=weights.device)
                
                # Generate all combinations of indices to prune
                for prune_indices in itertools.combinations(range(group_size), N):
                    # Create trial mask for this combination
                    trial_mask = torch.ones(group_size, device=weights.device)
                    trial_mask[list(prune_indices)] = 0
                    
                    # Extract pruned weights and their indices
                    pruned_weights = w_group * (1 - trial_mask)
                    
                    # Create selection matrix for pruned columns
                    e_p = torch.zeros(hessian_inverse.shape[1], N, device=weights.device)
                    
                    # Fill selection matrix for the specific row and columns
                    for idx, prune_idx in enumerate(prune_indices):
                        e_p[j + prune_idx, idx] = 1
                    
                    # Calculate the product of selection matrices
                    try:
                        # Calculate e_p^T @ H^-1 @ e_p
                        temp = torch.matmul(hessian_inverse, e_p)
                        e_p_H_inv_e_p = torch.matmul(e_p.t(), temp)
                        
                        # Calculate the inverse of e_p^T @ H^-1 @ e_p
                        e_p_H_inv_e_p_inv = torch.inverse(e_p_H_inv_e_p)
                        
                        # Calculate loss from Equation 12: 0.5 * w_q @ [e_p^T @ H^-1 @ e_p]^-1 @ w_q^T
                        pruned_weights_vector = pruned_weights[pruned_weights != 0]
                        loss = 0.5 * torch.matmul(torch.matmul(pruned_weights_vector, e_p_H_inv_e_p_inv), 
                                                 pruned_weights_vector.t())
                        
                        # Keep the mask with the lowest loss
                        if loss < best_loss:
                            best_loss = loss
                            best_mask = trial_mask
                    except Exception as e:
                        # Skip this combination if there's a numerical error
                        continue
                
                # Update the final mask with the best mask for this group
                mask[i, j:end_idx] = best_mask
        
        return mask
    
    def _compensate_weights_solution_s(self, weights, mask, activations, hessian_inverse):
        """
        Compensate weights using Solution S (SparseGPT approach).
        
        Args:
            weights: Original weight matrix [output_dim, input_dim]
            mask: Binary pruning mask [output_dim, input_dim]
            activations: Input activations
            hessian_inverse: Pre-computed Hessian inverse
            
        Returns:
            Updated weight matrix
        """
        # Implementation similar to SparseGPT
        # Process each output dimension (row) separately
        for i in range(weights.shape[0]):
            # Skip if row is all zeros or all ones
            if torch.all(mask[i] == 0) or torch.all(mask[i] == 1):
                continue
                
            # Get pruned and kept indices for this row
            pruned_indices = (mask[i] == 0).nonzero().view(-1)
            kept_indices = (mask[i] == 1).nonzero().view(-1)
            
            if len(pruned_indices) == 0:
                continue
                
            # Compute weight updates for kept weights based on SparseGPT approach
            # Process each pruned weight one by one
            for p_idx in pruned_indices:
                # Original weight value
                w_p = weights[i, p_idx].item()
                
                # If weight is already 0, skip
                if w_p == 0:
                    continue
                
                # Calculate compensation for each kept weight
                for k_idx in kept_indices:
                    # Calculate OBS update for this kept weight
                    h_kp = hessian_inverse[k_idx, p_idx]
                    h_pp = hessian_inverse[p_idx, p_idx]
                    
                    # Update formula from OBS
                    delta = -w_p * h_kp / h_pp
                    weights[i, k_idx] += delta
                
                # Zero out the pruned weight
                weights[i, p_idx] = 0
        
        return weights
    
    def _compensate_weights_solution_m(self, weights, mask, activations, hessian_inverse):
        """
        Compensate weights using Solution M (full MRP).
        
        This implements the optimal weight update from Equation 11 in the paper.
        
        Args:
            weights: Original weight matrix [output_dim, input_dim]
            mask: Binary pruning mask [output_dim, input_dim]
            activations: Input activations
            hessian_inverse: Pre-computed Hessian inverse
            
        Returns:
            Updated weight matrix
        """
        # Process each output dimension (row) separately
        for i in range(weights.shape[0]):
            # Skip if row is all zeros or all ones
            if torch.all(mask[i] == 0) or torch.all(mask[i] == 1):
                continue
                
            # Get pruned and kept indices for this row
            pruned_indices = torch.where(mask[i] == 0)[0]
            kept_indices = torch.where(mask[i] == 1)[0]
            
            if len(pruned_indices) == 0:
                continue
                
            # Extract pruned weights
            w_pruned = weights[i, pruned_indices]
            
            # Create selection matrix for pruned columns
            e_p = torch.zeros(hessian_inverse.shape[1], len(pruned_indices), device=weights.device)
            
            # Fill selection matrix
            for j, idx in enumerate(pruned_indices):
                e_p[idx, j] = 1
            
            try:
                # Calculate e_p^T @ H^-1 @ e_p
                temp = torch.matmul(hessian_inverse, e_p)
                e_p_H_inv_e_p = torch.matmul(e_p.t(), temp)
                
                # Calculate [e_p^T @ H^-1 @ e_p]^-1
                e_p_H_inv_e_p_inv = torch.inverse(e_p_H_inv_e_p)
                
                # Calculate e_p^T @ H^-1
                e_p_H_inv = torch.matmul(e_p.t(), hessian_inverse)
                
                # Calculate optimal weight update for this row per Equation 13:
                # δw* = -w_q @ [e_p^T @ H^-1 @ e_p]^-1 @ e_p^T @ H^-1
                delta_w = -torch.matmul(torch.matmul(w_pruned, e_p_H_inv_e_p_inv), e_p_H_inv)
                
                # Apply updates to the kept weights only
                # First zero out the pruned weights
                weights[i, pruned_indices] = 0
                
                # Then add delta_w to all weights (only kept weights will be non-zero)
                weights[i] += delta_w
                
            except Exception as e:
                print(f"Error in compensating weights for row {i}: {str(e)}")
                # Fallback to Solution S for this row
                print("Falling back to Solution S for this row")
                self._compensate_weights_solution_s(weights[i:i+1], mask[i:i+1], 
                                                   activations, hessian_inverse)
        
        return weights

    # APT Pruning Implementation
    def prune_apt_ss(self, ubatch, keep_ratio=0.9, dampening=0.01):
        """
        Prune ViT model using APT (Accurate Post-Training Pruning) with Solution S for mask selection
        and Solution S for weight compensation.
        
        This method implements the approach from Zhao et al. (2024), which formulates pruning
        as a Multiple Removal Problem (MRP) and solves it optimally.
        
        Args:
            ubatch: Batch of input data for calibration (determines activation patterns)
            keep_ratio: Percentage of weights to keep (0-1)
            dampening: Dampening factor for numerical stability of Hessian inverse
        
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
        
        print(f"Input tensor shape: {ubatch.shape}")
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
        
        # Now compute Hessian inverse and apply APT pruning
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
            
            # Calculate Hessian inverse
            hessian_inverse = self._calculate_hessian_inverse(activations, dampening)
            
            # Select mask using Solution S (Equation 14)
            weights = layer.weight.data
            mask = self._select_mask_solution_s(weights, activations, hessian_inverse, keep_ratio)
            masks[name] = mask
            
            # Apply weight compensation using Solution S
            updated_weights = self._compensate_weights_solution_s(
                weights.clone(), mask, activations, hessian_inverse)
            
            # Update layer weights
            layer.weight.data = updated_weights
            
            # Print statistics
            density = mask.sum().item() / mask.numel()
            print(f"Layer:{name} => Density: {density:.4f}")
        
        # Convert state dict to expected format
        state_dict = net.state_dict()
        weights = {}
        for key, val in state_dict.items():
            weights[key] = val
        
        return weights
    
    def prune_apt_sm(self, ubatch, keep_ratio=0.9, dampening=0.01):
        """
        Prune ViT model using APT with Solution S for mask selection and Solution M for weight compensation.
        
        This combination provides a good balance of performance and efficiency.
        
        Args:
            ubatch: Batch of input data for calibration (determines activation patterns)
            keep_ratio: Percentage of weights to keep (0-1)
            dampening: Dampening factor for numerical stability of Hessian inverse
        
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
        
        print(f"Input tensor shape: {ubatch.shape}")
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
        
        # Now compute Hessian inverse and apply APT pruning
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
            
            # Calculate Hessian inverse
            hessian_inverse = self._calculate_hessian_inverse(activations, dampening)
            
            # Select mask using Solution S (Equation 14)
            weights = layer.weight.data
            mask = self._select_mask_solution_s(weights, activations, hessian_inverse, keep_ratio)
            masks[name] = mask
            
            # Apply weight compensation using Solution M (optimal compensation)
            updated_weights = self._compensate_weights_solution_m(
                weights.clone(), mask, activations, hessian_inverse)
            
            # Update layer weights
            layer.weight.data = updated_weights
            
            # Print statistics
            density = mask.sum().item() / mask.numel()
            print(f"Layer:{name} => Density: {density:.4f}")
        
        # Convert state dict to expected format
        state_dict = net.state_dict()
        weights = {}
        for key, val in state_dict.items():
            weights[key] = val
        
        return weights
    
    def prune_apt_iterative(self, ubatch, final_keep_ratio=0.3, steps=3, mini_test_batch=None, 
                           solution_mask='s', solution_comp='m', dampening=0.01):
        """
        Iterative APT pruning with multiple steps for higher sparsity.
        
        Gradually prunes the model in steps, which often achieves higher sparsity
        with less accuracy degradation than one-shot pruning.
        
        Args:
            ubatch: Batch of input data for calibration (determines activation patterns)
            final_keep_ratio: Final percentage of weights to keep (0-1)
            steps: Number of pruning steps to use
            mini_test_batch: Optional validation batch for monitoring accuracy between steps
            solution_mask: Which solution to use for mask selection ('s' or 'm')
            solution_comp: Which solution to use for weight compensation ('s' or 'm')
            dampening: Dampening factor for numerical stability of Hessian inverse
            
        Returns:
            Dictionary of pruned weights compatible with the pipeline
        """
        print(f"Starting iterative APT pruning to target keep ratio {final_keep_ratio} in {steps} steps")
        print(f"Using Solution {solution_mask.upper()} for mask selection and Solution {solution_comp.upper()} for compensation")
        
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
        
        # Determine which pruning method to use
        if solution_mask.lower() == 's' and solution_comp.lower() == 's':
            prune_method = net.prune_apt_ss
        elif solution_mask.lower() == 's' and solution_comp.lower() == 'm':
            prune_method = net.prune_apt_sm
        else:
            # Default to SM if an invalid combination is provided
            prune_method = net.prune_apt_sm
            print(f"Warning: Unsupported solution combination: {solution_mask}-{solution_comp}. Using S-M instead.")
        
        # Perform iterative pruning
        for step in range(steps):
            current_keep_ratio -= step_size
            print(f"Pruning step {step+1}/{steps}, keep_ratio = {current_keep_ratio:.2f}")
            
            # Apply APT pruning at current sparsity level
            weights = prune_method(ubatch, current_keep_ratio, dampening)
            
            # Update model for next iteration
            net.load_state_dict(weights)
            
            # Track accuracy degradation if test batch provided
            if mini_test_batch is not None:
                step_acc = net._quick_eval(mini_test_batch)
                print(f"Accuracy after step {step+1}: {step_acc:.4f} (delta: {step_acc - initial_acc:.4f})")
                
                # Potentially back off if accuracy drops too much
                if step_acc < initial_acc - 0.20 and step < steps - 1:
                    print(f"Warning: Large accuracy drop detected. Adjusting remaining pruning steps.")
                    remaining_steps = steps - step - 1
                    if remaining_steps > 0:
                        step_size = step_size * 0.7  # Reduce pruning aggressiveness
        
        # Return final weights
        return weights
    
    def prune_apt_and_calibrate(self, ubatch, calib_loader, keep_ratio=0.3, 
                               solution_mask='s', solution_comp='m', dampening=0.01,
                               calib_steps=100, calib_lr=1e-5):
        """
        Apply APT pruning followed by gradient-based calibration (fine-tuning).
        
        This combines the benefits of APT pruning with additional supervised fine-tuning
        to regain accuracy at high sparsity levels.
        
        Args:
            ubatch: Batch of input data for calibration (APT pruning)
            calib_loader: DataLoader for calibration (fine-tuning)
            keep_ratio: Percentage of weights to keep (0-1)
            solution_mask: Which solution to use for mask selection ('s' or 'm')
            solution_comp: Which solution to use for weight compensation ('s' or 'm')
            dampening: Dampening factor for numerical stability of Hessian inverse
            calib_steps: Number of optimization steps for calibration
            calib_lr: Learning rate for calibration
            
        Returns:
            Dictionary of pruned and calibrated weights
        """
        print(f"Applying APT pruning with Solution {solution_mask.upper()}-{solution_comp.upper()} "
              f"at keep_ratio={keep_ratio} followed by calibration")
        
        # First apply APT pruning
        net = copy.deepcopy(self)
        
        # Determine which pruning method to use
        if solution_mask.lower() == 's' and solution_comp.lower() == 's':
            prune_method = net.prune_apt_ss
        elif solution_mask.lower() == 's' and solution_comp.lower() == 'm':
            prune_method = net.prune_apt_sm
        else:
            # Default to SM if an invalid combination is provided
            prune_method = net.prune_apt_sm
            print(f"Warning: Unsupported solution combination: {solution_mask}-{solution_comp}. Using S-M instead.")
        
        # Apply APT pruning
        weights = prune_method(ubatch, keep_ratio=keep_ratio, dampening=dampening)
        net.load_state_dict(weights)
        
        print(f"Starting calibration for {calib_steps} steps with lr={calib_lr}")
        
        # Store masks for each layer
        masks = {}
        for name, module in net.named_modules():
            if isinstance(module, nn.Linear):
                # Create binary mask (1 for non-zero weights, 0 for zero weights)
                mask = (module.weight.data != 0).float()
                masks[name] = mask
        
        # Calibration (fine-tuning) phase
        net.train()  # Set model to training mode
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=calib_lr)
        
        # Calibration loop
        for step in range(calib_steps):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(calib_loader):
                device = next(net.parameters()).device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Apply masks before optimizer step (keep pruned weights at zero)
                for name, module in net.named_modules():
                    if isinstance(module, nn.Linear) and name in masks:
                        module.weight.grad.data.mul_(masks[name])
                
                optimizer.step()
                
                # Logging
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(targets).sum().item()
                correct += batch_correct
                total += targets.size(0)
                
                # Break if we've processed enough batches
                if batch_idx * calib_loader.batch_size >= calib_steps * 32:
                    break
            
            # Print calibration progress
            if (step + 1) % 10 == 0 or step == 0:
                print(f"Calibration step {step+1}/{calib_steps}, "
                      f"loss: {total_loss/(batch_idx+1):.4f}, "
                      f"acc: {100.0*correct/total:.2f}%")
        
        # Final check to ensure zeros stay zeros
        for name, module in net.named_modules():
            if isinstance(module, nn.Linear) and name in masks:
                module.weight.data.mul_(masks[name])
        
        # Return final weights
        state_dict = net.state_dict()
        final_weights = {}
        for key, val in state_dict.items():
            final_weights[key] = val
        
        return final_weights
    
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