"""ViT Transformers with WANDA Pruning and Calibration.

This module implements the WANDA (Weight-Activation Norm Data-Aware) pruning approach
for Vision Transformers, based on the paper "A Simple and Effective Pruning Approach
for Large Language Models" (Sun et al., ICLR 2024). It extends the original implementation
with calibration functionality to recover accuracy after pruning.
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

# Import from original implementation
from .vit_wanda import ViTLayerShard, ViTModelShard, _WEIGHTS_URLS

logger = logging.getLogger(__name__)


class ViTShardForImageClassificationV2(ModuleShard):
    """Module shard based on `ViTForImageClassification` with calibration support."""
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
    
    # New calibration methods
    def calibrate_weights(self, calib_loader, criterion=None, optimizer=None, 
                          num_steps=100, learning_rate=1e-5, device=None):
        """
        Fine-tune a pruned model to recover accuracy.
        
        Args:
            calib_loader: DataLoader with calibration samples
            criterion: Loss function to use (defaults to CrossEntropyLoss)
            optimizer: Optimizer to use (defaults to Adam)
            num_steps: Number of calibration steps (batches)
            learning_rate: Learning rate for the optimizer
            device: Device to use (defaults to model's device)
            
        Returns:
            Dictionary of calibrated weights 
        """
        print(f"Starting calibration with {num_steps} steps, lr={learning_rate}")
        
        # Use model's device if not specified
        if device is None:
            device = next(self.parameters()).device
        
        # Set model to training mode
        self.train()
        
        # Default criterion
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()
        
        # Create pruning mask from the pruned weights to maintain sparsity
        weight_masks = {}
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:  # Only for weight matrices
                weight_masks[name] = (param != 0).float()  # 1 where weights remain, 0 where pruned
        
        # Default optimizer with low learning rate
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Fine-tuning loop
        running_loss = 0.0
        processed_steps = 0
        
        for step, (inputs, targets) in enumerate(calib_loader):
            if step >= num_steps:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = self(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Apply masks before step to maintain sparsity pattern
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if name in weight_masks:
                        param.grad *= weight_masks[name]  # Zero-out gradients for pruned weights
            
            optimizer.step()
            
            # Log progress
            running_loss += loss.item()
            processed_steps += 1
            
            if (step + 1) % 10 == 0:
                print(f'Step {step+1}/{num_steps}, Loss: {running_loss/10:.4f}')
                running_loss = 0.0
        
        # Evaluate on the last batch as a sanity check
        self.eval()
        with torch.no_grad():
            outputs = self(inputs)
            _, predicted = outputs.max(1)
            accuracy = predicted.eq(targets).sum().item() / targets.size(0)
        print(f"Calibration completed. Final batch accuracy: {accuracy:.4f}")
        
        # Return the calibrated weights
        weights = {}
        for name, param in self.state_dict().items():
            weights[name] = param.cpu().numpy()
            
        return weights

    def prune_and_calibrate(self, ubatch, calib_loader, keep_ratio=0.9, 
                           calib_steps=100, calib_lr=1e-5, device=None):
        """
        Combined pruning and calibration in one step.
        
        Args:
            ubatch: Batch of input data for pruning calibration
            calib_loader: DataLoader with samples for post-pruning calibration
            keep_ratio: Percentage of weights to keep (0-1)
            calib_steps: Number of calibration steps
            calib_lr: Learning rate for calibration
            device: Device to use (defaults to model's device)
            
        Returns:
            Dictionary of pruned and calibrated weights
        """
        print(f"Starting pruning (keep_ratio={keep_ratio}) followed by calibration")
        
        # First prune the model
        pruned_weights = self.prune_wanda(ubatch, keep_ratio)
        
        # Load the pruned weights
        self.load_state_dict(pruned_weights)
        
        # Then calibrate the pruned model
        calibrated_weights = self.calibrate_weights(
            calib_loader=calib_loader,
            num_steps=calib_steps,
            learning_rate=calib_lr,
            device=device
        )
        
        return calibrated_weights
    
    def prune_iterative_and_calibrate(self, ubatch, calib_loader, final_keep_ratio=0.3, 
                                      prune_steps=3, calib_steps=100, calib_lr=1e-5, 
                                      mini_test_batch=None, device=None):
        """
        Combined iterative pruning and calibration in one step.
        
        Args:
            ubatch: Batch of input data for pruning calibration
            calib_loader: DataLoader with samples for post-pruning calibration
            final_keep_ratio: Final percentage of weights to keep (0-1)
            prune_steps: Number of pruning steps
            calib_steps: Number of calibration steps
            calib_lr: Learning rate for calibration
            mini_test_batch: Optional validation batch for monitoring during pruning
            device: Device to use (defaults to model's device)
            
        Returns:
            Dictionary of pruned and calibrated weights
        """
        print(f"Starting iterative pruning (final_keep_ratio={final_keep_ratio}, steps={prune_steps}) followed by calibration")
        
        # First iteratively prune the model
        pruned_weights = self.prune_wanda_iterative(
            ubatch, 
            final_keep_ratio=final_keep_ratio, 
            steps=prune_steps, 
            mini_test_batch=mini_test_batch
        )
        
        # Load the pruned weights
        self.load_state_dict(pruned_weights)
        
        # Then calibrate the pruned model
        calibrated_weights = self.calibrate_weights(
            calib_loader=calib_loader,
            num_steps=calib_steps,
            learning_rate=calib_lr,
            device=device
        )
        
        return calibrated_weights 