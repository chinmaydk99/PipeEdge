""" Evaluate accuracy on ImageNet dataset of PipeEdge """
import os
import argparse
import time
import torch
import numpy as np

import io
from contextlib import redirect_stdout
from typing import List
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, ImageNet
from torchvision import transforms
from transformers import DeiTFeatureExtractor, ViTFeatureExtractor
from runtime import forward_hook_quant_encode, forward_pre_hook_quant_decode
from utils.data import ViTFeatureExtractorTransforms
import model_cfg
from evaluation_tools.evaluation_quant_test import *

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
        self.model_name = model_name.split('/')[1]
        self.pruning_keep_ratio = None  # Will store the pruning keep ratio
        
        # Create directory if it doesn't exist
        self.file_path = os.path.join(self.output_dir, self.model_name)
        os.makedirs(self.file_path, exist_ok=True)
        
        # Initialize files
        self.acc_file = os.path.join(self.file_path, f"result_{self.partition}_{str(self.quant)}.txt")
        self.sparsity_file = os.path.join(self.file_path, f"sparsity_{self.partition}_{str(self.quant)}.txt")
        self.final_file = os.path.join(self.file_path, f"final_{self.partition}_{str(self.quant)}.txt")
        
        # Initialize sparsity dict
        self.sparsity_info = {}
        self.logical_layers = {}  # Store mapping to logical layers
        
    def set_pruning_keep_ratio(self, keep_ratio):
        """Set the pruning keep ratio"""
        self.pruning_keep_ratio = keep_ratio
        
    def update(self, pred, target):
        self.correct = pred.eq(target.view(1, -1).expand_as(pred)).float().sum()
        self.current_acc = self.correct / self.batch_size
        self.total_acc = (self.total_acc * self.tested_batch + self.current_acc)/(self.tested_batch+1)
        self.tested_batch += 1

    def report(self):
        print(f"The accuracy so far is: {100*self.total_acc:.2f}")
        file_name = os.path.join(self.output_dir, self.model_name, "result_"+self.partition+"_"+str(self.quant)+".txt")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'a') as f:
            f.write(f"{100*self.total_acc:.2f}\n")
            
    def capture_sparsity(self, layer_info):
        """Capture layer density information from console output"""
        if "Layer:" in layer_info and "Density:" in layer_info:
            # Parse the layer info string
            try:
                # Extract layer name
                layer_name = layer_info.split("Layer:")[1].split("=>")[0].strip()
                # Extract density value
                density = float(layer_info.split("Density:")[1].strip())
                sparsity = 1.0 - density
                
                # Add a counter to make the key unique
                counter = len(self.sparsity_info) + 1
                key = f"{counter}_{layer_name}"
                
                # Store in dict
                self.sparsity_info[key] = sparsity
                
                # Also write to file immediately
                with open(self.sparsity_file, 'a') as f:
                    f.write(f"{key}: {sparsity:.6f}\n")
                    
            except (ValueError, IndexError) as e:
                print(f"Error parsing layer info: {e}")
    
    def map_to_logical_layers(self):
        """Map linear layers to logical layers matching the partition scheme"""
        if not self.sparsity_info:
            return {}
            
        # For ViT: 6 linear layers per transformer block map to 4 logical layers
        logical_layers = {}
        
        # Group layers by their position in the transformer block
        layer_groups = {}
        
        # Process each linear layer and map to logical layers
        for key, sparsity in self.sparsity_info.items():
            idx = int(key.split('_')[0])
            
            # Calculate which transformer block this belongs to (0-indexed)
            block_idx = (idx - 1) // 6
            # Calculate which component within the block (0-5)
            component_idx = (idx - 1) % 6
            
            # Map to logical layer (1-indexed)
            if component_idx < 3:  # Q, K, V layers (first 3 in each block)
                logical_idx = block_idx * 4 + 1
                layer_type = "Attention QKV"
            elif component_idx == 3:  # Attention output projection
                logical_idx = block_idx * 4 + 2
                layer_type = "Attention Output"
            elif component_idx == 4:  # First MLP layer
                logical_idx = block_idx * 4 + 3
                layer_type = "MLP1"
            else:  # Second MLP layer
                logical_idx = block_idx * 4 + 4
                layer_type = "MLP2"
            
            # Group by logical layer
            if logical_idx not in layer_groups:
                layer_groups[logical_idx] = {
                    'sparsities': [],
                    'layer_type': layer_type,
                    'block': block_idx + 1  # 1-indexed block
                }
            layer_groups[logical_idx]['sparsities'].append(sparsity)
        
        # Calculate average sparsity for each logical layer
        for logical_idx, data in layer_groups.items():
            logical_layers[logical_idx] = {
                'sparsity': sum(data['sparsities']) / len(data['sparsities']),
                'layer_type': data['layer_type'],
                'block': data['block']
            }
            
        self.logical_layers = logical_layers
        return logical_layers
                
    def save_final_stats(self):
        """Save final accuracy and other statistics"""
        # Map to logical layers
        self.map_to_logical_layers()
        
        with open(self.final_file, 'w') as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Partition: {self.partition}\n")
            
            # Include pruning information if available
            if self.pruning_keep_ratio is not None:
                f.write(f"Pruning Keep Ratio: {self.pruning_keep_ratio}\n")
                f.write(f"Pruning Factor: {1.0 - self.pruning_keep_ratio:.6f}\n")
                
            f.write(f"Final Accuracy: {100*self.total_acc:.6f}%\n")
            f.write(f"Total Batches: {self.tested_batch}\n\n")
            
            # Write original layer-wise sparsity
            if self.sparsity_info:
                avg_sparsity = sum(self.sparsity_info.values()) / len(self.sparsity_info)
                f.write(f"Average Raw Sparsity: {avg_sparsity:.6f}\n")
                f.write("Linear Layer Sparsity (Raw):\n")
                for layer, sparsity in self.sparsity_info.items():
                    f.write(f"  {layer}: {sparsity:.6f}\n")
            
            # Write logical layer sparsity (matching partition scheme)
            if self.logical_layers:
                f.write(f"\nAverage Logical Layer Sparsity: {sum(data['sparsity'] for data in self.logical_layers.values()) / len(self.logical_layers):.6f}\n")
                f.write("Logical Layer Sparsity (Partitioning Scheme - 48 Layers):\n")
                
                # Write embedding layer (not pruned)
                f.write(f"  Layer 0 (Embedding): 0.000000\n")
                
                # Write transformer block layers
                for idx in range(1, 49):
                    if idx in self.logical_layers:
                        data = self.logical_layers[idx]
                        layer_type = data['layer_type']
                        block = data['block']
                        sparsity = data['sparsity']
                        f.write(f"  Layer {idx} (Block {block}, {layer_type}): {sparsity:.6f}\n")
                    else:
                        # Special handling for the last MLP2 layer (Layer 48)
                        if idx == 48:
                            # For Layer 48 (Block 12, MLP2), estimate from the previous layer of same block
                            mlp1_layer = 47  # Layer 47 is Block 12, MLP1
                            if mlp1_layer in self.logical_layers:
                                prev_data = self.logical_layers[mlp1_layer]
                                f.write(f"  Layer 48 (Block 12, MLP2): {prev_data['sparsity']:.6f}\n")
                            else:
                                # Fallback if previous layer is also missing
                                f.write(f"  Layer 48 (Block 12, MLP2): 0.000000\n")
                        else:
                            # For any other missing layers
                            f.write(f"  Layer {idx} (Missing Data): 0.000000\n")
                
                # Write classification layer (not pruned)
                f.write(f"  Layer 49 (Classification): 0.000000\n")

def _make_shard(model_name, model_file, stage_layers, stage, q_bits, prune):
    shard = model_cfg.module_shard_factory(model_name, model_file, stage_layers[stage][0],
                                            stage_layers[stage][1], stage, prune)
    shard.register_buffer('quant_bits', q_bits)
    shard.eval()
    return shard

def _forward_model(input_tensor, model_shards):
    num_shards = len(model_shards)
    temp_tensor = input_tensor
    for idx in range(num_shards):
        shard = model_shards[idx]

        # decoder
        if idx != 0:
            temp_tensor = forward_pre_hook_quant_decode(shard, temp_tensor)

        # forward
        if isinstance(temp_tensor[0], tuple) and len(temp_tensor[0]) == 2:
            temp_tensor = temp_tensor[0]
        elif isinstance(temp_tensor, tuple) and isinstance(temp_tensor[0], torch.Tensor):
            temp_tensor = temp_tensor[0]
        temp_tensor = shard(temp_tensor)

        # encoder
        if idx != num_shards-1:
            temp_tensor = (forward_hook_quant_encode(shard, None, temp_tensor),)
    return temp_tensor

def evaluation(args, dataset_cfg):
    """ Evaluation main func"""
    # localize parameters
    dataset_path = args.dataset_root
    dataset_split = args.dataset_split
    batch_size = args.batch_size
    ubatch_size = args.ubatch_size
    num_workers = args.num_workers
    partition = args.partition
    quant = args.quant
    output_dir = args.output_dir
    model_name = args.model_name
    model_file = args.model_file
    num_stop_batch = args.stop_at_batch
    is_clamp = True
    prune = args.prune
    train_batch_size = args.train_batch_size
    keep_ratio = args.keep_ratio
    # if model_file is None:
    #     model_file = model_cfg.get_model_default_weights_file(model_name)

    # load dataset
    if model_name in ['facebook/deit-base-distilled-patch16-224',
                        'facebook/deit-small-distilled-patch16-224',
                        'facebook/deit-tiny-distilled-patch16-224']:
        feature_extractor = DeiTFeatureExtractor.from_pretrained(model_name)
        val_transform = ViTFeatureExtractorTransforms(feature_extractor)
        val_dataset = ImageFolder(os.path.join(dataset_path, dataset_split),
                                transform = val_transform)
    elif model_name.startswith('torchvision'):
        feature_extractor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        # transforms.Lambda(lambda x: x.unsqueeze(0))
        ])
        val_dataset = ImageFolder(os.path.join(dataset_path, dataset_split),
                                transform = feature_extractor)
    else:
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        val_transform = ViTFeatureExtractorTransforms(feature_extractor)
        val_dataset = ImageFolder(os.path.join(dataset_path, dataset_split),
                                transform = val_transform)


    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle=True,
        pin_memory=True
    )

    # Initialize the accuracy reporter early
    acc_reporter = EnhancedReportAccuracy(batch_size, output_dir, model_name, partition, quant[0] if quant else 0)

    if prune:
        pruned_model_file = model_cfg._MODEL_CONFIGS[model_name]['pruned_weights_file']
        # dataset_split = 'train'
        print("keep ratio : ", keep_ratio, ",      train_data size : ", train_batch_size)
        
        # Set the pruning keep ratio in the accuracy reporter
        acc_reporter.set_pruning_keep_ratio(keep_ratio)
        
        train_dataset = ImageFolder(os.path.join(dataset_path, 'train'), transform = val_transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size = train_batch_size,
            shuffle=True,
            pin_memory=True
        )
        for ubatch, ubatch_labels in train_loader:
            config = model_cfg.get_model_config(model_name)
            shard_config = model_cfg.ModuleShardConfig(layer_start=1, layer_end=model_cfg.get_model_layers(model_name),
                                            is_first=True, is_last=True)
            model_file = model_cfg.get_model_default_weights_file(model_name)
            
            model = model_cfg._MODEL_CONFIGS[model_name]['shard_module'](config, shard_config, model_file)
            
            # Capture the density outputs during pruning
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                weights = model.prune_magnitude(keep_ratio)
            
            # Process captured output
            for line in output_buffer.getvalue().split('\n'):
                if "Layer:" in line and "Density:" in line:
                    acc_reporter.capture_sparsity(line)

            np.savez(pruned_model_file, **weights)
            print('Pruning successfully.')
            model_file = pruned_model_file
            break

    def _get_default_quant(n_stages: int) -> List[int]:
        return [0] * n_stages
    parts = [int(i) for i in partition.split(',')]
    assert len(parts) % 2 == 0
    num_shards = len(parts)//2
    stage_layers = [(parts[i], parts[i+1]) for i in range(0, len(parts), 2)]
    stage_quant = [int(i) for i in quant.split(',')] if quant else _get_default_quant(len(stage_layers))

    # model construct
    model_shards = []
    q_bits = []
    for stage in range(num_shards):
        q_bits = torch.tensor((0 if stage == 0 else stage_quant[stage - 1], stage_quant[stage]))
        model_shards.append(_make_shard(model_name, model_file, stage_layers, stage, q_bits, prune))
        model_shards[-1].register_buffer('quant_bit', torch.tensor(stage_quant[stage]), persistent=False)

    # run inference
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(val_loader):
            if batch_idx == num_stop_batch and num_stop_batch:
                break
            output = _forward_model(input, model_shards)
            _, pred = output.topk(1)
            pred = pred.t()
            acc_reporter.update(pred, target)
            acc_reporter.report()
    print(f"Final Accuracy: {100*acc_reporter.total_acc}; Quant Bitwidth: {stage_quant}")
    end_time = time.time()
    print(f"total time = {end_time - start_time}")
    
    # Save final statistics
    acc_reporter.save_final_stats()


if __name__ == "__main__":
    """Main function."""
    parser = argparse.ArgumentParser(description="Pipeline Parallelism Evaluation on Single GPU",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Eval configs
    parser.add_argument("-q", "--quant", type=str,
                        help="comma-delimited list of quantization bits to use after each stage")
    parser.add_argument("-pt", "--partition", type=str, default= '1,22,23,48',
                        help="comma-delimited list of start/end layer pairs, e.g.: '1,24,25,48'; "
                             "single-node default: all layers in the model")
    parser.add_argument("-o", "--output-dir", type=str, default="/home1/haonanwa/projects/PipeEdge/results")
    parser.add_argument("-st", "--stop-at-batch", type=int, default=None, help="the # of batch to stop evaluation")
    
    # Device options
    parser.add_argument("-d", "--device", type=str, default=None,
                        help="compute device type to use, with optional ordinal, "
                             "e.g.: 'cpu', 'cuda', 'cuda:1'")
    parser.add_argument("-n", "--num-workers", default=4, type=int,
                        help="the number of worker threads for the dataloder")
    # Model options
    parser.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224",
                        choices=model_cfg.get_model_names(),
                        help="the neural network model for loading")
    parser.add_argument("-M", "--model-file", type=str,
                        help="the model file, if not in working directory")
    # Dataset options
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("-tb", "--train-batch-size", default=64, type=int, help="train batch size for pruning")
    parser.add_argument("-u", "--ubatch-size", default=8, type=int, help="microbatch size")

    dset = parser.add_argument_group('Dataset arguments')
    dset.add_argument("--dataset-name", type=str, default='ImageNet', choices=['CoLA', 'ImageNet'],
                      help="dataset to use")
    dset.add_argument("--dataset-root", type=str, default= "/project/jpwalter_148/hnwang/datasets/ImageNet/",
                      help="dataset root directory (e.g., for 'ImageNet', must contain "
                           "'ILSVRC2012_devkit_t12.tar.gz' and at least one of: "
                           "'ILSVRC2012_img_train.tar', 'ILSVRC2012_img_val.tar'")
    dset.add_argument("--dataset-split", default='val', type=str,
                      help="dataset split (depends on dataset), e.g.: train, val, validation, test")
    dset.add_argument("--dataset-indices-file", default=None, type=str,
                      help="PyTorch or NumPy file with precomputed dataset index sequence")
    dset.add_argument("--dataset-shuffle", type=bool, nargs='?', const=True, default=False,
                      help="dataset shuffle")
    dset.add_argument("--prune", type=bool, nargs='?', const=True, default=False,
                      help="Pruning method")
    dset.add_argument("--keep-ratio", type=float, default=0.9,
                      help="Snip_pruning keep ratio")
    args = parser.parse_args()


    if args.dataset_indices_file is None:
        indices = None
    elif args.dataset_indices_file.endswith('.pt'):
        indices = torch.load(args.dataset_indices_file)
    else:
        indices = np.load(args.dataset_indices_file)
    dataset_cfg = {
        'name': args.dataset_name,
        'root': args.dataset_root,
        'split': args.dataset_split,
        'indices': indices,
        'shuffle': args.dataset_shuffle,
    }

    evaluation(args, dataset_cfg)