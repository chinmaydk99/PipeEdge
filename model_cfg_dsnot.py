"""Model configurations and default parameters for DSnoT pruning."""
import logging
from typing import Any, Callable, List, Optional, Tuple
from torch.distributed import rpc as trpc
from torchvision import models
from transformers import AutoConfig
from pipeedge.comm import p2p, rpc
from pipeedge.models import ModuleShard, ModuleShardConfig
from pipeedge.models.cnn import alexnet, resnet
# Import vit_dsnot instead of vit_wanda
from pipeedge.models.transformers import bert, deit, vit_dsnot
import devices

_logger = logging.getLogger(__name__)

_MODEL_CONFIGS = {}

# Modify to handle DSnoT pruned weights file if specified
def _model_cfg_add(name, layers, weights_file, shard_module, pruned_weights_file=''):
    _MODEL_CONFIGS[name] = {
        'name': name,
        'layers': layers,
        'weights_file': weights_file, # Original dense weights
        'shard_module': shard_module,
        'pruned_weights_file': pruned_weights_file # Specific file for DSnoT results
    }

# Transformer blocks can be split 4 ways, e.g., where ViT-Base has 12 layers, we specify 12*4=48
# Use vit_dsnot module and define DSnoT pruned weight filenames
_model_cfg_add('google/vit-base-patch16-224-dsnot', 48, 'ViT-B_16-224.npz',
               vit_dsnot.ViTShardForImageClassification, 'ViT-B_16-224_DSNOT_pruned.npz')
_model_cfg_add('google/vit-large-patch16-224-dsnot', 96, 'ViT-L_16-224.npz',
               vit_dsnot.ViTShardForImageClassification, 'ViT-L_16-224_DSNOT_pruned.npz')
_model_cfg_add('google/vit-huge-patch14-224-in21k-dsnot', 128, 'ViT-H_14.npz',
               vit_dsnot.ViTShardForImageClassification, 'ViT-H_14_DSNOT_pruned.npz')

# Keep other models as they are (assuming they don't use DSnoT for now)
_model_cfg_add('bert-base-uncased', 48, 'BERT-B.npz',
               bert.BertModelShard)
_model_cfg_add('bert-large-uncased', 96, 'BERT-L.npz',
               bert.BertModelShard)
_model_cfg_add('textattack/bert-base-uncased-CoLA', 48, 'BERT-B-CoLA.npz',
               bert.BertShardForSequenceClassification)
_model_cfg_add('facebook/deit-base-distilled-patch16-224', 48, 'DeiT_B_distilled.npz',
               deit.DeiTShardForImageClassification)
_model_cfg_add('facebook/deit-small-distilled-patch16-224', 48, 'DeiT_S_distilled.npz',
               deit.DeiTShardForImageClassification)
_model_cfg_add('facebook/deit-tiny-distilled-patch16-224', 48, 'DeiT_T_distilled.npz',
               deit.DeiTShardForImageClassification)
_model_cfg_add('torchvision/resnet18', 21, 'resnet18.pt',
               resnet.ResNet18ModelShard)
_model_cfg_add('torchvision/resnet34', 37, 'resnet34.pt',
               resnet.ResNet34ModelShard)
_model_cfg_add('torchvision/resnet50', 54, 'resnet50.pt',
               resnet.ResNet50ModelShard)
_model_cfg_add('torchvision/resnet101', 105, 'resnet101.pt',
               resnet.ResNet101ModelShard)
_model_cfg_add('torchvision/alexnet', 5, 'alexnet.pt',
               alexnet.AlexNetModelShard)

def get_model_names() -> List[str]:
    """Get a list of available model names."""
    return list(_MODEL_CONFIGS.keys())

def get_model_dict(model_name: str) -> dict:
    """Get a model's key/value properties - modify at your own risk."""
    return _MODEL_CONFIGS[model_name]

def get_model_layers(model_name: str) -> int:
    """Get a model's layer count."""
    return _MODEL_CONFIGS[model_name]['layers']

def get_model_config(model_name: str, model_file = None) -> Any:
    """Get a model's config."""
    # We'll need more complexity if/when we add support for models not from `transformers`
    base_model_name = model_name.replace('-dsnot', '') # Use base name for config loading

    if base_model_name.split('/')[0] == 'torchvision':
        if base_model_name.split('/')[1].startswith('resnet'):
            config = resnet.ResnetConfig(base_model_name)
        if base_model_name.split('/')[1] == 'alexnet':
            config = alexnet.AlexNetConfig(base_model_name)
    else:
        config = AutoConfig.from_pretrained(base_model_name)
        # Config overrides
        if base_model_name == 'google/vit-huge-patch14-224-in21k':
            # ViT-Huge doesn't include classification, so we have to set this ourselves
            # NOTE: not setting 'id2label' or 'label2id'
            config.num_labels = 21843
    return config

def get_model_default_weights_file(model_name: str) -> str:
    """Get a model's default *original* dense weights file name."""
    return _MODEL_CONFIGS[model_name]['weights_file']

def get_model_pruned_weights_file(model_name: str) -> str:
    """Get the specific file name intended for the pruned weights."""
    return _MODEL_CONFIGS[model_name].get('pruned_weights_file', '') # Return empty if not defined


def save_model_weights_file(model_name: str, model_file: Optional[str]=None) -> None:
    """Save a model's *original* dense weights file by downloading if necessary."""
    # Use base name for downloading original weights
    base_model_name = model_name.replace('-dsnot', '')
    if model_file is None:
        # Get the filename associated with the original dense weights
        model_file = get_model_default_weights_file(model_name)
    # This works b/c all shard implementations have the same save_weights interface
    # Use the correct module (vit_dsnot) but call the static method
    module = _MODEL_CONFIGS[model_name]['shard_module']
    # Pass the base model name to find the correct download URL
    module.save_weights(base_model_name, model_file)

# Modify factory to remove the 'prune' argument as it's handled dynamically now
def module_shard_factory(model_name: str, model_file: Optional[str], layer_start: int,
                         layer_end: int, stage: int) -> ModuleShard:
    """Get a shard instance on the globally-configured `devices.DEVICE`."""
    # This works b/c all shard implementations have the same constructor interface
    if model_file is None:
        # Always load the original dense weights file for initialization
        model_file = get_model_default_weights_file(model_name)
    config = get_model_config(model_name, model_file)
    is_first = layer_start == 1
    is_last = layer_end == get_model_layers(model_name)
    shard_config = ModuleShardConfig(layer_start=layer_start, layer_end=layer_end,
                                     is_first=is_first, is_last=is_last)
    module = _MODEL_CONFIGS[model_name]['shard_module']
    # Pass only config, shard_config, model_file (dense weights)
    shard = module(config, shard_config, model_file)
    _logger.info("======= %s Stage %d =======", module.__name__, stage)
    shard.to(device=devices.DEVICE)
    shard.eval()
    return shard

# Keep RPC and P2P factory functions as they mainly depend on module interface,
# but ensure they call the modified module_shard_factory or construct the module correctly.

def _dist_rpc_pipeline_stage_factory(*args, **kwargs) -> rpc.DistRpcPipelineStage:
    """Get a `rpc.DistRpcPipelineStage` instance on the globally-configured `devices.DEVICE`."""
    stage = rpc.DistRpcPipelineStage(*args, **kwargs)
    stage.module_to(device=devices.DEVICE)
    return stage

def dist_rpc_pipeline_factory(model_name: str, model_file: Optional[str], stage_ranks: List[int],
                              stage_layers: List[Tuple[int, int]], results_to: int,
                              results_cb: Callable[[Any], None]) -> rpc.DistRpcPipeline:
    """Get an RPC pipeline instance."""
    # Always use original dense weights for initialization
    if model_file is None:
        model_file = get_model_default_weights_file(model_name)
    module_cls = _MODEL_CONFIGS[model_name]['shard_module'] # Get the class (e.g., ViTShardForImageClassification)
    stage_rrefs = []
    assert len(stage_ranks) > 0
    assert len(stage_ranks) == len(stage_layers)
    for i, (dst_rank, layers) in enumerate(zip(stage_ranks, stage_layers)):
        config = get_model_config(model_name)
        is_first = i == 0
        is_last = i == len(stage_ranks) - 1
        shard_config = ModuleShardConfig(layer_start=layers[0], layer_end=layers[1],
                                         is_first=is_first, is_last=is_last)
        # Ensure args match the constructor of vit_dsnot module (config, shard_config, model_weights)
        module_args = (config, shard_config, model_file)
        # Pass the actual module class to the factory
        rref = trpc.remote(dst_rank, _dist_rpc_pipeline_stage_factory, args=(module_cls,),
                           kwargs={ 'module_args': module_args })
        # Log the correct module name being used
        trpc.remote(dst_rank, _logger.info,
                    args=("======= %s Stage %d =======", module_cls.__name__, i))
        stage_rrefs.append(rref)
    return rpc.DistRpcPipeline(stage_rrefs, results_to, results_cb)

# P2P factory remains largely the same, relying on the module interface
def dist_p2p_pipeline_stage_factory(stage_ranks: List[int], data_rank: int, rank: int,
                                    stage: Optional[int], module: Optional[ModuleShard],
                                    handle_results_cb: Callable[[Any], None]) \
    -> p2p.DistP2pPipelineStage:
    """Get a P2P pipeline stage instance."""
    if rank == data_rank:
        if stage is None:
            # We're data_rank w/out a module shard
            rank_src = stage_ranks[-1]
            rank_dst = stage_ranks[0]
            work_cb = None
        else:
            # We're simultaneously data_rank and a pipeline stage
            # In this case, the current p2p design requires that we must be the first stage
            if stage != 0:
                raise ValueError(f"Data rank must be stage=0 or stage=None, but stage={stage}")
            # Degenerate case when we're both data_rank and the only stage
            rank_src = stage_ranks[-1] if len(stage_ranks) > 1 else None
            rank_dst = stage_ranks[1] if len(stage_ranks) > 1 else None
            work_cb = module
        # While the handle_results_cb parameter isn't optional, we should assert it anyway.
        # If None, DistP2pPipelineStage would loop results back to its input queue, then the first
        # module shard would try to process the results tensors, which it would fail to unpack.
        # It wouldn't be obvious from the error that the real problem was handle_results_cb=None.
        assert handle_results_cb is not None
        results_cb = handle_results_cb
    elif stage is None:
        # We're completely idle
        rank_src = None
        rank_dst = None
        work_cb = None
        results_cb = None
    else:
        # We're not data_rank, but we have a module shard (possibly first and/or last stage)
        rank_src = data_rank if stage == 0 else stage_ranks[(stage - 1)]
        rank_dst = data_rank if stage == len(stage_ranks) - 1 else stage_ranks[(stage + 1)]
        work_cb = module
        results_cb = None
    return p2p.DistP2pPipelineStage(rank_src, rank_dst, work_cb, results_cb) 