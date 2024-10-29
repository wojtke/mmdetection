import json
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.dist import get_dist_info

def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed"):
        return 0
    elif var_name.startswith("backbone.patch_embed"):
        return 0
    elif var_name.startswith("backbone.blocks"):
        layer_id = int(var_name.split('.')[2])  # Adjust based on model's layer structure
        return layer_id + 1
    else:
        return num_max_layer - 1

@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LayerDecayOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to param groups with layer-wise decay."""
        parameter_groups = {}
        
        # Retrieve settings from `paramwise_cfg`
        num_layers = self.paramwise_cfg.get('num_layers', 12) + 2  # Assumes extra layers
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate', 0.75)
        weight_decay = self.base_wd  # Base weight decay

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters
            
            # Determine group name and weight decay based on param type
            if len(param.shape) == 1 or name.endswith(".bias") or name in ('pos_embed', 'cls_token'):
                group_name = "no_decay"
                this_weight_decay = 0.0
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            # Layer-specific learning rate scaling
            layer_id = get_num_layer_for_vit(name, num_layers)
            group_name = f"layer_{layer_id}_{group_name}"

            if group_name not in parameter_groups:
                scale = layer_decay_rate ** (num_layers - layer_id - 1)
                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale,
                    "lr": scale * self.base_lr
                }

            # Append parameters to the appropriate group
            parameter_groups[group_name]["params"].append(param)

        # Display group info at rank 0 for transparency
        rank, _ = get_dist_info()
        if rank == 0:
            display_groups = {k: {"lr_scale": v["lr_scale"], "weight_decay": v["weight_decay"]}
                              for k, v in parameter_groups.items()}
            print(f"Parameter groups with layer-wise decay: {json.dumps(display_groups, indent=2)}")
        
        params.extend(parameter_groups.values())