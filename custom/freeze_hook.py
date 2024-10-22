from mmengine.model import is_model_wrapper
from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class FreezeHook(Hook):
    def before_run(self, runner):
        print("FREEEZE! "*10)
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        backbone = model.backbone
        for param in backbone.parameters():
            param.requires_grad = False
