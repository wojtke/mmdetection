_base_ = [
    '../configs/_base_/models/cascade-mask-rcnn_r50_fpn.py',
    'coco_224.py',
    'misc.py'
]

custom_imports = dict(
    imports=['mmpretrain.models', 'custom.freeze_hook'], allow_failed_imports=False
)
custom_hooks = [dict(type='FreezeHook')]



model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.TIMMBackbone',
        model_name='vit_small_patch16_224',
        features_only=True,
        pretrained=True,
        img_size=[224, 224],  # Custom image size | https://github.com/bytedance/ibot/blob/da316d82636a7a7356835ef224b13d5f3ace0489/evaluation/object_detection/configs/cascade_rcnn/vit_small_giou_4conv1f_coco_3x.py#L20
        patch_size=16,  # Patch size for ViT
        embed_dim=384,  # Embedding dimension for transformer
        depth=12,  # Number of transformer layers (depth)
        num_heads=6,  # Number of attention heads
        mlp_ratio=4.,  # MLP ratio
        qkv_bias=True,  # Enable bias for QKV projection
        drop_path_rate=0.1,  # Stochastic depth rate
        out_indices=(3, 5, 7, 11),  # Change in output feature map indices
        dynamic_img_size=False # Enables pos encoding interpolation https://github.com/huggingface/pytorch-image-models/blob/310ffa32c5758474b0a4481e5db1494dd419aa23/timm/models/vision_transformer.py#L657
    ),
    neck=dict(
        type='FPN',  # Keep FPN but modify the input channels
        in_channels=[384, 384, 384, 384],  # Adjust FPN input channels for ViT
        out_channels=256,  # Keep the output channels same as default
        num_outs=5  # Same as default, number of output scales
    ),
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',  # Change from 'Shared2FCBBoxHead' to 'ConvFCBBoxHead'
                num_shared_convs=4,  # Number of shared convolution layers
                num_shared_fcs=1,  # Number of fully connected layers
                conv_out_channels=256,  # Convolution output channels
                norm_cfg=dict(type='SyncBN', requires_grad=True),  # Add SyncBN for normalization
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0),  # Change loss_bbox from 'SmoothL1Loss' to 'GIoULoss'
                reg_class_agnostic=False,  # Change reg_class_agnostic from True to False
                reg_decoded_bbox=True,  # Set reg_decoded_bbox to True
            ),
            dict(
                type='ConvFCBBoxHead',  # Similar changes for the second stage
                num_shared_convs=4,
                num_shared_fcs=1,
                conv_out_channels=256,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
            ),
            dict(
                type='ConvFCBBoxHead',  # Similar changes for the third stage
                num_shared_convs=4,
                num_shared_fcs=1,
                conv_out_channels=256,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
            )
        ]
    ),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_across_levels=False,  # Disable NMS across levels in ViT config
            nms_pre=2000,  # Same as default, pre-NMS filtering
            nms_post=2000,  # Change to include post-NMS filtering
            max_per_img=2000,  # Same as default
            nms=dict(type='nms', iou_threshold=0.7),  # Same as default
            min_bbox_size=0  # Same as default
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,  # Disable NMS across levels in ViT config
            nms_pre=1000,  # Same as default, pre-NMS filtering
            nms_post=1000,  # Change to include post-NMS filtering
            max_per_img=1000,  # Same as default
            nms=dict(type='nms', iou_threshold=0.7),  # Same as default
            min_bbox_size=0  # Same as default
        )
    )
)

