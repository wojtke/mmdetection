_base_ = [
    '../configs/_base_/datasets/coco_instance.py',
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # Same normalization as before
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

train_dataloader = dict(batch_size=8, num_workers=16)
# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',  # New augmentation strategy with policies
         policies=[
             [
                 dict(type='Resize', img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                                (576, 1333), (608, 1333), (640, 1333),
                                                (672, 1333), (704, 1333), (736, 1333),
                                                (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize', img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value', keep_ratio=True),
                 dict(type='RandomCrop', crop_type='absolute_range',
                      crop_size=(384, 600), allow_negative_crop=True),
                 dict(type='Resize', img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                                (576, 1333), (608, 1333), (640, 1333),
                                                (672, 1333), (704, 1333), (736, 1333),
                                                (768, 1333), (800, 1333)],
                      multiscale_mode='value', keep_ratio=True, override=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
