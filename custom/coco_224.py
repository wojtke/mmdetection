_base_ = [
    '../configs/_base_/datasets/coco_instance.py',
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # Same normalization as before
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

#train_dataloader = dict(batch_size=8, num_workers=16)

# Training pipeline with augmentation
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize', scale=(224, 224))  # Keep the original aspect ratio here
             ],
             [
                 dict(type='RandomChoiceResize', scales=[(400, 600), (500, 750), (600, 900)]),
                 dict(type='RandomCrop', crop_type='absolute_range',
                      crop_size=(384, 384), allow_negative_crop=True),  # Crop with minimal distortion in aspect ratio
                 dict(type='Resize', scale=(224, 224), override=True)  # Final resize with aspect ratio kept
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
train_dataloader = dict(
        dataset=dict(
            pipeline=train_pipeline,
        ),
        batch_size=8,
        num_workers=16,
)

# Validation pipeline without augmentation for consistent evaluation
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(224, 224)),  # Resize to 224x224 while keeping the aspect ratio
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),  # Padding to match model requirements
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

val_dataloader = dict(
        dataset=dict(
             pipeline=val_pipeline,
        ),
        batch_size=8,
        num_workers=16,
)
