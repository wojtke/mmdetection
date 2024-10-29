_base_ = [
    'coco.py',
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize', scale=(224, 224))
             ],
             [
                 dict(type='RandomChoiceResize', scales=[(400, 600), (500, 750), (600, 900)]),
                 dict(type='RandomCrop', crop_type='absolute_range',
                      crop_size=(384, 384), allow_negative_crop=True), 
                 dict(type='Resize', scale=(224, 224))
             ]
         ]),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline = train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(224, 224)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

val_dataloader = dict(dataset=dict(pipeline = test_pipeline))
test_dataloader = dict(dataset=dict(pipeline = test_pipeline))