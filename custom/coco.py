_base_ = [
    '../configs/_base_/datasets/coco_instance.py',
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AutoAugment',  
        policies=[
            [
                dict(type='RandomChoiceResize', scales=[
                    (480, 1333), (512, 1333), (544, 1333), 
                    (576, 1333), (608, 1333), (640, 1333),
                    (672, 1333), (704, 1333), (736, 1333), 
                    (768, 1333), (800, 1333)
                    ], keep_ratio=True)
            ],
            [
                dict(type='RandomChoiceResize', scales=[
                    (400, 1333), (500, 1333), (600, 1333)
                    ], keep_ratio=True),
                dict(type='RandomCrop', crop_type='absolute_range',
                    crop_size=(384, 600), allow_negative_crop=True),
                dict(type='RandomChoiceResize', scales=[
                    (480, 1333), (512, 1333), (544, 1333), 
                    (576, 1333), (608, 1333), (640, 1333),
                    (672, 1333), (704, 1333), (736, 1333), 
                    (768, 1333), (800, 1333)
                    ], keep_ratio=True)
            ]
        ]),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=8,  
    num_workers=16,
    persistent_workers=True,
    dataset=dict(
        pipeline = train_pipeline
    )
)