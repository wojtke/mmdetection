_base_ = [
    'cmrcnn-vit.py',
    'coco_224.py',
]


model = dict(
    backbone=dict(
        model_name='vit_small_patch16_224',
        img_size=[224, 224], 
    ),
)