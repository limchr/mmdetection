_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']


# dataset settings
dataset_type = 'CocoDataset'

img_scale = (768, 1280)  # height, width
max_epochs = 300
num_last_epochs = 15
# resume_from = '/home/chris/src/mmdetection/work_dirs/converted_meta2/epoch_35.pth'
resume_from = None
interval = 5

# with open(data_root+'/okutama_action.yaml', 'r') as f:
#     l = f.readlines()
#     for ll in l:
#         if ll.startswith('names:'):
#             class_names = ll[ll.find('[')+1:ll.find(']')].split(',')
#             num_classes = len(class_names)
# f = None
# l = None
# model settings


# change this

# name = 'converted_min'
# class_names = ['Lying','Sitting']

name = 'converted_vanilla'
class_names = ['Calling','Carrying','Drinking','Hand','Hugging','Lying','Pushing/Pulling','Reading','Running','Shaking','Sitting','Standing','Walking']

# name = 'converted_meta2'
# class_names = ['Activity','Movement']

# name = 'converted_meta3'
# class_names = ['Activity','Movement','Resting']

# /change this


data_root = '/home/chris/data/okutama_action/'+name+'/'
# data_root_b1 = '/home/chris/data/okutama_action/'+name+'_b1/'
# data_root_b2 = '/home/chris/data/okutama_action/'+name+'_b2/'

work_dir = 'work_dirs/'+name+'/'
num_classes = len(class_names)



model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,    
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict( 
        type='YOLOXHead', num_classes=num_classes, in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5),

    ),
    init_cfg=dict(type='Pretrained',checkpoint='/home/chris/src/mmdetection/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [



    # # comment out for debugging
    # dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    # dict(
    #     type='RandomAffine',
    #     scaling_ratio_range=(0.1, 2),
    #     border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    # dict(
    #     type='MixUp',
    #     img_scale=img_scale,
    #     ratio_range=(0.8, 1.6),
    #     pad_val=114.0),
    # dict(type='YOLOXHSVRandomAug'),
    # dict(type='RandomFlip', flip_ratio=0.5),
    # # According to the official implementation, multi-scale
    # # training is not considered here but in the
    # # 'mmdet/models/detectors/yolox.py'.





    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=False,
        size_divisor=64,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),





    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'], 
    meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg'))
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'images/',
        classes=class_names,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline
    )


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='Normalize', **img_norm_cfg),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=False,
                size_divisor=64,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])  
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=[
        train_dataset,        
        ],
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline,
        classes=class_names,
        
        
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline,
        classes=class_names,
        
        ))

# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)


# learning policy
lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch CHANGED from 5
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='bbox')
log_config = dict(interval=50)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
