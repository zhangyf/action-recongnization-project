# model settings
model = dict(
    type='TSN3D',
    backbone=dict(
        type='ResNet_I3D_SlowFast',
        pretrained_slow=None,
        pretrained_fast=None,
        depth=50,
        alpha=8,
        beta_inv=8,
        num_stages=4,
        out_indices=[3],
        frozen_stages=-1,
        slow_inflate_freq=(0, 0, 1, 1),
        fast_inflate_freq=(1, 1, 1, 1),
        inflate_style='3x1x1',
        bn_eval=False,
        partial_bn=False,
        style='pytorch'),
    spatial_temporal_module=dict(
        type='SlowFastSpatialTemporalModule',
        adaptive_pool=True,
        spatial_type='avg',
        temporal_size=1,
        spatial_size=1),
    segmental_consensus=dict(
        type='SimpleConsensus',
        consensus_type='avg'),
    cls_head=dict(
        type='ClsHead',
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.5,
        in_channels=2304,  # 2048+256
        num_classes=400))
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'RawFramesDataset'
data_root_val = 'data/kinetics400/rawframes_val/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

data = dict(
    test=dict(
        type=dataset_type,
        ann_file='data/kinetics400/kinetics400_val_list_rawframes.txt',
        img_prefix=data_root_val,
        img_norm_cfg=img_norm_cfg,
        input_format="NCTHW",
        num_segments=10,
        new_length=32,
        new_step=2,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=256,
        input_size=256,
        div_255=False,
        flip_ratio=0,
        resize_keep_ratio=True,
        oversample='three_crop',
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=False,
        test_mode=True)
)

dist_params = dict(backend='nccl', port=16187)
