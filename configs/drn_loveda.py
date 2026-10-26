custom_imports = {'imports': ['mamba_vision', 'wsse', 'seg_datasets', 'masked_dice_loss'],
 'allow_failed_imports': False}

default_scope = 'mmseg'

model = {'backbone': {'depths': (1, 3, 8, 4),
              'dim': 80,
              'drop_path_rate': 0.3,
              'in_dim': 32,
              'layer_scale': None,
              'mlp_ratio': 4,
              'norm_layer': 'ln2d',
              'num_heads': (2, 4, 8, 16),
              'out_indices': (0, 1, 2, 3),
              'pretrained': './ckpts/mambavision_tiny_1k.pth.tar',
              'type': 'MM_mamba_vision',
              'window_size': (8, 8, 64, 32)},
 'data_preprocessor': {'bgr_to_rgb': True,
                       'mean': [123.675, 116.28, 103.53],
                       'pad_val': 0,
                       'seg_pad_val': 255,
                       'size_divisor': 32,
                       'std': [58.395, 57.12, 57.375],
                       'type': 'SegDataPreProcessor'},
 'decode_head': {'align_corners': False,
                 'channels': 256,
                 'dropout_ratio': 0.1,
                 'in_channels': [80, 160, 320, 640],
                 'in_index': [0, 1, 2, 3],
                 'loss_decode': [{'loss_weight': 1.0,
                                  'type': 'CrossEntropyLoss',
                                  'use_sigmoid': False},
                                 {'eps': 0.001,
                                  'ignore_index': 255,
                                  'loss_name': 'loss_dice',
                                  'loss_weight': 0.4,
                                  'naive_dice': False,
                                  'type': 'PixelMaskedDiceLoss',
                                  'use_sigmoid': True}],
                 'norm_cfg': {'requires_grad': True, 'type': 'SyncBN'},
                 'num_classes': 7,
                 'type': 'SegformerHead'},
 'neck': {'band_gate': False,
          'fusion': 'sum',
          'in_channels': [80, 160, 320, 640],
          'levels': (0, 1),
          'scan_mode': 'raster',
          'ssm_cfg': {'bidirectional': True, 'chunk': 64, 'd_state': 4, 'dt_limit': (0.0001, 0.1)},
          'streams': 'high_only',
          'type': 'WSSE'},
 'test_cfg': {'mode': 'whole'},
 'train_cfg': {},
 'type': 'EncoderDecoder'}

train_dataloader = {'batch_size': 4,
 'dataset': {'data_prefix': {'img_path': '', 'seg_map_path': ''},
             'data_root': 'data/loveda',
             'pipeline': [{'type': 'LoadImageFromFile'},
                          {'type': 'LoadAnnotations'},
                          {'keep_ratio': True,
                           'ratio_range': (0.5, 2.0),
                           'scale': (2048, 512),
                           'type': 'RandomResize'},
                          {'cat_max_ratio': 0.75, 'crop_size': (512, 512), 'type': 'RandomCrop'},
                          {'prob': 0.5, 'type': 'RandomFlip'},
                          {'type': 'PhotoMetricDistortion'},
                          {'type': 'PackSegInputs'}],
             'split': 'train',
             'type': 'LoveDADatasetRV'},
 'num_workers': 4,
 'persistent_workers': True,
 'sampler': {'shuffle': True, 'type': 'InfiniteSampler'}}

val_dataloader = {'batch_size': 1,
 'dataset': {'data_prefix': {'img_path': '', 'seg_map_path': ''},
             'data_root': 'data/loveda',
             'pipeline': [{'type': 'LoadImageFromFile'},
                          {'keep_ratio': True, 'scale': (1024, 1024), 'type': 'Resize'},
                          {'type': 'LoadAnnotations'},
                          {'type': 'PackSegInputs'}],
             'split': 'val',
             'type': 'LoveDADatasetRV'},
 'num_workers': 4,
 'persistent_workers': True,
 'sampler': {'shuffle': False, 'type': 'DefaultSampler'}}

val_evaluator = {'iou_metrics': ['mIoU', 'mFscore'], 'type': 'IoUMetric'}

train_cfg = {'max_iters': 80000, 'type': 'IterBasedTrainLoop', 'val_interval': 4000}

val_cfg = {'type': 'ValLoop'}

optim_wrapper = {'loss_scale': 'dynamic',
 'optimizer': {'betas': (0.9, 0.999), 'lr': 6e-05, 'type': 'AdamW', 'weight_decay': 0.01},
 'paramwise_cfg': {'custom_keys': {'absolute_pos_embed': {'decay_mult': 0.0},
                                   'head': {'lr_mult': 10.0},
                                   'norm': {'decay_mult': 0.0},
                                   'pos_block': {'decay_mult': 0.0},
                                   'pos_embed': {'decay_mult': 0.0},
                                   'relative_position_bias_table': {'decay_mult': 0.0}}},
 'type': 'AmpOptimWrapper'}

param_scheduler = [{'begin': 0, 'by_epoch': False, 'end': 1500, 'start_factor': 1e-06, 'type': 'LinearLR'},
 {'begin': 1500, 'by_epoch': False, 'end': 80000, 'eta_min': 0.0, 'power': 1.0, 'type': 'PolyLR'}]

default_hooks = {'checkpoint': {'by_epoch': False,
                'interval': 4000,
                'max_keep_ckpts': 3,
                'rule': 'greater',
                'save_best': 'mIoU',
                'type': 'CheckpointHook'},
 'logger': {'interval': 50, 'log_metric_by_epoch': False, 'type': 'LoggerHook'},
 'param_scheduler': {'type': 'ParamSchedulerHook'},
 'sampler_seed': {'type': 'DistSamplerSeedHook'},
 'timer': {'type': 'IterTimerHook'},
 'visualization': {'type': 'SegVisualizationHook'}}

env_cfg = {'cudnn_benchmark': False,
 'dist_cfg': {'backend': 'nccl'},
 'mp_cfg': {'mp_start_method': 'fork', 'opencv_num_threads': 0}}

log_processor = {'by_epoch': False}

test_dataloader = {'batch_size': 1,
 'dataset': {'data_prefix': {'img_path': '', 'seg_map_path': ''},
             'data_root': 'data/loveda',
             'pipeline': [{'type': 'LoadImageFromFile'},
                          {'keep_ratio': True, 'scale': (1024, 1024), 'type': 'Resize'},
                          {'type': 'LoadAnnotations'},
                          {'type': 'PackSegInputs'}],
             'split': 'val',
             'type': 'LoveDADatasetRV'},
 'num_workers': 4,
 'persistent_workers': True,
 'sampler': {'shuffle': False, 'type': 'DefaultSampler'}}

test_evaluator = {'iou_metrics': ['mIoU', 'mFscore'], 'type': 'IoUMetric'}

test_cfg = {'type': 'TestLoop'}

randomness = dict(seed=0)
work_dir = 'work_dirs/drn'
