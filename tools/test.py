# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmseg.utils import register_all_modules


def main():
    parser = argparse.ArgumentParser(description='Evaluate DRN segmentation.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--work-dir')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    cfg.model.backbone.pretrained = None
    cfg.load_from = args.checkpoint
    Runner.from_cfg(cfg).test()


if __name__ == '__main__':
    main()
