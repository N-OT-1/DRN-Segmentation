import argparse
import os
from pathlib import Path
import sys

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))


def main():
    from mmengine.config import Config
    from mmseg.apis import init_model,inference_model
    import mamba_vision,wsse,seg_datasets,masked_dice_loss
    from PIL import Image
    parser=argparse.ArgumentParser(description='Export an unchanged class-ID segmentation prediction.')
    parser.add_argument('config');parser.add_argument('checkpoint')
    parser.add_argument('--image',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    cfg=Config.fromfile(args.config);cfg.model.backbone.pretrained=None
    model=init_model(cfg,args.checkpoint,device=args.device)
    result=inference_model(model,args.image)
    labels=result.pred_sem_seg.data.squeeze().cpu().numpy().astype('uint8')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(labels).save(args.output)


if __name__=='__main__':main()
