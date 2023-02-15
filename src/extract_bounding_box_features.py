import os
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmdet.apis import inference_detector
from mmdet.datasets import build_dataset, build_dataloader

from argparse import ArgumentParser

import os
import argparse

import numpy as np
import torch

from src.deep_dream_demo2 import init_detector, show_result_pyplot



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU


# Just dataset of one directory full of unlabeled images
class unsupervised_ds(dset.ImageFolder):
    def __init__(
        self,
        img_dir,
        transform = None,
        target_transform = None,
    ):
        self.img_folder = img_dir[img_dir.rfind('/')+1:]
        super().__init__(img_dir[:img_dir.rfind('/')],transform=transform,target_transform=target_transform)

    def find_classes(self, directory):
        return (self.img_folder,{self.img_folder:0})

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', help='Config file', default='configs/yolox/coco.py')
    parser.add_argument('--checkpoint', help='Checkpoint file', default='configs/yolox/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth')
    parser.add_argument('--imgdir', help='Image directory', default='data/coco/train2017')
    parser.add_argument('--outdir', default='src/out/bbox_feats_extracted', help='Path to output dir')
    parser.add_argument('--imgheight', type=int, default=640, help='Image heights for YOLOX')
    parser.add_argument('--imgwidth', type=int, default=640, help='Image width for YOLOX')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args

def main(args):
    # dataset of all images within args.imgdir (e.g. coco data set image directory)
    dataset = unsupervised_ds(img_dir=args.imgdir,
                        transform=transforms.Compose([
                            transforms.Resize((args.imgheight,args.imgwidth)),
                            transforms.CenterCrop((args.imgheight,args.imgwidth)),
                            transforms.ToTensor(),
                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    dl = len(dataset)
    
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=DEVICE)
    
    # create out directory if not exist
    try:
        os.makedirs(args.outdir)
    except OSError:
        pass

    # iterate over dataset
    for xi,x in enumerate(dataset):
        print('processing sample %d/%d' % (xi+1,dl))
        x = x[0][None] * 255
        x = x[:,[2,1,0],:,:]
        x = x.cuda()
        
        # get grid cells intermediate output of YOLOX
        gc = model.forward_grid_cells(x)

        # retrieve result from GPU and stack it into 3 arrays of fine, medium and coarse granularity: The order of those arrays are: C, X,Y,W,H, [P] (class probability array)
        fine_grid = np.concatenate([gc[2][0].cpu().detach().numpy(),gc[1][0].cpu().detach().numpy(),gc[0][0].cpu().detach().numpy()],axis=1)
        medium_grid = np.concatenate([gc[2][1].cpu().detach().numpy(),gc[1][1].cpu().detach().numpy(),gc[0][1].cpu().detach().numpy()],axis=1)
        coarse_grid = np.concatenate([gc[2][2].cpu().detach().numpy(),gc[1][2].cpu().detach().numpy(),gc[0][2].cpu().detach().numpy()],axis=1)

        # dump the bbox result into separate file, with imagefilename and ending .pkl in outdir
        with open(os.path.join(args.outdir,os.path.basename(dataset.samples[xi][0])+'.pkl'),'wb') as f:
            pkl.dump([fine_grid,medium_grid,coarse_grid],f)

        # empty cache
        torch.cuda.empty_cache()

        #
        # # sanity check if the network is actually detecting reasonable things
        #
        # result = model.simple_test(x,img_metas=[{'scale_factor':1}])
        # imshowp = np.array(x.detach().cpu()).squeeze().transpose(1,2,0)
        # show_result_pyplot(model,imshowp,result=result[0],out_file='detection_result.jpg')


if __name__ == '__main__':
    args = parse_args()
    main(args)