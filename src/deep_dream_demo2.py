import warnings
from pathlib import Path

import mmcv
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
import shutil
import time


import numpy as np
import torch
import cv2 as cv



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file', default='/home/chris/data/okutama_action/samples/3.jpg')
    parser.add_argument('--config', help='Config file', default='configs/yolox/okutama.py')
    parser.add_argument('--checkpoint', help='Checkpoint file', default='work_dirs/converted_vanilla/latest.pth')
    parser.add_argument('--out-file', default='src/demo_result.jpg', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args



def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmcv.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if 'pretrained' in config.model:
        config.model.pretrained = None
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    if device == 'npu':
        from mmcv.device.npu import NPUDataParallel
        model = NPUDataParallel(model)
        model.cfg = config

    return model



def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0,
                       title='result',
                       wait_time=5,
                       palette=None,
                       out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param. Default: 0.
        palette (str or tuple(int) or :obj:`Color`): Color.
            The tuple of color should be in BGR order.
        out_file (str or None): The path to write the image.
            Default: None.
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        win_name=title,
        bbox_color=palette,
        text_color=(200, 200, 200),
        mask_color=palette,
        out_file=out_file)




def load_dataset(ds,model):
    val_dataset = build_dataset(ds, dict(test_mode=False))

    val_dataloader_default_args = dict(
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False,
        persistent_workers=False)

    val_dataloader_args = {
        **val_dataloader_default_args,
        **model.cfg.data.get('val_dataloader', {})
    }

    val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
    
    return val_dataset, val_dataloader


def get_ds_sample(ds,idx):
    img = ds[idx]
    ann = ds.get_ann_info(idx)
    # return img, ann
    return img['img'][0].data, ann['bboxes'], ann['labels']



def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    ds, dl = load_dataset(model.cfg.data.val,model)

    sample_idx = 0

    samp = ds[sample_idx]
    ann = ds.get_ann_info(sample_idx)

    fn = samp['img_metas'][0].data['filename']
    dd = dict(img_prefix=None, img_info=dict(filename=fn))
    
   
    testpi = Compose(model.cfg.data.test.pipeline)

    a = testpi(dd)
    a['img'] = collate(a['img'], samples_per_gpu=1)

    data = dict()
    data['img_metas'] = [img_metas.data for img_metas in a['img_metas']]
    data['imgs'] = [img.data for img in a['img'].data]

    device = next(model.parameters()).device
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]

    with torch.no_grad():
        interm_results = model.extract_feat(data['imgs'][0])


    img = torch.ones_like(a['img'].data[0])*128
    if next(model.parameters()).is_cuda:
        img = scatter(img, [device])[0]

    from src.train_dcgan import Discriminator
    netD = Discriminator(1, nc=3, ndf=64).to(device)
    netD.load_state_dict(torch.load('src/out/dcgan/netD_epoch_20.pth'))


    for i in range(50000):
        model.forward_dream(img, interm_results, lr=0.003, gan=netD, ratio=0.5)

        if i % 100 == 0:
            with torch.no_grad():   
                results = model.simple_test(img, img_metas=data['img_metas'])


            # resultsli = []
            # for ri,rbb in enumerate(results[0]):
            #     for rb in rbb:
            #         resultsli.append(rb + [ri])



            imshowp = np.array(img.detach().cpu()).squeeze().transpose(1,2,0)


            show_result_pyplot(model,imshowp,result=results[0],out_file='src/out/deep_detection_dream_result.jpg')
            # import matplotlib.pyplot as plt
            # plt.show()










    # imgs = ['/home/chris/data/okutama_action/samples/3.jpg']



    # model.cfg.data.test.pipeline = replace_ImageToTensor(model.cfg.data.test.pipeline)
    # test_pipeline = Compose(model.cfg.data.test.pipeline)

    # datas = []
    # for img in imgs:
    #     # prepare data
    #     if isinstance(img, np.ndarray):
    #         # directly add img
    #         data = dict(img=img)
    #     else:
    #         # add information into dict
    #         data = dict(img_info=dict(filename=img), img_prefix=None)
    #     # build the data pipeline
    #     data = test_pipeline(data)
    #     datas.append(data)

    # data = collate(datas, samples_per_gpu=len(imgs))
    # # just get the actual data from DataContainer
    # data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    # data['img'] = [img.data[0] for img in data['img']]
    # if next(model.parameters()).is_cuda:
    #     # scatter to specified GPU
    #     data = scatter(data, [DEVICE])[0]
    # else:
    #     for m in model.modules():
    #         assert not isinstance(
    #             m, RoIPool
    #         ), 'CPU inference with RoIPool is not supported currently.'



    
    
    # # test a single image
    # result = inference_detector(model, args.img)
    
    
    
    
    
    # # show the results
    # show_result_pyplot(
    #     model,
    #     args.img,
    #     result,
    #     palette=args.palette,
    #     score_thr=args.score_thr,
    #     out_file=args.out_file)
    
    
    
    
    # img = data['img'][0]
    # # forward the model
    # import matplotlib.pyplot as plt
    # for i in range(1000):
    #     print(i)
    #     img = model.forward_dream(img)


    # plt.imshow(img.cpu().detach().numpy().squeeze().transpose((1,2,0))/255)
    # plt.savefig('src/testi.png')






if __name__ == '__main__':
    args = parse_args()
    main(args)
