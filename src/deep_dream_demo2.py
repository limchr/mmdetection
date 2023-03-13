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

from PIL import Image


import numpy as np
import torch
import cv2 as cv

import torchvision.datasets as dset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

sthr = 0.02
from helper import setup_clean_directory


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file', default='/home/chris/data/okutama_action/samples/3.jpg')
    parser.add_argument('--config', help='Config file', default='configs/yolox/coco.py')
    parser.add_argument('--checkpoint', help='Checkpoint file', default='configs/yolox/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth')
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
    return model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=False,
        wait_time=wait_time,
        win_name=title,
        bbox_color=palette,
        text_color=(200, 200, 200),
        mask_color=palette,
        out_file=None)




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


def show_color_hist(img,nbins=50,outfile='src/out/colorhist.png'):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(img[:,:,0].flatten(), color = "red", ec="red",bins=nbins,alpha=0.3)
    plt.hist(img[:,:,1].flatten(), color = "green", ec="green",bins=nbins,alpha=0.3)
    plt.hist(img[:,:,2].flatten(), color = "blue", ec="blue",bins=nbins,alpha=0.3)
    plt.savefig(outfile)


def norm_contrast(img):
    pt = 0.1
    ip = np.percentile(img,q=[pt,100-pt])
    norm = (img-ip[0])*255/(ip[1]-ip[0])
    return norm

def ten2arr(ten):
    return np.array(ten.detach().cpu()).squeeze().transpose(1,2,0)

def save_img(arr,path,resize=None):
    im = Image.fromarray(np.array(arr,dtype=np.uint8))
    if not resize is None:
        im = im.resize(resize,Image.Resampling.LANCZOS)
    im.save(path)

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
import torchvision.transforms as transforms



def param_visu(args):
    out_dir = 'src/out/param_visu/'
    img_dir = 'src/data/param_plot'
    setup_clean_directory(out_dir)

    num_circles = 7000
    lr_decay = num_circles*0.95
    
    lrs = [0.01, 0.1, 1.0, 10.0]
    amps = [1, 10, 100]
    opt = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    # tva = [0.05, 0.33, 0.66, 0.95]
    # bnl = 0

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    device = next(model.parameters()).device
    model.init_dream()

    ds = unsupervised_ds(img_dir=img_dir,
                        transform=transforms.Compose([
                            transforms.Resize((640,640)),
                            transforms.CenterCrop((640,640)),
                            transforms.ToTensor(),
                        ]))
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)

    for si, sample in enumerate(dl):
        gs = sample[0] * 255
        gsc = gs.cuda()
        samp = gs.data.numpy().squeeze().transpose(1,2,0)

        with torch.no_grad():
            interm_results = model.forward_dummy(gsc)
        results_orig = model.simple_test(gsc,img_metas=None)
        save_img(samp,'src/out/param_visu/original_%d.jpg'%(si,))
        bboxes = show_result_pyplot(model,samp,result=results_orig[0],score_thr=sthr)
        save_img(bboxes,'src/out/param_visu/original_bboxes_%d.jpg'%(si,))



        for lr in lrs:
            for amp in amps:
                    for op in opt:
                        tv = 1-op
                        dlr = lr
                        img = torch.ones_like(gsc)*128
                        if next(model.parameters()).is_cuda:
                            img = scatter(img, [device])[0]
                            
                        for i in range(num_circles):
                            if i>0 and i%lr_decay==0:
                                dlr *= 0.1
                                print('LR DECAY: '+str(dlr))
                            model.simple_dream(img, ratios=[op,tv], target_feats=interm_results, lr=dlr, amp=amp)


                        # loss_count = 0
                        # last_loss = 10000
                        # for i in range(num_circles):
                        #     if i>0 and i%lr_decay==0:
                        #         dlr *= 0.1
                        #         print('LR DECAY: '+str(dlr))
                        #     imgr, loss = model.simple_dream(img, ratios=[op,tv], target_feats=interm_results, lr=dlr, amp=amp)
                        #     if loss >= last_loss:
                        #         loss_count += 1
                        #     else:
                        #         loss_count = 0
                        #     last_loss = loss
                        #     if loss_count > 20:
                        #         break
                          

                
                        with torch.no_grad():   
                            results = model.simple_test(img, img_metas=None)
                        norm = norm_contrast(ten2arr(img))
                        bboxes = show_result_pyplot(model,norm,result=results[0],score_thr=sthr)
                        rrso = np.concatenate([samp,norm,bboxes],axis=1)
                        save_img(norm,'src/out/param_visu/plain_%d_%f_%f_%f_%d.jpg'%(si,op,tv,lr,amp))
                        save_img(bboxes,'src/out/param_visu/bboxes_%d_%f_%f_%f_%d.jpg'%(si,op,tv,lr,amp))
                        save_img(rrso,'src/out/param_visu/combined_%d_%f_%f_%f_%d.jpg'%(si,op,tv,lr,amp))

def optim_visu(args):
    out_dir = 'src/out/optim_visu/'
    img_dir = 'data/optim_visu/'
    setup_clean_directory(out_dir)

    num_circles = 10000
    lr_decay = num_circles//4
    
    lr = 10.0
    amp = 1
    
    rpl = 0.6
    tvl = 0.4
    bnl = 0.0

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    device = next(model.parameters()).device
    model.init_dream()

    ds = unsupervised_ds(img_dir=img_dir,
                        transform=transforms.Compose([
                            transforms.Resize((640,640)),
                            transforms.CenterCrop((640,640)),
                            transforms.ToTensor(),
                        ]))
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)

    for si, sample in enumerate(dl):
        dlr = lr
        gs = sample[0] * 255
        gsc = gs.cuda()
        samp = gs.data.numpy().squeeze().transpose(1,2,0)

        img = torch.ones_like(gsc)*128
        if next(model.parameters()).is_cuda:
            img = scatter(img, [device])[0]
            
        results_orig = model.simple_test(gsc,img_metas=None)
        with torch.no_grad():
            interm_results = model.forward_dummy(gsc)

        for i in range(num_circles):
            if i>0 and i%lr_decay==0:
                dlr *= 0.1
                print('LR DECAY: '+str(dlr))
            model.simple_dream(img, ratios=[rpl,tvl], target_feats=interm_results, lr=dlr, amp=amp)


        with torch.no_grad():   
            results = model.simple_test(img, img_metas=None)

        bboxeso = show_result_pyplot(model,samp,result=results_orig[0],score_thr=sthr)
        norm = norm_contrast(ten2arr(img))
        bboxes = show_result_pyplot(model,norm,result=results[0],score_thr=sthr)
        rrso = np.concatenate([samp,bboxes,norm,bboxes],axis=1)
        save_img(samp,os.path.join(out_dir,'original_%d.jpg'%(si,)))
        save_img(samp,os.path.join(out_dir,'thumb_%d.jpg'%(si,)),resize=(200,200))
        save_img(bboxeso,os.path.join(out_dir,'orbb_%d.jpg'%(si,)))
        save_img(norm,os.path.join(out_dir,'optim_%d.jpg'%(si,)))
        save_img(bboxes,os.path.join(out_dir,'opbb_%d.jpg'%(si,)))
        # save_img(rrso,os.path.join(out_dir,'combined_%d.jpg'%(si,)))


def test(args):
    out_dir = 'src/out/test/'
    img_dir = 'data/coco/val2017'
    setup_clean_directory(out_dir)

    num_circles = 2000
    lr_decay = num_circles*0.9
    
    lr = 0.5
    amp = 10
    
    rpl = 0.40
    tvl = 0.60
    bnl = 0.0

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    device = next(model.parameters()).device
    model.init_dream()

    ds = unsupervised_ds(img_dir=img_dir,
                        transform=transforms.Compose([
                            transforms.Resize((640,640)),
                            transforms.CenterCrop((640,640)),
                            transforms.ToTensor(),
                        ]))
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)

    for si, sample in enumerate(dl):
        dlr = lr
        gs = sample[0] * 255
        gsc = gs.cuda()
        samp = gs.data.numpy().squeeze().transpose(1,2,0)

        # img = gsc.clone()
        img = torch.ones_like(gsc)*128

        if next(model.parameters()).is_cuda:
            img = scatter(img, [device])[0]
            
        results_orig = model.simple_test(gsc,img_metas=None)
        with torch.no_grad():
            interm_results = model.forward_dummy(gsc)

        # for i1,ir in enumerate(interm_results):
        #     for i2,iir in enumerate(ir):
        #         interm_results[i1][i2] *= 0
        #         interm_results[i1][i2] -= 1

        # interm_results[2][2] -= 10
        # interm_results[2][1] -= 10
        # interm_results[2][0] -= 10
        
        # # save_img(interm_results[2][2].squeeze().cpu(),'src/out/test.jpg')
        # interm_results[2][2][0,0,10,10] = 2.5
        # interm_results[0][2][0,0,10,10] = 2.5
        # interm_results[1][2][0,0,10,10] = 0.5
        # interm_results[1][2][0,1,10,10] = 0.5
        # interm_results[1][2][0,2,10,10] = 0.7
        # interm_results[1][2][0,3,10,10] = 0.7




        for i in range(num_circles):
            if i>0 and i%lr_decay==0:
                dlr *= 0.1
                print('LR DECAY: '+str(dlr))
            model.simple_dream(img, ratios=[rpl,tvl], target_feats=interm_results, lr=dlr, amp=amp)


        with torch.no_grad():   
            results = model.simple_test(img, img_metas=None)

        bboxeso = show_result_pyplot(model,samp,result=results_orig[0],score_thr=sthr)
        norm = norm_contrast(ten2arr(img))
        bboxes = show_result_pyplot(model,norm,result=results[0],score_thr=sthr)
        rrso = np.concatenate([samp,bboxes,norm,bboxes],axis=1)
        save_img(samp,os.path.join(out_dir,'%d_original.jpg'%(si,)))
        save_img(samp,os.path.join(out_dir,'%d_thumb.jpg'%(si,)),resize=(200,200))
        save_img(bboxeso,os.path.join(out_dir,'%d_orbb.jpg'%(si,)))
        save_img(norm,os.path.join(out_dir,'%d_optim.jpg'%(si,)))
        save_img(bboxes,os.path.join(out_dir,'%d_opbb.jpg'%(si,)))
        # save_img(rrso,os.path.join(out_dir,'%d_combined.jpg'%(si,)))



def art_demo_video(args):
    num_circles = 3001
    save_every = 2


    lr = 0.1
    amp = 10
    ratio = 0.7

    
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    device = next(model.parameters()).device

    ds = unsupervised_ds(img_dir='data/art_demo_video',
                        transform=transforms.Compose([
                            transforms.Resize(640),
                            transforms.CenterCrop((640,640)),
                            transforms.ToTensor(),
                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    dl = torch.utils.data.DataLoader(ds, batch_size=1,
                                        shuffle=False, num_workers=1)

    from helper import setup_clean_directory
    setup_clean_directory('src/out/art_demo_video/')
    frame_count = 0

    for si in range(len(ds)):
        sample = ds[si]
        gs = sample[0].unsqueeze(0) * 255
        gsc = gs.cuda()
        samp = gs.data.numpy().squeeze().transpose(1,2,0)

        with torch.no_grad():
            interm_results = model.forward_dummy(gsc)

        img = torch.ones_like(gsc)*128
        if next(model.parameters()).is_cuda:
            img = scatter(img, [device])[0]



        for ic in range(num_circles):
            model.forward_dream(img, ratios=[1-ratio,ratio,0.0], target_feats=interm_results, lr=lr, amp=amp)

            if ic % save_every == 0:
                norm = norm_contrast(ten2arr(img))
                rrso = np.concatenate([samp,norm],axis=1)
                save_img(norm,'src/out/art_demo_video/plain_%08d.jpg'%(frame_count,))
                save_img(rrso,'src/out/art_demo_video/concat_%08d.jpg'%(frame_count,))
                frame_count += 1
                if ic == num_circles-1:
                    for _ in range(60*3):
                        save_img(norm,'src/out/art_demo_video/plain_%08d.jpg'%(frame_count,))
                        save_img(rrso,'src/out/art_demo_video/concat_%08d.jpg'%(frame_count,))
                        frame_count += 1



def articlevideo(args):
    out_path = 'src/out/newvideo/'
    in_dir = 'data/articlevideo'
    num_circles = 5000
    save_every = 2


    lr = 0.1
    amp = 10
    rpl = 0.4
    tv = 0.6

    
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    device = next(model.parameters()).device

    ds = unsupervised_ds(img_dir=in_dir,
                        transform=transforms.Compose([
                            transforms.Resize(640),
                            transforms.CenterCrop((640,640)),
                            transforms.ToTensor(),
                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    dl = torch.utils.data.DataLoader(ds, batch_size=1,
                                        shuffle=False, num_workers=1)

    from helper import setup_clean_directory
    setup_clean_directory(out_path)
    frame_count = 0

    for si in range(len(ds)):
        sample = ds[si]
        gs = sample[0].unsqueeze(0) * 255
        gsc = gs.cuda()
        samp = gs.data.numpy().squeeze().transpose(1,2,0)

        with torch.no_grad():
            interm_results = model.forward_dummy(gsc)

        img = torch.ones_like(gsc)*128
        if next(model.parameters()).is_cuda:
            img = scatter(img, [device])[0]



        for ic in range(num_circles):
            model.simple_dream(img, ratios=[rpl,tv,0.0], target_feats=interm_results, lr=lr, amp=amp)

            if ic % save_every == 0:
                norm = norm_contrast(ten2arr(img))
                rrso = np.concatenate([samp,norm],axis=1)
                save_img(norm,os.path.join(out_path,'plain_%08d.jpg'%(frame_count,)))
                save_img(rrso,os.path.join(out_path,'concat_%08d.jpg'%(frame_count,)))
                frame_count += 1
                if ic == num_circles-1:
                    for _ in range(60*3):
                        save_img(norm,os.path.join(out_path,'plain_%08d.jpg'%(frame_count,)))
                        save_img(rrso,os.path.join(out_path,'concat_%08d.jpg'%(frame_count,)))
                        frame_count += 1




# ffmpeg -i "src/out/art_demo_video/%08d.jpg" -c:v libx264 -vf "fps=30,format=yuv420p" out.mp4



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
    # param_visu(args)
    # param_visu(args)
    # articlevideo(args)
    # video_visu(args)
    # art_demo_video(args)
    test(args)







# old pipeline

    # # build the model from a config file and a checkpoint file
    # model = init_detector(args.config, args.checkpoint, device=args.device)
    
    # ds, dl = load_dataset(model.cfg.data.val,model)


    # from helper import setup_clean_directory
    # setup_clean_directory('src/out/param_visu/')


    # for sample_idx in range(1000):
    #     samp = ds[sample_idx]
    #     orig = samp['img'][0].data.numpy().squeeze().transpose(1,2,0)

    #     ann = ds.get_ann_info(sample_idx)

    #     fn = samp['img_metas'][0].data['filename']
    #     dd = dict(img_prefix=None, img_info=dict(filename=fn))
        
    
    #     testpi = Compose(model.cfg.data.test.pipeline)

    #     a = testpi(dd)
    #     a['img'] = collate(a['img'], samples_per_gpu=1)

    #     data = dict()
    #     data['img_metas'] = [img_metas.data for img_metas in a['img_metas']]
    #     data['imgs'] = [img.data for img in a['img'].data]

    #     device = next(model.parameters()).device
    #     if next(model.parameters()).is_cuda:
    #         data = scatter(data, [device])[0]

    #     with torch.no_grad():
    #         interm_results = model.forward_dummy(data['imgs'][0])


    #     from src.train_dcgan import Discriminator
    #     netD = Discriminator(1, nc=3, ndf=64).to(device)
    #     netD.load_state_dict(torch.load('src/out/dcgan/netD_epoch_20.pth'))

    #     lrs = [0.001, 0.01, 0.1, 1.0, 10.0]
    #     amps = [1, 10, 100, 1000]
    #     ratios = [0.0, 0.25, 0.5, 0.75, 0.95]

    #     save_img(orig,'src/out/param_visu/original_%d.jpg'%(sample_idx,))


    #     for lr in lrs:
    #         for amp in amps:
    #             for ratio in ratios:
    #                 img = torch.ones_like(a['img'].data[0])*128
    #                 if next(model.parameters()).is_cuda:
    #                     img = scatter(img, [device])[0]
    #                 for i in range(10000):
    #                     model.forward_dream(img, ratios=[1-ratio,ratio,0.0], target_feats=interm_results, lr=lr, amp=amp)
            
    #                 with torch.no_grad():   
    #                     results = model.simple_test(img, img_metas=data['img_metas'])

    #                 # resultsli = []
    #                 # for ri,rbb in enumerate(results[0]):
    #                 #     for rb in rbb:
    #                 #         resultsli.append(rb + [ri])

    #                 norm = norm_contrast(ten2arr(img))
    #                 bboxes = show_result_pyplot(model,norm,result=results[0])
    #                 rrso = np.concatenate([orig,norm,bboxes],axis=1)
    #                 save_img(norm,'src/out/param_visu/plain_%d_%f_%f_%d.jpg'%(sample_idx,ratio,lr,amp))
    #                 save_img(bboxes,'src/out/param_visu/bboxes_%d_%f_%f_%d.jpg'%(sample_idx,ratio,lr,amp))
    #                 save_img(rrso,'src/out/param_visu/combined_%d_%f_%f_%d.jpg'%(sample_idx,ratio,lr,amp))

