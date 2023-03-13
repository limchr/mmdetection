# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info

from ...utils import log_img_scale
from ..builder import DETECTORS
from .single_stage import SingleStageDetector

from torch.autograd import Variable
from src.modelutils import Recorder, get_all_layers

@DETECTORS.register_module()
class YOLOX(SingleStageDetector):
    r"""Implementation of `YOLOX: Exceeding YOLO Series in 2021
    <https://arxiv.org/abs/2107.08430>`_

    Note: Considering the trade-off between training speed and accuracy,
    multi-scale training is temporarily kept. More elegant implementation
    will be adopted in the future.

    Args:
        backbone (nn.Module): The backbone module.
        neck (nn.Module): The neck module.
        bbox_head (nn.Module): The bbox head module.
        train_cfg (obj:`ConfigDict`, optional): The training config
            of YOLOX. Default: None.
        test_cfg (obj:`ConfigDict`, optional): The testing config
            of YOLOX. Default: None.
        pretrained (str, optional): model pretrained path.
            Default: None.
        input_size (tuple): The model default input image size. The shape
            order should be (height, width). Default: (640, 640).
        size_multiplier (int): Image size multiplication factor.
            Default: 32.
        random_size_range (tuple): The multi-scale random range during
            multi-scale training. The real training image size will
            be multiplied by size_multiplier. Default: (15, 25).
        random_size_interval (int): The iter interval of change
            image size. Default: 10.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 input_size=(640, 640),
                 size_multiplier=32,
                 random_size_range=(15, 25),
                 random_size_interval=10,
                 init_cfg=None):
        super(YOLOX, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained, init_cfg)
        log_img_scale(input_size, skip_square=True)
        self.rank, self.world_size = get_dist_info()
        self._default_input_size = input_size
        self._input_size = input_size
        self._random_size_range = random_size_range
        self._random_size_interval = random_size_interval
        self._size_multiplier = size_multiplier
        self._progress_in_iter = 0

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Multi-scale training
        img, gt_bboxes = self._preprocess(img, gt_bboxes)

        losses = super(YOLOX, self).forward_train(img, img_metas, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore)

        # random resizing
        if (self._progress_in_iter + 1) % self._random_size_interval == 0:
            self._input_size = self._random_resize(device=img.device)
        self._progress_in_iter += 1

        return losses

    @staticmethod
    def total_variation_loss(img):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


    def init_dream(self,):
        ml = get_all_layers(self)
        self.bn_layers = []
        for ll in ml:
            if isinstance(ll, torch.nn.BatchNorm2d):
                self.bn_layers.append(ll)
        
        # a = list(self.named_children())
        # backb = list(a[0][1].named_children())
        # bn0 = backb[1][1][0].bn
        # bn1 = list(backb[1][1][1].named_children())[0][1].bn
        # bn2 = list(backb[1][1][1].named_children())[1][1].bn
        # bn3 = list(backb[1][1][1].named_children())[2][1].bn
        # bn4 = list(list(backb[1][1][1].named_children())[3][1][0].named_children())[0][1].bn
        # bn5 = list(list(backb[1][1][1].named_children())[3][1][0].named_children())[1][1].bn
        # self.bn_layers = [bn0,bn1,bn2,bn3,bn4,bn5]
        # self.bn_layers = [bn0]
        self.recorders = [Recorder(layer, record_input = True, record_output = False, backward = False) for layer in self.bn_layers]

    # ratios: target_difference, total variation, bn
    def forward_dream(self, img, ratios, amp, target_feats=None, lr=0.3, gan=None,ampcls=None):
        img.requires_grad = True
        feat = self.forward_dummy(img)
        if target_feats is None:
            target_feats = torch.zeros_like(feat[0])
        
        # batch normalization loss
        if ratios[2]>0:
            reco = [r.recording[0].clone() for r in self.recorders]
            bn_losses = []
            for bnout,bnl in zip(reco,self.bn_layers):
                bn_mean = torch.mean(bnout - bnl.running_mean.unsqueeze(1).unsqueeze(0).unsqueeze(2),axis=[2,3]) - bnl.running_mean          
                # bn_channels = bnout.mean(axis=[2,3])
                # bn_var = torch.square(bn_channels - torch.mean(bn_channels)) - bnl.running_var
                bn_var = torch.mean(torch.square(bnout - torch.mean(bnout)),axis=[2,3]) - bnl.running_var

                bn_mean_loss = torch.mean(torch.square(bn_mean))
                bn_var_loss = torch.mean(torch.square(bn_var))

                bn_losses.append(bn_mean_loss+bn_var_loss)
        
            bnmasterloss = torch.mean(torch.stack(bn_losses))
        else:
            bnmasterloss = 0


        # masked loss
        mask_losses = []
        for ol in range(3):
            mul_mask = torch.ones_like(target_feats[2][ol])
            bin_mask = target_feats[2][ol] > 0.5
            mul_mask[bin_mask] = amp
            # bin_mask = target_feats[2][ol] < 0.1
            # mul_mask[bin_mask] = 0
            # if not ampcls is None:
            #     clsampmask = target_feats[0][ol].argmax(axis=1) == ampcls
            #     for ii in clsampmask.shape[-1]:
            #         for jj in clsampmask.shape[-1]:
            #             if clsampmask[0,ii,jj]:
            #                 target_feats[0][ol]

            for il in range(3):
                sq_err = torch.square(feat[il][ol] - target_feats[il][ol])
                sq_err *= mul_mask
                sq_errm = torch.mean(sq_err)
                mask_losses.append(sq_errm)
        final_mask_loss = torch.mean(torch.stack(mask_losses))        
                
        
        
        rpl_loss = ratios[0] * final_mask_loss
        tv_loss = ratios[1] * self.total_variation_loss(img) * 20
        bn_loss = ratios[2] * bnmasterloss / 5000
        end_loss = rpl_loss + tv_loss + bn_loss
        
        
        # loss calculation
        end_loss.backward()
        grad = img.grad
        g_std = torch.std(grad)
        g_mean = torch.mean(grad)
        grad = grad - g_mean
        grad = grad / g_std
        
        momentum = 0.0
        if not hasattr(self,'old_grad'):
            self.old_grad = grad.clone()
        grad_r = grad * (1-momentum) + momentum * self.old_grad
        self.old_grad = grad_r.clone().detach()

        # gradient descent
        img.data = img.data - lr * grad_r
        img.grad.data.zero_()

        print('rp-loss: %3.3f, tv-loss: %3.3f, bn-loss: %3.3f, total_loss: %3.3f' % (rpl_loss,tv_loss,bn_loss,end_loss))

        return img, end_loss







    # ratios: target_difference, total variation, bn
    def simple_dream(self, img, ratios, amp, target_feats=None, lr=0.3):
        img.requires_grad = True
        feat = self.forward_dummy(img)
        if target_feats is None:
            target_feats = torch.zeros_like(feat[0])
        
        # masked loss
        mask_losses = []
        for ol in range(3):
            mul_mask = torch.ones_like(target_feats[2][ol])
            bin_mask = target_feats[2][ol] > 0.6
            mul_mask[bin_mask] = amp
            for il in range(3):
                sq_err = torch.square(feat[il][ol] - target_feats[il][ol])
                sq_err *= mul_mask
                sq_errm = torch.mean(sq_err)
                mask_losses.append(sq_errm)
        final_mask_loss = torch.mean(torch.stack(mask_losses))        
                
        
        
        rpl_loss = ratios[0] * final_mask_loss
        tv_loss = ratios[1] * self.total_variation_loss(img) * 20
        end_loss = rpl_loss + tv_loss
        
        
        # loss calculation
        end_loss.backward()
        grad = img.grad
        g_std = torch.std(grad)
        g_mean = torch.mean(grad)
        grad = grad - g_mean
        grad = grad / g_std
        
        img.data = img.data - lr * grad
        img.grad.data.zero_()

        print('rp-loss: %3.3f, tv-loss: %3.3f, total_loss: %3.3f' % (rpl_loss,tv_loss,end_loss))

        return












    def forward_grid_cells(self, img):
        feat = self.extract_feat(img)
        return self.bbox_head.forward(feat)



    def _preprocess(self, img, gt_bboxes):
        scale_y = self._input_size[0] / self._default_input_size[0]
        scale_x = self._input_size[1] / self._default_input_size[1]
        if scale_x != 1 or scale_y != 1:
            img = F.interpolate(
                img,
                size=self._input_size,
                mode='bilinear',
                align_corners=False)
            for gt_bbox in gt_bboxes:
                gt_bbox[..., 0::2] = gt_bbox[..., 0::2] * scale_x
                gt_bbox[..., 1::2] = gt_bbox[..., 1::2] * scale_y
        return img, gt_bboxes

    def _random_resize(self, device):
        tensor = torch.LongTensor(2).to(device)

        if self.rank == 0:
            size = random.randint(*self._random_size_range)
            aspect_ratio = float(
                self._default_input_size[1]) / self._default_input_size[0]
            size = (self._size_multiplier * size,
                    self._size_multiplier * int(aspect_ratio * size))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if self.world_size > 1:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size


