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

    def forward_dream(self, img, target_feats=None, lr=0.3, ratio=0.7, gan=None):


        img.requires_grad = True
        feat = self.extract_feat(img)
        # outs = self.bbox_head.forward(feat)
        if target_feats is None:
            target_feats = torch.zeros_like(feat[0])
        losses = []
        for pf,tf in zip(feat,target_feats):
            loss_component = torch.nn.MSELoss(reduction='mean')(pf,tf)
            losses.append(loss_component)
        final_loss = torch.mean(torch.stack(losses))
        final_loss.backward()
        grad = img.grad.data
        g_std = torch.std(grad)
        g_mean = torch.mean(grad)
        grad = grad - g_mean
        grad = grad / g_std
        img.data = img.data - lr * ratio * grad
        img.grad.data.zero_()

        sr = torch.nn.functional.interpolate(img/255,size=(64,64), mode='bilinear')
        # sr.requires_grad = True
        sr.retain_grad()
        ganresult = gan.forward(sr)
        ganloss = torch.nn.MSELoss(reduction='mean')(ganresult, torch.ones_like(ganresult))
        ganloss.backward()
        gradg = sr.grad
        g_std = torch.std(gradg) + 0.00000000000000000000001
        g_mean = torch.mean(gradg)
        gradg = gradg - g_mean
        gradg = gradg / g_std
        gan_grad = torch.nn.functional.interpolate(gradg,size=(768,1280), mode='bilinear')
        img.data = img.data + lr * (1-ratio) * gan_grad
        sr.grad.data.zero_()

        print(final_loss)
        # print(ganloss)

        return img

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
