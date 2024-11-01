#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/8/10 上午11:30
# @Author  : Xinkai Ji
# @Email   : xinkaiji@hotmail.com
# @File    : detector
# @Software: PyCharm
# @Desc    :
import numpy as np
import torch

from mmdet.apis import inference_detector, init_detector
from mmrotate.utils import register_all_modules
from mmrotate.structures.bbox import rbox2qbox
class MMDet(object):
    def __init__(self, cfg_file, checkpoint_file, score_thresh=0.7,
                device_name='cuda:0'):
        # net definition
        register_all_modules()
        self.device = device_name
        self.net = init_detector(cfg_file, checkpoint_file, device=self.device)

        print('Loading weights from %s... Done!' % (checkpoint_file))

        #constants
        self.score_thresh = score_thresh


        # self.class_names = self.net.CLASSES
        # self.num_classes = len(self.class_names)

    def det_images_batch(self, ori_image_ls):
        # forward
        new_predictions_ls = []
        bbox_result = inference_detector(self.net, ori_image_ls)
        for det_result in bbox_result:
            pre_inst = det_result.pred_instances
            bboxes = pre_inst.bboxes
            scores = pre_inst.scores
            labels = pre_inst.labels
            # 获取符合阈值条件的索引
            keep = scores > self.score_thresh

            # 根据索引筛选bboxes和labels
            filtered_bboxes = bboxes[keep]
            filtered_qbboxes = rbox2qbox(filtered_bboxes)
            filtered_labels = labels[keep]
            filtered_scores = scores[keep]
            # concat
            concat_result = torch.cat([filtered_qbboxes, filtered_scores.unsqueeze(1), filtered_labels.unsqueeze(1)], dim=1).cpu().numpy()
            new_predictions_ls.append(concat_result)

        return new_predictions_ls