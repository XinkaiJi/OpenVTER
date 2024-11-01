#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2022-05-07 22:43
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : VehicleTrackingModule.py 
@Software: PyCharm
@desc: 
'''
import numpy as np
import torch
import time

class VehicleTrackingModule:
    _defaults = {
        "confidence": 0.3,
                 }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.mot_tracker = self.load_model()
        if 'device' in kwargs.keys():
            self.device = torch.device(self.device)
        else:
            self.device = torch.device('cpu')

    def load_model(self):
        if self.model_name == 'sort_r_angle':
            from tracking.sort_r import Sort
            mot_tracker = Sort(max_age=self.max_age,
                                    min_hits=self.min_hits,
                                    iou_threshold=self.iou_threshold)  # create instance of the SORT tracker
            print('using sort_r')
            return mot_tracker
        elif self.model_name == 'sort_r_match' or self.model_name == 'sort_r':
            from tracking.sort_r import Sort_R_match
            mot_tracker = Sort_R_match(max_age=self.max_age,
                               min_hits=self.min_hits,
                               iou_threshold=self.iou_threshold)  # create instance of the SORT tracker
            print('using sort_r_match')
            return mot_tracker
        elif self.model_name == 'deep_sort_r':
            from tracking.deep_sort_r import DeepSort
            use_cuda = self.use_cuda and torch.cuda.is_available()

            mot_tracker = DeepSort(model_path=self.REID_CKPT,
                            max_dist=self.MAX_DIST, min_confidence=self.MIN_CONFIDENCE,
                            nms_max_overlap=self.NMS_MAX_OVERLAP,
                            max_iou_distance=self.MAX_IOU_DISTANCE,
                            max_age=self.MAX_AGE, n_init=self.N_INIT, nn_budget=self.NN_BUDGET,
                            use_cuda=use_cuda,match_iou_threshold1=self.match_iou_threshold1,match_iou_threshold2=self.match_iou_threshold2)
            print('using deep_sort_r')
            return mot_tracker

    def update(self,dets = torch.empty((0,15)), ori_img=None):
        # center_x, center_y,bbox_w, bbox_h,theta,x1,y1,x1,y2,x2,y3,x4,y4,score,category
        # 统计时间
        dets = dets.cpu()
        # start_t = time.time()
        if self.model_name == 'deep_sort_r':
            result = self.mot_tracker.add_det_data(dets,ori_img,self.device)
        else:
            result = self.mot_tracker.add_det_data(dets,self.device)
        # end_t = time.time()
        # num = result.shape[0]
        # print('update time:%.3f,num:%d,each num:%.5f'%(end_t-start_t,num,(end_t-start_t)/num))
        return result