#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2022-05-16 14:11
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : centernet_bba.py 
@Software: PyCharm
@desc: 
'''
import torch
import cv2
import numpy as np
from .decoder import DecDecoder

class CenterNet_BBA(object):
    _defaults = {"checkpoint_type":"jit",
                 "checkpoint_jit":"checkpoints/centernet_bbavectors/centernet_bbavectors_oneclass_M_202206.jit",
                 "confidence":0.2,
                 "img_width": 640,
                 "img_height": 512,
                 "device_name": "cuda:0",
                 "class_names": ["car", "truck", "bus", "freight_car", "van"],
                 "color_pans": "[(204,78,210),(204,78,210),(204,78,210),(204,78,210),(204,78,210)]",
                 "down_ratio": 4
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

        # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        # self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.num_classes = len(self.class_names)
        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        if self.device_name == 'cpu':
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device(self.device_name)
            print('using %s'%self.device_name)


        if self.checkpoint_type == "jit":
            self.load_model_jit(self.checkpoint_jit)
        elif self.checkpoint_type == "pth":
            pass

    def load_model_jit(self,resume):
        self.model = torch.jit.load(resume)
        self.model = self.model.to(self.device)
        self.model.eval()
        print('{} model, and classes loaded.'.format(resume))

        self.decoder = DecDecoder(K=self.max_object_num,
                                     conf_thresh=self.confidence,
                                     num_classes=self.num_classes)

    def det_images_batch(self, images_ls):
        # 输入图片和模型输入尺寸保持一致
        new_images = []
        if len(images_ls) ==0:
            return None
        image_shape = images_ls[0].shape[:2]
        for image_data in images_ls:
            h, w, c = image_data.shape
            if self.img_height !=h or self.img_width != w:
                image_data = cv2.resize(image_data, (self.img_width, self.img_height))
            # image_data = preprocess_input(np.array(image_data, dtype='float32'))
            new_images.append(image_data)

        image_batch = torch.from_numpy(np.transpose(np.array(new_images, dtype=np.float32) / 255 - 0.5, (0, 3, 1, 2)))
        with torch.no_grad():
            images = image_batch.to(self.device)
            outputs = self.model(images)
            predictions = self.decoder.ctdet_decode(outputs)
            new_predictions_ls = self.decode_prediction_batch(predictions, images_ls[0], self.down_ratio)

        return new_predictions_ls

    def decode_prediction_batch(self,predictions_ls,img, down_ratio):
        h, w, c = img.shape
        # pts0_ls = []
        # scores0_ls = []
        new_prediction_ls = []
        for i in range(len(predictions_ls)):
            predictions = predictions_ls[i]
            if predictions.size == 0:
                new_prediction_ls.append(None)
                # pts0_ls.append(None)
                # scores0_ls.append(None)
                continue
            cen_pt = predictions[ :, :2]
            tt = predictions[:, 2:4]
            rr = predictions[:, 4:6]
            bb = predictions[:, 6:8]
            ll = predictions[:, 8:10]
            tl = tt + ll - cen_pt
            bl = bb + ll - cen_pt
            tr = tt + rr - cen_pt
            br = bb + rr - cen_pt
            score = predictions[:, 10:11]
            clse = predictions[:, 11:12]
            new_prediction = np.concatenate([tr, br, bl, tl, score, clse], axis=1)
            w_scale = down_ratio / self.img_width * w
            h_scale = down_ratio / self.img_height * h
            w_index = [0, 2, 4, 6]
            h_index = [1, 3, 5, 7]
            new_prediction[ :, w_index] = new_prediction[:, w_index] * w_scale
            new_prediction[:, h_index] = new_prediction[:, h_index] * h_scale
            new_prediction_ls.append(new_prediction)
        return new_prediction_ls

if __name__ == "__main__":
    c = CenterNet_BBA()
