#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2021-12-21 15:28
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : VehicleDetModule.py 
@Software: PyCharm
@desc: 
'''
import torch
import numpy as np
import cv2
from mmcv.ops import nms_rotated
from collections import deque


class VehicleDetModule:
    _defaults = {
        "device": "cuda:0",
        "confidence": 0.3,
        "nms_iou": 0.3,
        "img_width": 640,
        "img_height": 640,
        "category": ['car','truck','bus','freight_car','van'],
        "color_pans": [(204,78,210),(0,192,255), (0,131,0),(240,176,0), (254,100,38)],

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
        if self.device_name == 'cpu':
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device(self.device_name)
        else:
            print('no cuda!')
        # self.device = torch.device(self.device)
        self.color_pans = eval(self.color_pans)

        self.Vehicles = {}
        self.MAXLEN = 30


    def load_model(self):
        if self.model_name == 'yolox_r':
            from detection.yolox_r.yolo import YOLO
            yolo_dict = {"checkpoint_type":self.checkpoint_type,
                         "phi": self.phi,
                         "device_name": self.device_name,
                "checkpoint": self.checkpoint, "input_shape": [self.img_width,self.img_height],
                         "confidence": self.confidence,"nms_iou":self.nms_iou,"class_names":self.class_names}
            self.model = YOLO(**yolo_dict)
        elif self.model_name == 'centernet_bbavectors':
            from detection.centernet_bbavectors.centernet_bba import CenterNet_BBA
            centernet_dict = {"checkpoint_type":self.checkpoint_type,
                 "checkpoint_jit":self.checkpoint_jit,
                 "confidence":self.confidence,
                 "img_width": self.img_width,
                 "img_height": self.img_height,
                 "device_name": self.device_name,
                 "class_names": self.class_names,
                 "down_ratio": self.down_ratio,
                 "max_object_num":self.max_object_num
                 }

            self.model = CenterNet_BBA(**centernet_dict)
        elif self.model_name == 'mmrotate':
            from detection.mmrotate.detector import MMDet

            self.model = MMDet(self.cfg,self.checkpoint,self.score_thresh,self.device_name)

    def _process_img_batch(self, ori_image_ls):

        predictions = self.model.det_images_batch(ori_image_ls)
        # print('prediction:',predictions)
        return predictions

    def _nms_rotated_tensor(self,predictions):
        np_boxes = self.four_points2bbox_angle_tensor(predictions)
        boxes = torch.from_numpy(np_boxes).to(self.device)
        # labels = torch.from_numpy(np_labels).cuda()
        dets, keep_inds = nms_rotated(boxes[:, :5], boxes[:, 5], 0.3,  boxes[:, 6])
        # print(dets)
        return boxes[keep_inds]

    def four_points2bbox_angle_tensor(self,predictions):
        # predictions = predictions.cpu().numpy()
        bboxs_angle = []
        # predictions = predictions[0]
        for index,pt in enumerate(predictions):
            rect = cv2.minAreaRect(pt[:8].reshape(4,2))
            center_x, center_y = rect[0][0], rect[0][1]
            bbox_w, bbox_h = rect[1][0], rect[1][1]
            theta = rect[2]*np.pi/180

            pts_4 = cv2.boxPoints(((center_x, center_y), (bbox_w, bbox_h), rect[2]))  # 4 x 2
            score = pt[8]
            cat = pt[9]
            bbox_angle = [center_x, center_y,bbox_w, bbox_h,theta,
                          pts_4[0][0], pts_4[0][1], pts_4[1][0], pts_4[1][1], pts_4[2][0], pts_4[2][1], pts_4[3][0], pts_4[3][1],
                          score,cat]
            bboxs_angle.append(bbox_angle)
        np_boxes = np.array(bboxs_angle,dtype=np.float32)
        return np_boxes

    def _nms_rotated(self,pts_cat, scores_cat, cat_np):
        np_boxes = self.four_points2bbox_angle(pts_cat, scores_cat, cat_np)

        boxes = torch.from_numpy(np_boxes).to(self.device)
        # labels = torch.from_numpy(np_labels).cuda()

        dets, keep_inds = nms_rotated(boxes[:, :5], boxes[:, 5], 0.3,  boxes[:, 6])
        # print(dets)
        return boxes[keep_inds]

    def four_points2bbox_angle(self,pts_cat, scores_cat, cat_np):
        bboxs_angle = []
        for index,pt in enumerate(pts_cat):
            rect = cv2.minAreaRect(pt)
            center_x, center_y = rect[0][0], rect[0][1]
            bbox_w, bbox_h = rect[1][0], rect[1][1]
            theta = rect[2]*np.pi/180
            score = scores_cat[index]
            cat = cat_np[index]
            bbox_angle = [center_x, center_y,bbox_w, bbox_h,theta,score,cat]
            bboxs_angle.append(bbox_angle)
        np_boxes = np.asarray(bboxs_angle,dtype=np.float32)
        return np_boxes

    def inference_img_batch(self, image_ls):
        new_predictions_ls = self._process_img_batch(image_ls)

        nms_results_ls = []

        for new_predictions in new_predictions_ls:
            nms_results = None
            if new_predictions is not None and new_predictions.size:
                new_predictions = self._filter_edge_bbox(new_predictions)
                if new_predictions is not None and new_predictions.size:
                    nms_results = self._nms_rotated_tensor(new_predictions)

            nms_results_ls.append(nms_results)

        return nms_results_ls

    def _filter_edge_bbox(self,new_predictions):
        x_index = [0,2,4,6]
        y_index = [1,3,5,7]
        min_pixel = 3 #0.1*gap
        # nn_data = filter(lambda x: x[x_index].min() >0.1*gap and x[x_index].max() < self.img_width-0.1*gap and x[y_index].min() >0.1*gap and x[y_index].max() < self.img_height-0.1*gap, new_predictions)
        nn_data = new_predictions[np.where( (min_pixel< new_predictions[:, x_index].min(axis=1)) & (self.img_width-min_pixel > new_predictions[:, x_index].max(axis=1)) &
                                           (min_pixel < new_predictions[:, y_index].min(axis=1)) & (self.img_height-min_pixel > new_predictions[:, y_index].max(axis=1)))]
        return nn_data


    def draw_bboxs(self,ori_image,nms_results):
        for bbox in nms_results:
            bbox_id = int(bbox[-1])
            x1,y1 = int(bbox[0]), int(bbox[1])
            x2,y2 = int(bbox[2]), int(bbox[3])
            cv2.rectangle(ori_image, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(ori_image, '%d' % bbox_id, (x1, y1),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1, 1)
        return ori_image

    def add_veh(self,id,c_x,c_y,ori_image,color):
        if id in self.Vehicles.keys():
            self.Vehicles[id].append((c_x,c_y))
            traj = self.Vehicles[id]
            for i in range(len(traj)-1):
                cv2.line(ori_image, traj[i], traj[i+1], color, 1,lineType=cv2.LINE_AA)
        else:
            self.Vehicles[id] = deque(maxlen=self.MAXLEN)
            self.Vehicles[id].append((c_x, c_y))

    def draw_oriented_bboxs(self,ori_image, nms_results,bbox_label):
        '''
        绘制旋转bboxs
        :param ori_image:
        :param nms_results:
        :return:
        '''
        cat = None
        score = None
        bbox_id = 0
        id = None
        xy_pts = None
        lane_name = None
        for object_index in range(nms_results.shape[0]):
            pred = nms_results[object_index]
            if len(pred)==10:
                cat = self.category[int(pred[-1])]
                score = pred[-2]
            elif len(pred)==11:
                cat = self.category[int(pred[-2])]
                score = pred[-3]
                id = int(pred[-1])
            elif len(pred)==19 or len(pred)==20:
                cat = self.category[int(pred[9])]
                score = pred[8]
                id = int(pred[10])
                xy_pts = np.mean(pred[11:19].reshape(-1,2),axis=0)
                if len(pred)==20:
                    lane_name = pred[19]
            else:
                bbox_id = int(pred[-1])
            tl = np.asarray([pred[0], pred[1]], np.float32)
            tr = np.asarray([pred[2], pred[3]], np.float32)
            br = np.asarray([pred[4], pred[5]], np.float32)
            bl = np.asarray([pred[6], pred[7]], np.float32)

            # tt = (np.asarray(tl, np.float32) + np.asarray(tr, np.float32)) / 2
            # rr = (np.asarray(tr, np.float32) + np.asarray(br, np.float32)) / 2
            # bb = (np.asarray(bl, np.float32) + np.asarray(br, np.float32)) / 2
            # ll = (np.asarray(tl, np.float32) + np.asarray(bl, np.float32)) / 2

            box = np.asarray([tl, tr, br, bl], np.float32)
            cen_pts = np.mean(box, axis=0)

            # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(tt[0]), int(tt[1])), (0, 0, 255), 1, 1)
            # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(rr[0]), int(rr[1])), (255, 0, 255), 1, 1)
            # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(bb[0]), int(bb[1])), (0, 255, 0), 1, 1)
            # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(ll[0]), int(ll[1])), (255, 0, 0), 1, 1)


            if cat is not None:
                if id is not None:
                    if 'tail' in bbox_label:
                        self.add_veh(id,int(cen_pts[0]), int(cen_pts[1]),ori_image,(0,255,255))

                    cv2.drawContours(ori_image, [np.int0(box)], -1, self.color_pans[int(pred[9])], 2, 1)
                    x1 = int(cen_pts[0])
                    y1 = int(cen_pts[1])-10
                    ret, baseline = cv2.getTextSize('%d'%id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(ori_image, (x1, y1 - ret[1]),(x1 + ret[0], y1), (255, 0, 0), -1) # 添加一个蓝底使得id更清楚

                    if xy_pts is None:
                        cv2.putText(ori_image, '%d (%s,%.2f)'%(id,cat,score), (int(cen_pts[0]), int(cen_pts[1])-10),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1, 1)
                    else:
                        txt_label = ''
                        if 'id' in bbox_label:
                            txt_label += '%d '%id
                        if 'cat' in bbox_label:
                            txt_label += '%s ' % cat
                        if 'score' in bbox_label:
                            txt_label += '%.2f ' % score
                        cv2.putText(ori_image, txt_label,
                                    (int(cen_pts[0]), int(cen_pts[1]) - 10),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1, 1)
                        if 'xy' in bbox_label:
                            cv2.putText(ori_image, '(%.2f,%.2f)' % (xy_pts[0], xy_pts[1]),
                                        (int(cen_pts[0]), int(cen_pts[1]) + 10),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1, 1)

                else:
                    ori_image = cv2.drawContours(ori_image, [np.int0(box)], -1, self.color_pans[int(pred[-1])], 2, 1)
                    cv2.putText(ori_image, '%s,%.2f'%(cat, score), (int(box[1][0]), int(box[1][1])),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1, 1)

            else:
                ori_image = cv2.drawContours(ori_image, [np.int0(box)], -1, (204,78,210), 1, 1)
                cv2.putText(ori_image, '%d'%bbox_id, (int(box[1][0]), int(box[1][1])),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1, 1)
            if lane_name is not None and 'lane' in bbox_label:
                cv2.putText(ori_image, '(lane:%d)' % (lane_name),
                            (int(cen_pts[0]), int(cen_pts[1]) + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1, 1)
        return ori_image