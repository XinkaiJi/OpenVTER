"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
# import matplotlib

# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
import cv2
from mmcv.ops import box_iou_rotated
import torch
import lap
from scipy.optimize import linear_sum_assignment
np.random.seed(0)


def linear_assignment(cost_matrix):
    try:

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    # w = bbox[2] - bbox[0]
    # h = bbox[3] - bbox[1]
    # x = bbox[0] + w / 2.
    # y = bbox[1] + h / 2.
    # s = w * h  # scale is just area
    # r = w / float(h)
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    theta = bbox[4]
    s = w * h  # scale is just area
    r = w / float(h)
    if theta> np.pi/2.0 or theta<-np.pi/2.0:
        print('convert_bbox_to_z theta',theta)
        theta = (theta+np.pi/2.0)%np.pi-np.pi/2.0
    return np.array([x, y, s, r,theta]).reshape((5, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    theta = x[4]
    if theta> np.pi/2.0 or theta<-np.pi/2.0:
        print('convert_x_to_bbox theta',theta)
        theta = (theta+np.pi/2.0)%np.pi-np.pi/2.0

    if (score == None):
        return np.array([x[0], x[1], w, h,theta]).reshape((1, 5))
    else:
        return np.array([x[0], x[1], w, h,theta, score]).reshape((1,6))

def rbb_to_hbb(dets):
    '''
    convert rotated bounding boxes to horizontal bounding boxes
    :param dets: torch , [[center_x, center_y,bbox_w, bbox_h,theta,x1,y1,x1,y2,x2,y3,x4,y4,score,category],..]
    :return:numpy array,[[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    '''
    if dets.device.type == 'cpu':
        rotated_bbox_np = dets.numpy()
    else:
        rotated_bbox_np = dets.data.cpu().numpy()
    object_num = len(rotated_bbox_np)
    xy = rotated_bbox_np[:,5:13].reshape(object_num,-1,2)
    x1_y1 = xy.min(axis=1)
    x2_y2 = xy.max(axis=1)
    res = np.hstack((x1_y1,x2_y2,rotated_bbox_np[:,13:14]))
    return res

def iou_rotated_and_bbox_mmcv(o_bbox,h_bbox,device):
    np_boxes2 = []
    for t, trk in enumerate(h_bbox):
        x1, y1, x2, y2 = trk[0], trk[1], trk[2], trk[3]
        pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],dtype=np.float32)

        rect = cv2.minAreaRect(pts)
        center_x, center_y = rect[0][0], rect[0][1]
        bbox_w, bbox_h = rect[1][0], rect[1][1]
        theta = rect[2] * np.pi / 180
        trk_r_bbox =[center_x, center_y, bbox_w, bbox_h, theta]
        np_boxes2.append(trk_r_bbox)
    np_boxes2 = np.array(np_boxes2,dtype=np.float32)
    if device.type == 'cpu':
        boxes2 = torch.from_numpy(np_boxes2)
        if o_bbox.device.type != 'cpu':
            o_bbox = o_bbox.data.cpu()
    else:
        boxes2 = torch.from_numpy(np_boxes2).to(device)
        if o_bbox.device.type == 'cpu':
            o_bbox = o_bbox.to(device)

    ious = box_iou_rotated(boxes2,o_bbox[:,:5],'iou')
    if device.type == 'cpu':
        return ious.numpy()
    else:
        return ious.cpu().numpy()



def match_hbbox_to_obbox(o_bbox,h_bbox,device,iou_threshold1=0.9,iou_threshold2=0.2):
    # iou_matrix = iou_rotated_and_bbox(o_bbox, h_bbox)

    iou_matrix = iou_rotated_and_bbox_mmcv(o_bbox, h_bbox,device)
    if o_bbox.device.type == 'cpu':
        o_bbox_np = o_bbox.numpy()
    else:
        o_bbox_np = o_bbox.cpu().numpy()
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold2).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    ret = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] > iou_threshold1):
            r_bbox = o_bbox_np[m[1]]
            score = r_bbox[-2]
            cat = r_bbox[-1]
            trk = h_bbox[m[0]]
            x1, y1, x2, y2 = trk[0], trk[1], trk[2], trk[3]
            id = trk[4]
            res_bbox =np.array([x1, y1, x2, y1, x2, y2, x1, y2,score,cat,id]).reshape(1, -1)
            ret.append(res_bbox)
        else:
            trk = h_bbox[m[0]]
            id = trk[4]
            x = o_bbox_np[m[1]]

            score = x[-2]
            cat =x[-1]
            res_bbox = np.array([x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12], score, cat, id]).reshape(1, -1)
            ret.append(res_bbox)
    if (len(ret) > 0):
        return np.concatenate(ret)
    return np.empty((0, 11))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        dim_z = 5 # 观测变量
        dim_x = 8 # 状态变量
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        # self.kf.F = np.array(
        #     [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
        #      [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.F =  np.eye(dim_x)
        self.kf.F[0,5] = 1
        self.kf.F[1, 6] = 1
        self.kf.F[2, 7] = 1
        # self.kf.F[4, 8] = 1
        self.kf.H = np.zeros((dim_z, dim_x))
        self.kf.H[0, 0] = 1
        self.kf.H[1, 1] = 1
        self.kf.H[2, 2] = 1
        self.kf.H[3, 3] = 1
        self.kf.H[4, 4] = 1
        # self.kf.H = np.array(
        #     [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[5:, 5:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.Q[4, 4] *= 0
        TIME_INTERVAL = 1
        MAX_SPEED = 5
        MAX_TR = 3
        # self.kf.Q[0, 0] = (TIME_INTERVAL * MAX_SPEED) ** 2
        # self.kf.Q[1, 1] = (TIME_INTERVAL * MAX_SPEED) ** 2
        # self.kf.Q[4, 4] = (TIME_INTERVAL * MAX_TR) ** 2
        # self.kf.Q[3, 3] = MAX_SPEED ** 2
        # self.kf.Q[4, 4] = MAX_SPEED ** 2
        # self.kf.Q[5, 5] = MAX_TR ** 2
        # self.kf.Q[6, 6] = 1 ** 2
        # self.kf.Q[7, 7] = 1 ** 2

        self.kf.x[:5] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        # theta = self.kf.x[4]
        # theta = (theta + np.pi / 2.0) % np.pi - np.pi / 2.0
        # self.kf.x[4] = theta
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        theta = self.kf.x[4]
        if theta > np.pi / 2.0 or theta < -np.pi / 2.0:
            print('convert_x_to_bbox theta',theta)
            print(self.kf.x)
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # iou_matrix = iou_batch(detections, trackers)
    boxes1 = np.array(detections[:, :5], dtype=np.float32)
    boxes2 = np.array(trackers[:, :5], dtype=np.float32)
    boxes1 = torch.from_numpy(boxes1)
    boxes2 = torch.from_numpy(boxes2)
    iou_matrix = box_iou_rotated(boxes1, boxes2, 'iou').numpy()
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))


    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort_bak(object):
    '''
    匹配法
    '''
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3,match_iou_threshold1=0.9,match_iou_threshold2=0.2):
        """
        max_age表示在多少帧中没有检测，trackers就会中止。min_hits代表持续多少帧检测到，生成trackers。
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.match_iou_threshold1 = match_iou_threshold1
        self.match_iou_threshold2 = match_iou_threshold2

    def add_det_data(self, dets= torch.empty((0,15)), device=torch.device('cpu')):
        '''

        :param device:
        :param dets: numpy array, center_x, center_y,bbox_w, bbox_h,theta,x1,y1,x2,y2,x3,y3,x4,y4,score,category
        :return: numpy array, x1,y1,x2,y2,x3,y3,x4,y4,score,category,id
        '''
        if len(dets) == 0:
            self.update()
            return np.empty((0, 11)) #
        else:
            # if device.type == 'cpu' and dets.device.type != 'cpu':
            #     dets = dets.cpu()
            h_bbox = rbb_to_hbb(dets)
            # sort
            track_bbs_ids = self.update(h_bbox)
            # o_bboxs_res numpy array: x1, y1, x2, y1, x2, y2, x1, y2,score,cat,id
            o_bboxs_res = match_hbbox_to_obbox(dets, track_bbs_ids, device,iou_threshold1=self.match_iou_threshold1,iou_threshold2=self.match_iou_threshold2)
            return o_bboxs_res

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))


class Sort():
    '''
    长边90法表示
    带角度表示
    '''
    def __init__(self,max_age=1, min_hits=3, iou_threshold=0.3,match_iou_threshold1=0.9,match_iou_threshold2=0.2 ):
        """
        max_age表示在多少帧中没有检测，trackers就会中止。min_hits代表持续多少帧检测到，生成trackers。
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.match_iou_threshold1 = match_iou_threshold1
        self.match_iou_threshold2 = match_iou_threshold2

    def update(self, dets=np.empty((0, 7))):
        """
                Params:
                  dets - a numpy array of detections in the format [[x,y,w,h,t,score,cls],[x,y,w,h,t,score],...]
                Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 6)) for frames without detections).
                Returns the a similar array, where the last column is the object ID.

                NOTE: The number of objects returned may differ from the number of detections provided.
                """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t][0].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3],pos[4], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]][0].update(dets[m[0], :])
            self.trackers[m[1]][1] =dets[m[0], -3:]

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            bbox_info = dets[i, -3:] # theta,score cls
            self.trackers.append([trk,bbox_info])
        i = len(self.trackers)
        for trk,bbox_info in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d[:4], [d[4],bbox_info[1],bbox_info[2],trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 7))

    def add_det_data(self, dets= torch.empty((0,15)), device=torch.device('cpu')):
        '''
        :param device:
        :param dets: numpy array, center_x, center_y,bbox_w, bbox_h,theta,x1,y1,x2,y2,x3,y3,x4,y4,score,category
        :return: numpy array, x1,y1,x2,y2,x3,y3,x4,y4,score,category,id
        '''
        if len(dets) == 0:
            self.update()
            return np.empty((0, 11)) #
        else:
            # if device.type == 'cpu' and dets.device.type != 'cpu':
            #     dets = dets.cpu()
            rbbangle = det_to_rbbangle(dets)
            # sort
            track_bbs_ids = self.update(rbbangle)
            o_bboxs_res = rbox2corner(track_bbs_ids)
            # o_bboxs_res numpy array: x1, y1, x2, y1, x2, y2, x1, y2,score,cat,id

            return o_bboxs_res

def det_to_rbbangle(dets):
    '''
    convert rotated bounding boxes to horizontal bounding boxes
    :param dets: torch , [[center_x, center_y,bbox_w, bbox_h,theta,x1,y1,x2,y2,x2,y3,x4,y4,score,category],..]
    :return:numpy array,[[x,y,w,h,t,score,cls],[x,y,w,h,t,score,cls],...]
    '''
    if dets.device.type == 'cpu':
        rotated_bbox_np = dets.numpy()
    else:
        rotated_bbox_np = dets.data.cpu().numpy()
    # object_num = len(rotated_bbox_np)
    # xy = rotated_bbox_np[:,5:13].reshape(object_num,-1,2)
    res = corner2rbb(rotated_bbox_np[:,5:13])
    res = np.hstack((res,rotated_bbox_np[:,13:]))
    return res

def corner2rbb(corners):
    original_shape = corners.shape[0]
    points = corners.reshape(-1, 4, 2)
    rboxes = []
    for pts in points:
        (x, y), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([x, y, w, h, angle / 180 * np.pi])
    rboxes = np.array(rboxes)
    start_angle = -90
    start_angle = start_angle / 180 * np.pi
    width_longer = True
    x, y, w, h, t = rboxes[:, 0], rboxes[:, 1], rboxes[:, 2], rboxes[:, 3], rboxes[:, 4]
    if width_longer:
        # swap edge and angle if h >= w
        w_ = np.where(w > h, w, h)
        h_ = np.where(w > h, h, w)
        t = np.where(w > h, t, t + np.pi / 2)
        t = ((t - start_angle) % np.pi) + start_angle
    rboxes = np.stack([x, y, w_, h_, t], axis=-1)
    return rboxes.reshape(original_shape, 5)


def rbox2corner(rboxes):
    """
    Convert rotated box (x, y, w, h, t) to corners ((x1, y1), (x2, y1),
    (x1, y2), (x2, y2)).

    Args:
        boxes (ndarray): Rotated box array with shape of (..., 5).

    Returns:
        ndarray: Corner array with shape of (..., 8).
    """
    # start_angle = -90
    # start_angle = start_angle / 180 * np.pi
    # x, y, w, h, t = rboxes[:, 0], rboxes[:, 1], rboxes[:, 2], rboxes[:, 3], rboxes[:, 4]
    # t = ((t - start_angle) % np.pi)
    # w_ = np.where(t < np.pi / 2, w, h)
    # h_ = np.where(t < np.pi / 2, h, w)
    # t = np.where(t < np.pi / 2, t, t - np.pi / 2) + start_angle
    # rboxes = np.stack([x, y, w_, h_, t], axis=-1)
    ctr = rboxes[..., :2]
    w = rboxes[..., 2:3]
    h = rboxes[..., 3:4]
    theta = rboxes[..., 4:5]

    cos_value = np.cos(theta)
    sin_value = np.sin(theta)

    vec1 = np.concatenate([w / 2 * cos_value, w / 2 * sin_value], axis=-1)
    vec2 = np.concatenate([-h / 2 * sin_value, h / 2 * cos_value], axis=-1)

    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    corners = np.stack([pt1, pt2, pt3, pt4], axis=-2).reshape(-1, 8)
    if rboxes.shape[1] > 5:
        corners = np.concatenate([corners, rboxes[..., 5:]], axis=1)
    return corners


import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
def plot_rotated_rectangles(rectangles):
    fig, ax = plt.subplots()
    for rect in rectangles:
        corners = np.reshape(rect, (4, 2))
        polygon = Polygon(corners, closed=True, edgecolor='r', fill=None)
        ax.add_patch(polygon)
        for (x, y) in corners:
            ax.plot(x, y, 'bo')
    plt.gca().invert_yaxis()
    ax.set_aspect('equal', 'box')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Rotated Rectangles")
    plt.show()
# 示例顶点坐标的 numpy 矩阵，每行表示一个矩形
# rectangles = np.array([
#     [1, 1, 4,1, 5, 6, 2, 6],
#     [2, 3, 5, 4, 4, 7, 1, 6],
#     [1, 2, 4, 2, 4, 6, 1, 6],
# [1, 1, 4,1, 3, 6, 0, 6],
# [5, 6, 5,4, 8, 4, 8, 6]
# ])
# rbbs = corner2rbb(rectangles)
# print(rbbs)
# corners = rbox2corner(rbbs)
# print(corners)
# print(corner2rbb(rectangles))
# 计算所有矩形的长边、短边和角度
# long_edges, short_edges, angles = calculate_rotated_bbox_properties(rectangles)

# print("长边长度:", long_edges)
# print("短边长度:", short_edges)
# print("角度 (弧度):", angles)
# print("角度 (度数):", np.degrees(angles))
# 可视化矩形
# plot_rotated_rectangles(rectangles)
# plot_rotated_rectangles(corners)
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    print('hello world')
