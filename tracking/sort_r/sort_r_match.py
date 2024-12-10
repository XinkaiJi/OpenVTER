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
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

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
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
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

    iou_matrix = iou_batch(detections, trackers)

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


class Sort_R_match(object):
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
