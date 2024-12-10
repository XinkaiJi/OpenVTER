import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv.ops import box_iou_rotated
import cv2
import lap
from scipy.optimize import linear_sum_assignment
from .deep.feature_extractor import Extractor, FastReIDExtractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker

__all__ = ['DeepSort']

def linear_assignment(cost_matrix):
    try:

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def rbb_to_hbb(dets):
    '''
    convert rotated bounding boxes to horizontal bounding boxes
    :param dets: torch , [[center_x, center_y,bbox_w, bbox_h,theta,x1,y1,x1,y2,x2,y3,x4,y4,score,category],..]
    :return:numpy array bbox_xywh, confidences, classes
    '''
    if dets.device.type == 'cpu':
        rotated_bbox_np = dets.numpy()
    else:
        rotated_bbox_np = dets.data.cpu().numpy()
    object_num = len(rotated_bbox_np)
    xy = rotated_bbox_np[:, 5:13].reshape(object_num, -1, 2)
    x1_y1 = xy.min(axis=1)
    x2_y2 = xy.max(axis=1)
    # res = np.hstack((x1_y1, x2_y2, rotated_bbox_np[:, 13:14]))
    w = x2_y2[:, 0] - x1_y1[:, 0]
    h = x2_y2[:, 1] - x1_y1[:, 1]
    cx = (x1_y1[:, 0] + x2_y2[:, 0]) / 2
    cy = (x1_y1[:, 1] + x2_y2[:, 1]) / 2

    bbox_xywh = np.stack((cx, cy, w, h), axis=1)
    confidences = rotated_bbox_np[:, 13]
    classes = rotated_bbox_np[:, 14]

    return bbox_xywh, confidences, classes



def match_hbbox_to_obbox(o_bbox, h_bbox, device, iou_threshold1=0.9, iou_threshold2=0.2):
    # iou_matrix = iou_rotated_and_bbox(o_bbox, h_bbox)

    iou_matrix = iou_rotated_and_bbox_mmcv(o_bbox, h_bbox, device)
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
            id = trk[-1]
            res_bbox = np.array([x1, y1, x2, y1, x2, y2, x1, y2, score, cat, id]).reshape(1, -1)
            ret.append(res_bbox)
        else:
            trk = h_bbox[m[0]]
            id = trk[-1]
            x = o_bbox_np[m[1]]

            score = x[-2]
            cat = x[-1]
            res_bbox = np.array([x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], score, cat, id]).reshape(1, -1)
            ret.append(res_bbox)
    if (len(ret) > 0):
        return np.concatenate(ret)
    return np.empty((0, 11))



def iou_rotated_and_bbox_mmcv(o_bbox, h_bbox, device):
    np_boxes2 = []
    for t, trk in enumerate(h_bbox):
        x1, y1, x2, y2 = trk[0], trk[1], trk[2], trk[3]
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        rect = cv2.minAreaRect(pts)
        center_x, center_y = rect[0][0], rect[0][1]
        bbox_w, bbox_h = rect[1][0], rect[1][1]
        theta = rect[2] * np.pi / 180
        trk_r_bbox = [center_x, center_y, bbox_w, bbox_h, theta]
        np_boxes2.append(trk_r_bbox)
    np_boxes2 = np.array(np_boxes2, dtype=np.float32)
    for i in range(o_bbox.shape[0]):
        x1, y1, x2, y2 = o_bbox[i, 5], o_bbox[i, 6], o_bbox[i, 7], o_bbox[i, 8]
        x3, y3, x4, y4 = o_bbox[i, 9], o_bbox[i, 10], o_bbox[i, 11], o_bbox[i, 12]
        pts = np.array([[x1, y1], [x2, y1], [x3, y3], [x4, y4]], dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        center_x, center_y = rect[0][0], rect[0][1]
        bbox_w, bbox_h = rect[1][0], rect[1][1]
        theta = rect[2] * np.pi / 180
        o_bbox[i, :5] = torch.tensor([center_x, center_y, bbox_w, bbox_h, theta])
    if device.type == 'cpu':
        boxes2 = torch.from_numpy(np_boxes2)
        if o_bbox.device.type != 'cpu':
            o_bbox = o_bbox.data.cpu()
    else:
        boxes2 = torch.from_numpy(np_boxes2).to(device)
        if o_bbox.device.type == 'cpu':
            o_bbox = o_bbox.to(device)



    ious = box_iou_rotated(boxes2, o_bbox[:, :5], 'iou')
    if device.type == 'cpu':
        return ious.numpy()
    else:
        return ious.cpu().numpy()
class DeepSort(object):
    def __init__(self, model_path, model_config=None, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0,
                 max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True,match_iou_threshold1=0.9,match_iou_threshold2=0.2):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.match_iou_threshold1 = match_iou_threshold1
        self.match_iou_threshold2 = match_iou_threshold2
        if model_config is None:
            self.extractor = Extractor(model_path, use_cuda=use_cuda)
        else:
            self.extractor = FastReIDExtractor(model_config, model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, classes, ori_img, masks=None):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, label, features[i], None if masks is None else masks[i])
                      for i, (conf, label) in enumerate(zip(confidences, classes))
                      if conf > self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        mask_outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            track_cls = track.cls
            outputs.append(np.array([x1, y1, x2, y2, track_cls, track_id], dtype=np.int32))
            if track.mask is not None:
                mask_outputs.append(track.mask)
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs, mask_outputs

    def add_det_data(self, dets= torch.empty((0,15)),ori_img=None, device=torch.device('cpu')):
        '''

        :param device:
        :param dets: numpy array, center_x, center_y,bbox_w, bbox_h,theta,x1,y1,x2,y2,x3,y3,x4,y4,score,category
        :return: numpy array, x1,y1,x2,y2,x3,y3,x4,y4,score,category,id
        '''
        if len(dets) == 0:
            self.update(np.empty((0, 4)),np.empty((0, 1)),np.empty((0, 1)),ori_img)
            return np.empty((0, 11)) #
        else:
            # if device.type == 'cpu' and dets.device.type != 'cpu':
            #     dets = dets.cpu()
            bbox_xywh, confidences, classes = rbb_to_hbb(dets)
            # sort
            track_bbs_ids,_ = self.update(bbox_xywh, confidences, classes,ori_img)
            # o_bboxs_res numpy array: x1, y1, x2, y1, x2, y2, x1, y2,score,cat,id
            o_bboxs_res = match_hbbox_to_obbox(dets, track_bbs_ids, device,iou_threshold1=self.match_iou_threshold1,iou_threshold2=self.match_iou_threshold2)
            return o_bboxs_res



    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    @staticmethod
    def _xyxy_to_tlwh(bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            # plt.figure()
            # plt.imshow(im)
            # plt.show()
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
