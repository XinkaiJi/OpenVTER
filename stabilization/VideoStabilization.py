#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2022-01-01 21:44
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : VideoStabilization.py 
@Software: PyCharm
@desc: 根据跟踪点进行视频稳定
'''
import time

import cv2
import os
import numpy as np
import pickle
from utils.VideoTool import get_all_video_info


class VideoStabilization:
    def __init__(self):
        self.mask = None
        self.raw_transforms = {}
        self.transforms = {}


    def init_stabilize(self, mask):
        self.mask = mask
        self.colors = []
        self.tracker_fixed_points = []

        # self._init_multitracker(tracker_type, first_frame)
        self.feature_params = dict(maxCorners=200,
                                   qualityLevel=0.05,
                                   minDistance=7,
                                   blockSize=10,
                                   useHarrisDetector=True,
                                   k=0.04)
        self.stabilize_gap = 30 # 所稳定到帧的间隔


    def _get_transform(self, prev_frame_gray, curr_gray, prev_pts):
        # 仿射变换
        # 6参数。三个点计算放射矩阵。
        if len(prev_pts)>=3:
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, curr_gray, prev_pts, None)
            assert prev_pts.shape == curr_pts.shape
            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]
            affine_matrix, inlier = cv2.estimateAffinePartial2D(curr_pts, prev_pts)
            affine_matrix_pre2curr, inlier = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
            # Extract traslation
            dx = affine_matrix[0, 2]
            dy = affine_matrix[1, 2]
            # Extract rotation angle
            da = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0])
            return [dx, dy, da], affine_matrix,affine_matrix_pre2curr
        else:
            print('Warning! Not enough key points, prev_pts<3 !')
            dx = 0
            dy = 0
            da = 0
            default_m = np.zeros((2, 3), np.float64)
            default_m[0, 0] = np.cos(da)
            default_m[0, 1] = -np.sin(da)
            default_m[1, 0] = np.sin(da)
            default_m[1, 1] = np.cos(da)
            default_m[0, 2] = dx
            default_m[1, 2] = dy
            return [dx, dy, da], default_m, default_m


        # affine_image = cv2.warpAffine(frame, affine_matrix, (self.cols,self.rows))
        # return affine_image

    def _raw_transform_to_xya(self):
        for key, m in self.raw_transforms.items():
            dx = m[0, 2]
            dy = m[1, 2]
            # Extract rotation angle
            da = np.arctan2(m[1, 0], m[0, 0])
            self.transforms[key] = [dx, dy, da]


    def stabilize_video(self, video_file_ls, save_folder, step=1, output_video=True, stab_file='raw_traj.pkl',
                        scale=True,video_start_frame=0,video_end_frame=0,stabilize_frame=None,smooth_xya=True,
                        translate=True,video_output_fps=1,skip_interval=None):
        '''

        :param stab_file:
        :param video_file_ls:
        :param save_folder:
        :param step: 1:output the stabilize pkl file 2:output stabilize video 3：output the stabilize pkl file and stabilize video
        :param output_video: output video or images
        :return:
        '''
        assert (step == 1 or step == 2 or step ==3 ),"step should be 1 or 2"
        if step == 1:
            print('start calculate video stabilization transformers')
        elif step == 2:
            print('start stable videos')
        elif step == 3:
            print('start calculate video stabilization transformers and stable videos')

        if isinstance(video_file_ls, str):
            video_file_ls = [video_file_ls]
        num_frame_ls, all_num_frame, width, height, fps = get_all_video_info(video_file_ls)
        first_video_name = video_file_ls[0]
        _, video_name_ext = os.path.split(first_video_name)
        video_name, extension = os.path.splitext(video_name_ext)
        if len(video_file_ls) == 1:
            save_path = os.path.join(save_folder, video_name)
        else:
            save_path = os.path.join(save_folder, video_name + '_Num_%d' % len(video_file_ls))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # pkl_path = os.path.join(save_path, 'sm_traj.pkl')
        raw_pkl_path = os.path.join(save_path, stab_file)

        frame_index = video_start_frame
        output_frame_gap = round(fps/video_output_fps)
        prev_transforms = None
        prev_transforms_pre2curr = None
        affine_matrix_pre2curr_dict = {}
        for video_index, video_file in enumerate(video_file_ls):
            cap = cv2.VideoCapture(video_file)
            valid_frames = num_frame_ls[video_index]
            # 第一个视频跳过video_start_frame帧
            if video_index == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES,video_start_frame)
                valid_frames -= video_start_frame
            if video_index == len(video_file_ls)-1:
                valid_frames -= video_end_frame
            print('\nInput Video:%s' % video_file)
            # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # fps = cap.get(cv2.CAP_PROP_FPS)
            # num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            print("Input Video:width:{} height:{} fps:{} num_frames:{} valid_frames:{}".format(width, height, fps,
                                                                               num_frame_ls[video_index],valid_frames))

            if (step == 2 or step ==3) and video_index == 0:
                if step == 2:
                    self.load_transforms(raw_pkl_path)
                if output_video:
                    video_format = 'mp4v'
                    out_path = os.path.join(save_path, 'stabilize_output_no_smooth.mp4')
                    fourcc = cv2.VideoWriter_fourcc(*video_format)
                    writer = cv2.VideoWriter(out_path, fourcc, video_output_fps, (int(width), int(height)))

            if not stabilize_frame is None:
                prev_frame_gray = cv2.cvtColor(stabilize_frame, cv2.COLOR_BGR2GRAY)
                prev_pts = cv2.goodFeaturesToTrack(
                    prev_frame_gray, mask=self.mask, **self.feature_params)
            video_frame_index = 0
            s_time = time.time()
            remain_time = 0
            current_mask = self.mask
            while video_frame_index<valid_frames:
                success, frame = cap.read()
                if not success:
                    cap.release()
                    break
                # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if step == 1 or step == 3:

                    if stabilize_frame is None and frame_index == video_start_frame:
                        prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        prev_pts = cv2.goodFeaturesToTrack(
                            prev_frame_gray, mask=current_mask, **self.feature_params)
                    else:
                        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # self.transforms[frame_index], self.raw_transforms[frame_index] = self._get_transform(
                        #     prev_frame_gray, curr_gray, prev_pts)

                        self.transforms[frame_index], raw_step_transforms,affine_matrix_pre2curr = self._get_transform(prev_frame_gray,
                                                                                                curr_gray, prev_pts)

                        if (frame_index % self.stabilize_gap == 0 or (prev_transforms is None)) and (not isInskip(frame_index,skip_interval)):
                            if not prev_transforms is None:
                                self.raw_transforms[frame_index] = np.matmul(np.row_stack((prev_transforms, [0, 0, 1])),
                                                                             np.row_stack((raw_step_transforms, [0, 0, 1])))[:2,:]
                                affine_matrix_pre2curr_dict[frame_index] = np.matmul(np.row_stack((prev_transforms_pre2curr, [0, 0, 1])),
                                                                             np.row_stack((affine_matrix_pre2curr, [0, 0, 1])))[:2,:]
                            else:
                                self.raw_transforms[frame_index] = raw_step_transforms
                                affine_matrix_pre2curr_dict[frame_index] = affine_matrix_pre2curr

                            prev_transforms = self.raw_transforms[frame_index]
                            prev_frame_gray = curr_gray

                            rows, cols = current_mask.shape
                            current_mask = cv2.warpAffine(self.mask, affine_matrix_pre2curr_dict[frame_index],
                                                          (cols, rows))
                            prev_pts = cv2.goodFeaturesToTrack(
                                prev_frame_gray, mask=current_mask, **self.feature_params)
                            prev_transforms_pre2curr = affine_matrix_pre2curr_dict[frame_index]


                        else:
                            self.raw_transforms[frame_index] = np.matmul(
                                np.row_stack((prev_transforms, [0, 0, 1])),
                                np.row_stack((raw_step_transforms, [0, 0, 1])))[:2, :]
                            affine_matrix_pre2curr_dict[frame_index] = np.matmul(
                                np.row_stack((prev_transforms_pre2curr, [0, 0, 1])),
                                np.row_stack((affine_matrix_pre2curr, [0, 0, 1])))[:2, :]


                if step ==2 or step ==3:
                    if frame_index % output_frame_gap == 0:
                        if step == 2:
                            affine_image = self._affine_transform(frame, frame_index, scale,smooth_xya,translate=translate)
                        else:
                            affine_image = self._affine_transform(frame, frame_index, scale, smooth_xya=False,
                                                                  translate=translate)
                        if not output_video:
                            file_folder = os.path.join(save_path, 'stabilize_img')
                            if not os.path.exists(file_folder):
                                os.makedirs(file_folder)
                            img_name = os.path.join(file_folder, '%06d.jpg' % frame_index)
                            cv2.imwrite(img_name, affine_image)
                        else:
                            writer.write(affine_image)
                frame_index += 1
                video_frame_index += 1
                if frame_index%10==0:
                    remain_time = (time.time() - s_time) * (all_num_frame - frame_index - 1)/10
                    cal_fps = 10 / (time.time() - s_time)
                    s_time = time.time()
                    # print('\rprocess frame:%d/%d, FPS:%.1f, remain time:%.2f min' % (
                    # frame_index, all_num_frame - video_end_frame, fps, remain_time / 60))
                    print('\rprocess frame:%d/%d, FPS:%.1f, remain time:%.2f min' % (frame_index, all_num_frame-video_end_frame,cal_fps, remain_time / 60), end="", flush=True)

        if step == 1 or step == 3:
            self._save_transforms(raw_pkl_path)
            print('\nsave transforms file:%s'%raw_pkl_path)
        if step ==2 or step == 3:
            if output_video:
                writer.release()

    def _smooth_transforms(self, SMOOTHING_RADIUS):
        trajectory = self.transforms
        smoothed_trajectory = smooth(trajectory, SMOOTHING_RADIUS=SMOOTHING_RADIUS)
        # Calculate difference in smoothed_trajectory and trajectory
        # difference = smoothed_trajectory - trajectory
        # Calculate newer transformation array
        self.transforms_smooth = smoothed_trajectory # self.transforms + difference

    def _save_transforms(self, raw_pkl_path):
        # with open(pkl_path,'wb') as f:
        #     pickle.dump(self.transforms,f)
        with open(raw_pkl_path, 'wb') as f:
            pickle.dump(self.raw_transforms, f)

    def load_transforms(self, raw_path):
        # with open(path,'rb') as f:
        #     self.transforms = pickle.load(f)
        with open(raw_path, 'rb') as f:
            self.raw_transforms = pickle.load(f)
        self._raw_transform_to_xya()
        # self.transforms_smooth = self.transforms
        self._smooth_transforms(3)

    def _affine_transform(self, frame, frame_index,scale=True,smooth_xya=False,translate=True):
        dx = 0
        dy = 0
        da = 0
        default_m = np.zeros((2, 3), np.float64)
        default_m[0, 0] = np.cos(da)
        default_m[0, 1] = -np.sin(da)
        default_m[1, 0] = np.sin(da)
        default_m[1, 1] = np.cos(da)
        default_m[0, 2] = dx
        default_m[1, 2] = dy
        if smooth_xya:
            m = self.transforms_smooth.get(frame_index, default_m)
            if not translate:
                m[0, 2] = 0
                m[1, 2] = 0
        else:
            m = self.raw_transforms.get(frame_index, default_m)
            if not scale:
                if translate:
                    # Extract traslation
                    dx = m[0, 2]
                    dy = m[1, 2]
                else:
                    dx = 0
                    dy = 0
                # Extract rotation angle
                da = np.arctan2(m[1, 0], m[0, 0])
                new_m = np.zeros((2, 3), np.float64)
                new_m[0, 0] = np.cos(da)
                new_m[0, 1] = -np.sin(da)
                new_m[1, 0] = np.sin(da)
                new_m[1, 1] = np.cos(da)
                new_m[0, 2] = dx
                new_m[1, 2] = dy
                m = new_m
        rows, cols, _ = frame.shape
        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (cols, rows))
        return frame_stabilized

    def stabilize_frame(self, frame, frame_index,scale=True,smooth_xya=False,translate=True):
        return self._affine_transform(frame, frame_index,scale=scale,smooth_xya=smooth_xya,translate=translate)

def isInskip(frame_index,skip_interval):
    if skip_interval is None:
        return False
    for interval in skip_interval:
        if frame_index>= interval[0] and frame_index <= interval[1]:
            return True
    return False

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth(trajectory, SMOOTHING_RADIUS=5):
    start_frame_index = np.min(list(trajectory.keys()))
    num_frame = len(trajectory)
    np_trajectory = []
    for i in range(start_frame_index,start_frame_index+num_frame):
        np_trajectory.append(trajectory[i])
    np_trajectory = np.array(np_trajectory)
    smoothed_trajectory = np.copy(np_trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(np_trajectory[:, i], radius=SMOOTHING_RADIUS)
    res_trajectory = {}
    for i in range(start_frame_index,start_frame_index+num_frame):
        np_index = i-start_frame_index
        dx,dy,da = smoothed_trajectory[np_index].tolist()
        default_m = np.zeros((2, 3), np.float64)
        default_m[0, 0] = np.cos(da)
        default_m[0, 1] = -np.sin(da)
        default_m[1, 0] = np.sin(da)
        default_m[1, 1] = np.cos(da)
        default_m[0, 2] = dx
        default_m[1, 2] = dy
        res_trajectory[i] = default_m

    return res_trajectory


import json


def load_fixed_points(file_name):
    mask = None
    fixed_points = []
    with open(file_name, 'r') as f:
        data = json.load(f)
        if data['imageWidth'] and data['imageHeight'] and data['shapes']:
            width = data['imageWidth']
            height = data['imageHeight']
            mask = np.zeros((height, width))
            for shape in data['shapes']:
                if shape['label'] == 'fp':
                    points = shape['points']
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    mask[int(y1):int(y2), int(x1):int(x2)] = 255
                    w = x2 - x1
                    h = y2 - y1
                    fixed_points.append([x1, y1, w, h])
    return np.uint8(mask), fixed_points


if __name__ == '__main__':
    # fixed_points =

    # mask, fixed_points = load_fixed_points('/home/data2/maopeipei/20211228/20211228_3_A_498.6_1.json')
    # vs = VideoStabilization()
    # vs.init_stabilize(mask)
    # video_folder = '/home/data2/maopeipei/20211228'
    # # video_name_ls = ['20220108_a_A_399.6_B_1.MP4','20220108_a_A_399.6_B_2.MP4','20220108_a_A_399.6_B_3.MP4','20220108_a_A_399.6_B_4.MP4']
    # video_name_ls = ["20211228_3_A_498.6_1.MP4", "20211228_3_A_498.6_2.MP4", "20211228_3_A_498.6_3.MP4"]
    # video_file_ls = []
    # for video_name in video_name_ls:
    #     video_file_ls.append(os.path.join(video_folder, video_name))
    # save_folder = '/home/data2/peipeimao/DJIProcessVideo'
    # step = 2
    # vs.stabilize_video(video_file_ls, save_folder, step, output_video=True)
    vs = VideoStabilization()
