#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2021-12-20 21:24
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : video_process.py 
@Software: PyCharm
@desc: 多视频拼接处理
'''
import os

import cv2
import json
import pickle
import numpy as np
import torch
from mmcv.ops import nms_rotated


from detection.VehicleDetModule import VehicleDetModule
from tracking.VehicleTrackingModule import VehicleTrackingModule

from stabilization.VideoStabilization import VideoStabilization

import time
import datetime
from utils import Config,RoadConfig,isPointinPolygon
from utils.VideoTool import get_all_video_info,get_srt,splitbase
from utils.MultiVideos import MultiVideos

class DroneVideoProcess:
    def __init__(self,config_json):
        config_dict = Config.fromfile(config_json)

        self.split_gap = config_dict.get('split_gap',100)
        subsize_height = config_dict.get('subsize_height',640)
        subsize_width = config_dict.get('subsize_width',640)
        self.split = splitbase('', '', gap=self.split_gap, subsize_height=subsize_height, subsize_width=subsize_width)


        self.output_video = config_dict.get('output_video',1) # 是否输出视频,0:不输出，1:输出
        self.output_img = config_dict.get('output_img',0) # 是否输出图片,0:不输出，1:输出

        self.sub_positions = None
        self.out_fps = config_dict.get('out_fps') #输出数据的fps，考虑抽帧的方式
        self.conf_thresh = config_dict.get('conf_thresh',0.25) # 目标检测置信度

        self.save_folder = config_dict.get('save_folder') # 输出路径
        self.inference_batch_size = config_dict.get('inference_batch_size',1) # 推理时候的batch大小，batch中图片是小图片



        self.stabilize_scale = config_dict.get('stabilize_scale', 1)
        self.stabilize_smooth = config_dict.get('stabilize_smooth', 1)
        self.stabilize_translate = config_dict.get('stabilize_translate', 1)
        self.output_background = config_dict.get('output_background', 1) # 输出背景图片
        self.background_image_ls = []

        self.bbox_label = config_dict.get('bbox_label',['id','score','xy'])

        self.multi_videos = self._init_multi_videos(config_json)

        self.video_name = self.multi_videos.video_name
        self.save_folder = os.path.join(self.save_folder,self.video_name)

        os.makedirs(self.save_folder,exist_ok=True)

        self.road_config = self.multi_videos.main_video.road_config
        # 目标检测
        self.det_model = self._get_det_model(config_dict.get('detection'))

        # 跟踪模型
        self.mot_tracker = self._get_tracking_model(config_dict.get('tracking'))

        self.det_bbox_result = {'video_info': [], 'output_info': {'output_fps': self.out_fps}, 'traj_info': [],
                                'process_time': datetime.datetime.now(),'raw_det':[]}



    def _init_multi_videos(self,multi_config_json):
        '''
        初始化多视频
        :return:
        '''
        multi_video = MultiVideos(multi_config_json)
        return multi_video

    def _get_det_model(self,config):
        '''
        获取检测模型 并加载参数
        :param config:
        :return:
        '''
        det_model = VehicleDetModule(**config)
        det_model.load_model()
        return det_model

    def _get_tracking_model(self,config):
        '''
        获取跟踪模型
        :param config:
        :return:
        '''
        tracking_model = VehicleTrackingModule(**config)
        return tracking_model

    def process_multi_video(self):
        '''
        多视频处理
        :return:
        '''
        # 初始化
        gap_length = self.road_config['length_per_pixel']*self.split_gap
        print('gap_length:%f m'%gap_length)
        num_frame_ls,all_num_frame,width,height,fps = self.multi_videos.get_video_info()


        # video_info = {'video_name': self.video_file, 'width': width, 'height': height, 'fps': fps,
        #               'total_frames': all_num_frame}
        # self.det_bbox_result['video_info'].append(video_info)
        print("Output Folder:%s" % self.save_folder)

        gap = round(fps/self.out_fps)
        print('Frame gap:%d'%gap)
        video_writer = None
        if self.output_video:
            video_writer,self.output_video_width, self.output_video_height = self.multi_videos.init_video_write(self.save_folder,self.out_fps)

        read_frame_index = -1
        output_frame_index = 0
        s_time = time.time()
        try:
            while 1:
                read_frame_index += 1
                sub_imgs,stitch_img,main_frame_index, main_unix_time = self._read_multi_video_frame()
                if read_frame_index == 0:
                    first_frame_name = os.path.join(self.save_folder, 'first_frame_' + self.video_name + '.jpg')
                    cv2.imwrite(first_frame_name, stitch_img)
                if len(sub_imgs) == 0:
                    break
                if read_frame_index % gap == 0:
                    # self._process_sub_imgs(sub_imgs,stitch_img,output_frame_index,main_frame_index,main_unix_time,video_writer)
                    # output_frame_index += 1
                    # e_time = time.time()
                    # remain_time = (e_time - s_time) * (all_num_frame // gap - 1)
                    # process_fps = 1 / (e_time - s_time)
                    # print('\rprocess main frame:%d/%d, FPS:%.1f, remain time:%.2f min' % (
                    # main_frame_index, all_num_frame,
                    # process_fps, remain_time / 60), end="", flush=True)
                    self._process_sub_imgs(sub_imgs,stitch_img,output_frame_index,main_frame_index,main_unix_time,video_writer)
                    output_frame_index += 1
                    e_time = time.time()
                    remain_time = (e_time - s_time) * (all_num_frame // gap - 1)
                    process_fps = 1 / (e_time - s_time)
                    print('\rprocess main frame:%d/%d, FPS:%.1f, remain time:%.2f min' % (
                    main_frame_index, all_num_frame,
                    process_fps, remain_time / 60))
                    s_time = e_time
        finally:
            if self.output_video:
                video_writer.release()
            self._save_det_bbox(self.save_folder)
            print('save video')


    def _read_multi_video_frame(self):
        '''
        读取多视频的合成画面
        :return:
        '''
        #
        ret,stitch_img,main_frame_index, main_unix_time = self.multi_videos.get_align_frame()
        if ret:
            # 切分图片
            if self.sub_positions is None:
                sub_imgs, sub_positions = self.split.split_image(stitch_img)
                self.sub_positions = sub_positions
            else:
                sub_imgs = self.split.split_image_with_position(stitch_img, self.sub_positions)
            return sub_imgs,stitch_img,main_frame_index, main_unix_time
        else:
            return [],None,None,None

    def _process_sub_imgs(self,sub_imgs,frame,output_frame,frame_index,unix_time,video_writer):

        # 输出背景图片
        if self.output_background and (not self.background_image_ls is None):
            if len(self.background_image_ls)<50:
                self.background_image_ls.append(frame.copy())
            else:
                background_img = np.zeros(frame.shape)
                # 图片叠加
                for image_b in self.background_image_ls:
                    background_img += image_b
                background_img = background_img / len(self.background_image_ls)
                background_path = os.path.join(self.save_folder, 'background_%s.jpg'%self.video_name)
                cv2.imwrite(background_path, background_img)
                print('output background image to:%s'%background_path)
                self.background_image_ls = None

        # 检测
        new_nms_ls = []
        for i in range(0,len(sub_imgs),self.inference_batch_size):
            s = i
            e = min(i+self.inference_batch_size,len(sub_imgs))
            select_imgs = sub_imgs[s:e]
            select_positions = self.sub_positions[s:e]
            nms_results_ls = self.det_model.inference_img_batch(select_imgs)
            for nms_results,position in zip(nms_results_ls,select_positions):
                if nms_results is None:
                    continue
                x, y = position

                position_arr = np.array([x, y, 0, 0, 0, x, y, x, y, x, y, x, y,0, 0],dtype=np.float32)
                position_arr_t = torch.from_numpy(position_arr).to(self.det_model.device)
                new_nms = nms_results + position_arr_t
                new_nms_ls.append(new_nms)
        # 添加坐标轴
        # if self.axis_image is not None:
        #     frame = cv2.add(frame, self.axis_image)
        # 跟踪
        det_raw = np.empty((0, 15))
        if len(new_nms_ls) == 0:
            self.mot_tracker.update()
            nms_img = frame
            if self.road_config['pixel2xy_matrix'] is not None:
                if len(self.road_config['lane']) == 0:
                    o_bboxs_res = np.empty((0, 19))
                else:
                    o_bboxs_res = np.empty((0, 20))
            else:
                o_bboxs_res = np.empty((0, 11))
        else:

            all_bbox = torch.vstack(new_nms_ls)
            dets, keep_inds = nms_rotated(all_bbox[:, :5], all_bbox[:, 5], 0.3)
            nms_all_bbox = all_bbox[keep_inds]
            # 根据面积求nms 保留面积大的
            # area = nms_all_bbox[:, 2]*nms_all_bbox[:, 3]
            # dets, keep_inds = nms_rotated(nms_all_bbox[:, :5], area, 0.1)
            # nms_all_bbox = nms_all_bbox[keep_inds]
            if nms_all_bbox.device.type == 'cpu':
                det_raw = nms_all_bbox.numpy()
            else:
                det_raw = nms_all_bbox.data.cpu().numpy()  # shape n*15

            o_bboxs_res = self.mot_tracker.update(nms_all_bbox)  # output shape n*11

            # 像素坐标转地理坐标
            o_bboxs_res = self._pixel_to_xy(o_bboxs_res)  # n*19
            o_bboxs_res = self._get_lane_id(o_bboxs_res)

            nms_img = self.det_model.draw_oriented_bboxs(frame, o_bboxs_res, self.bbox_label)

        if self.output_img:
            nms_img = cv2.resize(nms_img, (self.output_video_width, self.output_video_height))
            file_folder = os.path.join(self.save_folder, 'det_img_' + self.video_name)
            if not os.path.exists(file_folder):
                os.makedirs(file_folder)
            img_name = os.path.join(file_folder, '%06d.jpg' % output_frame)
            cv2.imwrite(img_name, nms_img)
        if self.output_video:
            nms_img = cv2.resize(nms_img, (self.output_video_width, self.output_video_height))
            video_writer.write(nms_img)
        # 检测结果存储

        self.det_bbox_result['traj_info'].append((frame_index, output_frame, o_bboxs_res, unix_time))
        self.det_bbox_result['raw_det'].append((frame_index, output_frame, det_raw, unix_time))



    def _save_det_bbox(self,save_file_folder):
        if save_file_folder is None:
            return

        if not os.path.exists(save_file_folder):
            os.makedirs(save_file_folder)
        file_path = os.path.join(save_file_folder,'stitch_bbox_result_'+self.video_name+'.pkl')
        print('\nstart writing detection result:%s'%file_path)
        with open(file_path,'wb') as f:
            pickle.dump(self.det_bbox_result,f)


    def _rotated_bbox_to_bbox(self,rotated_bbox):
        # from mmcv.ops import box_iou_rotated
        object_num = rotated_bbox.shape[0]
        rotated_bbox_np = rotated_bbox.data.cpu().numpy()
        res = []
        for i in range(object_num):
            x = rotated_bbox_np[i]
            cen_x, cen_y = x[0], x[1]
            bbox_w, bbox_h = x[2], x[3]
            theta = x[4] * 180 / np.pi
            pts_4 = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))  # 4 x 2
            x1_y1 = pts_4.min(0)
            x2_y2 = pts_4.max(0)
            x1,y1 = x1_y1[0],x1_y1[1]
            x2, y2 = x2_y2[0], x2_y2[1]
            score = x[5]
            h_bbox = [x1,y1, x2,y2, score]
            res.append(h_bbox)

        # bbox = rotated_bbox[:,:8].view(object_num,4,2)
        # score = rotated_bbox[:,-2].view(-1,1)
        # x1_y1 = bbox.min(1)
        # x2_y2 = bbox.max(1)

        res = np.array(res,dtype=np.float32)
        return res


    def _load_road_config(self,path):
        if not path:
            return
        return RoadConfig.fromfile(path)

    def _pixel_to_xy(self,nms_result):
        '''
        像素坐标转换为xy坐标,采用仿射变换矩阵
        :return:
        '''
        pixel2xy_matrix = self.road_config['pixel2xy_matrix']
        if pixel2xy_matrix is not None:
            pixel_data = nms_result[:,:8].copy()
            pixel_data = pixel_data.reshape(-1, 2)
            b = np.ones(pixel_data.shape[0])
            pixel_data = np.column_stack((pixel_data, b))
            xy_data = np.matmul(pixel2xy_matrix, pixel_data.T).T.reshape(-1,8)
            return np.hstack((nms_result,xy_data))
        else:
            return nms_result

    def _get_lane_id(self,nms_result):
        lanes = self.road_config['lane']
        if len(lanes) == 0:
            return nms_result
        else:
            lane_name_ls = []
            for object_index in range(nms_result.shape[0]):
                pred = nms_result[object_index]
                pixel_ct = np.mean(pred[:8].reshape(-1,2),axis=0)
                lane_name = -1
                for key, lane_polygon in lanes.items():
                    if isPointinPolygon(pixel_ct,lane_polygon):
                        lane_name = key
                        break
                lane_name_ls.append(lane_name)
            return np.hstack((nms_result,np.array(lane_name_ls).reshape(-1,1)))




if __name__ == '__main__':

    config_json = '../config/yingtianstreet/0708/multi_20220708_F1.json'
    v = DroneVideoProcess(config_json)
    # v.process_img()
    v.process_multi_video()