#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2022-05-10 9:59
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : video_stabilization.py 
@Software: PyCharm
@desc: 
'''

import os
import cv2
from utils import Config,RoadConfig
from stabilization.VideoStabilization import VideoStabilization,load_fixed_points

class DroneVideoStab:
    def __init__(self,config_json):
        config_dict = Config.fromfile(config_json)
        road_config = config_dict.get('road_config')
        road_config_dict = RoadConfig.fromfile(road_config)
        mask = road_config_dict['stab_mask']
        # fixed_points = road_config_dict['fixed_points']
        # mask, fixed_points = load_fixed_points(road_config)
        self.vs = VideoStabilization()
        self.vs.init_stabilize(mask)
        self.video_file_ls = self._get_video_file(config_dict)
        self.save_folder = config_dict.get('save_folder')
        self.stabilize_file = config_dict.get('stabilize_file')

        self.stabilize_frame = config_dict.get('stabilize_frame',None)
        self.video_start_frame = config_dict.get('video_start_frame',0)
        self.video_end_frame = config_dict.get('video_end_frame',0)
        self.stabilize_scale = config_dict.get('stabilize_scale', 1)
        self.stabilize_smooth = config_dict.get('stbilize_smooth',1)
        self.stabilize_translate = config_dict.get('stabilize_translate',1)

        self.out_fps = config_dict.get('out_fps',10)  # 输出数据的fps，考虑抽帧的方式
        self.skip_interval = config_dict.get('stabilize_skip', None)  # 该区间内不计算特征点
        if not self.skip_interval is None:
            print('skip_interval:',self.skip_interval)
        if not self.stabilize_frame is None:
            self.stabilize_frame = cv2.imread(self.stabilize_frame)

    def _get_video_file(self,config_dict):
        '''
        获取需处理的视频文件
        :param config_dict:
        :return:
        '''
        if 'video_file' in config_dict:
            return [config_dict.get('video_file')]
        elif 'first_video_name' in config_dict and 'video_num' in config_dict:
            video_folder = config_dict.get('video_folder')
            first_video_name = config_dict.get('first_video_name')
            video_num = int(config_dict.get('video_num'))
            video_file_ls = []
            for i in range(video_num):
                video_name = first_video_name.format(i+1)
                video_file_ls.append(os.path.join(video_folder, video_name))
            return video_file_ls
        else:
            video_folder = config_dict.get('video_folder')
            video_name_ls = config_dict.get('video_name')
            video_file_ls = []
            for video_name in video_name_ls:
                video_file_ls.append(os.path.join(video_folder, video_name))
            return video_file_ls

    def process(self,step=1):
        '''

        :param step: 1:output the stabilize pkl file 2:output stabilize video
        :return:
        '''
        self.vs.stabilize_video(self.video_file_ls, self.save_folder, step, output_video=True,
                                stab_file=self.stabilize_file,scale=self.stabilize_scale,
                                video_start_frame=self.video_start_frame,video_end_frame=self.video_end_frame,
                                stabilize_frame=self.stabilize_frame,smooth_xya=self.stabilize_smooth,
                                translate=self.stabilize_translate,video_output_fps=self.out_fps,
                                skip_interval=self.skip_interval)