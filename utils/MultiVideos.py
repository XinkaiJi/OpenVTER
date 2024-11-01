#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2022-07-20 20:57
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : MultiVideos.py 
@Software: PyCharm
@desc: 多视频类,将多个视频抽象为一个视频,用于多视频拼接
'''
import sys
root_path = "/home/xinkaiji/Workspace/CV/openVTER_V2"
sys.path.append(root_path)

import os
import cv2
import json
import math
import numpy as np

from stabilization.VideoStabilization import VideoStabilization
from utils.VideoTool import get_all_video_info,get_srt,str2unixtime,get_transformer_matrix
from utils import Config,RoadConfig,MultiVideosConfig

class SingleVideo:

    def __init__(self,config_file,stab=True):
        config_dict = Config.fromfile(config_file)
        self.video_file_ls = Config.get_video_file(config_dict)
        self.road_config_json_file = config_dict.get('road_config',None)
        self.road_config = RoadConfig.fromfile(config_dict.get('road_config',None))
        self.mask = self.road_config['det_mask']
        self.axis_image = self.road_config['axis_image']
        self.pipeline = config_dict.get('pipeline', ['stab','det'])
        self.video_start_frame = config_dict.get('video_start_frame', 0)  # 视频开始帧
        self.video_end_frame = config_dict.get('video_end_frame', 0)  # 视频结尾截取掉的帧
        self.save_folder = config_dict.get('save_folder')  # 输出路径
        # 稳定
        self.stabilize_scale = config_dict.get('stabilize_scale', 1)
        self.stabilize_smooth = config_dict.get('stbilize_smooth', 1)
        self.stabilize_translate = config_dict.get('stabilize_translate', 1)

        _, video_name_ext = os.path.split(self.video_file_ls[0])
        self.video_name, extension = os.path.splitext(video_name_ext)
        if len(self.video_file_ls)==1:
            self.save_folder = os.path.join(self.save_folder,self.video_name)
        else:
            self.save_folder = os.path.join(self.save_folder, self.video_name+"_Num_%d"%len(self.video_file_ls))

        if 'stab' in self.pipeline and stab:
            self.stabilize_transformers_path = os.path.join(self.save_folder, config_dict.get('stabilize_file', None))
            self._init_stabilizer()

        self.num_frame_ls, self.all_num_frame, self.width,self.height, self.fps = get_all_video_info(self.video_file_ls)
        self.srt_info_ls = get_srt(self.video_file_ls)

        # 读取第一个视频
        cap = cv2.VideoCapture(self.video_file_ls[0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_start_frame)
        self.current_cap = cap
        self.current_cap_index = 0
        self.current_frame_index = self.video_start_frame-1

        self.offset = 0
        self.play_gap = 1 # 播放的帧间隔,默认为1帧,即连续
        self.flight_name = None
        for key,item in self.road_config['drivingline'].items():
            if key.endswith('line'):
                self.flight_name = key.split('_')[-2]


    def __str__(self):
        video_info = "VideoName:%s,VideoNum:%d,Width:%d,Height:%d,fps:%d"%(self.video_name,len(self.video_file_ls),self.width,self.height,self.fps)
        return video_info

    def get_video_info(self):
        return self.num_frame_ls, self.all_num_frame, self.width,self.height, self.fps


    def set_offset(self,offset):
        self.offset = offset

    def _init_stabilizer(self):
        '''
        加载稳定器
        :return:
        '''
        self.video_stabilizer = VideoStabilization()
        self.video_stabilizer.load_transforms(self.stabilize_transformers_path) # 导入偏移数据

    def get_frame(self,stabilized=True,add_road_mask=True):
        if self.play_gap == 1:
            return self._get_frame(stabilized=stabilized, add_road_mask=add_road_mask)
        else:
            for i in range(self.play_gap-1):
                self._get_frame(stabilized=False, add_road_mask=False)
            return self._get_frame(stabilized=stabilized, add_road_mask=add_road_mask)

    def _get_frame(self,stabilized=True,add_road_mask=True):
        '''
         获取视频帧
        :param stabilized: 是否对视频稳定
        :param add_road_mask: 是否添加道路mask
        :return:
        '''
        ret, frame = self.current_cap.read()
        self.current_frame_index += 1
        unix_time = self.get_unixtime_srt()
        if ret:
            # 获取到了画面
            # 该帧已到达需要结束位置
            if self.current_frame_index >= self.all_num_frame-self.video_end_frame:
                return False,None,None,None
        else:
            if self.current_cap_index+1 < len(self.video_file_ls):
                self.current_cap_index += 1
                self.current_cap.release()
                self.current_cap = cv2.VideoCapture(self.video_file_ls[self.current_cap_index])
                ret, frame = self.current_cap.read()
                assert ret,'read video error:%s'%self.video_file_ls[self.current_cap_index]
            else:
                self.current_cap.release()
                return False,None,None,None # 到达了最后一个视频结尾
        # 稳定视频
        if stabilized and 'stab' in self.pipeline:
            frame = self.video_stabilizer.stabilize_frame(frame, self.current_frame_index,
                                                          scale=self.stabilize_scale,
                                                          smooth_xya=self.stabilize_smooth,
                                                          translate=self.stabilize_translate)
        # 添加mask
        if add_road_mask and self.mask is not None:
            frame = cv2.bitwise_and(frame, self.mask)

        unix_time += self.offset*(1000/self.fps)
        return ret, frame, self.current_frame_index, unix_time


    def get_unixtime_srt(self):
        '''
        从字幕文件中获取时间
        :return:
        '''
        if len(self.srt_info_ls) == 0:
            return None
        else:
            if self.current_frame_index >= len(self.srt_info_ls):
                frame_time = self.srt_info_ls[-1][0]
            else:
                frame_time = self.srt_info_ls[self.current_frame_index][0]
        frame_unix_time = str2unixtime(frame_time)
        return frame_unix_time

    def move_current_frame(self,seconds):
        move_frame_num = int(seconds*self.fps)
        target_frame_index = self.current_frame_index+move_frame_num
        all_num = sum(self.num_frame_ls[:self.current_cap_index+1])
        if target_frame_index < all_num:
            self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
            self.current_frame_index = target_frame_index - 1
        else:
            assert target_frame_index < sum(self.num_frame_ls),'can not align'
            for i in range(1,len(self.num_frame_ls)-self.current_cap_index):
                all_num = sum(self.num_frame_ls[:self.current_cap_index + 1+i])
                if target_frame_index < all_num:
                    break
            self.current_cap_index += i
            self.current_cap.release()
            self.current_cap = cv2.VideoCapture(self.video_file_ls[self.current_cap_index])
            move_index = target_frame_index-sum(self.num_frame_ls[:self.current_cap_index])
            self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, move_index)
            self.current_frame_index = target_frame_index - 1

    def release(self):
        self.current_cap.release()


class MultiVideos:

    def __init__(self,mutli_video_json):
        mutli_video_json = os.path.abspath(os.path.expanduser(mutli_video_json))
        multi_video_config_dict = MultiVideosConfig.fromfile(mutli_video_json)
        main_video_json = multi_video_config_dict['MainVideo']
        file_folder,file_name = os.path.split(mutli_video_json)
        main_video_json_path = os.path.join(file_folder,main_video_json)
        self.main_video = SingleVideo(main_video_json_path)
        self.sub_video_ls = []
        self.sub_video_offset_ls = []
        sub_video_json_ls = multi_video_config_dict['SubVideos']
        for json_file,offset in sub_video_json_ls:
            sub_video_json_path = os.path.join(file_folder, json_file)
            sub_video = SingleVideo(sub_video_json_path)
            sub_video.set_offset(offset)
            self.sub_video_ls.append(sub_video)
            self.sub_video_offset_ls.append(offset)


        self.crop_region = None
        self.split_x_position = None
        self.affine_matrix = None
        self._cal_affine_matrix()
        self.is_align = self._align_frame()
        if self.sub_video_ls[0].fps != self.main_video.fps:
            print('Main video fps:%d,sub video fps:%d'%(self.main_video.fps,self.sub_video_ls[0].fps))
            print('Two videos have different fps!!')

            self.multi_video_fps = math.gcd(self.main_video.fps, self.sub_video_ls[0].fps)

            self.main_video.play_gap = self.main_video.fps//self.multi_video_fps
            self.sub_video_ls[0].play_gap = self.sub_video_ls[0].fps//self.multi_video_fps
        else:
            self.multi_video_fps = self.main_video.fps
        print('Multi_video_fps:', self.multi_video_fps)
        if not self.is_align:
            print('not align!!!')
        else:
            self.get_align_frame()
        main_video_name = self.main_video.video_name
        sub_video_name = self.sub_video_ls[0].video_name
        self.video_name = 'M-%s-S-%s' % (main_video_name, sub_video_name)

    def __str__(self):
        main_str = str(self.main_video)
        res_str = 'MainVideo %s\n'%main_str
        for i,sub_video in enumerate(self.sub_video_ls):
            res_str += 'SubVideo(%d) %s\n'%(i,str(sub_video))
        return res_str

    def get_video_info(self):
        num_frame_ls,all_num_frame,width,height,fps = self.main_video.get_video_info()
        all_num_frame = all_num_frame/(fps/self.multi_video_fps)
        return num_frame_ls,all_num_frame,width,height,self.multi_video_fps

    def _align_frame(self):
        '''
        对齐两个视频
        :return:
        '''
        main_ret, main_frame, main_frame_index, main_unix_time = self.main_video.get_frame(stabilized=False,add_road_mask=False)
        sub_video = self.sub_video_ls[0]

        sub_ret, sub_frame, sub_frame_index, sub_unix_time = sub_video.get_frame(stabilized=False,add_road_mask=False)

        aligen = False
        gap_time = abs(main_unix_time-sub_unix_time)

        if gap_time<=1000/self.main_video.fps:
            aligen = True

        main_ahead = True
        if main_unix_time < sub_unix_time:
            main_ahead = False
            self.main_video.move_current_frame(max(gap_time/1000-1,gap_time / 1000))
        else:
            sub_video.move_current_frame(max(gap_time / 1000 - 1,gap_time / 1000))

        while main_ret and sub_ret and not aligen:

            if main_ahead:
                sub_ret, sub_frame, sub_frame_index, sub_unix_time = sub_video.get_frame(stabilized=False, add_road_mask=False)
                # print('\rgap time: %f s'%((main_unix_time-sub_unix_time)/1000),end="",flush=True)
                print('\rgap time: %f s' % ((main_unix_time - sub_unix_time) / 1000))
                if main_unix_time<=sub_unix_time:
                    aligen = True
            else:
                main_ret, main_frame, main_frame_index, main_unix_time = self.main_video.get_frame(stabilized=False,
                                                                                                   add_road_mask=False)
                # print('\rgap time: %f s' % ((sub_unix_time-main_unix_time) / 1000),end="",flush=True)
                print('\rgap time: %f s' % ((sub_unix_time - main_unix_time) / 1000))
                if main_unix_time>=sub_unix_time:
                    aligen = True
        return aligen

    def get_align_frame(self,stabilized=True,add_road_mask=False,add_frame_index = False):
        '''
        获取对齐的视频帧
        :return:ret,stitch_img,frame_index,unix_time
        '''
        main_ret, main_frame, main_frame_index, main_unix_time = self.main_video.get_frame(stabilized=stabilized,
                                                                                                   add_road_mask=add_road_mask)
        sub_video = self.sub_video_ls[0]

        sub_ret, sub_frame, sub_frame_index, sub_unix_time = sub_video.get_frame(stabilized=stabilized,
                                                                                                   add_road_mask=add_road_mask)
        if main_ret and sub_ret:
            stitch_img = self.image_stitch(main_frame, sub_frame)
            if add_frame_index:
                text = 'M:%d,S:%d'%(main_frame_index,sub_frame_index)
                h,w,c = stitch_img.shape
                cv2.putText(stitch_img,text,(w//2,h//2),cv2.FONT_HERSHEY_SIMPLEX,fontScale=3,color=(255,0,0),thickness=2)

            return True,stitch_img,main_frame_index, main_unix_time
        else:
            if main_ret:
                self.main_video.current_cap.release()
            if sub_ret:
                sub_video.current_cap.release()
            return False,None,None,None
        # stitch_file = '/home/xinkaiji/temp_videos/align/stitch_%d.jpg'%main_unix_time
        # main_file = '/home/xinkaiji/temp_videos/align/main_%d.jpg'%main_unix_time
        # sub_file = '/home/xinkaiji/temp_videos/align/sub_%d.jpg' % sub_unix_time
        # cv2.imwrite(stitch_file, stitch_img)
        # cv2.imwrite(main_file,main_frame)
        # cv2.imwrite(sub_file, sub_frame)
        # print('save main:%s'%main_file)
        # print('save sub:%s' % sub_file)
        # print('save stitch:%s' % stitch_file)

    def _get_transformer_pts(self,road_config_json_file,main_name):
        # if main_name is None:
        #     label = 'stitch_'
        # else:
        #     label = 'stitch_%s'%main_name
        label = 'stitch_A3A4'
        with open(road_config_json_file, 'r',encoding='utf-8') as f:
            road_config = json.load(f)
        for shape in road_config['shapes']:
            if shape['label'].startswith(label):  # 稳定跟踪固定区域
                pts = shape["points"]
                return pts

    def _cal_affine_matrix(self):
        '''
        计算仿射变换矩阵
        :param json_main:
        :param json_sub:
        :return:
        '''

        main_pts = self._get_transformer_pts(self.main_video.road_config_json_file,self.main_video.flight_name)
        sub_pts = self._get_transformer_pts(self.sub_video_ls[0].road_config_json_file,self.main_video.flight_name)
        affine_matrix = get_transformer_matrix(sub_pts,main_pts,use_three=True)
        self.split_x_position = int(np.max(np.array(main_pts), axis=0)[0])
        self.affine_matrix = affine_matrix


    def image_stitch(self,main_img,sub_img):
        width = main_img.shape[1] + sub_img.shape[1]
        height = main_img.shape[0]

        # print(F1_img.shape[1])
        result = cv2.warpAffine(sub_img, self.affine_matrix, (width, height))
        # gap = 600

        result[0:main_img.shape[0], 0:self.split_x_position] = main_img[:, :self.split_x_position, :]
        if self.crop_region:
            x, y, w, h = self.crop_region
        else:
            # transform the panorama image to grayscale and threshold it
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

            # get a bbox from the contour area
            self.crop_region = cv2.boundingRect(thresh)
            x, y, w, h = self.crop_region
            # crop the image to the bbox coordinates
        result = result[y:y + h, x:x + w]
        return result


    def init_video_write(self,save_folder,output_fps):
        '''
        初始化视频输出
        :param save_folder:
        :return:
        '''
        x, y, width, height = self.crop_region
        new_height = int(height*self.main_video.width/width)
        self.output_width = self.main_video.width
        self.output_height = new_height
        video_format = 'mp4v'
        out_path = os.path.join(save_folder,
                                'paper_stitch_'+ self.video_name + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*video_format)
        video_writer = cv2.VideoWriter(out_path, fourcc, output_fps, (self.output_width, self.output_height))
        print("Output Tracking Video:%s" % out_path)
        return video_writer, self.output_width, self.output_height

    def release(self):
        self.main_video.release()
        for sub_video in self.sub_video_ls:
            sub_video.release()

if __name__ == '__main__':

    main_json = '../config/mixed_roads/multi_20220617_A3A4_F1.json'

    multi_video = MultiVideos(main_json)
    print(multi_video)
    print('x_position:',multi_video.split_x_position)
    ret,stitch_img,sub_frame_index, sub_unix_time = multi_video.get_align_frame()
    video_writer,output_width, output_height = multi_video.init_video_write('/home/xinkaiji/temp_videos/align',multi_video.multi_video_fps)
    try:
        while ret:
            ret,stitch_img,main_frame_index, main_unix_time = multi_video.get_align_frame(add_frame_index=False)
            if not ret:
                break
            stitch_img = cv2.resize(stitch_img,(output_width,output_height))
            video_writer.write(stitch_img)
    finally:
        video_writer.release()

