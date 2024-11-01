#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2021-12-20 21:24
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : video_process.py 
@Software: PyCharm
@desc: 
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

class DroneVideoProcess:
    def __init__(self,config_json):
        config_dict = Config.fromfile(config_json)
        self.pipeline = config_dict.get('pipeline',['det'])
        self.video_file = self._get_video_file(config_dict)
        self.road_config = self._load_road_config(config_dict.get('road_config',None))
        self.mask = self.road_config['det_mask']
        self.axis_image = self.road_config['axis_image']
        # self.split = splitbase('','',subsize_height=812,
        #          subsize_width=940)
        self.split_gap = config_dict.get('split_gap',100)
        subsize_height = config_dict.get('subsize_height',640)
        subsize_width = config_dict.get('subsize_width',640)
        self.split = splitbase('', '',gap=self.split_gap,subsize_height=subsize_height,subsize_width=subsize_width)

        # self.num_classes = config_dict.get('num_classes') # 分类类别
        # self.checkpoints = config_dict.get('checkpoints') # 模型路径
        # self.phi = config_dict.get('phi')

        self.output_video = config_dict.get('output_video',1) # 是否输出视频,0:不输出，1:输出
        self.output_img = config_dict.get('output_img',0) # 是否输出图片,0:不输出，1:输出

        self.sub_positions = None
        self.out_fps = config_dict.get('out_fps') #输出数据的fps，考虑抽帧的方式
        self.conf_thresh = config_dict.get('conf_thresh',0.25) # 目标检测置信度

        self.save_folder = config_dict.get('save_folder') # 输出路径
        self.inference_batch_size = config_dict.get('inference_batch_size',1) # 推理时候的batch大小，batch中图片是小图片

        self.video_start_frame = config_dict.get('video_start_frame',0) # 视频开始帧
        self.video_end_frame = config_dict.get('video_end_frame',0) # 视频结尾截取掉的帧

        self.stabilize_scale = config_dict.get('stabilize_scale', 1)
        self.stabilize_smooth = config_dict.get('stabilize_smooth', 1)
        self.stabilize_translate = config_dict.get('stabilize_translate', 1)
        self.output_background = config_dict.get('output_background', 1) # 输出背景图片
        self.background_image_ls = []

        self.bbox_label = config_dict.get('bbox_label',['id','score','xy'])
        _,video_name_ext = os.path.split(self.video_file[0])
        self.video_name, extension = os.path.splitext(video_name_ext)
        if len(self.video_file)==1:
            self.save_folder = os.path.join(self.save_folder,self.video_name)
        else:
            self.save_folder = os.path.join(self.save_folder, self.video_name+"_Num_%d"%len(self.video_file))
        os.makedirs(self.save_folder,exist_ok=True)
        # 目标检测
        self.det_model = self._get_det_model(config_dict.get('detection'))

        # 跟踪模型
        self.mot_tracker = self._get_tracking_model(config_dict.get('tracking'))

        # 稳定
        if 'stab' in self.pipeline:
            self.stabilize_transformers_path = os.path.join(self.save_folder,config_dict.get('stabilize_file',None))
            self._init_stabilizer()

        self.det_bbox_result = {'video_info': [], 'output_info': {'output_fps': self.out_fps}, 'traj_info': [],
                                'process_time': datetime.datetime.now(),'raw_det':[]}

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

    def _get_mask(self,mask_json_file):
        if mask_json_file is None or mask_json_file == '':
            return None
        with open(mask_json_file, "r",encoding='utf-8') as f:
            tmp = f.read()
        annotation = json.loads(tmp)
        imageHeight = annotation['imageHeight']
        imageWidth = annotation['imageWidth']
        mask = np.zeros((imageHeight,imageWidth,3))
        if annotation['shapes']:
            for s in annotation["shapes"]:
                if s['label'] == 'road':
                    points = s["points"]
                    points = np.array(points, np.int32)
                    cv2.fillPoly(mask, [points], (1, 1, 1))
        return mask

    def process_video(self):
        '''
        处理视频
        :return:
        '''
        gap_length = self.road_config['length_per_pixel']*self.split_gap
        print('gap_length:%f m'%gap_length)
        num_frame_ls,all_num_frame,width,height,fps = get_all_video_info(self.video_file)
        srt_info_ls = get_srt(self.video_file)
        if len(srt_info_ls)==0:
            print('SRT file is not used')
        video_info = {'video_name': self.video_file, 'width': width, 'height': height, 'fps': fps,
                      'total_frames': all_num_frame}
        self.det_bbox_result['video_info'].append(video_info)
        print("Output Folder:%s" % self.save_folder)
        print('Process Pipeline:','->'.join(self.pipeline))
        # assert round(fps) % self.out_fps == 0, "fps:%d"%fps
        gap = round(fps/self.out_fps)
        print('Frame gap:%d'%gap)
        frame_index = self.video_start_frame
        output_frame = 0
        video_writer = None
        if self.output_video:
            video_format = 'mp4v'
            out_path = os.path.join(self.save_folder, 'tracking_output_' +"_".join(self.pipeline)+"_"+ self.video_name + '.mp4')
            fourcc = cv2.VideoWriter_fourcc(*video_format)
            video_writer = cv2.VideoWriter(out_path, fourcc, self.out_fps, (int(width), int(height)))
            print("Output Tracking Video:%s" % out_path)
        try:
            s_time = time.time()
            for video_index,video_file in enumerate(self.video_file):
                cap = cv2.VideoCapture(video_file)
                valid_frames = num_frame_ls[video_index]
                # 第一个视频跳过video_start_frame帧
                if video_index == 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,self.video_start_frame)
                    valid_frames -= self.video_start_frame
                if video_index == len(self.video_file) - 1:
                    valid_frames -= self.video_end_frame
                print('Input Video:%s'%video_file)
                print("Input Video:width:{} height:{} fps:{} num_frames:{} valid_frames:{}".format(width, height, fps, num_frame_ls[video_index], valid_frames))
                current_video_frame = 0
                video_frame_index = 0
                while video_frame_index<valid_frames:
                    ret, frame = cap.read()
                    # frame = cv2.subtract(frame, np.ones(frame.shape, dtype='uint8') * 50) # 降低亮度
                    if not ret:
                        break
                    if frame_index == self.video_start_frame:
                        first_frame_name = os.path.join(self.save_folder,'first_frame_'+self.video_name+'.jpg')
                        if self.axis_image is not None:
                            new_frame = cv2.add(frame, self.axis_image)
                            cv2.imwrite(first_frame_name, new_frame)
                        else:
                            cv2.imwrite(first_frame_name, frame)
                    if frame_index%gap==0:

                        self._process_img(frame,output_frame,frame_index,video_writer,srt_info_ls,self.save_folder)
                        output_frame += 1
                        e_time = time.time()
                        remain_time = (e_time - s_time) * (all_num_frame//gap - output_frame - 1)
                        process_fps = 1/(e_time-s_time)
                        print('\rvideo index:%d,process frame:%d/%d,current video:%d/%d, FPS:%.1f, remain time:%.2f min'%(video_index,frame_index,all_num_frame-self.video_end_frame,current_video_frame,valid_frames,process_fps,remain_time/60),end="",flush=True)
                        # print(
                        #     '\rvideo index:%d,process frame:%d/%d,current video:%d/%d, FPS:%.1f, remain time:%.2f min' % (
                        #     video_index, frame_index, all_num_frame - self.video_end_frame, current_video_frame,
                        #     valid_frames, process_fps, remain_time / 60))
                        s_time = e_time
                    video_frame_index += 1
                    frame_index += 1
                    current_video_frame += 1
                cap.release()
        finally:
            if self.output_video:
                video_writer.release()
            self._save_det_bbox(self.save_folder)
            print('save video')


    def _process_img(self,frame,output_frame,frame_index,video_writer,srt_info_ls,save_file_folder=None):
        '''
        处理一帧画面
        :param frame:
        :param output_frame:
        :param frame_index:
        :param video_writer:
        :param srt_info_ls:
        :param save_file_folder:
        :return:
        '''
        t1 = time.time()
        # 视频稳定
        if 'stab' in self.pipeline:
            frame = self.video_stabilizer.stabilize_frame(frame,frame_index,scale=self.stabilize_scale,
                                                          smooth_xya=self.stabilize_smooth,
                                                          translate=self.stabilize_translate)
        # 输出背景图片
        if self.output_background and (not self.background_image_ls is None):
            if len(self.background_image_ls)<50:
                self.background_image_ls.append(frame)
            else:
                background_img = np.zeros(frame.shape)
                # 图片叠加
                for image_b in self.background_image_ls:
                    background_img += image_b
                background_img = background_img / len(self.background_image_ls)
                background_path = os.path.join(save_file_folder, 'background_%s.jpg'%self.video_name)
                cv2.imwrite(background_path, background_img)
                print('output background image to:%s'%background_path)
                self.background_image_ls = None

        # 添加mask
        if self.mask is not None:
            frame = cv2.bitwise_and(frame, self.mask)
            # frame = np.asarray(frame * self.mask, np.uint8)

        # 切分图片
        if frame_index == self.video_start_frame:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sub_imgs, sub_positions = self.split.split_image(frame_rgb)
            self.sub_positions = sub_positions
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sub_imgs = self.split.split_image_with_position(frame_rgb,self.sub_positions)

        # 目标检测
        t2 = time.time()
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
        t3 = time.time()
        # 添加坐标轴
        if self.axis_image is not None:
            frame = cv2.add(frame, self.axis_image)
        # 跟踪
        det_raw = np.empty((0, 15))
        if len(new_nms_ls)==0:
            self.mot_tracker.update()
            nms_img = frame
            if self.road_config['pixel2xy_matrix'] is not None:
                if len(self.road_config['lane'])==0:
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
                det_raw = nms_all_bbox.data.cpu().numpy() # shape n*15

            o_bboxs_res = self.mot_tracker.update(nms_all_bbox,frame) # output shape n*11


            # 像素坐标转地理坐标
            o_bboxs_res = self._pixel_to_xy(o_bboxs_res) # n*19
            o_bboxs_res = self._get_lane_id(o_bboxs_res)

            nms_img = self.det_model.draw_oriented_bboxs(frame, o_bboxs_res,self.bbox_label)
        # nms_img = self.det_model.draw_oriented_bboxs(frame, nms_all_bbox)
        # 图像或视频输出


        t4 = time.time()
        if self.output_img:
            file_folder = os.path.join(save_file_folder,'det_img_'+self.video_name)
            if not os.path.exists(file_folder):
                os.makedirs(file_folder)
            img_name = os.path.join(file_folder,'%06d.jpg'%output_frame)
            cv2.imwrite(img_name, nms_img)
        if self.output_video:
            video_writer.write(nms_img)
        # 检测结果存储
        if len(srt_info_ls) == 0:
            self.det_bbox_result['traj_info'].append((frame_index,output_frame,o_bboxs_res))
            self.det_bbox_result['raw_det'].append((frame_index,output_frame,det_raw))
        else:
            if frame_index >= len(srt_info_ls):
                frame_time = srt_info_ls[-1][0]
            else:
                frame_time = srt_info_ls[frame_index][0]
            self.det_bbox_result['traj_info'].append((frame_index,output_frame, o_bboxs_res,frame_time))
            self.det_bbox_result['raw_det'].append((frame_index, output_frame, det_raw,frame_time))
        t5 = time.time()

        # print('t1:%.2f,t2:%.2f,t3:%.2f,t4:%.2f'%(t2-t1,t3-t2,t4-t3,t5-t4))

    def _save_det_bbox(self,save_file_folder):
        if save_file_folder is None:
            return

        if not os.path.exists(save_file_folder):
            os.makedirs(save_file_folder)
        file_path = os.path.join(save_file_folder,'det_bbox_result_'+self.video_name+'.pkl')
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


    # def process_img(self):
    #     # 处理一帧画面
    #     frame = cv2.imread('/home/peipeimao/Workspace/CV/BBAVectors-Oriented-Object-Detection/example_data/20211221200930.jpg')
    #     if self.mask is not None:
    #         frame = np.asarray(frame * self.mask,np.uint8)
    #         print('using mask')
    #     sub_imgs, sub_positions = self.split.split_image(frame)
    #     new_nms_ls = []
    #     for img, position in zip(sub_imgs, sub_positions):
    #         nms_results = self.det_model.inference_img(img)
    #         if nms_results is None:
    #             continue
    #         x, y = position
    #         position_arr = np.array([x, y, x, y, x, y, x, y, 0, 0])
    #         new_nms = nms_results + position_arr
    #         new_nms_ls.append(new_nms)
    #     all_bbox = np.vstack(new_nms_ls)
    #     nms_all_bbox = func_utils.non_maximum_suppression_result(all_bbox)
    #     nms_img = self.det_model.draw_oriented_bboxs(frame, nms_all_bbox)
    #     import matplotlib.pyplot as plt  # plt 用于显示图片
    #     plt.imshow(nms_img)  # 显示图片
    #     cv2.imwrite('demo_test.png',nms_img)
    #     # plt.savefig('demo_test.png')
    #     plt.show()

    def _init_stabilizer(self):
        '''
        加载稳定器
        :return:
        '''
        self.video_stabilizer = VideoStabilization()
        self.video_stabilizer.load_transforms(self.stabilize_transformers_path) # 导入偏移数据

    def _save_result_video(self):
        pass

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

    config_json = '../config/mixed_roads_deepsort/A/20220617_A3_F2.json'
    v = DroneVideoProcess(config_json)
    # v.process_img()
    v.process_video()