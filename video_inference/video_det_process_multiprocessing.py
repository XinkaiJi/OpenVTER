#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2022-09-27
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : video_process.py 
@Software: PyCharm
@desc: 多进程处理单视频检测
'''
import os
import cv2
import pickle
import numpy as np
import torch
import time
import datetime
import multiprocessing as mp

from mmcv.ops import nms_rotated

from detection.VehicleDetModule import VehicleDetModule
from tracking.VehicleTrackingModule import VehicleTrackingModule

from utils import Config,RoadConfig,isPointinPolygon
from utils.VideoTool import get_all_video_info,get_srt,splitbase
from utils.MultiVideos import SingleVideo

def queue_img_put(q_put,config_json,save_folder):
    '''
    读取视频进程
    :param q_put:输入队列
    :param config_json: 配置config文件
    :return:
    '''
    # 初始化
    main_video = SingleVideo(config_json)
    config_dict = Config.fromfile(config_json)
    split_gap = config_dict.get('split_gap', 100)
    subsize_height = config_dict.get('subsize_height', 640)
    subsize_width = config_dict.get('subsize_width', 640)
    out_fps = config_dict.get('out_fps')  # 输出数据的fps，考虑抽帧的方式
    video_name = main_video.video_name

    output_background = config_dict.get('output_background', 1)  # 输出背景图片

    num_frame_ls, all_num_frame, width, height, fps = main_video.get_video_info()
    road_config = main_video.road_config
    gap_length = road_config['length_per_pixel'] * split_gap
    split = splitbase('', '', gap=split_gap, subsize_height=subsize_height, subsize_width=subsize_width)
    gap = round(fps / out_fps)
    print('gap_length:%f m' % gap_length)
    read_frame_index = -1
    sub_positions = None

    background_image_ls = []
    try:
        while 1:
            read_frame_index += 1
            ret, stitch_img, main_frame_index, main_unix_time = main_video.get_frame()
            if ret:
                # 输出第一张图
                if read_frame_index == 0:
                    first_frame_name = os.path.join(save_folder, 'first_frame_' + video_name + '.jpg')
                    cv2.imwrite(first_frame_name, stitch_img)

                # 切分图片
                if sub_positions is None:
                    sub_imgs, sub_positions = split.split_image(stitch_img)
                else:
                    sub_imgs = split.split_image_with_position(stitch_img, sub_positions)
                input_frame = [sub_imgs, stitch_img, main_frame_index, main_unix_time,sub_positions]
            else:
                input_frame = [[], None, None, None,None]
            if len(input_frame[0]) == 0: # 退出该进程
                q_put.put(input_frame) # 判断是否结束
                break
            if read_frame_index % gap == 0:
                while q_put.qsize() >= q_put._maxsize-1:
                    while q_put.qsize() > 5:
                        time.sleep(1)
                if output_background and (not background_image_ls is None):
                    if len(background_image_ls) < 50:
                        background_image_ls.append(stitch_img.copy())
                    else:
                        background_img = np.zeros(stitch_img.shape)
                        # 图片叠加
                        for image_b in background_image_ls:
                            background_img += image_b
                        background_img = background_img / len(background_image_ls)
                        background_path = os.path.join(save_folder, 'unstabilize_background_%s.jpg' % video_name)
                        cv2.imwrite(background_path, background_img)
                        print('output background image to:%s' % background_path)
                        background_image_ls = None
                q_put.put(input_frame)
    finally:
        main_video.release()


def queue_img_process(origin_img_q,config_json,video_name,video_info,save_folder):
    '''
    视频处理进程
    :param origin_img_q:
    :param result_img_q:
    :param config_json:
    :param video_name:
    :param video_info:
    :param save_folder:
    :return:
    '''
    config_dict = Config.fromfile(config_json)
    out_fps = config_dict.get('out_fps')  # 输出数据的fps，考虑抽帧的方式
    inference_batch_size = config_dict.get('inference_batch_size', 1)  # 推理时候的batch大小，batch中图片是小图片


    det_model = VehicleDetModule(**config_dict.get('detection'))
    det_model.load_model()

    num_frame_ls, all_num_frame, width, height, fps = video_info
    gap = round(fps / out_fps)

    det_bbox_result = {'video_info': [], 'output_info': {'output_fps': out_fps}, 'traj_info': [],
                            'process_time': datetime.datetime.now(), 'raw_det': []}
    output_frame = -1
    try:
        s_time = time.time()
        while 1:
            output_frame += 1
            sub_imgs, frame,frame_index, unix_time,sub_positions = origin_img_q.get()  # 获取数据
            if len(sub_imgs) == 0:  # 退出该进程
                break
            # 检测
            new_nms_ls = []
            for i in range(0, len(sub_imgs),inference_batch_size):
                s = i
                e = min(i + inference_batch_size, len(sub_imgs))
                select_imgs = sub_imgs[s:e]
                select_positions = sub_positions[s:e]
                nms_results_ls = det_model.inference_img_batch(select_imgs)
                for nms_results, position in zip(nms_results_ls, select_positions):
                    if nms_results is None:
                        continue
                    x, y = position

                    position_arr = np.array([x, y, 0, 0, 0, x, y, x, y, x, y, x, y, 0, 0], dtype=np.float32)
                    position_arr_t = torch.from_numpy(position_arr).to(det_model.device)
                    new_nms = nms_results + position_arr_t
                    new_nms_ls.append(new_nms)
            det_raw = np.empty((0, 15))

            if len(new_nms_ls) > 0:
                all_bbox = torch.vstack(new_nms_ls)
                dets, keep_inds = nms_rotated(all_bbox[:, :5], all_bbox[:, 5], 0.3)
                nms_all_bbox = all_bbox[keep_inds]
                if nms_all_bbox.device.type == 'cpu':
                    det_raw = nms_all_bbox.numpy()
                else:
                    det_raw = nms_all_bbox.data.cpu().numpy()  # shape n*15
            det_bbox_result['raw_det'].append((frame_index, output_frame, det_raw, unix_time))
            e_time = time.time()
            remain_time = (e_time - s_time) * ((all_num_frame-frame_index) // gap)
            process_fps = 1 / (e_time - s_time)
            print(
                '\rprocess origin queue size:%d, main frame:%d/%d, FPS:%.1f, remain time:%.2f min' % (
                    origin_img_q.qsize(),  frame_index, all_num_frame, process_fps,
                    remain_time / 60), end="", flush=True)
            s_time = e_time
    finally:
        _save_det_bbox(save_folder, video_name, det_bbox_result)



def _save_det_bbox(save_file_folder,video_name,det_bbox_result):
    if save_file_folder is None:
        return
    if not os.path.exists(save_file_folder):
        os.makedirs(save_file_folder)
    file_path = os.path.join(save_file_folder,'det_bbox_result_'+video_name+'.detpkl')
    print('\nstart writing detection result:%s'%file_path)
    with open(file_path,'wb') as f:
        pickle.dump(det_bbox_result,f)

def run(config_json):

    main_video = SingleVideo(config_json)
    road_config = main_video.road_config
    video_name = main_video.video_name
    num_frame_ls, all_num_frame, width, height, fps = main_video.get_video_info()
    main_video_width = main_video.width
    main_video.release()
    video_info = [num_frame_ls, all_num_frame, width, height, fps]

    save_folder = main_video.save_folder
    os.makedirs(save_folder, exist_ok=True)


    mp.set_start_method(method='spawn')

    origin_img_q = mp.Queue(maxsize=100)


    processes = [
        mp.Process(target=queue_img_put, args=(origin_img_q, config_json,save_folder)),
        mp.Process(target=queue_img_process, args=(origin_img_q,config_json, video_name,video_info,save_folder))
    ]


    [setattr(process, "daemon", True) for process in processes]
    [process.start() for process in processes]
    [process.join() for process in processes]



if __name__ == '__main__':

    config_json = '../config/yingtianstreet/0707/Y1/20220707_Y1_A_F1.json'
    # v = DroneVideoProcess(config_json)
    # v.process_img()
    # v.process_multi_video()
    run(config_json)