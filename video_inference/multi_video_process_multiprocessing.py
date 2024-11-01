#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2022-09-27
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : video_process.py 
@Software: PyCharm
@desc: 多进程处理多视频
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
from utils.MultiVideos import MultiVideos

def queue_img_put(q_put,config_json):
    '''
    读取视频
    :param q_put:
    :param config_json:
    :return:
    '''
    # 初始化
    multi_videos = MultiVideos(config_json)
    config_dict = Config.fromfile(config_json)
    split_gap = config_dict.get('split_gap', 100)
    subsize_height = config_dict.get('subsize_height', 640)
    subsize_width = config_dict.get('subsize_width', 640)
    out_fps = config_dict.get('out_fps')  # 输出数据的fps，考虑抽帧的方式
    video_name = multi_videos.video_name
    save_folder = config_dict.get('save_folder')  # 输出路径
    save_folder = os.path.join(save_folder, video_name)
    output_background = config_dict.get('output_background', 1)  # 输出背景图片

    num_frame_ls, all_num_frame, width, height, fps = multi_videos.get_video_info()
    road_config = multi_videos.main_video.road_config
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
            ret, stitch_img, main_frame_index, main_unix_time = multi_videos.get_align_frame()
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
                    time.sleep(0.001)
                if output_background and (not background_image_ls is None):
                    if len(background_image_ls) < 50:
                        background_image_ls.append(stitch_img.copy())
                    else:
                        background_img = np.zeros(stitch_img.shape)
                        # 图片叠加
                        for image_b in background_image_ls:
                            background_img += image_b
                        background_img = background_img / len(background_image_ls)
                        background_path = os.path.join(save_folder, 'background_%s.jpg' % video_name)
                        cv2.imwrite(background_path, background_img)
                        print('output background image to:%s' % background_path)
                        background_image_ls = None
                q_put.put(input_frame)
    finally:
        multi_videos.release()


def queue_img_process(origin_img_q,result_img_q,config_json,road_config,video_name,video_info):

    config_dict = Config.fromfile(config_json)
    out_fps = config_dict.get('out_fps')  # 输出数据的fps，考虑抽帧的方式
    inference_batch_size = config_dict.get('inference_batch_size', 1)  # 推理时候的batch大小，batch中图片是小图片
    save_folder = config_dict.get('save_folder')  # 输出路径
    save_folder = os.path.join(save_folder, video_name)
    os.makedirs(save_folder, exist_ok=True)

    det_model = VehicleDetModule(**config_dict.get('detection'))
    det_model.load_model()

    num_frame_ls, all_num_frame, width, height, fps = video_info
    gap = round(fps / out_fps)
    mot_tracker = VehicleTrackingModule(**config_dict.get('tracking'))

    det_bbox_result = {'video_info': [], 'output_info': {'output_fps': out_fps}, 'traj_info': [],
                            'process_time': datetime.datetime.now(), 'raw_det': []}
    output_frame = -1
    try:
        s_time = time.time()
        while 1:
            output_frame += 1
            sub_imgs, frame,frame_index, unix_time,sub_positions = origin_img_q.get()  # 获取数据
            if len(sub_imgs) == 0:  # 退出该进程
                result_img_q.put([None,None,None])
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
            if len(new_nms_ls) == 0:
                mot_tracker.update()
                # nms_img = frame
                if road_config['pixel2xy_matrix'] is not None:
                    if len(road_config['lane']) == 0:
                        o_bboxs_res = np.empty((0, 19))
                    else:
                        o_bboxs_res = np.empty((0, 20))
                else:
                    o_bboxs_res = np.empty((0, 11))
            else:

                all_bbox = torch.vstack(new_nms_ls)
                dets, keep_inds = nms_rotated(all_bbox[:, :5], all_bbox[:, 5], 0.3)
                nms_all_bbox = all_bbox[keep_inds]
                if nms_all_bbox.device.type == 'cpu':
                    det_raw = nms_all_bbox.numpy()
                else:
                    det_raw = nms_all_bbox.data.cpu().numpy()  # shape n*15

                o_bboxs_res = mot_tracker.update(nms_all_bbox)  # output shape n*11

                # 像素坐标转地理坐标
                o_bboxs_res = _pixel_to_xy(o_bboxs_res,road_config['pixel2xy_matrix'])  # n*19
                o_bboxs_res = _get_lane_id(o_bboxs_res,road_config['lane'])

                # nms_img = det_model.draw_oriented_bboxs(frame, o_bboxs_res, bbox_label)
            det_bbox_result['traj_info'].append((frame_index, output_frame, o_bboxs_res, unix_time))
            det_bbox_result['raw_det'].append((frame_index, output_frame, det_raw, unix_time))
            while result_img_q.qsize() >= result_img_q._maxsize - 1:
                time.sleep(0.001)
            result_img_q.put([frame, o_bboxs_res,output_frame])
            e_time = time.time()
            remain_time = (e_time - s_time) * (all_num_frame // gap - 1)
            process_fps = 1 / (e_time - s_time)
            # print('\rprocess origin queue size:%d,result queue size:%d, main frame:%d/%d, FPS:%.1f, remain time:%.2f min' % (origin_img_q.qsize(),
            # result_img_q.qsize(),frame_index, all_num_frame, process_fps, remain_time / 60), end="", flush=True)
            print(
                '\rprocess origin queue size:%d,result queue size:%d, main frame:%d/%d, FPS:%.1f, remain time:%.2f min' % (
                origin_img_q.qsize(),
                result_img_q.qsize(), frame_index, all_num_frame, process_fps, remain_time / 60))
            s_time = e_time
    finally:
        _save_det_bbox(save_folder, video_name, det_bbox_result)



def queue_img_output(result_img_q,config_json,video_name, main_video_width):

    config_dict = Config.fromfile(config_json)
    output_video = config_dict.get('output_video', 1)  # 是否输出视频,0:不输出，1:输出
    output_img = config_dict.get('output_img', 0)  # 是否输出图片,0:不输出，1:输出
    save_folder = config_dict.get('save_folder')  # 输出路径
    out_fps = config_dict.get('out_fps')  # 输出数据的fps，考虑抽帧的方式
    bbox_label = config_dict.get('bbox_label', ['id', 'score', 'xy'])
    det_model = VehicleDetModule(**config_dict.get('detection'))
    save_folder = os.path.join(save_folder, video_name)
    crop_region = None
    video_writer = None
    try:
        while 1:
            frame, o_bboxs_res,output_frame = result_img_q.get()
            if frame is None:
                break
            if len(o_bboxs_res)>0:
                nms_img = det_model.draw_oriented_bboxs(frame, o_bboxs_res, bbox_label)
            else:
                nms_img = frame
            if output_img or output_video:
                if crop_region:
                    x, y, w, h = crop_region
                else:
                    # transform the panorama image to grayscale and threshold it
                    gray = cv2.cvtColor(nms_img, cv2.COLOR_BGR2GRAY)
                    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
                    # get a bbox from the contour area
                    crop_region = cv2.boundingRect(thresh)
                    x, y, w, h = crop_region
                nms_img = nms_img[y:y + h, x:x + w]
            if video_writer is None:
                x, y, width, height = crop_region
                new_height = int(height * main_video_width / width)
                output_width = main_video_width
                output_height = new_height
                video_format = 'mp4v'
                out_path = os.path.join(save_folder,
                                        'stitch_' + video_name + '.mp4')
                fourcc = cv2.VideoWriter_fourcc(*video_format)
                video_writer = cv2.VideoWriter(out_path, fourcc, out_fps, (output_width, output_height))
                print("Output Tracking Video:%s" % out_path)
            if output_img:
                nms_img = cv2.resize(nms_img, (output_width, output_height))
                file_folder = os.path.join(save_folder, 'det_img_' + video_name)
                if not os.path.exists(file_folder):
                    os.makedirs(file_folder)
                img_name = os.path.join(file_folder, '%06d.jpg' % output_frame)
                cv2.imwrite(img_name, nms_img)
            if output_video:
                nms_img = cv2.resize(nms_img, (output_width,output_height))
                video_writer.write(nms_img)
    finally:
        if output_video:
            video_writer.release()


def _pixel_to_xy(nms_result,pixel2xy_matrix):
    '''
    像素坐标转换为xy坐标,采用仿射变换矩阵
    :return:
    '''
    if pixel2xy_matrix is not None:
        pixel_data = nms_result[:,:8].copy()
        pixel_data = pixel_data.reshape(-1, 2)
        b = np.ones(pixel_data.shape[0])
        pixel_data = np.column_stack((pixel_data, b))
        xy_data = np.matmul(pixel2xy_matrix, pixel_data.T).T.reshape(-1,8)
        return np.hstack((nms_result,xy_data))
    else:
        return nms_result

def _get_lane_id(nms_result,lanes):
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

def _save_det_bbox(save_file_folder,video_name,det_bbox_result):
    if save_file_folder is None:
        return
    if not os.path.exists(save_file_folder):
        os.makedirs(save_file_folder)
    file_path = os.path.join(save_file_folder,'stitch_bbox_result_'+video_name+'.pkl')
    print('\nstart writing detection result:%s'%file_path)
    with open(file_path,'wb') as f:
        pickle.dump(det_bbox_result,f)

def run(config_json):

    multi_videos = MultiVideos(config_json)
    road_config = multi_videos.main_video.road_config
    video_name = multi_videos.video_name
    num_frame_ls, all_num_frame, width, height, fps = multi_videos.get_video_info()
    main_video_width = multi_videos.main_video.width
    multi_videos.release()
    video_info = [num_frame_ls, all_num_frame, width, height, fps]
    config_dict = Config.fromfile(config_json)
    save_folder = config_dict.get('save_folder')  # 输出路径
    save_folder = os.path.join(save_folder, video_name)
    os.makedirs(save_folder, exist_ok=True)

    mp.set_start_method(method='spawn')

    origin_img_q = mp.Queue(maxsize=50)

    result_img_q = mp.Queue(maxsize=50)

    processes = [
        mp.Process(target=queue_img_put, args=(origin_img_q, config_json)),
        mp.Process(target=queue_img_process, args=(origin_img_q, result_img_q, config_json, road_config, video_name,video_info)),
        mp.Process(target=queue_img_output, args=(result_img_q,config_json,video_name, main_video_width)),
    ]


    [setattr(process, "daemon", True) for process in processes]
    [process.start() for process in processes]
    [process.join() for process in processes]



if __name__ == '__main__':

    config_json = '../config/yingtianstreet/0711/multi_20220711_A_F1.json'
    # v = DroneVideoProcess(config_json)
    # v.process_img()
    # v.process_multi_video()
    run(config_json)