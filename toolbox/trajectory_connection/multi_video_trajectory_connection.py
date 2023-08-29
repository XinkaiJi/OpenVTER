#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2022-09-29 21:04
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : multi_video_trajectory_connection.py 
@Software: PyCharm
@desc: 多视频中车辆轨迹的拼接
'''


import os
import pickle
import json
import cv2
import shutil
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import linear_sum_assignment
# import lap
from utils import Config, RoadConfig, MultiVideosConfig
from utils.VideoTool import get_transformer_matrix

from toolbox.module.DrivingLine import DrivingLineList

INF = 10**5
def get_tppkl(video_json_path):
    '''
    加载tppkl文件
    :param video_json_path:
    :return:
    '''
    config_dict = Config.fromfile(video_json_path)
    video_file_ls = Config.get_video_file(config_dict)
    save_folder = config_dict.get('save_folder')  # 输出路径
    _, video_name_ext = os.path.split(video_file_ls[0])
    video_name, extension = os.path.splitext(video_name_ext)
    if len(video_file_ls) == 1:
        save_folder = os.path.join(save_folder, video_name)
    else:
        save_folder = os.path.join(save_folder, video_name + "_Num_%d" % len(video_file_ls))
    road_config_json = config_dict.get('road_config', None)
    _, file_name = os.path.split(road_config_json)
    json_name, _ = os.path.splitext(file_name)
    tppkl_path = os.path.join(save_folder, 'tp_result_%s.tppkl' % json_name)
    img_path = os.path.join(save_folder, 'first_frame_%s.jpg' % json_name)
    img = cv2.imread(img_path) if os.path.exists(img_path) else None

    # vehicles_data = None
    with open(tppkl_path, 'rb') as f:
        vehicles_data = pickle.load(f)
    print('load success:%s' % tppkl_path)
    return vehicles_data, img, json_name


def get_affine_matrix(main_video_json_path, sub_video_json_path, main_name=None):
    '''
    计算仿射变换
    :param main_video_json_path:
    :param sub_video_json_path:
    :param main_name:
    :return:
    '''
    main_config_dict = Config.fromfile(main_video_json_path)
    main_video_road_config_json_file = main_config_dict.get('road_config', None)
    sub_config_dict = Config.fromfile(sub_video_json_path)
    sub_video_road_config_json_file = sub_config_dict.get('road_config', None)
    main_pts = _get_transformer_pts(main_video_road_config_json_file, main_name)
    sub_pts = _get_transformer_pts(sub_video_road_config_json_file, main_name)
    affine_matrix = get_transformer_matrix(sub_pts, main_pts, use_three=True)
    split_x_position = int(np.max(np.array(main_pts), axis=0)[0])

    main_road_config = RoadConfig.fromfile(main_video_road_config_json_file)
    sub_road_config = RoadConfig.fromfile(sub_video_road_config_json_file)
    roi = get_roi(main_road_config, sub_road_config, affine_matrix)
    return affine_matrix, split_x_position, roi


def _get_transformer_pts(road_config_json_file, main_name):
    '''
    获取用于计算仿射变换的点
    :param road_config_json_file:
    :param main_name:
    :return:
    '''
    if main_name is None:
        label = 'stitch_'
    else:
        label = 'stitch_%s' % main_name
    if not os.path.isabs(road_config_json_file):
        print(road_config_json_file)
        if road_config_json_file.startswith('.'):
            road_config_json_file = road_config_json_file[1:]
            print(road_config_json_file)
        road_config_json_file = root_path + road_config_json_file
        print(road_config_json_file)
    with open(road_config_json_file, 'r', encoding='utf-8') as f:
        road_config = json.load(f)
    for shape in road_config['shapes']:
        if shape['label'].startswith(label):  #
            pts = shape["points"]
            return pts


def image_stitch(main_img, sub_img, affine_matrix, split_x_position, node_f, node_t,img_save_folder):
    '''
    图片拼接
    :param main_img:
    :param sub_img:
    :param affine_matrix:
    :param split_x_position:
    :return:
    '''
    width = main_img.shape[1] + sub_img.shape[1]
    height = main_img.shape[0]

    # print(F1_img.shape[1])
    result = cv2.warpAffine(sub_img, affine_matrix, (width, height))
    # gap = 600
    result[0:main_img.shape[0], 0:split_x_position] = main_img[:, :split_x_position, :]

    img_name = 'stitch_%s_to_%s.jpg' % (node_f, node_t)
    save_path = os.path.join(img_save_folder,img_name)
    cv2.imwrite(save_path, result)
    print('save stitch img:%s' % save_path)


def image_stitch_multi(main_node_id, tppkl_dict, node_to_main,img_save_folder):
    main_img = tppkl_dict[main_node_id][1]
    width = main_img.shape[1] * (len(node_to_main) + 1)
    height = main_img.shape[0] * (len(node_to_main) + 1)
    result = None
    img_name_ls = [main_node_id]
    for video_id, trans_data in node_to_main.items():
        img_name_ls.append(video_id)
        sub_img = tppkl_dict[video_id][1]
        affine_matrix = trans_data['affine_matrix'][:2, :]
        aff_img = cv2.warpAffine(sub_img, affine_matrix, (width, height))
        if result is None:
            result = aff_img
        else:
            result = np.maximum(result, aff_img)
    result[0:main_img.shape[0], 0:main_img.shape[1]] = main_img
    result = crop_img(result)
    img_name = 'stitch' + '_'.join(img_name_ls) + '.jpg'
    save_path = os.path.join(img_save_folder,img_name)
    cv2.imwrite(save_path, result)
    print('save stitch img:%s' % save_path)

def crop_img(image):
    '''
    裁剪图像 但保留左上角区域使得坐标不变
    :param image:
    :return:
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 将图像转换为二值图像
    _, binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    # 寻找图像的轮廓
    # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # 寻找最小边界框
    # min_rect = cv2.minAreaRect(contours[0])
    # # 获取最小边界框的四个顶点坐标
    # box = cv2.boxPoints(min_rect)
    # # 将坐标转换为整数类型
    # box = np.int0(box)
    # # 计算最小边界框的宽度和高度
    # width = int(min_rect[1][0])
    # height = int(min_rect[1][1])
    # # 左上角点需要保留，使得图像区域像素坐标不变
    # cropped_image = image[0:min(box[:, 1]) + height, 0:min(box[:, 0]) + width]
    x,y,w,h = cv2.boundingRect(binary_image)
    cropped_image = image[0:y + h, 0:x + w]
    # print('box',box)
    # print('h:%d,w:%d'%(height,width))
    return cropped_image


def get_roi(main_road_config, sub_road_config, affine_matrix):
    '''
    获取仿射变换后重叠的区域
    :param main_road_config:
    :param sub_road_config:
    :param affine_matrix:
    :return:
    '''
    main_road_mask = main_road_config['det_mask']
    sub_road_mask = sub_road_config['det_mask']
    width = main_road_mask.shape[1]
    height = main_road_mask.shape[0]
    result = cv2.warpAffine(sub_road_mask, affine_matrix, (width, height))
    roi = cv2.bitwise_and(main_road_mask, result)
    # cv2.imwrite('/home/xinkaiji/temp_videos/stitch_mask.jpg',roi)
    return roi


def apply_affine_transform(points, A):
    # 将二维坐标转换为齐次坐标
    homogenous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    # 应用仿射变换
    transformed_points = np.dot(A, homogenous_points.T).T
    # 转换回二维坐标
    transformed_points = transformed_points[:, :2]
    return transformed_points

def affine_sub_tppkl(sub_vehicle_data, affine_matrix):
    '''
    将sub tppkl的数据变换到main的坐标下
    :param sub_vehicle_data:
    :param affine_matrix:
    :return:
    '''

    # new_affine_matrix = np.row_stack((affine_matrix, [0, 0, 1]))
    new_affine_matrix = affine_matrix
    for veh_id, veh_data in sub_vehicle_data.items():
        veh_data['geo_cpos_x'] = None
        veh_data['geo_cpos_y'] = None
        # new_pixel_cpos_x = []
        # new_pixel_cpos_y = []
        pixel_cpos = []
        if 'old_pixel_cpos_x' in veh_data:
            for i in range(len(veh_data['pixel_cpos_x'])):
                x = veh_data['old_pixel_cpos_x'][i]
                y = veh_data['old_pixel_cpos_y'][i]
                pixel_cpos.append([x, y])
            transformed_points = apply_affine_transform(np.array(pixel_cpos), new_affine_matrix)
            veh_data['pixel_cpos_x'] = transformed_points[:, 0].tolist()
            veh_data['pixel_cpos_y'] = transformed_points[:, 1].tolist()
        else:
            for i in range(len(veh_data['pixel_cpos_x'])):
                x = veh_data['pixel_cpos_x'][i]
                y = veh_data['pixel_cpos_y'][i]
                pixel_cpos.append([x, y])
                # point = np.array([x, y, 1])
                # new_point = np.matmul(new_affine_matrix, point.T)
                # new_x = new_point[0]
                # new_y = new_point[1]
                # new_pixel_cpos_x.append(new_x)
                # new_pixel_cpos_y.append(new_y)
            transformed_points = apply_affine_transform(np.array(pixel_cpos), new_affine_matrix)
            veh_data['old_pixel_cpos_x'] = veh_data['pixel_cpos_x'][:]
            veh_data['old_pixel_cpos_y'] = veh_data['pixel_cpos_y'][:]
            # veh_data['pixel_cpos_x'] = new_pixel_cpos_x
            # veh_data['pixel_cpos_y'] = new_pixel_cpos_y
            veh_data['pixel_cpos_x'] = transformed_points[:, 0].tolist()
            veh_data['pixel_cpos_y'] = transformed_points[:, 1].tolist()
    return sub_vehicle_data


def convert_sub_frame_index(sub_vehicle_data, f_s, f_m, det_s, det_m, u_s, u_m, offset_time):
    for veh_id, veh_data in sub_vehicle_data.items():
        if 'old_frame_index' in veh_data:
            frame_index = veh_data['old_frame_index']
            new_index_ls = []
            for index in frame_index:
                new_index = round(((index - f_s) * det_s + u_s - u_m) / det_m + f_m)
                new_index_ls.append(new_index)
            veh_data['frame_index'] = new_index_ls
            veh_data['start_unix_time'] = [veh_data['old_start_unix_time'][0] + offset_time,
                                           veh_data['old_start_unix_time'][1]]
        else:
            veh_data['old_frame_index'] = veh_data['frame_index'][:]
            veh_data['old_start_unix_time'] = veh_data['start_unix_time'][:]
            frame_index = veh_data['frame_index']
            new_index_ls = []
            for index in frame_index:
                new_index = round(((index - f_s) * det_s + u_s - u_m) / det_m + f_m)
                new_index_ls.append(new_index)
            veh_data['frame_index'] = new_index_ls
            veh_data['start_unix_time'] = [veh_data['start_unix_time'][0] + offset_time, veh_data['start_unix_time'][1]]
    return sub_vehicle_data


def select_roi(vehicle_data, roi):
    '''
    筛选roi内的轨迹数据
    :param vehicle_data:
    :param roi:
    :return:
    '''
    h, w, _ = roi.shape
    select_vehicle = {}
    vehicle_id_time_order = []
    for veh_id, veh_data in vehicle_data.items():
        frame_index = veh_data['frame_index']
        x_ls = veh_data['pixel_cpos_x']
        y_ls = veh_data['pixel_cpos_y']
        lane_id_ls = veh_data['lane_id']
        # start_unix_time = veh_data['start_unix_time'][0]
        select_frame_index = []
        select_position = []
        for i in range(len(frame_index)):
            x, y = int(x_ls[i]), int(y_ls[i])
            lane_id = lane_id_ls[i]
            if x < w and y < h and roi[y, x, 0] > 0:  # in roi
                select_frame_index.append(frame_index[i])
                select_position.append([x, y, lane_id])
        if len(select_frame_index) > 5:
            start_index, end_index = _longestConsecutive(select_frame_index)
            select_frame_index = select_frame_index[start_index:end_index + 1]
            select_position = select_position[start_index:end_index + 1]
            select_vehicle[veh_id] = [select_frame_index, select_position]
            vehicle_id_time_order.append([veh_id, select_frame_index[0]])
    if len(vehicle_id_time_order) > 0:
        vehicle_id_time_order.sort(key=lambda x: x[1], reverse=False)  # 升序
    return select_vehicle, vehicle_id_time_order


def _longestConsecutive(frame_index):
    '''
    list中获取最长的连续序列
    :param frame_index:
    :return:
    '''
    nums = frame_index[:]
    index_ls = [0]
    num_ls = []
    nums.append(INF)
    for i in range(len(nums) - 1):
        if abs(nums[i + 1] - nums[i]) > 1:
            num = i + 1 - index_ls[-1]
            num_ls.append(num)
            index_ls.append(i + 1)
    if len(num_ls) == 0:
        print(frame_index)
    index = np.argmax(num_ls)
    end_index = index_ls[index + 1] - 1
    start_index = index_ls[index]
    return start_index, end_index


def get_meta_data(vehicle_data):
    start_unix_time_ls = []
    end_unix_time_ls = []
    start_frame_index_ls = []
    end_frame_index_ls = []
    detaT = 0
    for veh_id, veh_data in vehicle_data.items():
        if 'old_frame_index' in veh_data:
            frame_index_key = 'old_frame_index'
            start_unix_time_key = 'old_start_unix_time'
        else:
            frame_index_key = 'frame_index'
            start_unix_time_key = 'start_unix_time'
        if isinstance(veh_data[frame_index_key], np.ndarray):
            veh_data[frame_index_key] = [round(x) for x in veh_data[frame_index_key]]
        start_unix_time = veh_data[start_unix_time_key][0]
        detaT = veh_data[start_unix_time_key][1]
        end_unix_time = start_unix_time + len(veh_data[frame_index_key]) * detaT

        start_frame_index_ls.append(veh_data[frame_index_key][0])
        end_frame_index_ls.append(veh_data[frame_index_key][-1])
        start_unix_time_ls.append(start_unix_time)
        end_unix_time_ls.append(end_unix_time)
    start_unix_time = min(start_unix_time_ls)
    end_unix_time = max(end_unix_time_ls)
    start_frame_index = min(start_frame_index_ls)
    end_frame_index = max(end_frame_index_ls)
    return start_unix_time, end_unix_time, start_frame_index, end_frame_index, detaT


def match_vehicle(selected_main_vehicle, main_vehicle_id_time_order, selected_sub_vehicle, sub_vehicle_id_time_order,
                  main_start_index, main_end_index, similarity_func=None):
    if similarity_func is None:
        similarity_func = _cal_similarity
    main_veh_id_ls = _get_horizon_data(main_vehicle_id_time_order, main_start_index, main_end_index)
    sub_veh_id_ls = _get_horizon_data(sub_vehicle_id_time_order, main_start_index, main_end_index)
    cos_matrix = []
    for main_veh_id in main_veh_id_ls:
        cos = []
        for sub_veh_id in sub_veh_id_ls:
            main_data = selected_main_vehicle[main_veh_id]
            sub_data = selected_sub_vehicle[sub_veh_id]
            dist = similarity_func(main_data, sub_data)
            cos.append(dist)
        cos_matrix.append(cos)
    cos_matrix = np.array(cos_matrix)
    h, w = cos_matrix.shape
    if h > w:
        temp = np.ones((h, h - w)) * INF
        cos_matrix = np.hstack((cos_matrix, temp))
    elif h < w:
        temp = np.ones((w - h, w)) * INF
        cos_matrix = np.vstack((cos_matrix, temp))
    step_match = True
    donot_matched = [[], []]
    if step_match:
        matched_m = []
        matched_s = []
        matched_dist = []
        pixel_dist = 300
        temp_cos = np.array(cos_matrix, copy=True)
        raw_cos = np.array(cos_matrix, copy=True)
        temp_cos[cos_matrix > pixel_dist] = INF
        matched_indices = linear_assignment(temp_cos)

        for m in matched_indices:
            if m[0] < h and m[1] < w:
                dist = temp_cos[m[0], m[1]]
                if dist <= pixel_dist:
                    m_id = main_veh_id_ls[m[0]]
                    s_id = sub_veh_id_ls[m[1]]
                    dist = raw_cos[m[0], m[1]]
                    matched_m.append(m_id)
                    matched_s.append(s_id)
                    matched_dist.append(dist)
                    cos_matrix[m[0], :] = INF
                    cos_matrix[:, m[1]] = INF
        matched_indices = linear_assignment(cos_matrix)
        matched_main = {}
        for m in matched_indices:
            if m[0] < h and m[1] < w:
                m_id = main_veh_id_ls[m[0]]
                s_id = sub_veh_id_ls[m[1]]
                dist = raw_cos[m[0], m[1]]
                if (not m_id in matched_m) and (not s_id in matched_s):
                    matched_m.append(m_id)
                    matched_s.append(s_id)
                    matched_dist.append(dist)
                if (not m_id in matched_m) and (s_id in matched_s):
                    donot_matched[0].append(m_id)
                if (m_id in matched_m) and (not s_id in matched_s):
                    donot_matched[1].append(s_id)
        for i in range(len(matched_m)):
            matched_main[matched_m[i]] = [matched_s[i], matched_dist[i]]
    else:
        matched_indices = linear_assignment(cos_matrix)
        matched_main = {}
        for m in matched_indices:
            if m[0] < h and m[1] < w:
                m_id = main_veh_id_ls[m[0]]
                s_id = sub_veh_id_ls[m[1]]
                dist = cos_matrix[m[0], m[1]]
                matched_main[m_id] = [s_id, dist]

    return matched_main,donot_matched


def _get_horizon_data(vehicle_id_time_order, start_index, end_index):
    veh_id_ls = []
    for veh_id, start_frame in vehicle_id_time_order:
        if start_index <= start_frame < end_index:
            veh_id_ls.append(veh_id)
        if start_frame > end_index:
            break
    return veh_id_ls


def _cal_similarity(main_data, sub_data):
    '''
    计算相似度
    :param main_data: [select_frame_index,select_position]
    :param sub_data:
    :return:
    '''
    sim = INF
    main_frame, main_position = main_data
    sub_frame, sub_position = sub_data
    s_frame = max(main_frame[0], sub_frame[0])
    e_frame = min(main_frame[-1], sub_frame[-1])
    if e_frame - s_frame < 5:
        return sim
    s_p_index = main_frame.index(s_frame)
    e_p_index = main_frame.index(e_frame)
    main_select_position = main_position[s_p_index:e_p_index + 1]
    s_p_index = sub_frame.index(s_frame)
    e_p_index = sub_frame.index(e_frame)
    sub_select_position = sub_position[s_p_index:e_p_index + 1]
    p_m = np.array(main_select_position)
    p_s = np.array(sub_select_position)
    xy_m = p_m[:, :2]
    xy_s = p_s[:, :2]
    direction = np.dot((xy_m[-1] - xy_m[0]), (xy_s[-1] - xy_s[0]))
    if direction < 0:  # 同个方向
        return sim
    dist = np.mean(np.sqrt(np.sum((xy_m - xy_s) ** 2, axis=0)))
    if 0<np.max(p_m[:, 2])<50 or 0<np.max(p_s[:, 2])<50: # 主路上考虑车道变化
        dist += np.mean(np.abs(p_m[:, 2] - p_s[:, 2])) * 100
    return dist


def linear_assignment(cost_matrix):
    '''
    匈牙利算法进行匹配
    :param cost_matrix:
    :return:
    '''
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def get_matched_file(matched_main):
    id_pair_ls = []
    threshold = 1000
    not_connect_ls = [[-1,-1,-1]]
    for main_id, item in matched_main.items():
        sub_id, dist = item
        if dist < threshold:
            id_pair_ls.append([main_id, sub_id, dist])
        else:
            not_connect_ls.append([main_id, sub_id, dist])
    matched_df = pd.DataFrame(np.array(id_pair_ls), columns=['main_id', 'sub_id', 'dist'])
    print('dist mean:%f,num:%d' % (matched_df['dist'].mean(), len(id_pair_ls)))
    mean_dist = matched_df['dist'].mean()
    pair_num = len(id_pair_ls)
    not_matched_df = pd.DataFrame(np.array(not_connect_ls), columns=['main_id', 'sub_id', 'dist'])
    return matched_df, mean_dist, pair_num,not_matched_df


def generate_new_tppkl(main_vehicle_data, sub_vehicle_data, matched_main, dist_threshold, only_connected_veh=True):
    new_veh_data = []
    dont_connect = []
    main_id_ls = []
    sub_id_ls = []
    drivingline_ls = []
    lane_name = ''
    for veh_id, veh_data in main_vehicle_data.items():
        dl = list(veh_data['drivingline'].keys())
        if len(dl) == 1:
            lane_name = dl[0]
        break
    for main_id, item in matched_main.items():
        sub_id, dist = item
        if dist > dist_threshold:
            continue
        main_data = main_vehicle_data[main_id]
        sub_data = sub_vehicle_data[sub_id]
        new_data = connect_vehicle_data(main_data, sub_data)
        if new_data is None:
            # print('can not match:m:%f,s:%f' % (main_id,sub_id))
            dont_connect.append([main_id, sub_id, dist])
        else:
            new_veh_data.append(new_data)
            main_id_ls.append(round(main_id))
            sub_id_ls.append(round(sub_id))
            drivingline_ls.append(new_data['drivingline'][lane_name][0][0])
            drivingline_ls.append(new_data['drivingline'][lane_name][-1][0])
    max_drivingline = max(drivingline_ls)
    min_drivingline = min(drivingline_ls)
    gap_dist = 30
    if not only_connected_veh:
        for veh_id, veh_data in main_vehicle_data.items():
            if not round(veh_id) in main_id_ls:
                if veh_data['frame_index'][0] < 10 or min_drivingline - gap_dist < \
                        veh_data['drivingline'][lane_name][0][0] < min_drivingline + gap_dist \
                        or max_drivingline - gap_dist < veh_data['drivingline'][lane_name][0][
                    0] < max_drivingline + gap_dist:
                    new_veh_data.append(veh_data)
                # if veh_data['lane_id'][0]==20 and veh_data['drivingline']['mainline'][0][0]>-200 and  veh_data['drivingline']['mainline'][0][0]<0:
                #     print('m',round(veh_id))
        for veh_id, veh_data in sub_vehicle_data.items():
            if not round(veh_id) in sub_id_ls:
                # new_veh_data.append(veh_data)

                if veh_data['frame_index'][0] < 10 or min_drivingline - gap_dist < \
                        veh_data['drivingline'][lane_name][0][0] < min_drivingline + gap_dist \
                        or max_drivingline - gap_dist < veh_data['drivingline'][lane_name][0][
                    0] < max_drivingline + gap_dist:
                    new_veh_data.append(veh_data)
                # if veh_data['lane_id'][0]==20 and veh_data['drivingline']['mainline'][-1][0]>0 and  veh_data['drivingline']['mainline'][-1][0]<200:
                #     print('s',round(veh_id))
                # else:
                #     new_veh_data.append(veh_data)
    new_veh_data.sort(key=lambda x: x['frame_index'][0], reverse=False)  # 升序
    new_veh_data_dict = {index: x for index, x in enumerate(new_veh_data)}
    return new_veh_data_dict, dont_connect


def connect_vehicle_data(main_data, sub_data):
    m_f = main_data['frame_index']
    s_f = sub_data['frame_index']
    line_name = list(main_data['drivingline'].keys())[0]

    if s_f[0] < m_f[0] < s_f[-1] < m_f[-1]:
        # 从sub开到main
        main_data, sub_data = sub_data, main_data
    if (s_f[0] < m_f[0] < s_f[-1] < m_f[-1]) or (s_f[-1] > m_f[-1] > s_f[0] > m_f[0]):
        start_index = main_data['frame_index'].index(sub_data['frame_index'][0])
        end_index = sub_data['frame_index'].index(main_data['frame_index'][-1])
        pos_x = main_data['pixel_cpos_x'][:start_index].copy()
        pos_y = main_data['pixel_cpos_y'][:start_index].copy()
        drivingline = main_data['drivingline'][line_name][:start_index].copy()
        for i in range(len(main_data['frame_index']) - start_index):
            x_m = main_data['pixel_cpos_x'][start_index + i]
            y_m = main_data['pixel_cpos_y'][start_index + i]
            x_s = sub_data['pixel_cpos_x'][i]
            y_s = sub_data['pixel_cpos_y'][i]
            x = x_m + (-x_m + x_s) * i / (len(main_data['frame_index']) - start_index)
            y = y_m + (-y_m + y_s) * i / (len(main_data['frame_index']) - start_index)
            pos_x.append(x)
            pos_y.append(y)
            d_m = main_data['drivingline'][line_name][start_index + i]
            d_s = sub_data['drivingline'][line_name][i]
            d_0 = d_m[0] + (-d_m[0] + d_s[0]) * i / (len(main_data['frame_index']) - start_index)
            d_1 = d_m[1] + (-d_m[1] + d_s[1]) * i / (len(main_data['frame_index']) - start_index)
            d = (d_0, d_1)
            drivingline.append(d)
        pos_x.extend(sub_data['pixel_cpos_x'][end_index + 1:].copy())
        pos_y.extend(sub_data['pixel_cpos_y'][end_index + 1:].copy())
        drivingline.extend(sub_data['drivingline'][line_name][end_index + 1:].copy())
        split_num = end_index // 2
        try:
            lane_id = main_data['lane_id'][:-split_num] + sub_data['lane_id'][end_index + 1 - split_num:]
        except:
            print(main_data['lane_id'][:-split_num])
            print(sub_data['lane_id'][end_index + 1 - split_num:])
        lane_dist = main_data['lane_dist'][:-split_num] + sub_data['lane_dist'][end_index + 1 - split_num:]
        frame_index = main_data['frame_index'][:-split_num] + sub_data['frame_index'][end_index + 1 - split_num:]
        start_unix_time = main_data['start_unix_time']
        detaT = main_data['detaT']
        vehicle_length = main_data["vehicle_length"]
        new_veh_data = {'frame_index': frame_index, 'pixel_cpos_x': pos_x, 'pixel_cpos_y': pos_y,
                        'drivingline': {line_name: drivingline}, 'lane_id': lane_id, 'lane_dist': lane_dist,
                        'start_unix_time': start_unix_time, 'detaT': detaT, 'vehicle_length': vehicle_length}
        for i in range(len(frame_index) - 1):
            if frame_index[i + 1] - frame_index[i] != 1:
                print('dont continue')
            # if drivingline[i+1][0]<drivingline[i][0]:
            #     print('e')
    else:
        return None
    return new_veh_data


def optimize_offset(roi, main_vehicle_data, sub_vehicle_data, sub_start_frame_index, main_start_frame_index,
                    sub_detaT, main_detaT, sub_start_unix_time, main_start_unix_time, raw_fps, windows=300,
                    horizon=100):
    min_dist_thred = 100 # 当最小距离小于min_dist_thred且目前距离大于min_scale被最小距离就停止搜索
    min_scale = 2
    print('optimize offset')
    threshold = 1000
    matched_ls = []
    main_start_t, main_end_t = main_start_unix_time, main_start_unix_time + horizon * 1000
    select_sub_data = {}
    for veh_id, veh_data in sub_vehicle_data.items():
        s_time = veh_data['start_unix_time'][0]
        e_time = s_time + len(veh_data['frame_index']) * veh_data['start_unix_time'][1]
        if main_start_t <= s_time <= main_end_t or main_start_t <= e_time <= main_end_t:
            select_sub_data[veh_id] = veh_data
    print('raw sub num:%d,select sub num:%d' % (len(sub_vehicle_data), len(select_sub_data)))
    min_dist = INF
    offset_list = [0]
    for offset in range(1, windows):
        offset_list.append(offset)
        offset_list.append(-offset)
    min_direction = 0
    early_stop = False
    for offset in offset_list:
        # for offset in range(-35,0):
        offset_time = offset * (1000 / raw_fps)
        u_s = sub_start_unix_time + offset_time
        # new_sub_vehicle_data = copy.deepcopy(select_sub_data)
        new_sub_vehicle_data = select_sub_data
        new_sub_vehicle_data = convert_sub_frame_index(new_sub_vehicle_data, sub_start_frame_index,
                                                       main_start_frame_index,
                                                       sub_detaT, main_detaT, u_s, main_start_unix_time, offset_time)
        selected_main_vehicle, main_vehicle_id_time_order = select_roi(main_vehicle_data, roi)
        selected_sub_vehicle, sub_vehicle_id_time_order = select_roi(new_sub_vehicle_data, roi)
        roll_horizon_frame = int(horizon * 1000 / main_detaT)
        main_start_index, main_end_index = main_start_frame_index, main_start_frame_index + roll_horizon_frame
        matched_main,donot_matched = match_vehicle(selected_main_vehicle, main_vehicle_id_time_order, selected_sub_vehicle,
                                     sub_vehicle_id_time_order,
                                     main_start_index, main_end_index)
        dist_ls = []
        for main_id, item in matched_main.items():
            sub_id, dist = item
            if dist < threshold:
                dist_ls.append(dist)
        if len(dist_ls) > 0:
            mean_dist = np.mean(dist_ls)
        else:
            mean_dist = INF
        matched_ls.append([offset, mean_dist, matched_main])
        if mean_dist < min_dist:
            min_dist = mean_dist
            min_direction = offset

        print('offset:%d,mean dist:mean_dist:%.2f,min_dist:%.2f' % (offset, mean_dist, min_dist))

        if (not early_stop) and (offset*min_direction>0) and min_dist < min_dist_thred and mean_dist>max(min_scale*min_dist,min_dist_thred):
            early_stop = True
        if early_stop: # 把这个注释掉就可以展示early stop之后的
            print('optimize problem early stop')
            break
        # if min_dist < min_dist_thred and mean_dist>max(min_scale*min_dist,min_dist_thred):
        #     print('optimize problem early stop')
        #     break
    matched_ls.sort(key=lambda x: x[1], reverse=False)  # 升序
    return matched_ls[0]


def remove_short_trajectory(vehicle_data):
    '''
    移除时间长度短的轨迹
    :param vehicle_data:
    :return:
    '''
    min_time = 10  # second
    del_key = []
    for key, item in vehicle_data.items():
        # min_frame = int(min_time/item['detaT'])
        # if len(item['frame_index']) < min_frame:
        #     del_key.append(key)
        lane_id_ls = item['lane_id']
        if max(set(lane_id_ls), key=lane_id_ls.count) == -1:
            del_key.append(key)
    for key in del_key:
        vehicle_data.pop(key)
    for key, item in vehicle_data.items():  # 因为双向两车道 所以车道不会变
        lane_id_ls = item['lane_id']
        lane_id = max(set(lane_id_ls), key=lane_id_ls.count)
        if lane_id == -1:
            print('ss')
        item['lane_id'] = [lane_id for _ in range(len(lane_id_ls))]


def save_data2pkl(vehicles_data, file_name):
    '''
    保存tppkl文件
    :param vehicles_data:
    :param file_name:
    :return:
    '''
    print('start save data:%s' % file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(vehicles_data, f)


def convert_drivingline(sub_vehicle_data, main_road_config_json_file, sub_road_config_json_file):
    # 获取sub上的drivingline为0的像素点

    main_dl_ls = DrivingLineList([main_road_config_json_file])
    sub_dl_ls = DrivingLineList([sub_road_config_json_file])
    # 计算该点在main视频中的drivingline长度
    assert len(main_dl_ls.drivingline_name_list) == 1, 'too many drivingline'
    line_name = main_dl_ls.drivingline_name_list[0]
    main_name = list(main_dl_ls.drivingline[line_name].base_dist_dict.keys())[0]
    sub_name = list(sub_dl_ls.drivingline[line_name].base_dist_dict.keys())[0]
    main_pts = _get_transformer_pts(main_road_config_json_file, main_name)
    sub_pts = _get_transformer_pts(sub_road_config_json_file, main_name)

    main_d_pts = main_dl_ls.get_global_distance(main_pts, main_name, line_name, True)
    sub_d_pts = sub_dl_ls.get_global_distance(sub_pts, sub_name, line_name, True)

    # 变换sub_vehiucle_data中所有的drivingline
    gap_dist_ls = []
    for main_d, sub_d in zip(main_d_pts, sub_d_pts):
        gap = main_d[0] - sub_d[0]
        gap_dist_ls.append(gap)
    mean_gap = max(gap_dist_ls)
    for veh_id, veh_data in sub_vehicle_data.items():
        for index, drivingline in enumerate(veh_data['drivingline'][line_name]):
            veh_data['drivingline'][line_name][index] = [drivingline[0] + mean_gap, drivingline[1]]
    return sub_vehicle_data




def get_video_edges(multi_video_config):
    '''
    获取需要处理的视频边
    :param multi_video_config:
    :return:
    '''
    # 从config文件中读取视频网络连接表路径
    multi_video_config_dict = MultiVideosConfig.fromfile(multi_video_config)
    video_graph_vg = multi_video_config_dict['video_graph']
    file_folder, file_name = os.path.split(multi_video_config)
    video_graph_path = os.path.join(file_folder, video_graph_vg)
    # 获取主视频编号
    # main_node = multi_video_config_dict['MainVideo'][1]
    # 创建一个空的有向图
    G = nx.DiGraph()
    # 从文本文件中读取连接关系
    with open(video_graph_path, "r") as f:
        for line in f:
            if line.strip():
                edge = line.strip().split(',')
                G.add_edge(edge[0], edge[1])
    # 遍历所有节点
    process_edges = []
    # 遍历边
    for edge in G.edges():
        source = edge[0]  # 源节点
        target = edge[1]  # 目标节点
        process_edges.append([source, target])
    # for node in G.nodes():
    #     if node == main_node:
    #         continue
    #     shortest_path = nx.shortest_path(G, node, main_node)
    #     # 提取路径中的边
    #     edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
    #     all_edge.extend(edges)
    # process_edges = list(set(all_edge))
    print(process_edges)
    return process_edges, G


def load_all_tppkl(video_id_dict):
    tppkl_dict = {}
    for video_id, video_json_path in video_id_dict.items():
        vehicle_data, img, json_name = get_tppkl(video_json_path)
        tppkl_dict[video_id] = [vehicle_data, img, json_name]
    return tppkl_dict


def get_edge_parameter(node_f, node_t, video_id_dict, multi_video_config_dict, offset_dict, tppkl_dict,img_save_folder):
    auto_optimize_offset = multi_video_config_dict.get('auto_optimize_offset', False)
    raw_fps = multi_video_config_dict.get('raw_video_fps', 30)  # 原始视频的fps
    # 获取main和sub的tppkl轨迹数据
    main_video_json_path = video_id_dict[node_t]
    sub_video_json_path = video_id_dict[node_f]
    # main_vehicle_data, main_img, main_json_name = get_tppkl(main_video_json_path)
    # sub_vehicle_data, sub_img, sub_json_name = get_tppkl(sub_video_json_path)
    main_vehicle_data, main_img, main_json_name = tppkl_dict[node_t]
    sub_vehicle_data, sub_img, sub_json_name = tppkl_dict[node_f]
    # 获取sub映射到main的仿射变换矩阵
    stitch_name = "%s%s" % (node_f, node_t)
    affine_matrix, split_x_position, roi = get_affine_matrix(main_video_json_path, sub_video_json_path, stitch_name)
    sub_vehicle_data = affine_sub_tppkl(sub_vehicle_data, affine_matrix)
    image_stitch(main_img, sub_img, affine_matrix, split_x_position, node_f, node_t,img_save_folder)  # 图片拼接
    # if multi_video_config_dict.get('convert_sub_drivingline', False):
    #     config_dict = Config.fromfile(main_video_json_path)
    #     main_road_config_json_file = config_dict['road_config']
    #     config_dict = Config.fromfile(sub_video_json_path)
    #     sub_road_config_json_file = config_dict['road_config']
    #     sub_vehicle_data = convert_drivingline(sub_vehicle_data, main_road_config_json_file, sub_road_config_json_file)

    main_start_unix_time, main_end_unix_time, main_start_frame_index, main_end_frame_index, main_detaT = get_meta_data(
        main_vehicle_data)
    sub_start_unix_time, sub_end_unix_time, sub_start_frame_index, sub_end_frame_index, sub_detaT = get_meta_data(
        sub_vehicle_data)
    # optimal offset
    # offset = None
    offset = offset_dict[node_f]
    # if auto_optimize_offset:
    if offset is None:
        print('optimize %s to %s' % (node_f, node_t))
        offset_best, mean_dist, matched_main = optimize_offset(roi, main_vehicle_data, sub_vehicle_data,
                                                               sub_start_frame_index, main_start_frame_index,
                                                               sub_detaT, main_detaT, sub_start_unix_time,
                                                               main_start_unix_time, raw_fps)
        print('%s to %s :best offset:%d,mean_dist:%.2f' % (node_f, node_t,offset_best, mean_dist))
        offset = offset_best
    #
    # offset = 62
    offset_time = offset * (1000 / raw_fps)
    u_s = sub_start_unix_time + offset_time

    sub_vehicle_data = convert_sub_frame_index(sub_vehicle_data, sub_start_frame_index, main_start_frame_index,
                                               sub_detaT, main_detaT, u_s, main_start_unix_time, offset_time)

    # 筛选出重叠区域的轨迹数据
    selected_main_vehicle, main_vehicle_id_time_order = select_roi(main_vehicle_data, roi)
    selected_sub_vehicle, sub_vehicle_id_time_order = select_roi(sub_vehicle_data, roi)
    print(len(selected_main_vehicle), len(selected_sub_vehicle))
    # 轨迹排序，在滚动窗口内，计算两两轨迹重叠时间内的相似度
    # 匈牙利算法进行匹配

    roll_win = 100000  # seconds 秒
    step_gap = 1000
    main_total_frame = int((main_end_unix_time - main_start_unix_time) / main_detaT)
    roll_horizon_frame = int(roll_win * 1000 / main_detaT)
    step_frame = int(step_gap * 1000 / main_detaT)

    main_start_index = 0
    # main_end_index = main_total_frame + 1
    main_end_index = 5*60*10 + 1
    matched_main,donot_matched_single = match_vehicle(selected_main_vehicle, main_vehicle_id_time_order, selected_sub_vehicle,
                                 sub_vehicle_id_time_order,
                                 main_start_index, main_end_index)
    matched_df, mean_dist, pair_num,not_matched_df = get_matched_file(matched_main)

    return offset, matched_df, mean_dist, pair_num, affine_matrix, split_x_position, roi,not_matched_df,donot_matched_single


def get_to_main(G, main_node, tran_dict):
    node_to_main = {}
    for node in G.nodes():
        if node == main_node:
            continue
        shortest_path = nx.shortest_path(G, node, main_node)
        # 提取路径中的边
        edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
        offset_to_main = 0
        affine_to_main = np.eye(3)
        for node_f, node_t in edges:
            tran_info = tran_dict[(node_f, node_t)]
            offset_to_main += tran_info["offset"]
            new_affine_matrix = np.row_stack((tran_info["affine_matrix"], [0, 0, 1]))
            affine_to_main = np.matmul(new_affine_matrix, affine_to_main)
        node_to_main[node] = {"offset": offset_to_main, "affine_matrix": affine_to_main}
    return node_to_main


def combine_vehicles(edge_ls, match_df_ls):
    all_list = []
    matched_veh = {}
    for (node_f, node_t), match_df in zip(edge_ls, match_df_ls):
        data_list = match_df.values.tolist()
        for main_node_veh, sub_node_veh, _ in data_list:
            new_t_name = (node_t, main_node_veh)
            new_f_name = (node_f, sub_node_veh)
            all_list.append([new_f_name, new_t_name])
            if node_t in matched_veh:
                matched_veh[node_t].append(main_node_veh)
            else:
                matched_veh[node_t] = [main_node_veh]
            if node_f in matched_veh:
                matched_veh[node_f].append(sub_node_veh)
            else:
                matched_veh[node_f] = [sub_node_veh]
    res_lists = merge_lists(all_list)
    return res_lists, matched_veh


# 时间对齐
def time_space_alignment(tppkl_dict, node_to_main, raw_fps, main_video_id):
    main_vehicle_data, _, _ = tppkl_dict[main_video_id]
    main_start_unix_time, main_end_unix_time, main_start_frame_index, main_end_frame_index, main_detaT = get_meta_data(
        main_vehicle_data)
    new_tppkl_dict = {}
    for video_id, sub_vehicle_data in tppkl_dict.items():
        sub_vehicle_data = sub_vehicle_data[0]
        if video_id in node_to_main:
            offset_to_main = node_to_main[video_id]["offset"]
            affine_to_main = node_to_main[video_id]["affine_matrix"]

            sub_start_unix_time, sub_end_unix_time, sub_start_frame_index, sub_end_frame_index, sub_detaT = get_meta_data(
                sub_vehicle_data)
            sub_vehicle_data = affine_sub_tppkl(sub_vehicle_data, affine_to_main)  # 在计算车辆匹配时候已经变换过了

            offset_time = offset_to_main * (1000 / raw_fps)
            u_s = sub_start_unix_time + offset_time
            sub_vehicle_data = convert_sub_frame_index(sub_vehicle_data, sub_start_frame_index, main_start_frame_index,
                                                       sub_detaT, main_detaT, u_s, main_start_unix_time, offset_time)
        new_tppkl_dict[video_id] = sub_vehicle_data
    return new_tppkl_dict


def combine_tppkl(combined_veh_lists, tppkl_dict, matched_veh):
    '''

    :param combined_veh_lists:
    :param tppkl_dict: 时间对齐后的tppkl
    :return:
    '''
    veh_id_index = 0
    new_veh_dict = {}
    for combine_v in combined_veh_lists:
        data_list = []
        for video_id, veh_id in combine_v:
            veh_data = tppkl_dict[video_id][veh_id]
            data_list.append(veh_data)
        new_veh_data = combine_vehicle_data(data_list)
        new_veh_dict[veh_id_index] = new_veh_data
        veh_id_index += 1
    for video_id, veh_data in tppkl_dict.items():
        for veh_id, veh_info in veh_data.items():
            if veh_id not in matched_veh[video_id]:  # 不在已经匹配的车辆中
                new_veh_dict[veh_id_index] = veh_info
                veh_id_index += 1
    return new_veh_dict


def combine_vehicle_data(data_list):
    new_veh_data = {}
    frame_index_ls = []
    start_unix_time_0 = []
    start_unix_time_1 = []
    detaT = data_list[0]['detaT']
    vehicle_length = data_list[0]['vehicle_length']
    for vehicle_data in data_list:
        frame_index = vehicle_data['frame_index']
        frame_index_ls.extend(frame_index)
        start_unix_time_0.append(vehicle_data['start_unix_time'][0])
        start_unix_time_1.append(vehicle_data['start_unix_time'][1])
    frame_index_ls = sorted(set(frame_index_ls))
    assert frame_index_ls[-1] - frame_index_ls[0] + 1 == len(frame_index_ls), "frame index not continue"
    # if frame_index_ls[-1] - frame_index_ls[0] + 1 != len(frame_index_ls):
    #     print('sss')
    pixel_cpos_x_ls = []
    pixel_cpos_y_ls = []
    lane_id_ls = []
    for f_index in frame_index_ls:
        pixel_cpos_x = []
        pixel_cpos_y = []
        lane_id = []
        for vehicle_data in data_list:
            frame_index = vehicle_data['frame_index']
            if frame_index[0] <= f_index <= frame_index[-1]:
                sec_index = f_index - frame_index[0]
                x_s = vehicle_data['pixel_cpos_x'][sec_index]
                y_s = vehicle_data['pixel_cpos_y'][sec_index]
                l_id = vehicle_data['lane_id'][sec_index]
                pixel_cpos_x.append(x_s)
                pixel_cpos_y.append(y_s)
                lane_id.append(l_id)
        pixel_cpos_x_ls.append(np.mean(pixel_cpos_x))
        pixel_cpos_y_ls.append(np.mean(pixel_cpos_y))
        lane_id = list(set(lane_id))
        # if len(lane_id)>1:
        #     print('lane id error')
        if len(lane_id) > 2:
            print('sss', lane_id)
        lane_id_ls.append(lane_id[0])
    new_start_unix_time_0 = np.min(start_unix_time_0)
    assert np.max(start_unix_time_1) - np.min(start_unix_time_1)<0.01,"detaT is error"

    new_start_unix_time_1 = start_unix_time_1[0]
    new_veh_data["frame_index"] = frame_index_ls
    new_veh_data["pixel_cpos_x"] = pixel_cpos_x_ls
    new_veh_data["pixel_cpos_y"] = pixel_cpos_y_ls
    new_veh_data["lane_id"] = lane_id_ls
    new_veh_data["start_unix_time"] = (new_start_unix_time_0,new_start_unix_time_1)
    new_veh_data['detaT'] = detaT
    new_veh_data['vehicle_length'] = vehicle_length
    return new_veh_data


def merge_lists(lists_list):
    merged_lists = []
    # 遍历输入列表中的每个列表
    for list_1 in lists_list:
        merged = False
        # 遍历已合并的列表列表
        for list_2 in merged_lists:
            # 如果list_1和list_2有共同元素
            if any(elem in list_1 for elem in list_2):
                # 合并两个列表并更新merged标志
                merged_lists.remove(list_2)
                merged_lists.append(list(set(list_1 + list_2)))
                merged = True
                break
        # 如果list_1没有与已合并的列表共同元素，则将其添加到merged_lists中
        if not merged:
            merged_lists.append(list_1)
    return merged_lists


def affine_drivingline(video_json_dict, node_to_main):
    trans_drivingline_dict = {}
    for video_id, affine_dict in node_to_main.items():
        affine = affine_dict['affine_matrix']
        video_json_path = video_json_dict[video_id]
        config_dict = Config.fromfile(video_json_path)
        video_road_config_json_file = config_dict.get('road_config', None)
        pts = _get_drivingline(video_road_config_json_file)
        transformed_points = apply_affine_transform(np.array(pts), affine)
        trans_drivingline_dict[video_id] = transformed_points
    # 连接


def _get_drivingline(road_config_json_file):
    '''
    获取用于计算仿射变换的点
    :param road_config_json_file:
    :param main_name:
    :return:
    '''
    label = 'drivingline_'
    with open(road_config_json_file, 'r', encoding='utf-8') as f:
        road_config = json.load(f)
    for shape in road_config['shapes']:
        if shape['label'].startswith(label) and shape['label'].endswith("line"):  #
            pts = shape["points"]
            return pts


def save_node_to_main(node_to_main, multi_video_config,save_folder):
    file_folder, file_name = os.path.split(multi_video_config)
    file_name_without_ext, file_ext = os.path.splitext(file_name)
    node_name = 'node_to_main_' + file_name_without_ext + '.nm'
    node_name = os.path.join(save_folder, node_name)
    print('save offset time and affine transformation matrix to:%s' % node_name)
    with open(node_name, 'wb') as f:
        pickle.dump(node_to_main, f)

def save_match_df(match_df_ls,not_match_df_ls,edge_ls,save_folder):
    '''

    :param match_df_ls:
    :param save_folder:
    :return:
    '''
    match_folder = os.path.join(save_folder, 'match_table')
    os.makedirs(match_folder, exist_ok=True)
    for index,df in enumerate(match_df_ls):
        node_f, node_t = edge_ls[index]
        tb_name = "match_%s_%s.csv"%(node_f,node_t)
        tb_name = os.path.join(match_folder,tb_name)
        df.to_csv(tb_name, index=False)
    for index, df_single in enumerate(not_match_df_ls):
        df,single = df_single
        node_f, node_t = edge_ls[index]
        tb_name = "not_match_%s_%s.csv" % (node_f, node_t)
        tb_name = os.path.join(match_folder, tb_name)
        df.to_csv(tb_name, index=False)
        # 打开文件并写入列表数据
        output_file = "not_match_single_%s_%s.csv" % (node_f, node_t)
        output_file = os.path.join(match_folder, output_file)
        with open(output_file, 'w') as file:
            # 写入第一个列表的内容
            file.write('M:'+','.join(str(item) for item in single[0]))
            file.write('\n')  # 换行
            # 写入第二个列表的内容
            file.write('S:'+','.join(str(item) for item in single[1]))
            file.write('\n')  # 换行

def run_mainV2(multi_video_config):
    output_file_name = []
    video_id_dict = {}
    file_folder, file_name = os.path.split(multi_video_config)
    json_name, extension = os.path.splitext(file_name)

    multi_video_config_dict = MultiVideosConfig.fromfile(multi_video_config)
    video_json, main_node_id = multi_video_config_dict['MainVideo']
    video_json_path = os.path.join(file_folder, video_json)
    video_id_dict[main_node_id] = video_json_path
    sub_video_json_ls = multi_video_config_dict['SubVideos']

    save_folder = multi_video_config_dict["save_folder"]
    save_folder = os.path.join(save_folder, json_name)
    os.makedirs(save_folder, exist_ok=True)
    img_save_folder = os.path.join(save_folder, 'stitch')
    os.makedirs(img_save_folder, exist_ok=True)
    # 复制json文件到保存的文件夹
    video_graph_vg = multi_video_config_dict['video_graph']
    file_folder, file_name = os.path.split(multi_video_config)
    video_graph_path = os.path.join(file_folder, video_graph_vg)
    shutil.copy(multi_video_config,save_folder)
    shutil.copy(video_graph_path, save_folder)

    offset_dict = {}
    output_file_name.append(main_node_id)
    for sub_video_json in sub_video_json_ls:
        sub_json = sub_video_json[0]
        offset_temp = sub_video_json[1]
        sub_id = sub_video_json[2]
        sub_json = os.path.join(file_folder, sub_json)
        video_id_dict[sub_id] = sub_json
        offset_dict[sub_id] = offset_temp
        output_file_name.append(sub_id)
    process_edges, G = get_video_edges(multi_video_config)  # 获取要处理的边
    tppkl_dict = load_all_tppkl(video_id_dict)
    tran_dict = {}
    edge_ls = []
    match_df_ls = []
    not_match_df_ls = []
    for node_f, node_t in process_edges:
        offset, matched_df, mean_dist, pair_num, affine_matrix, split_x_position, roi,not_matched_df,donot_matched_single = get_edge_parameter(node_f,
                                                                                                           node_t,
                                                                                                           video_id_dict,
                                                                                                           multi_video_config_dict,
                                                                                                           offset_dict,
                                                                                                           tppkl_dict,
                                                                                                           img_save_folder)
        tran_dict[(node_f, node_t)] = {"offset": offset, "affine_matrix": affine_matrix}
        edge_ls.append([node_f, node_t])
        match_df_ls.append(matched_df)
        not_match_df_ls.append([not_matched_df,donot_matched_single])
    node_to_main = get_to_main(G, main_node_id, tran_dict)  # 获得变换到main的
    save_node_to_main(node_to_main, multi_video_config,save_folder)
    save_match_df(match_df_ls,not_match_df_ls,edge_ls,save_folder)
    # affine_drivingline(video_id_dict, node_to_main)
    image_stitch_multi(main_node_id, tppkl_dict, node_to_main,img_save_folder)

    raw_fps = multi_video_config_dict.get('raw_video_fps', 30)  # 原始视频的fps
    new_tppkl_dict = time_space_alignment(tppkl_dict, node_to_main, raw_fps, main_node_id)
    new_tppkl_dict[main_node_id] = tppkl_dict[main_node_id][0]
    combined_veh_lists, matched_veh_dict = combine_vehicles(edge_ls, match_df_ls)
    all_veh_data = combine_tppkl(combined_veh_lists, new_tppkl_dict, matched_veh_dict)
    output_file_name = os.path.join(save_folder,'stitch_tppkl_' + json_name+ ".tppkl")
    print('save connected tppkl to:%s'%output_file_name)
    with open(output_file_name , "wb") as file:
        pickle.dump(all_veh_data, file)


if __name__ == '__main__':
    # multi_video_json = '../../config/Neihuanxi/multi_20221101_NS_F5.json'
    # run_main(multi_video_json)
    multi_video_json = '../../config/mixed_roads/multi_20220617_D1toA3_F1.json'
    run_mainV2(multi_video_json)

    # lists_list = [[("A",1), ("B",1)], [("B",1), ("C",1)], [("D",1), ("E",1)]]
    # merged_lists = merge_lists(lists_list)
    # print(merged_lists)

    # nums = [1,2,3,5,6,7,9,10,11,12,13,14,18]
    # start_index,end_index = longestConsecutive(nums)
    #
    # print(start_index,end_index)
    # print(nums)
    # print(nums[start_index:end_index+1])
