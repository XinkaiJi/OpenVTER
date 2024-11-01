#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2022-01-03 20:22
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : config.py 
@Software: PyCharm
@desc: 
'''
import base64
import json
import os
import sys
import os.path as osp
from importlib import import_module
import ast
import numpy as np
import cv2
from shapely.geometry import Point
from shapely.geometry import LineString

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

def Merge(dict1, dict2):
    return (dict2.update(dict1))

class Config:

    @staticmethod
    def fromfile(filename):
        return Config._file2dict(filename)

    @staticmethod
    def _file2dict(filename):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in ['.json']:
            raise IOError('Only json type are supported now!')
        folder, video_name_ext = os.path.split(filename)
        with open(filename,'r') as f:
            config_dict = json.load(f)
        if 'base' in config_dict:
            base_json = config_dict['base']
            base_json_filename = os.path.join(folder,base_json)
            check_file_exist(base_json_filename)
            with open(base_json_filename, 'r') as f:
                base_dict = json.load(f)
            Merge(config_dict,base_dict)
            base_dict.pop('base', 'no base(key)')
            return base_dict
        if 'detection' in config_dict:
            detection_json = config_dict['detection']
            detection_json_filename = os.path.join(folder, detection_json)
            check_file_exist(detection_json_filename)
            with open(detection_json_filename, 'r') as f:
                detection_dict = json.load(f)
            config_dict['detection'] = detection_dict
        if 'tracking' in config_dict:
            tracking_json = config_dict['tracking']
            tracking_json_filename = os.path.join(folder, tracking_json)
            check_file_exist(tracking_json_filename)
            with open(tracking_json_filename, 'r') as f:
                tracking_dict = json.load(f)
            config_dict['tracking'] = tracking_dict

        return config_dict

    @staticmethod
    def get_video_file(config_dict):
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

class RoadConfig:

    @staticmethod
    def fromfile(filename):
        return RoadConfig._file2dict(filename)

    @staticmethod
    def _file2dict(filename):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in ['.json']:
            raise IOError('Only json type are supported now!')
        stab_mask = None
        # fixed_points = []
        length_per_pixel_ls = []
        lane_dict = {}
        base_points = None
        x_axis_end = None
        x_axis_vector = None
        y_point = None
        y_axis_vector = None
        axis_image = None
        pixel2xy_matrix = None
        mean_length_per_pixel = None
        img_np = None
        intersection_region = {}
        drivingline = {}
        laneline_dict = {}
        with open(filename, 'r',encoding='utf-8') as f:
            data = json.load(f)
            if data['imageWidth'] and data['imageHeight'] and data['shapes']:
                width = data['imageWidth']
                height = data['imageHeight']
                stab_mask = np.zeros((height, width,3),dtype = "uint8")
                det_mask = np.zeros((height, width, 3),dtype = "uint8")
                if data['imageData']:
                    imageData = data['imageData']
                    img = base64.b64decode(imageData)
                    nparr = np.frombuffer(img, np.uint8)
                    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                for shape in data['shapes']:
                    if shape['label'] == 'fp': # 稳定跟踪固定区域
                        # points = shape['points']
                        # x1, y1 = points[0]
                        # x2, y2 = points[1]
                        # stab_mask[int(y1):int(y2), int(x1):int(x2)] = 255
                        # w = x2 - x1
                        # h = y2 - y1
                        points = shape["points"]
                        points = np.array(points, np.int32)
                        cv2.fillPoly(stab_mask, [points], (255, 255, 255))
                        # fixed_points.append([x1, y1, w, h])
                    if shape['label'] == 'road': # 目标检测区域
                        points = shape["points"]
                        points = np.array(points, np.int32)
                        cv2.fillPoly(det_mask, [points], (255, 255, 255))

                    if shape['label'].startswith('lane_'): # 车道区域
                        lane_index = int(shape['label'].split('_')[-1])
                        lane_dict[lane_index] = shape['points']

                    if shape['label'].startswith('laneline_'): # 车道区域
                        key_name = shape['label']
                        value_points = shape["points"]
                        laneline_dict[key_name] = value_points
                    if shape['label'].startswith('length'): # 单位像素距离
                        length = float(shape['label'].split('_')[-1])
                        points = shape['points']
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        pixel_length = ((x1-x2)**2+(y1-y2)**2)**0.5
                        length_per_pixel = length/pixel_length
                        length_per_pixel_ls.append(length_per_pixel)
                    if shape['label'] == 'x': # x轴
                        points = shape['points']
                        base_points = points[0] # 原点
                        x_axis_end = points[1]
                        x_axis_vector = [x_axis_end[0]-base_points[0],x_axis_end[1]-base_points[1]] #
                    if shape['label'] == 'y':
                        points = shape['points']
                        y_point = points[0]
                    if shape['label'].startswith('int'):
                        id = shape['label']
                        intersection_region[id] = shape["points"]
                    if shape['label'].startswith('drivingline'):
                        key_name = shape['label']
                        value_points = shape["points"]
                        drivingline[key_name] = value_points
                if base_points is not None and y_point is not None:
                    footpoint = getFootPoint(y_point,base_points,x_axis_end)  # 获取垂足
                    y_axis_vector = [y_point[0]-footpoint[0],y_point[1]-footpoint[1]]

                if len(length_per_pixel_ls)>0:
                    mean_length_per_pixel = np.mean(length_per_pixel_ls)
                    if x_axis_vector and y_axis_vector:
                        unit =10
                        axis_image = RoadConfig._generate_axis_image(width, height, base_points, x_axis_vector, y_axis_vector, mean_length_per_pixel, unit)
                        pixel2xy_matrix = RoadConfig._get_pixel2xy_matrix(base_points, x_axis_vector, y_axis_vector, mean_length_per_pixel, unit)
                # if len(laneline_dict)>0:
                #     axis_image = RoadConfig._draw_laneline(axis_image,laneline_dict,width, height)
        res = {'det_mask':det_mask,'stab_mask':stab_mask[:,:,0],
               'length_per_pixel':mean_length_per_pixel,'base_points':base_points,
               'x_axis_vector':x_axis_vector,'y_axis_vector':y_axis_vector,'lane':lane_dict,'axis_image':axis_image,
               'pixel2xy_matrix':pixel2xy_matrix,"intersection_region":intersection_region,
               "drivingline":drivingline,"laneline":laneline_dict,"image":img_np}
        # img_logo_gray = cv2.cvt Color(axis_image, cv2.COLOR_BGR2GRAY)
        # ret, img_logo_mask = cv2.threshold(img_logo_gray, 10, 255, cv2.THRESH_BINARY)  # 二值化函数
        # img_logo_mask = cv2.bitwise_not(img_logo_mask)
        # img_res0 = cv2.bitwise_and(img_np, img_np, mask=img_logo_mask)
        # img_res2 = cv2.add(img_np,axis_image)
        # cv2.imwrite('demo_axis.png', img_res2)
        return res

    @staticmethod
    def _generate_axis_image(w,h,base_points,x_axis_vector,y_axis_vector,mean_length_per_pixel,unit=10):
        '''
        生成带有坐标轴的黑色底图
        :param w: 图片宽
        :param h: 图片高
        :param base_points: 原点坐标
        :param x_axis_vector: x轴向量
        :param y_axis_vector: y轴向量
        :param mean_length_per_pixel: 每个像素的长度
        :param unit: 刻度的长度，默认每10米一个刻度
        :return: 带有坐标轴的黑色地图
        '''
        black_image = np.zeros((int(h),int(w),3),dtype=np.uint8)
        pixel_length = unit/mean_length_per_pixel # 10米的像素长度

        # x axis
        RoadConfig._draw_axis(black_image, w, h, base_points, x_axis_vector, pixel_length, unit)
        RoadConfig._draw_axis(black_image, w, h, base_points, y_axis_vector, pixel_length, unit)
        cv2.circle(black_image, (int(base_points[0]), int(base_points[1])), 10, (0, 255, 0), -1)  # 原点

        return black_image

    @staticmethod
    def _get_pixel2xy_matrix(base_points,x_axis_vector,y_axis_vector,mean_length_per_pixel,unit):
        '''
        获取像素坐标转地理坐标的仿射变换矩阵
        :param base_points: 原点
        :param x_axis_vector: x轴向量
        :param y_axis_vector: y轴向量
        :param mean_length_per_pixel:  每个像素的长度
        :param unit: 刻度的长度，默认每10米一个刻度
        :return: 仿射变换矩阵
        '''
        pixel_length = unit / mean_length_per_pixel  # 10米的像素长度
        axis_length = (x_axis_vector[0] ** 2 + x_axis_vector[1] ** 2) ** 0.5
        u_x = pixel_length / axis_length * x_axis_vector[0]
        u_y = pixel_length / axis_length * x_axis_vector[1]
        y_axis_length = (y_axis_vector[0] ** 2 + y_axis_vector[1] ** 2) ** 0.5
        y_u_x = pixel_length / y_axis_length * y_axis_vector[0]
        y_u_y = pixel_length / y_axis_length * y_axis_vector[1]
        src_points = np.float32([base_points, [base_points[0]+u_x, base_points[1]+u_y], [base_points[0]+y_u_x, base_points[1]+y_u_y]])
        dst_points = np.float32([[0, 0], [unit, 0],
                                 [0, unit]])
        # 6参数。三个点计算放射矩阵。
        affine_matrix = cv2.getAffineTransform(src_points, dst_points)
        return affine_matrix

    @staticmethod
    def _draw_axis(img,w,h,base_points,axis_vector,pixel_length,unit):
        '''
        在图片上绘制坐标轴
        :param img: 图片
        :param w: 图片宽
        :param h: 图片高
        :param base_points: 原点
        :param axis_vector: 坐标轴向量
        :param pixel_length: 每个像素的长度
        :param unit: 刻度的长度，默认每10米一个刻度
        :return:
        '''
        axis_length = (axis_vector[0] ** 2 + axis_vector[1] ** 2) ** 0.5
        u_x = pixel_length / axis_length * axis_vector[0]
        u_y = pixel_length / axis_length * axis_vector[1]
        positive_direction = int(min((w - base_points[0]) / abs(u_x), (h - base_points[1]) /abs(u_y)))
        negative_direction = int(min(base_points[0] / abs(u_x), base_points[1] / abs(u_y)))
        num_direction = max(positive_direction,negative_direction)
        cv2.line(img,
                        (int(base_points[0] - num_direction * u_x), int(base_points[1] - num_direction * u_y)),
                        (int(base_points[0] + num_direction * u_x), int(base_points[1] + num_direction * u_y)),
                        (255, 0, 0), 10)
        #cv2.arrowedLine
        i = 0
        while 1:
            x1 = int(base_points[0] + (i+1) * u_x)
            y1 = int(base_points[1] + (i+1) * u_y)
            if x1 < 0 or x1 > w or y1 < 0 or y1 > h:
                break
            i+=1
            cv2.circle(img, (x1, y1), 5, (0, 255, 0), -1)
            cv2.putText(img, '%d' % (i*unit), (x1-20, y1 - 10),
                        cv2.FONT_HERSHEY_COMPLEX,0.5, (255,255, 0), 1, 1)

        i = 0
        while 1:
            x1 = int(base_points[0] - (i+1) * u_x)
            y1 = int(base_points[1] - (i+1) * u_y)
            if x1<0 or x1>w or y1<0 or y1>h:
                break
            i+=1
            cv2.circle(img, (x1, y1), 5, (0, 255, 0), -1)
            cv2.putText(img, '%d' % (-i*unit), (x1-20, y1 - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, 1)

    @staticmethod
    def _draw_laneline(img,laneline_dict,w, h):
        if img is None:
            img = np.zeros((int(h), int(w), 3), dtype=np.uint8)
        line_color = (0, 200, 0)
        thickness = 2
        for laneline_id,pts in laneline_dict.items():
            cv2.polylines(img,[np.int_(pts)],isClosed=False,color=line_color,thickness=thickness)
        for laneline_id, pts in laneline_dict.items():
            # pt_m = pts[int(len(pts)/2)]
            pt_m = pts[0]
            center_x = int(pt_m[0])
            center_y = int(pt_m[1])
            cv2.putText(img, laneline_id, (center_x, center_y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1, 1)

        return img

class MultiVideosConfig:
    '''
    多视频的config
    '''
    @staticmethod
    def fromfile(filename):
        return MultiVideosConfig._file2dict(filename)

    @staticmethod
    def _file2dict(filename):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in ['.json']:
            raise IOError('Only json type are supported now!')
        with open(filename, 'r',encoding='utf-8') as f:
            res = json.load(f)

        return res


def getFootPoint(point, line_p1, line_p2):
    """
    获得垂足
    @point, line_p1, line_p2 : [x, y]
    """
    x0 = point[0]
    y0 = point[1]
    x1 = line_p1[0]
    y1 = line_p1[1]
    x2 = line_p2[0]
    y2 = line_p2[1]
    k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / \
        ((x2 - x1) ** 2 + (y2 - y1) ** 2) * 1.0

    xn = k * (x2 - x1) + x1
    yn = k * (y2 - y1) + y1
    return (xn, yn)

color_map = [(128,0,0),(0,128,0),(128,128,0),(0, 0, 128),
             (128,0,128),(0,128,128),(128,128,128),(64,0,0),
             (192,0,0),(64,128,0),(192,128,0),(64,0,128),
             (192,0,128),(64,128,128),(192,128,128),(0,64,0 ),(128,64,0)]


def get_centerpoint(lis):
    '''
    计算多边形中点
    :param lis: [[x1,y1],[x2,y2],...,[xn,yn]]
    :return:
    '''
    area = 0.0
    x, y = 0.0, 0.0
    a = len(lis)
    for i in range(a):
        lat = lis[i][0]  # weidu
        lng = lis[i][1]  # jingdu
        if i == 0:
            lat1 = lis[-1][0]
            lng1 = lis[-1][1]
        else:
            lat1 = lis[i - 1][0]
            lng1 = lis[i - 1][1]
        fg = (lat * lng1 - lng * lat1) / 2.0
        area += fg
        x += fg * (lat + lat1) / 3.0
        y += fg * (lng + lng1) / 3.0
    x = int(x / area)
    y = int(y / area)
    return x, y


def visualize_config(config_json, img_path=None):
    '''
    可视化road config
    :param road_config: 道路config文件
    :param img: config文件对应的图片
    :return:
    '''
    road_config = RoadConfig.fromfile(config_json)
    if img_path is None:
        img = road_config['image']
    else:
        img = cv2.imread(img_path)
    h,w,c = img.shape
    # 绘制车道线
    axis_image = road_config['axis_image']
    if axis_image is not None:
        img = cv2.add(img, axis_image)
    line_string_dict = road_config['laneline']
    # 首先提取车道线的label并排序
    laneline_label_list = []
    for key in line_string_dict.keys():
        if isinstance(key, str):
            laneline_label_list.append(key)
    laneline_label_list.sort(key=lambda x: int(x.split('_')[-1]), reverse=False)
    # 构造多边形并与车道线对应
    polygon_cor_ls = []
    for i in range(len(laneline_label_list) - 1):
        # 提取第一条线的坐标
        first_laneline_label = laneline_label_list[i]
        first_laneline_cor = line_string_dict[first_laneline_label]  # LineString

        # 提取第二条线的坐标
        second_laneline_label = laneline_label_list[i + 1]
        second_laneline_cor = line_string_dict[second_laneline_label]  # LineString

        # 坐标拼接生成Polygon
        if abs(int(first_laneline_label.split('_')[-1]) - int(second_laneline_label.split('_')[-1])) == 1:
            polygon_cor = first_laneline_cor + second_laneline_cor[::-1]
            polygon_cor_ls.append([polygon_cor,first_laneline_label])
    # 绘制车道区域
    background = np.zeros((int(h), int(w), 3), dtype=np.uint8)
    lane = road_config['lane']
    for lane_index,(polygon_cor,first_laneline_label) in enumerate(polygon_cor_ls):
        color = color_map[lane_index % len(color_map)]
        cv2.fillPoly(background, [np.int_(polygon_cor)], color)
        center_x, center_y = get_centerpoint(polygon_cor)
        cv2.putText(img, first_laneline_label, (center_x, center_y),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, 1)

    for index, (lane_id, pts) in enumerate(lane.items()):
        lane_index = index+len(polygon_cor_ls)
        color = color_map[lane_index%len(color_map)]
        cv2.fillPoly(background, [np.int_(pts)],color)
        center_x, center_y = get_centerpoint(pts)
        cv2.putText(img, 'lane_%d'%lane_id, (center_x, center_y),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, 1)
    img = cv2.addWeighted(img, 0.8, background, 0.4, 0)

    #绘制drivingline
    drivingline = road_config['drivingline']
    line_color = (100, 30, 220)
    thickness = 2
    length_per_pixel = road_config['length_per_pixel']

    from toolbox.module.DrivingLine import DrivingLineList
    drivingline_ls = DrivingLineList([config_json])
    drivingline_name = drivingline_ls.drivingline_name_list[0]
    if len(drivingline_ls.drivingline[drivingline_name].base_dist_dict) == 1:
        position_id = list(drivingline_ls.drivingline[drivingline_name].base_dist_dict.keys())[0]

    length_m = 50 # 10 meters
    pixel_length = length_m / length_per_pixel
    for drivingline_id,pts in drivingline.items():
        if drivingline_id.endswith('line'):
            cv2.polylines(img, [np.int_(pts)], isClosed=False, color=line_color, thickness=thickness)

            # 绘制刻度
            base_pixel = drivingline_ls.drivingline[drivingline_name].base_pixel_dict[position_id]
            points_ls = []
            line_strip = drivingline_ls.drivingline[drivingline_name].line_strip_dict[position_id]
            pt = Point(base_pixel[0], base_pixel[1])
            base_distance = line_strip.project(pt)
            line_strip_length = line_strip.length

            for i in np.arange(base_distance,line_strip_length,pixel_length):
                ip = line_strip.interpolate(i)
                points_ls.append([ip.x,ip.y])

            drivingline_dist = drivingline_ls.get_global_distance(points_ls, position_id, drivingline_name, True)
            for i in range(len(drivingline_dist)):
                cv2.circle(img,(int(points_ls[i][0]), int(points_ls[i][1])), 2, (0,0,0), 2)
                cv2.putText(img, '%.1f'%drivingline_dist[i][0], (int(points_ls[i][0]), int(points_ls[i][1])),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, 1)

    return img



# def get_multi_videos(yaml_file_path):
#     import yaml
#     with open(yaml_file_path,'r',encoding='utf8') as f:
#         multi_video = yaml.safe_load(f)
#         print(multi_video)

if __name__ == '__main__':
    # filename = '/data2/xinkaiji/mixedroad_process/20220617/merge_config/merge_F1/20220617_0700_A1_F1_374_1.json'
    # res = RoadConfig.fromfile(filename)
    # img = cv2.imread("/data2/xinkaiji/mixedroad_process/20220617/road_config/20220617_0700_A1_F1_374_1.jpg")
    # det_mask = res["det_mask"]
    # axis_image = res["axis_image"]
    # img = cv2.bitwise_and(img, det_mask)
    #
    # img = cv2.add(img,axis_image)
    # cv2.imwrite("/home/xinkaiji/temp_videos/20220617_0700_A1_F1_374_1_mask.jpg",img)

    road_config_file = '/data3/DJIData/HurongHighway/20220617/B1-B2/M-20220617_0700_B2_F1_372_1-S-20220617_0700_B1_F1_370_1/first_frame_M-20220617_0700_B2_F1_372_1-S-20220617_0700_B1_F1_370_1.json'
    img_path = '/data3/DJIData/HurongHighway/20220617/B1-B2/M-20220617_0700_B2_F1_372_1-S-20220617_0700_B1_F1_370_1/first_frame_M-20220617_0700_B2_F1_372_1-S-20220617_0700_B1_F1_370_1.jpg'


    img = visualize_config(road_config_file, img_path)
    folder,file_path = os.path.split(img_path)
    img_file_path ='labeled_'+file_path
    output_path = os.path.join(folder,img_file_path)
    cv2.imwrite(output_path, img)

    # y = [50,60]
    # x1 = [10,10]
    # x2 = [100,0]
    # print(getFootPoint(y,x1,x2))

    # p1 = [10,10]
    # p2 = [11,20]
    # p3 = [21,30]
    # p4 = [20,40]
    # print(findIntersection(p1, p2, p3, p4))
    # get_multi_videos('../config/yingtianstreet/0708/multi_20220708_F1.yaml')
    pass