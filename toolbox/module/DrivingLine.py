#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2022-07-03 21:04
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : CoordinateSystem.py 
@Software: PyCharm
@desc: 坐标系工具 实现各类坐标系的转换
'''
import os

from shapely.geometry import Point
from shapely.geometry import LineString
from utils import isPointinPolygon
# class CoordinateSystem:
#
#     @staticmethod
#     def pixelPt2drivingDist(pixel_pts,driving_line,base_point=None):
#         '''
#         根据像素点坐标计算该像素点在多段线上投影距离多段线起点的距离
#         :param pixel_pts:[[px,py],..,[]]
#         :param driving_line: [[x1,y1],[x2,y2],..,[xn,yn]]
#         :param base_point:[[x1,y1],global_dist]
#         :return:pixel_dist:[dist1,dist2,..,]
#
#         '''
#         pixel_dist_ls = []
#         line_strip = LineString(driving_line)
#         gap_dist = 0
#         if base_point:
#             global_dist = base_point[1]
#             base_pt = Point(base_point[0][0],base_point[0][1])
#             base_dist = line_strip.project(base_pt) #
#             gap_dist = global_dist - base_dist
#         for pixel_pt in pixel_pts:
#             pt = Point(pixel_pt[0],pixel_pt[1])
#             dist = line_strip.project(pt)  # 距离多段线起点的距离
#             dist = dist + gap_dist
#             pixel_dist_ls.append(dist)
#         return pixel_dist_ls
#
#     @staticmethod
#     def get_connect_point_position(file_list):
#         from utils import RoadConfig
#         # 读取所有road config文件
#         # 计算第一个road config文件中的base点和连接点
#         # 根据上一个连接点，计算第二个road config文件中的连接点，以此类推
#         drivingline_dict = {}
#         drivingline_base_position = {}
#         for file_name in file_list:
#             road_config = RoadConfig.fromfile(file_name)
#             drivingline = road_config['drivingline']
#             length_per_pixel = road_config['length_per_pixel']
#
#             for driveingline_key,driveingline_value in drivingline.items():
#                 if driveingline_key.endswith('line'):
#                     drivingline_line = driveingline_key.split('_')
#                     line_name = drivingline_line[1]
#                     position_id = drivingline_line[2]
#                     if line_name in drivingline_dict.keys():
#                         if position_id in drivingline_dict[line_name].keys():
#                             drivingline_dict[line_name][position_id]['line'] = driveingline_value
#                         else:
#                             drivingline_dict[line_name][position_id] = {'line':driveingline_value}
#                     else:
#                         drivingline_dict[line_name] = {position_id:{'line':driveingline_value}}
#                     drivingline_dict[line_name][position_id]['length_per_pixel'] = length_per_pixel
#                 if 'point' in driveingline_key:
#                     drivingline_point = driveingline_key.split('_')
#                     line_name = drivingline_point[1]
#                     position_id = drivingline_point[2]
#                     position = float(drivingline_point[4])
#                     assert not line_name in drivingline_base_position.keys(),'double base point'
#                     drivingline_base_position[line_name] = position_id # 记录base point所在位置
#                     if line_name in drivingline_dict.keys():
#                         if position_id in drivingline_dict[line_name].keys():
#                             drivingline_dict[line_name][position_id]['base'] = (driveingline_value[0],position)
#                         else:
#                             drivingline_dict[line_name][position_id] = {'base':(driveingline_value[0],position)}
#                     else:
#                         drivingline_dict[line_name] = {position_id:{'base':(driveingline_value[0],position)}}
#                 if 'connect' in driveingline_key:
#                     drivingline_connect = driveingline_key.split('_')
#                     line_name = drivingline_connect[1]
#                     position_id = drivingline_connect[2]
#                     position_id_from = drivingline_connect[4]
#                     position_id_to = drivingline_connect[5]
#
#                     if line_name in drivingline_dict.keys():
#                         if position_id in drivingline_dict[line_name].keys():
#                             if 'connect' in drivingline_dict[line_name][position_id].keys():
#                                 drivingline_dict[line_name][position_id]['connect'].append((position_id_from,position_id_to,driveingline_value[0]))
#                             else:
#                                 drivingline_dict[line_name][position_id]['connect'] = [(position_id_from,position_id_to,driveingline_value[0])]
#                         else:
#                             drivingline_dict[line_name][position_id] = {'connect':[(position_id_from,position_id_to,driveingline_value[0])]}
#                     else:
#                         drivingline_dict[line_name] = {position_id:{'connect':[(position_id_from,position_id_to,driveingline_value[0])]}}
#         return drivingline_dict,drivingline_base_position


class DrivingLine:

    def __init__(self,line_name,line_dict,base_position):
        self.drivingline_name = line_name
        base_fp = line_dict[base_position]


        self.line_strip_dict = {base_position:LineString(base_fp['line'])}
        self.lengthpixel_dict = {base_position:base_fp['length_per_pixel']}
        self.base_pixel_dict = {base_position:base_fp['base'][0]}
        self.base_dist_dict = {base_position:base_fp['base'][1]}
        self.region_dict = {}
        if 'region' in base_fp.keys():
            self.region_dict[base_position] = base_fp['region']

        self._get_all_fp_connect_point_dist(base_fp,line_dict)




    def get_global_drivingline_position(self,pixel_position_ls,position_id,output_vertical_line=False):
        '''
        输入像素坐标列表和当前拍摄位置id，输出该像素坐标列表对应的驾驶线距离,
        如果存在region则会先判断是否在region内,不在region内的像素坐标点对应距离为None
        :param pixel_position_ls:像素坐标点列表[[x1,y1],[x2,y2],...,[x3,y3]]
        :param position_id:当前拍摄位置id
        :return: pixel_dist_ls:像素坐标列表对应的驾驶线距离
        '''
        new_pixel_position_ls = []
        if position_id in self.region_dict.keys():
            # 选取区域内的点计算距离
            region  = self.region_dict[position_id]
            for pts in pixel_position_ls:
                if isPointinPolygon(pts,region):
                    new_pixel_position_ls.append(pts)
                else:
                    new_pixel_position_ls.append(None)
        else:
            new_pixel_position_ls = pixel_position_ls
        if position_id is None and len(list(self.line_strip_dict.keys()))==1:
            position_id = list(self.line_strip_dict.keys())[0]
        line_strip = self.line_strip_dict[position_id]
        length_per_pixel = self.lengthpixel_dict[position_id]
        base_pixel = self.base_pixel_dict[position_id]
        base_dist = self.base_dist_dict[position_id]
        pixel_dist_ls = self._pixelPt2drivingDist(new_pixel_position_ls, line_strip, base_pixel,base_dist,length_per_pixel,output_vertical_line)
        return pixel_dist_ls

    def _get_all_fp_connect_point_dist(self,base_fp,line_dict):
        base_pixel_point, base_dist = base_fp['base']
        base_line = LineString(base_fp['line'])
        length_per_pixel = base_fp['length_per_pixel']
        if not 'connect' in base_fp.keys():
            return
        assert len(base_fp['connect'])==1,'too many connnect in the first image'
        position_id_from, position_id_to, connect_pixel_point = base_fp['connect'][0]
        connect_point_dist = self._pixelPt2drivingDist([connect_pixel_point], base_line, base_pixel_point, base_dist,
                                                      length_per_pixel)[0]
        while position_id_to in line_dict.keys():
            current_fp_name = position_id_to
            current_fp = line_dict[current_fp_name]
            current_dist = connect_point_dist
            c_length_per_pixel = current_fp['length_per_pixel']
            c_line_strip = LineString(current_fp['line'])
            if 'region' in current_fp.keys():
                self.region_dict[current_fp_name] = current_fp['region']
            if len(current_fp['connect']) == 2:
                for connect in current_fp['connect']:
                    c_id_from, c_id_to, c_pixel_point = connect
                    if c_id_to == current_fp_name:
                        c_base_pixel_point = c_pixel_point
                for connect in current_fp['connect']:
                    c_id_from, c_id_to, c_pixel_point = connect
                    if c_id_to == current_fp_name:
                        self.line_strip_dict[current_fp_name] = c_line_strip
                        self.lengthpixel_dict[current_fp_name] = c_length_per_pixel
                        self.base_pixel_dict[current_fp_name] = c_pixel_point
                        self.base_dist_dict[current_fp_name] = current_dist

                    else:
                        connect_point_dist = self._pixelPt2drivingDist([c_pixel_point], c_line_strip, c_base_pixel_point, current_dist,
                                                 c_length_per_pixel)[0]
                        position_id_to = c_id_to
            elif len(current_fp['connect']) == 1:
                c_id_from, c_id_to, c_pixel_point = current_fp['connect'][0]
                assert c_id_to == current_fp_name,'the last fp error'
                self.line_strip_dict[current_fp_name] = c_line_strip
                self.lengthpixel_dict[current_fp_name] = c_length_per_pixel
                self.base_pixel_dict[current_fp_name] = c_pixel_point
                self.base_dist_dict[current_fp_name] = current_dist
                position_id_to = None

    def _pixelPt2drivingDist(self,pixel_position_ls, line_strip, base_pixel,base_dist,length_per_pixel,output_vertical_line=False):
        '''
        图片中车辆的像素坐标转换到全局的行车线距离
        :param pixel_position_ls:
        :param line_strip:
        :param base_pixel:
        :param base_dist:
        :param length_per_pixel:
        :return:
        '''
        pixel_dist_ls = []
        base_pt = Point(base_pixel[0],base_pixel[1])
        base_pixel_dist = line_strip.project(base_pt) #

        for pixel_pt in pixel_position_ls:
            if pixel_pt:
                pt = Point(pixel_pt[0],pixel_pt[1])
                pixel_dist = line_strip.project(pt)  # 距离多段线起点的距离
                dist = (pixel_dist - base_pixel_dist)*length_per_pixel + base_dist
                if output_vertical_line:
                    vertical_line_length = pt.distance(line_strip)*length_per_pixel # 点到多段线距离
                    pixel_dist_ls.append([dist,vertical_line_length])
                else:
                    pixel_dist_ls.append(dist)
            else:
                pixel_dist_ls.append(None)
        return pixel_dist_ls



class DrivingLineList:

    def __init__(self,file_list):
        self.drivingline_name_list = []
        self.drivingline = {}
        drivingline_dict, drivingline_base_position = self._get_connect_point_position(file_list)
        for line_name, line_dict in drivingline_dict.items():
            base_fp = drivingline_base_position[line_name]
            dl = DrivingLine(line_name, line_dict, base_fp)
            self.drivingline_name_list.append(line_name)
            self.drivingline[line_name] = dl

    def __str__(self):
        return ','.join(self.drivingline_name_list)

    def get_global_distance(self,pixel_position_ls, position_id,drivingline_name,output_vertical_line):
        '''

        :param pixel_position_ls:像素坐标
        :param position_id: 位置id
        :param drivingline_name: 行车线名称
        :param output_vertical_line:输出距行车线的距离
        :return:
        '''
        dl = self.drivingline[drivingline_name]
        return dl.get_global_drivingline_position(pixel_position_ls, position_id, output_vertical_line=output_vertical_line)

    def _get_connect_point_position(self,file_list):
        from utils import RoadConfig
        # 读取所有road config文件
        # 计算第一个road config文件中的base点和连接点
        # 根据上一个连接点，计算第二个road config文件中的连接点，以此类推
        drivingline_dict = {}
        drivingline_base_position = {}
        for file_name in file_list:
            road_config = RoadConfig.fromfile(file_name)
            drivingline = road_config['drivingline']
            length_per_pixel = road_config['length_per_pixel']

            for driveingline_key,driveingline_value in drivingline.items():
                if driveingline_key.endswith('region'):
                    drivingline_line = driveingline_key.split('_')
                    line_name = drivingline_line[1]
                    position_id = drivingline_line[2]
                    if line_name in drivingline_dict.keys():
                        if position_id in drivingline_dict[line_name].keys():
                            drivingline_dict[line_name][position_id]['region'] = driveingline_value
                        else:
                            drivingline_dict[line_name][position_id] = {'region':driveingline_value}

                if driveingline_key.endswith('line'):
                    drivingline_line = driveingline_key.split('_')
                    line_name = drivingline_line[1]
                    position_id = drivingline_line[2]
                    if line_name in drivingline_dict.keys():
                        if position_id in drivingline_dict[line_name].keys():
                            drivingline_dict[line_name][position_id]['line'] = driveingline_value
                        else:
                            drivingline_dict[line_name][position_id] = {'line':driveingline_value}
                    else:
                        drivingline_dict[line_name] = {position_id:{'line':driveingline_value}}
                    drivingline_dict[line_name][position_id]['length_per_pixel'] = length_per_pixel
                if 'point' in driveingline_key:
                    drivingline_point = driveingline_key.split('_')
                    line_name = drivingline_point[1]
                    position_id = drivingline_point[2]
                    position = float(drivingline_point[4])
                    assert not line_name in drivingline_base_position.keys(),'double base point'
                    drivingline_base_position[line_name] = position_id # 记录base point所在位置
                    if line_name in drivingline_dict.keys():
                        if position_id in drivingline_dict[line_name].keys():
                            drivingline_dict[line_name][position_id]['base'] = (driveingline_value[0],position)
                        else:
                            drivingline_dict[line_name][position_id] = {'base':(driveingline_value[0],position)}
                    else:
                        drivingline_dict[line_name] = {position_id:{'base':(driveingline_value[0],position)}}
                if 'connect' in driveingline_key:
                    drivingline_connect = driveingline_key.split('_')
                    line_name = drivingline_connect[1]
                    position_id = drivingline_connect[2]
                    position_id_from = drivingline_connect[4]
                    position_id_to = drivingline_connect[5]

                    if line_name in drivingline_dict.keys():
                        if position_id in drivingline_dict[line_name].keys():
                            if 'connect' in drivingline_dict[line_name][position_id].keys():
                                drivingline_dict[line_name][position_id]['connect'].append((position_id_from,position_id_to,driveingline_value[0]))
                            else:
                                drivingline_dict[line_name][position_id]['connect'] = [(position_id_from,position_id_to,driveingline_value[0])]
                        else:
                            drivingline_dict[line_name][position_id] = {'connect':[(position_id_from,position_id_to,driveingline_value[0])]}
                    else:
                        drivingline_dict[line_name] = {position_id:{'connect':[(position_id_from,position_id_to,driveingline_value[0])]}}
        return drivingline_dict,drivingline_base_position




if __name__ == '__main__':
    file_folder = r'/data2/xinkaiji/mixedroad_process/20220616/merge_config/merge_F1'
    file_list = os.listdir(file_folder)
    file_ls = []
    for file in file_list:
        file_name = os.path.join(file_folder,file)
        file_ls.append(file_name)
    # file_ls = ['../../config/mixed_roads_0616/road_config/B/20220616_0700_B1_F1_373_1.json']
    dl_ls = DrivingLineList(file_ls)
    print(dl_ls.drivingline['mainroad'].base_dist_dict)
    print(dl_ls)
    print(dl_ls.get_global_distance([[5471,1718]],'A1','mainroad',True))
    # coor = CoordinateSystem()
    # drivingline_dict,drivingline_base_position = coor.get_connect_point_position(file_ls)
    # for line_name,line_dict in drivingline_dict.items():
    #     base_fp = drivingline_base_position[line_name]
    #     dl = DrivingLine(line_name,line_dict,base_fp)
    #     print(dl.get_global_drivingline_position([[200,40]],'B1'))
    #     print('success')



    # from shapely.geometry import Point
    # from shapely.geometry import LineString
    # lines = [
    #     [
    #       10,
    #       1312.820512820513
    #     ],
    #     [
    #       400,
    #       1312.820512820513
    #     ],
    #     [
    #       745.2991452991454,
    #       1307.6923076923078
    #     ],
    #     [
    #       1074.3589743589744,
    #       1305.1282051282053
    #     ],
    #     [
    #       1413.6752136752139,
    #       1305.1282051282053
    #     ],
    #     [
    #       1752.9914529914531,
    #       1300.854700854701
    #     ],
    #     [
    #       2091.4529914529917,
    #       1299.1452991452993
    #     ],
    #     [
    #       2430.769230769231,
    #       1300.0
    #     ],
    #     [
    #       2773.504273504274,
    #       1299.1452991452993
    #     ],
    #     [
    #       3112.820512820513,
    #       1297.4358974358975
    #     ],
    #     [
    #       3794.871794871795,
    #       1295.7264957264958
    #     ],
    #     [
    #       4123.076923076924,
    #       1296.5811965811968
    #     ],
    #     [
    #       4452.136752136752,
    #       1296.5811965811968
    #     ],
    #     [
    #       4779.48717948718,
    #       1297.4358974358975
    #     ],
    #     [
    #       5103.418803418804,
    #       1297.4358974358975
    #     ]
    #   ]
    # point = Point(100, 1200)
    # line_strip = LineString(lines)
    # print('line_strip.length',line_strip.length)
    # dist = line_strip.project(point,True)
    # print(dist)
    # print(dist*line_strip.length)
    # point = line_strip.interpolate(dist,normalized=True)
    # point2 = line_strip.interpolate(dist)
    # l = list(point.coords)
    # print(l)
    # print(list(point2.coords))



