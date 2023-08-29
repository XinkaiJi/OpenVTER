#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2022-01-05 14:09
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : PolygonTool.py 
@Software: PyCharm
@desc: 
'''
from shapely.geometry import Point
from shapely.geometry import LineString

def isPointinPolygon(point, rangelist):  #
    '''
    判断是否在外包矩形内，如果不在，直接返回false
    :param point: [x,y]
    :param rangelist: [[0,0],[1,1],[0,1],[0,0]] [1,0.8]]
    :return:
    '''

    lnglist = []
    latlist = []
    for i in range(len(rangelist)):
        lnglist.append(rangelist[i][0])
        latlist.append(rangelist[i][1])
    maxlng = max(lnglist)
    minlng = min(lnglist)
    maxlat = max(latlist)
    minlat = min(latlist)
    if (point[0] > maxlng or point[0] < minlng or
            point[1] > maxlat or point[1] < minlat):
        return False
    count = 0
    point1 = rangelist[0]
    for i in range(1, len(rangelist)+1):
        if i == len(rangelist):
            i = 0
        point2 = rangelist[i]
        # 点与多边形顶点重合
        if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
            return False
        # 判断线段两端点是否在射线两侧 不在肯定不相交 射线（-∞，lat）（lng,lat）
        if (point1[1] < point[1] and point2[1] >= point[1]) or (point1[1] >= point[1] and point2[1] < point[1]):
            # 求线段与射线交点 再和lat比较
            point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0]) / (point2[1] - point1[1])
            # print(point12lng)
            # 点在多边形边上
            if (point12lng == point[0]):
                return False
            if (point12lng < point[0]):
                count += 1
        point1 = point2
    # print(count)
    if count % 2 == 0:
        return False
    else:
        return True


def get_lane_id_from_linestring(point, line_string_dict):
    """

    point表示车辆中心点坐标，元组(x,y)
    line_string_dict表示车道线，字典{'label':[(x1,y1),(x2,y2),(x3,y3)...(xn,yn)]} 'label'='laneline_x'
    return lane_id int
    """
    vehicle_point = Point(point)
    distance_dict = {}
    for key, value in line_string_dict.items():
        lane_line = LineString(value)
        distance = vehicle_point.distance(lane_line)
        distance_dict[key] = distance
    lane_id = -1
    if len(distance_dict) > 0:
        sorted_distance_dict = sorted(distance_dict.items(), key=lambda x: x[1], reverse=False)
        first_lane_line_label = sorted_distance_dict[0][0].split('_')[-1]
        second_lane_line_label = sorted_distance_dict[1][0].split('_')[-1]
        if abs(int(first_lane_line_label) - int(second_lane_line_label)) == 1:
            lane_id = min(int(first_lane_line_label), int(second_lane_line_label))
    return lane_id

def get_lane_id_from_polygon(point, lane_polygon_dict):
    pass

if __name__ == '__main__':
    polyhon = [
        [
            5.813953488372093,
            953.1007751937984
        ],
        [
            3397.841726618705,
            941.7266187050359
        ],
        [
            3400.975609756098,
            979.0243902439025
        ],
        [
            2700.487804878049,
            980.0000000000001
        ],
        [
            8.270676691729323,
            987.2180451127819
        ]
      ]
    point = [500,960]
    print(isPointinPolygon(point,polyhon))
