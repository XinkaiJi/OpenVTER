#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2022-01-12 16:48
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : VideoTool.py 
@Software: PyCharm
@desc: 
'''
import cv2
import os
import time
from datetime import datetime

def get_all_video_info(video_file_ls):
    all_num_frame = 0
    num_frame_ls = []
    old_width = 0
    old_height = 0
    old_fps = 0
    for video_index, video_file in enumerate(video_file_ls):
        cap = cv2.VideoCapture(video_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if video_index > 0 and (old_width != width or old_height != height or old_fps != fps):
            print(
                '%s and %sthe videos is not same in width,height or fps' % (video_file_ls[video_index - 1], video_file))
        old_width = width
        old_height = height
        old_fps = fps
        all_num_frame += num_frames
        num_frame_ls.append(num_frames)
    return num_frame_ls, all_num_frame, old_width, old_height, old_fps

def get_srt(video_file_ls):
    import srt
    srt_info = []
    for video_index, video_file in enumerate(video_file_ls):
        filepath, tmpfilename = os.path.split(video_file)
        shotname, extension = os.path.splitext(tmpfilename)
        srt_file = os.path.join(filepath,shotname+'.SRT')
        if os.path.exists(srt_file):
            with open(srt_file,'r') as f:
                lines = f.read()
            subtitle_generator = srt.parse(lines)
            subtitles = list(subtitle_generator)
            for x in subtitles:
                content = x.content
                t_s = content.find("\n")
                t_e = content.find("\n", t_s + 1)
                t = content[t_s + 1:t_e]
                srt_info.append((t,content))
        else:
            print('srt file %s is not exist'%srt_file)
            return []
    return srt_info

def str2unixtime(timestr):
    '''
    字符串转为unixtime(毫秒)
    :param timestr:  '2022-06-17 06:59:46,837,622'
    :return:
    '''
    timestr = timestr[:-4]
    datetime_obj = datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S,%f")
    int_unix_time = int(time.mktime(datetime_obj.timetuple())*1e3+datetime_obj.microsecond/1e3)
    return int_unix_time

def get_transformer_matrix(src_points, dst_points,use_three=False):
    '''
    计算仿射变换矩阵
    :param src_points:
    :param dst_points:
    :param use_three:
    :return:
    '''
    assert len(src_points)>=3,'小于三个校准点'
    src_points = np.array(src_points).astype(np.float32)
    dst_points = np.array(dst_points).astype(np.float32)
    if len(src_points)>3 and use_three:
        src_points = src_points[:3]
        dst_points = dst_points[:3]
    if len(src_points)==3:
        affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    else:
        src_points = np.array(src_points).astype(np.float32)
        dst_points = np.array(dst_points).astype(np.float32)
        affine_matrix, inlier = cv2.estimateAffinePartial2D(src_points, dst_points)
    return affine_matrix


# a = get_srt(["/home/data2/maopeipei/20220108/20220108_a_A_400.6_A_1.MP4","/home/data2/maopeipei/20220108/20220108_a_A_400.6_A_1.MP4"])
# print(a)
import copy
import numpy as np
import json

class splitbase():
    '''
    分割图像
    '''
    def __init__(self,
                 srcpath,
                 dstpath,
                 gap=100,
                 subsize_height=512,
                 subsize_width=640,
                 ext='.jpg'):
        self.srcpath = srcpath
        self.outpath = dstpath
        self.gap = gap
        self.subsize_height = subsize_height
        self.subsize_width = subsize_width

        self.slide_height = self.subsize_height - self.gap
        self.slide_width = self.subsize_width - self.gap
        self.srcpath = srcpath
        self.dstpath = dstpath
        self.ext = ext
        if dstpath!='' and not os.path.exists(self.outpath):
            os.makedirs(self.outpath)

    def saveimagepatches(self, img, subimgname, left, up, ext='.png'):
        subimg = copy.deepcopy(img[up: (up + self.subsize_height), left: (left + self.subsize_width)])
        outdir = os.path.join(self.dstpath, subimgname + ext)
        cv2.imwrite(outdir, subimg)
        print('save:%s'%outdir)

    def SplitSingle(self, name, rate, extent):
        img = cv2.imread(os.path.join(self.srcpath, name + extent))
        assert np.shape(img) != ()

        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '_' + str(rate) + '_'

        # weight = np.shape(resizeimg)[1]
        # height = np.shape(resizeimg)[0]
        height,width,_ = resizeimg.shape
        left, up = 0, 0
        while (left < width):
            if (left + self.subsize_width >= width):
                left = max(width - self.subsize_width, 0)
            up = 0
            while (up < height):
                if (up + self.subsize_height >= height):
                    up = max(height - self.subsize_height, 0)
                subimgname = outbasename + str(left) + '_' + str(up)
                self.saveimagepatches(resizeimg, subimgname, left, up)
                if (up + self.subsize_height >= height):
                    break
                else:
                    up = up + self.slide_height
            if (left + self.subsize_width >= width):
                break
            else:
                left = left + self.slide_width

    def splitdata(self, rate,img_name=None):
        imagelist = os.listdir(self.srcpath)
        imagenames = [os.path.splitext(x)[0] for x in imagelist if os.path.splitext(x)[1] == self.ext]
        if img_name is None:
            for name in imagenames:
                self.SplitSingle(name, rate, self.ext)
        else:
            self.SplitSingle(os.path.splitext(img_name)[0], rate, self.ext)

    def split_image(self,img):
        subimg_list = []
        subimg_positions = []
        height,width,_ = img.shape
        left, up = 0, 0
        total = 0
        while (left < width):
            if (left + self.subsize_width >= width):
                left = max(width - self.subsize_width, 0)
            up = 0
            while (up < height):
                if (up + self.subsize_height >= height):
                    up = max(height - self.subsize_height, 0)

                position = (left,up)
                # subimg = copy.deepcopy(img[up: (up + self.subsize_height), left: (left + self.subsize_width)])
                subimg = img[up: (up + self.subsize_height), left: (left + self.subsize_width)]
                total += 1
                if self.is_useful_img(subimg):
                    subimg_list.append(subimg)
                    subimg_positions.append(position)
                if (up + self.subsize_height >= height):
                    break
                else:
                    up = up + self.slide_height
            if (left + self.subsize_width >= width):
                break
            else:
                left = left + self.slide_width
        # print('t:%d,after:%d'%(total,len(subimg_list)))
        return subimg_list,subimg_positions

    def split_image_with_position(self,img,sub_positions):
        subimg_list = []
        for position in sub_positions:
            left,up = position
            # subimg = copy.deepcopy(img[up: (up + self.subsize_height), left: (left + self.subsize_width)])
            subimg = img[up: (up + self.subsize_height), left: (left + self.subsize_width)]
            subimg_list.append(subimg)
        return subimg_list

    def is_useful_img(self,img,threshold=0.98):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.size
        no_black = cv2.countNonZero(gray)
        black_ratio = (size-no_black)/size
        if black_ratio > threshold:
            return False
        else:
            return True


    def add_mask(self,img,mask_json_file):
        '''
        labelme工具标记出道路区域
        :param img:
        :param mask_json_file:
        :return:
        '''
        with open(mask_json_file, "r") as f:
            tmp = f.read()
        tmp = json.loads(tmp)
        mask = np.zeros_like(img)
        for s in tmp["shapes"]:
            points = s["points"]
            points = np.array(points, np.int32)
            cv2.fillPoly(mask, [points], (1, 1, 1))
        img_add = mask * img
        return img_add

