#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2021-12-20 21:16
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : video_inference_main.py 
@Software: PyCharm
@desc: 
'''
from video_inference.video_process import DroneVideoProcess
from video_inference.video_stabilization import DroneVideoStab
from video_inference.video_process_multiprocessing import run
from video_inference.video_det_process_multiprocessing import run as run_det
import argparse
import logging  # 引入logging模块
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s-%(filename)s[%(lineno)d]-%(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置

def parse_args():
    parser = argparse.ArgumentParser(description='OpenVTER Implementation')
    parser.add_argument("-c",'--config_json', type=str, help='config')
    parser.add_argument("-s","--step",
                        type=int,
                        help="1:stabilize 2:detect video without stabilization 3: detect and tracking video")
    parser.add_argument("-e", "--config_parameter",
                        type=int,
                        help="1:output the stabilize pkl file 2:output stabilize video")
    parser.add_argument("-m", "--multiprocessing",
                        action="store_true", default=False,
                        help="multiprocessing trajectory extraction")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.step == 1:
        video_stab = DroneVideoStab(args.config_json)
        video_stab.process(step=args.config_parameter)
    elif args.step == 2:
        run_det(args.config_json)
    elif args.step == 3:
        if args.multiprocessing:
            run(args.config_json)
        else:
            v = DroneVideoProcess(args.config_json)
            # v.process_img()
            v.process_video()


