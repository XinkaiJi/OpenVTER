import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont


from .utils.utils import cvtColor, get_classes, preprocess_input, resize_image,preprocess_input_tensor
from .utils.utils_bbox3 import decode_outputs, non_max_suppression
import cv2
'''
训练自己的数据集必看注释！
'''
def RHIoU(points):
    def compute_polygon_area(points):
        point_num = len(points)
        if (point_num < 3):
            return 0.0
        s = points[0][1] * (points[point_num - 1][0] - points[1][0])
        # for i in range(point_num): # (int i = 1 i < point_num ++i):
        for i in range(1, point_num):  # 有小伙伴发现一个bug，这里做了修改，但是没有测试，需要使用的亲请测试下，以免结果不正确。
            s += points[i][1] * (points[i - 1][0] - points[(i + 1) % point_num][0])
        return abs(s / 2.0)
    pts_4 = points.reshape(4,2)
    x1 = np.min(pts_4[:, 0])
    x2 = np.max(pts_4[:, 0])
    y1 = np.min(pts_4[:, 1])
    y2 = np.max(pts_4[:, 1])
    w_hbbox, h_hbbox = x2 - x1, y2 - y1
    r_area = compute_polygon_area(pts_4)

    # vector1 = pts_4[1, :] - pts_4[0, :]
    # vector2 = pts_4[2, :] - pts_4[1, :]
    # length_vector1 = np.linalg.norm(vector1)
    # legnth_vector2 = np.linalg.norm(vector2)
    # if length_vector1 > legnth_vector2:
    #     np.arcsin()
    if r_area/(w_hbbox*h_hbbox)>0.9:
        return np.array([x1,y1,x2,y1,x2,y2,x1,y2])
    else:
        return points
def toRect(points):
    pts_4 = points.reshape(4, 2)
    rect = cv2.minAreaRect(pts_4)
    box = cv2.boxPoints(rect)
    return box.flatten()

class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "checkpoint_type":"pth",
        "checkpoint"        : 'model_data/yolox_s.pth',
        "classes_path"      : 'model_data/coco_classes.txt',
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        # "input_shape"       : [640, 640],
        "img_width": 640,
        "img_height": 640,
        #---------------------------------------------------------------------#
        #   所使用的YoloX的版本。nano、tiny、s、m、l、x
        #---------------------------------------------------------------------#
        "phi"               : 's',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
        "device_name": "cuda:0"
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #------------------------------------------------ ---#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        # self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.num_classes = len(self.class_names)
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        if self.device_name == 'cpu':
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device(self.device_name)
        else:
            print('no cuda!')
        self.preprocess_image_rgb = [torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device),
                                     torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)]

        if self.checkpoint_type == "jit":
            self.load_model_jit()
        elif self.checkpoint_type == "pth":
            print('Do not support pth file!')

    def load_model_jit(self):
        self.net = torch.jit.load(self.checkpoint)
        self.net = self.net.to(self.device)
        self.net.eval()
        print('{} model'.format(self.checkpoint))




    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image,result_type='data',get_img=False):
        #---------------------------------------------------#
        #   获得输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.img_height,self.img_width), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs,  (self.img_width, self.img_height))
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, (self.img_width, self.img_height),
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return image


            top_label   = np.array(results[0][:, 15], dtype = 'int32')
            top_conf    = results[0][:, 13] * results[0][:, 14]
            top_boxes   = results[0][:, :4]
        new_prediction = self.convert2points(results[0])
        for object_index in range(new_prediction.shape[0]):
            pred = new_prediction[object_index]
            pred[:8] = toRect(pred[:8])
        if result_type == 'data':
            if get_img:
                image_cv = self.draw_nms(np.array(image, dtype='uint8'), new_prediction,
                                         ['car', 'truck', 'bus', 'freight_car', 'van'])

                return new_prediction,image_cv
            else:
                return new_prediction
        for object_index in range(new_prediction.shape[0]):
            pred = new_prediction[object_index]
            cat = self.class_names[int(pred[-1])]
            score = pred[-2]
            res = [cat, score, pred[0], pred[1], pred[2], pred[3], pred[4], pred[5], pred[6], pred[7]]
            print('{} {:.12f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}'.format(
                res[0], res[1], res[2] , res[3], res[4], res[5],
                                res[6] , res[7] , res[8], res[9] ))
        image_cv = self.draw_nms(np.array(image, dtype='uint8'),new_prediction, ['car','truck','bus','freight_car','van'])


        import matplotlib.pyplot as plt  # plt 用于显示图片

        plt.imshow(image_cv)  # 显示图片
        plt.show()
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean( (self.img_width, self.img_height)), 1))
        
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            left,top,right,bottom = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            draw.rectangle([left, top, right , bottom], outline=self.colors[c])
            # for i in range(thickness):
            #     draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def det_images_batch(self,images_ls):
        # 输入图片和模型输入尺寸保持一致
        new_images = []
        if len(images_ls) ==0:
            return None
        image_shape = images_ls[0].shape[:2]
        for image_data in images_ls:
            h, w, c = image_data.shape
            if self.img_height !=h or self.img_width != w:
                image_data = cv2.resize(image_data, (self.img_width, self.img_height))
            # image_data = preprocess_input(np.array(image_data, dtype='float32'))
            new_images.append(image_data)
        image_batch = torch.from_numpy(np.transpose(np.array(new_images, dtype=np.float32) , (0, 3, 1, 2)))

        with torch.no_grad():

            images = image_batch.to(self.device)
            images = preprocess_input_tensor(images,self.preprocess_image_rgb)
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs,  (self.img_width, self.img_height))
            results = non_max_suppression(outputs, self.num_classes,  (self.img_width, self.img_height),
                                          image_shape, False, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)
        new_prediction_ls = []
        for index,det in enumerate(results):
            if det is None:
                new_prediction_ls.append(None)
            else:
                new_prediction_ls.append(self.convert2points(det))
        return new_prediction_ls

    def convert2points(self,bbox_result):
        '''
        输出：x1,y1,x2,y2,x3,y3,x4,y4,score, clse
        '''
        # x1, y1, x2, y2, obj_conf, class_conf, class_pred

        alpha = bbox_result[:,12]
        alpha_index = (alpha<0.8).squeeze()
        xy = bbox_result[:,:4].reshape(-1,2,2)
        cen_pt = np.mean(xy,axis=1)
        ct = bbox_result[:, 4:6]
        cr = bbox_result[:, 6:8]
        cb = bbox_result[:, 8:10]
        cl = bbox_result[:, 10:12]
        tl = ct + cl + cen_pt
        bl = cb + cl + cen_pt
        tr = ct + cr + cen_pt
        br = cb + cr + cen_pt
        score = bbox_result[:, 13:14]*bbox_result[:, 14:15]
        clse = bbox_result[:, 15:16]
        new_prediction = np.concatenate([tl, tr, br, bl, score, clse], 1)
        bbox_pre = np.concatenate([bbox_result[:,[0,1]], bbox_result[:,[2,1]], bbox_result[:,[2,3]], bbox_result[:,[0,3]], score, clse], 1)
        new_prediction[alpha_index] = bbox_pre[alpha_index]
        return new_prediction


    def draw_nms(self,ori_image, nms_results, category):
        for object_index in range(nms_results.shape[0]):
            pred = nms_results[object_index]
            cat = category[int(pred[-1])]
            cat_color = self.colors[int(pred[-1])]
            score = pred[-2]
            tl = np.asarray([pred[0], pred[1]], np.float32)
            tr = np.asarray([pred[2], pred[3]], np.float32)
            br = np.asarray([pred[4], pred[5]], np.float32)
            bl = np.asarray([pred[6], pred[7]], np.float32)

            tt = (np.asarray(tl, np.float32) + np.asarray(tr, np.float32)) / 2
            rr = (np.asarray(tr, np.float32) + np.asarray(br, np.float32)) / 2
            bb = (np.asarray(bl, np.float32) + np.asarray(br, np.float32)) / 2
            ll = (np.asarray(tl, np.float32) + np.asarray(bl, np.float32)) / 2

            box = np.asarray([tl, tr, br, bl], np.float32)
            # cen_pts = np.mean(box, axis=0)

            # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(tt[0]), int(tt[1])), (0, 0, 255), 1, 1)
            # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(rr[0]), int(rr[1])), (255, 0, 255), 1, 1)
            # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(bb[0]), int(bb[1])), (0, 255, 0), 1, 1)
            # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(ll[0]), int(ll[1])), (255, 0, 0), 1, 1)

            # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(tl[0]), int(tl[1])), (0,0,255),1,1)
            # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(tr[0]), int(tr[1])), (255,0,255),1,1)
            # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(br[0]), int(br[1])), (0,255,0),1,1)
            # cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(bl[0]), int(bl[1])), (255,0,0),1,1)
            ori_image = cv2.drawContours(ori_image, [np.int0(box)], -1, cat_color, 1, 1)
            # box = cv2.boxPoints(cv2.minAreaRect(box))
            # ori_image = cv2.drawContours(ori_image, [np.int0(box)], -1, (0,255,0),1,1)
            cv2.putText(ori_image, '{:.2f} {}'.format(score, cat), (int(box[1][0]), int(box[1][1])),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1, 1)
        return ori_image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.img_height,self.img_width), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs,  (self.img_width, self.img_height))
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes,  (self.img_width, self.img_height),
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                  
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = decode_outputs(outputs,  (self.img_width, self.img_height))
                #---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                #---------------------------------------------------------#
                results = non_max_suppression(outputs, self.num_classes,  (self.img_width, self.img_height),
                            image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.img_height,self.img_width), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs,  (self.img_width, self.img_height))
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes,  (self.img_width, self.img_height),
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
