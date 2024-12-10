# 前提条件
本节中，我们将演示如何使用 PyTorch 准备环境。

OpenVTER 支持在 Linux 和 Windows 上运行。它需要 Python 3.7+、CUDA 9.2+、PyTorch 1.6+ 和 mmcv。

我们建议使用 conda 虚拟环境进行安装。
## 安装
### 步骤 1：创建 conda 虚拟环境
```
conda create --name OpenVTER python=3.8
conda activate OpenVTER
```
### 步骤 2：安装 PyTorch 和 mmcv

安装 PyTorch 可参考 [官方指南](https://pytorch.org/get-started/locally/) 

安装 mmcv 可参考 [官方指南](https://github.com/open-mmlab/mmcv)

以下是一些示例安装代码。

针对 CUDA 12.4
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -U openmim
mim install mmcv-full
```

### 步骤 3：安装其他包
```
pip install srt
pip install filterpy
pip install shapely
pip install lap
pip install tqdm
```

### 步骤 4：下载 OpenVTER
```
git clone https://github.com/XinkaiJi/OpenVTER.git
```

## 模型文件准备
下载模型文件 [Link](https://1drv.ms/f/s!AgYoU6qx6kykhc1834iwdhxaRuP83A?e=5yHuei) [password:OpenVTER202410]

将 “centernet_bbavectors” 和 “yolox-r” 文件夹放入 “checkpoints” 文件夹中。

最终的文件目录如下：

- **checkpoints**
  - **centernet_bbavectors**
    - centernet_bbavectors_oneclass_L_202207.jit
    - centernet_bbavectors_oneclass_M_202206.jit
  - **yolox-r**
    - yoloxr_X_202206.jit


## UAV Demo Video 准备
下载UAV demo video 

| Video Name        |                                         Download                                         | Remark |
|:------------------|:----------------------------------------------------------------------------------------:|-------:|
| 20220303_5_E_300  | [Onedrive](https://1drv.ms/f/s!AgYoU6qx6kykhc4D_3dC7ddo0nFR2Q) [password:OpenVTER202410] | Straight Road |

将视频放入您的数据文件夹，例如 "/data1/UAV_Videos/20220303_5_E_300"

所下载的视频中包含了MP4视频文件和srt字幕文件，字幕文件为视频每一帧提供时间戳。
# 开始
## 准备配置文件
OpenVTER需要两个配置文件，分别为road config和video config。

road config: 用于标注道路区域范围，比例尺，车道线等信息。具体详见 [road config](docs/zh_cn/road_config.md)

video config: 用于配置视频数据处理所需要的信息，比如视频文件路径，所采用的检测模型，跟踪模型等。具体详见 [video config](docs/zh_cn/video_config.md)



## 视频稳定
```
python video_inference_main.py -s 1 -e 1 -c config/demo_config/video_config/20220303_5_E_300.json
```
## 车辆检测与跟踪
```
python video_inference_main.py -s 3 -c config/demo_config/video_config/20220303_5_E_300.json
```
We will get the detection and tracking result in the "/data1/UAV_Videos/20220303_5_E_300/output/20220303_5_E_300_1_Num_3" folder.

"tracking_output_stab_det_20220303_5_E_300_1.mp4": Video containing results of vehicle detection and tracking
"det_bbox_result_20220303_5_E_300_1.pkl": Structured detection and tracking results