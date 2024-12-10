# Prerequisites
In this section we demonstrate how to prepare an environment with PyTorch.

OpenVTER works on Linux and Windows. It requires Python 3.7+, CUDA 9.2+, PyTorch 1.6+ and mmcv.

We recommend using the conda virtual environment to install.
## Install
### Step 1: Create a conda virtual environment
```
conda create --name OpenVTER python=3.8
conda activate OpenVTER
```
### Step 2: Install pytorch and mmcv

Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/) 
Install mmcv following [official instructions](https://github.com/open-mmlab/mmcv)
Some example installation code.

For CUDA 12.4
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -U openmim
mim install mmcv-full
```

### Step 3: Install other packages
```
pip install srt
pip install filterpy
pip install shapely
pip install lap
pip install tqdm
```

### Step 4: Download OpenVTER
```
git clone https://github.com/XinkaiJi/OpenVTER.git
```

## Model checkpooints Preparation
Download model checkpoints [Link](https://1drv.ms/f/s!AgYoU6qx6kykhc1834iwdhxaRuP83A?e=5yHuei) [password:OpenVTER202410]

Put the "centernet_bbavectors" and "yolox-r" folders into the "checkpoints" folder.

The final file directory is as follows:

- **checkpoints**
  - **centernet_bbavectors**
    - centernet_bbavectors_oneclass_L_202207.jit
    - centernet_bbavectors_oneclass_M_202206.jit
  - **yolox-r**
    - yoloxr_X_202206.jit


## UAV Demo Video Preparation
Dowload UAV demo video 

| Video Name        |                                         Download                                         | Remark |
|:------------------|:----------------------------------------------------------------------------------------:|-------:|
| 20220303_5_E_300  | [Onedrive](https://1drv.ms/f/s!AgYoU6qx6kykhc4D_3dC7ddo0nFR2Q) [password:OpenVTER202410] | Straight Road |

Put the vides in your data folder, such as "/data1/UAV_Videos/20220303_5_E_300"

# Get started
## Preparing configuration files




## Video stabilization
```
python video_inference_main.py -s 1 -e 1 -c config/demo_config/video_config/20220303_5_E_300.json
```
## Vehicle detection and tracking
```
python video_inference_main.py -s 3 -c config/demo_config/video_config/20220303_5_E_300.json
```
We will get the detection and tracking result in the "/data1/UAV_Videos/20220303_5_E_300/output/20220303_5_E_300_1_Num_3" folder.

"tracking_output_stab_det_20220303_5_E_300_1.mp4": Video containing results of vehicle detection and tracking
"det_bbox_result_20220303_5_E_300_1.pkl": Structured detection and tracking results