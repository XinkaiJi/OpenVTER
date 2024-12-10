# Video config

```
{
  "detection": "centernet_bbavectors.json",
  "tracking": "sort_r.json",
  "video_folder": "/data1/UAV_Videos/20220303_5_E_300",
  "first_video_name": "20220303_5_E_300_{0}.MP4",
  "video_num": 3,
  "road_config": "config/demo_config/road_config/20220303_5_E_300_1.json",
  "save_folder": "/data1/UAV_Videos/20220303_5_E_300/output",
  "out_fps": 25,
  "output_video": 1,
  "inference_batch_size": 10,
  "bbox_label": ["id","score","lane"],
  "pipeline": ["stab","det"],
  "stabilize_file": "affine_trans_matrix_20220303_5_E_300.pkl",
  "split_gap": 100,
  "subsize_height": 512,
  "stabilize_scale": 0,
  "video_start_frame": 0,
  "video_end_frame": 0
}
```