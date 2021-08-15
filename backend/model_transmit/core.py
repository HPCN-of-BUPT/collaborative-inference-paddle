import numpy as np
# from pathlib import Path

CLOUD_HOST = "127.0.0.1" # 云ip
EDGE_HOST = "127.0.0.1" # 端ip
CLOUD_SENTTO_EDGE = 8080 # 云接收端口
EDGE_SENDTO_CLOUD = 8081 # 端接收端口

BUFFER_SIZE = 1024 # 传输大文件时单次接收buffer大小
NUMPY_TYPE = np.float32 # tensor数据类型
ERROR_RATE = 0 # 信道误码率

SAVE_DIR = "./data/output" # 检测结果保存路径
LOAD_DIR = "./data/test" # 待检测文件保存路径
CLOUD_MODEL_DIR = "./data/send/server_infer_yolov3" # 云端加载模型路径
EDGE_MODEL_DIR = "./data/receive/client_infer_yolov3" # 端加载模型路径
# TOTAL = 0
# CORRECT = 0

PIXEL_MEANS = [0.485, 0.456, 0.406] # 归一化均值
PIXEL_STDS = [0.229, 0.224, 0.225] # 归一化方差
DRAW_THRESHOLD = 0.5 # IOU阈值
