import numpy as np
# from pathlib import Path

CLOUD_HOST = "127.0.0.1" # 云ip
EDGE_HOST = "127.0.0.1" # 端ip
BACKEND_HOST = "127.0.0.1" # 后台服务ip

CLOUD_MODEL_PORT = 8083 # 后台发送云端模型端口
CLOUD_TENSOR_PORT = 8081 # 边端发送特征端口
EDGE_MODEL_PORT = 8082 # 后台发送边端模型端口

BUFFER_SIZE = 1024 # 传输大文件时单次接收buffer大小
NUMPY_TYPE = np.float32 # tensor数据类型
ERROR_RATE = 0 # 信道误码率

SAVE_DIR = "./data/output" # 检测结果保存路径
LOAD_DIR = "./data/input" # 待检测文件保存路径
CLOUD_MODEL_DIR = "./data/cloud/server_infer_pruned" # 云端加载模型路径
EDGE_MODEL_DIR = "./data/edge/client_infer_pruned" # 端加载模型路径

PIXEL_MEANS = [0.485, 0.456, 0.406] # 归一化均值
PIXEL_STDS = [0.229, 0.224, 0.225] # 归一化方差
DRAW_THRESHOLD = 0.5 # IOU阈值

LABELS = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 
                'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
