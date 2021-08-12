import numpy as np

CLOUD_HOST = "127.0.0.1"
EDGE_HOST = "127.0.0.1"
CLOUD_SENTTO_EDGE = 8080
EDGE_SENDTO_CLOUD = 8081

BUFFER_SIZE = 1024
NUMPY_TYPE = np.float32
ERROR_RATE = 0

SAVE_DIR = "../data/output"
LOAD_DIR = "../data/test"
CLOUD_MODEL_DIR = "./data/send/server_infer_yolov3"
EDGE_MODEL_DIR = "./data/receive/client_infer_yolov3"
# TOTAL = 0
# CORRECT = 0

PIXEL_MEANS = [0.485, 0.456, 0.406]
PIXEL_STDS = [0.229, 0.224, 0.225]
DRAW_THRESHOLD = 0.5
