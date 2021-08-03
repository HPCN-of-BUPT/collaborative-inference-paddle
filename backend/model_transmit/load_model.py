import paddle
import glob
import numpy as np
import time
import cv2
import paddle.vision.transforms as T
# from paddlelite.lite import *

def image_size(img, height, width):
    image = cv2.imread(img)
    # pre_height, pre_width = 
    # height_ratio, width_ratio = 

def image_preprocess(img):
    # print(img)
    image = cv2.imread(img)
    image = cv2.resize(image, dsize=(32,32), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    trans = T.Compose([T.Transpose(), T.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))])
    image = trans(image)
    image = np.expand_dims(image,0)
    image = np.array(image, dtype=np.float32)
    return image

def edge_load_model(path_prefix,img):
    paddle.enable_static()
    startup_prog = paddle.static.default_startup_program()
    start_time = time.time()

    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(startup_prog)

    # 保存预测模型

    [inference_program, feed_target_names, fetch_targets] = (
        paddle.static.load_inference_model(path_prefix, exe))
    # tensor_img = np.array(np.random.random((1, 3, 32, 32)), dtype=np.float32)
    # print(tensor_img)
    results = exe.run(inference_program,
              feed={feed_target_names[0]: image_preprocess(img)},
              fetch_list=fetch_targets)
    end_time = time.time()
    return np.array(results[0]), round(end_time - start_time, 3)

def cloud_load_tensor(path_prefix, tensor):
    paddle.enable_static()
    startup_prog = paddle.static.default_startup_program()
    start_time = time.time()

    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(startup_prog)

    # 保存预测模型

    [inference_program, feed_target_names, fetch_targets] = (
        paddle.static.load_inference_model(path_prefix, exe))

    results = exe.run(inference_program,
              feed={feed_target_names[0]: tensor},
              fetch_list=fetch_targets)
    result = results[0].tolist()
    end_time = time.time()
    return [result[i].index(max(result[i])) for i in range(len(result))], round(end_time - start_time, 3)

# def edge_load_model_by_lite(path_prefix):
#     config = MobileConfig()

if __name__ == "__main__":
    tensor,edge_infer_time = edge_load_model(
        path_prefix="../data/send/model/client_infer_resnet18_cifar10",
        img="../data/test/air.jpeg")

    result,cloud_infer_time = cloud_load_tensor(
        path_prefix="../data/send/model/server_infer_resnet18_cifar10",tensor=tensor)
    # print(result)






