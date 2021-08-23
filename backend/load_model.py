import os, json
import numpy as np
import time
import cv2
import random
import paddle
import paddle.vision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import core

def read_image(img):
    origin = img
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32')
    
    h, w, _ = img.shape
    im_scale_x = 608 / float(w)
    im_scale_y = 608 / float(h)
    img = cv2.resize(img, None, None, 
                                 fx=im_scale_x, fy=im_scale_y, 
                                 interpolation=cv2.INTER_CUBIC)
    mean = np.array(core.PIXEL_MEANS).reshape((1, 1, -1))
    std = np.array(core.PIXEL_STDS).reshape((1, 1, -1))
    resized_img = img.copy()
    img = (img / 255.0 - mean) / std
    img = np.array(img).astype('float32').transpose((2, 0, 1))
    img = img[np.newaxis, :]
    return origin, img, resized_img

def resize_img(img, target_size):
    img = img.resize(target_size[1:], Image.BILINEAR)

    return img

def image_preprocess(img):
    # print(img)
    image = cv2.imread(img)
    image = cv2.resize(image, dsize=(32,32), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    trans = T.Compose([T.Transpose(), T.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))])
    image = trans(image)
    image = np.expand_dims(image,0)
    image = np.array(image, dtype=np.float32)
    return image

def image_preprocess_yolo(img):
    image = cv2.imread(img)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    origin, tensor_img, _ = read_image(image)
    input_w, input_h = origin.size[0], origin.size[1]
    image_shape = np.array([input_h, input_w], dtype='int32')
    return image_shape, tensor_img

def draw_bbox_image(img, boxes, labels, scores,label_names,thre,gt=False):
    color = ['FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7']
    
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    line_thickness = max(int(min(img.size) / 200), 2)
    # win:arial.ttf
    font = ImageFont.truetype("Arial.ttf", size=max(round(max(img.size) / 40), 12))

    for box, label,score in zip(boxes, labels, scores):
        if score >= thre:
            c = random.randint(0,19)
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
            draw.rectangle((xmin, ymin, xmax, ymax), None, "#" + color[c], width=line_thickness)
            draw.text(( xmin + 5, ymin + 5), 
                        label_names[int(label)] + ' ' + str(round(score * 100, 2)) + "%", 
                        "#" + color[c], font=font)
    return img

def edge_load_model_yolo(model_path, img_dir, img_name):
    paddle.enable_static()
    start_time = time.time()

    exe = paddle.static.Executor(paddle.CPUPlace())
    [inference_program, feed_target_names, fetch_targets] = (
        paddle.static.load_inference_model(model_path, exe))
    image_shape, tensor_image = image_preprocess_yolo(os.path.join(img_dir, img_name))

    results = exe.run(inference_program,
              feed={feed_target_names[0]: tensor_image},
              fetch_list=fetch_targets)
    end_time = time.time()
    return image_shape, results, round(end_time - start_time, 3)

def cloud_load_tensor_yolo(image_shape, tensor, model_path, img_dir,img_name):
    paddle.enable_static()
    start_time = time.time()

    exe = paddle.static.Executor(paddle.CPUPlace())
    [inference_program, feed_target_names, fetch_targets] = (
        paddle.static.load_inference_model(model_path, exe))
    # 自适应输入tensor
    feed = {}
    for index, t in enumerate(tensor):
        feed[feed_target_names[index]] = t
    feed[feed_target_names[-1]] = image_shape[np.newaxis, :]
    
    outputs = exe.run(inference_program,
                      feed=feed,
              fetch_list=fetch_targets,
              return_numpy=False)
    bboxes = np.array(outputs[0])
    # 仅传输标注框
    result = []
    for index, box in enumerate(bboxes):
        if box[1].astype('float32') >= core.DRAW_THRESHOLD:
            result.append(box.tolist())
    result = str(result)
    # 本地标框
    # if bboxes.shape[1] != 6:
    #     print("No object found in {}".format(img_name))
    # labels = bboxes[:, 0].astype('int32')
    # scores = bboxes[:, 1].astype('float32')
    # boxes = bboxes[:, 2:].astype('float32')      
    # img = cv2.imread(os.path.join(img_dir, img_name))
    # img = draw_bbox_image(img, boxes, labels, scores, labels_name, thre=core.DRAW_THRESHOLD)
    # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # # output_dir = core.SAVE_DIR + '/' + img_name
    # output_dir = os.path.join(core.SAVE_DIR , img_name)
    # cv2.imwrite(output_dir, img)

    end_time = time.time()
    return result,round(end_time - start_time, 3)

def edge_load_model(path_prefix,img):
    paddle.enable_static()
    start_time = time.time()
    exe = paddle.static.Executor(paddle.CPUPlace())

    [inference_program, feed_target_names, fetch_targets] = (
        paddle.static.load_inference_model(path_prefix, exe))

    results = exe.run(inference_program,
              feed={feed_target_names[0]: image_preprocess(img)},
              fetch_list=fetch_targets)
    end_time = time.time()
    return np.array(results[0]), round(end_time - start_time, 3)

def cloud_load_tensor(path_prefix, tensor):
    paddle.enable_static()
    start_time = time.time()
    exe = paddle.static.Executor(paddle.CPUPlace())

    [inference_program, feed_target_names, fetch_targets] = (
        paddle.static.load_inference_model(path_prefix, exe))

    results = exe.run(inference_program,
              feed={feed_target_names[0]: tensor},
              fetch_list=fetch_targets)
    result = results[0].tolist()
    end_time = time.time()
    return [result[i].index(max(result[i])) for i in range(len(result))], round(end_time - start_time, 3)

import requests
if __name__ == "__main__":
    # tensor,edge_infer_time = edge_load_model(
    #     path_prefix="../data/send/model/client_infer_resnet18_cifar10",
    #     img="../data/test/air.jpeg")

    # result,cloud_infer_time = cloud_load_tensor(
    #     path_prefix="../data/send/model/server_infer_resnet18_cifar10",tensor=tensor)
    # print(result)
    image_shape, tensor, edge_infer_time = edge_load_model_yolo(
            model_path="./data/edge/client_infer_yolov3", 
            img_dir="./data/test",
            img_name = "dog.jpg")
    
    output, cloud_infer_time  = cloud_load_tensor_yolo(
            image_shape=image_shape, 
            tensor=tensor, 
            model_path="./data/cloud/server_infer_yolov3",
            img_dir="./data/test",
            img_name="dog.jpg")
    # print("Result saved in " + output)
    infos = {'filename':"dog.jpg",
             'edgetime':edge_infer_time,
             'cloudtime':cloud_infer_time,
             'result':output}
    
    print(json.loads(output))
    r = requests.get("http://127.0.0.1:5000/receive_result", params=infos)
    # print(r.text)





