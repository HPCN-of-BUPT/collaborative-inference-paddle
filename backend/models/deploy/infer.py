import os
import numpy as np
import time
import cv2
import random
import paddle
import paddle.vision.transforms as T
from PIL import Image, ImageDraw, ImageFont
#import core
from paddlelite.lite import *
from paddle_serving_client import Client

labels_name = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
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
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, -1))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, -1))
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
    #win:arial.ttf
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

    config1 = MobileConfig()
    config1.set_model_from_file(model_path)
    image_shape, tensor_image = image_preprocess_yolo(os.path.join(img_dir, img_name))
    predictor = create_paddle_predictor(config1)
    input_tensor = predictor.get_input(0)
    input_tensor.from_numpy(tensor_image)
    predictor.run()
    result1 = predictor.get_output(0).numpy()
    result1 = np.array(result1)
    result2 = predictor.get_output(1).numpy()
    result2 = np.array(result2)
    results = [result1, result2]
    end_time = time.time()
    #print(results)
    return image_shape, results   #, round(end_time - start_time, 3)

def cloud_load_tensor_yolo(image_shape, tensor, model_path, img_dir,img_name):
    paddle.enable_static()
    start_time = time.time()

    client = Client()
    client.load_client_config(model_path)
    client.connect(["127.0.0.1:9393"])
    # 自适应输入tensor
    feed = {}
    feed_target_names = ['image1','image2','image_shape']
    image_shape = np.array(list(image_shape)).astype("int32")
    for index, t in enumerate(tensor):
        feed[feed_target_names[index]] = t[0]
    feed[feed_target_names[-1]] = image_shape#image_shape[np.newaxis, :]
    #print(feed['image1'])
    #print(feed['image_shape'])
    
    outputs = client.predict(feed=feed, fetch=["save_infer_model/scale_0.tmp_0"],batch=False)
    #print(outputs['save_infer_model/scale_0.tmp_0'])
    
    
    bboxes = np.array(outputs['save_infer_model/scale_0.tmp_0'])
    if bboxes.shape[1] != 6:
        print("No object found in {}".format(img_name))
    labels = bboxes[:, 0].astype('int32')
    scores = bboxes[:, 1].astype('float32')
    boxes = bboxes[:, 2:].astype('float32')

    img = cv2.imread(os.path.join(img_dir, img_name))
    img = draw_bbox_image(img, boxes, labels, scores, labels_name, thre=0.5)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # output_dir = core.SAVE_DIR + '/' + img_name
    output_dir = os.path.join('' , img_name)
    cv2.imwrite(output_dir, img)

    end_time = time.time()
    return output_dir,round(end_time - start_time, 3)

if __name__ == "__main__":
    
    image_shape, results = edge_load_model_yolo(
            model_path="edge_model/client_opt.nb", 
            img_dir="test/",
            img_name = "kite.jpg")
    
    #print(results)
    
    
    output, cloud_infer_time  = cloud_load_tensor_yolo(
            image_shape=image_shape, 
            tensor=results, 
            model_path="serving_client/serving_client_conf.prototxt",
            img_dir="test/",
            img_name="kite.jpg")
    print("Result saved in " + output)






