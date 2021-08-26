import paddle_eval
# 此example.py为示例。本文件夹以外的地方，可以import performance_test来代替直接引入paddle_eval。
# 所有函数都有详细注释。

import cv2
from PIL import Image
import numpy as np

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

def image_preprocess_yolo(img_path):
    image = cv2.imread(img_path)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    origin, img_tensor, _ = read_image(image)
    input_w, input_h = origin.size[0], origin.size[1]
    image_shape = np.array([input_h, input_w], dtype='int32')
    return image_shape, img_tensor


if __name__ == '__main__':
    print("Client Model Analyse:")
    c_infer_model_filepath = '../model_transmit/data/send/client_infer_yolov3'
    c_flops, c_model_size, c_output_size = paddle_eval.model_analyse(c_infer_model_filepath,
                                                         input_list=[paddle_eval.make_fake_input((1, 3, 608, 608))]
                                                         )
    print("Server Model Analyse:")
    s_infer_model_filepath = '../model_transmit/data/send/server_infer_yolov3'
    s_flops, s_model_size, s_output_size = paddle_eval.model_analyse(s_infer_model_filepath,
                                                         input_list=[paddle_eval.make_fake_input((1, 256, 76, 76), dtype='float32'),
                                                                     paddle_eval.make_fake_input((1, 512, 38, 38), dtype='float32'),
                                                                     np.asarray([576, 768]).reshape(1,2)
                                                                     ],
                                                         dtype='float'
                                                         )