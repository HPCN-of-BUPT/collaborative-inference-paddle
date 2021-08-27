import os, glob, json, random
import base64
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from flask import *
from flask_cors import *
from flask_sqlalchemy import SQLAlchemy
import pymysql
pymysql.version_info = (1, 4, 13, "final", 0)
pymysql.install_as_MySQLdb()

from db_utils.db_predict import *
from db_utils.db_eval import *
from db_utils.db_model import *
from db_utils import config
from performance import *
from export_client import *
from export_server import *
import core

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['SQLALCHEMY_DATABASE_URI'] = config.URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# 查询时会显示原始SQL语句
app.config['SQLALCHEMY_ECHO'] = True
db = SQLAlchemy(app)

import status


@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers["Access-Control-Allow-Methods"] = "PUT,GET,POST,DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,Accept,Origin,Referer,User-Agent"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    return response

#图像测试文件上传（前端-后端）
@app.route('/image_upload',methods=['POST'])
def image_upload():
    file = request.files.get('file')
    input_dir = core.LOAD_DIR
    file_path = input_dir + file.filename
    print(file_path)
    file.save(file_path)
    return jsonify({'msg': 'success'})


# 传输预处理后的待检测图片
# image_list = []
@app.route('/transmit_image', methods=['POST','GET'])
def transmit_result():
    file_list = []
    number = 0
    for filename in glob.glob(os.path.join(core.LOAD_DIR, "*.jpg")):
        if filename not in file_list:
            image = cv2.imread(filename)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            origin, tensor_img, _ = read_image(image)
            input_w, input_h = origin.size[0], origin.size[1]
            image_shape = np.array([input_h, input_w], dtype='int32').tolist()
            info = { "filename": filename, 
                    "shape": str(image_shape),
                    "tensor":str(tensor_img.tolist())}
            file_list.append(info)
            # image_list.append(filename)
            number += 1
    status.EDGE_STATUS = 1
    return {"file_list" : file_list, "number": number}

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

#检测边端是否已接收图片（前端-后端）
@app.route('/edge',methods=['POST','GET'])
def edge():
    if status.EDGE_STATUS == 0:
        return jsonify({'msg':'false'})
    else:
        return jsonify({'msg':'true'})


# 接收检测结果
@app.route('/receive_result',methods=['POST','GET'])
def receive_result():
    results = request.args
    # flag = n, n object detected; flag = 0, no object detected
    flag = draw_box(results['result'], results['filename'])
    # status.ID = add_result(db, System, results['result']) #结果添加数据库
    # status.TEST_STATUS = 1
    return "success"

def draw_box(bboxes,filename):
    if bboxes == '[]':
        return 0
    bboxes = np.array(json.loads(bboxes))
    labels = bboxes[:, 0].astype('int32')
    scores = bboxes[:, 1].astype('float32')
    boxes = bboxes[:, 2:].astype('float32')
    img = cv2.imread(os.path.join(core.LOAD_DIR,filename))
    color = ['FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7']
    
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    line_thickness = max(int(min(img.size) / 200), 2)
    # win:arial.ttf
    # font = ImageFont.truetype("arial.ttf", size=max(round(max(img.size) / 40), 12))
    
    font = ImageFont.truetype("Arial.ttf", size=max(round(max(img.size) / 40), 12))

    for box, label,score in zip(boxes, labels, scores):
        c = random.randint(0,19)
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        draw.rectangle((xmin, ymin, xmax, ymax), None, "#" + color[c], width=line_thickness)
        draw.text(( xmin + 5, ymin + 5), 
                    core.LABELS[int(label)] + ' ' + str(round(score * 100, 2)) + "%", 
                    "#" + color[c], font=font)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    output_dir = os.path.join(core.SAVE_DIR, filename)
    cv2.imwrite(output_dir, img)
    return len(labels)

#请求检测结果（前端）
@app.route('/get_result',methods=['POST','GET'])
def getResult():
    if status.TEST_STATUS == 0:
        return jsonify({'msg':'false'})
    else:
        status.EDGE_STATUS = 0
        status.TEST_STATUS = 0
        results = get_result(db,System,status.ID)

        f = open(os.path.join(core.LOAD_DIR, results['filename']), 'rb')
        base64_str = base64.b64encode(f.read())
        return jsonify({'msg':'true','result':results,'img_base64':str(base64_str,'utf-8')})

# 模型切割请求
@app.route('/cut', methods=['POST'])
def cut():
    print(request.data)
    args = parse_args()
    print_arguments(args)
    eval_client('./data/send')
    eval_server('./data/send')
    return "success"

# 模型切割结果
@app.route('/cut_result', methods=['GET'])
def cut_result():
    if status.CLIENT_MODEL_STATUS == 1 & status.SERVER_MODEL_STATUS == 1:
        results = get_eval_result(db,Submodel,1)
        status.CLIENT_MODEL_STATUS = status.SERVER_MODEL_STATUS = 0
        return jsonify({'msg': 'success',
                        'results': results
                        })
    else:
        return "fail"


# 请求模型评估结果
@app.route('/perform_client_result', methods=['POST','GET'])
def perform_client_result():
    if status.CLIENT_MODEL_STATUS == 0:
        infos = request.args
        print("client:")
        print(infos)
        add_client(db, Submodel, infos)
        status.CLIENT_MODEL_STATUS = 1
        return 'success'
    else:
        return 'fail'


# 请求模型评估结果
@app.route('/perform_server_result', methods=['POST','GET'])
def perform_server_result():
    if status.CLIENT_MODEL_STATUS == 0:
        infos = request.args
        print("server:")
        print(infos)
        add_server(db, Submodel, infos)
        status.SERVER_MODEL_STATUS = 1
        return 'success'
    else:
        return 'fail'


# 模型部署上传ip
@app.route('/cloud_model_arrange', methods=['POST'])
def cloud_model_arrange():
    print(request.data)
    #request.data['cloud_ip']
    return "success"

# 模型部署上传ip
@app.route('/edge_model_arrange', methods=['POST'])
def edge_model_arrange():
    print(request.data)
    # request.data['edge_ip']
    return "success"

if __name__ == '__main__':
    app.run()
