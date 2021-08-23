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

from db_utils import db_op
from db_utils import db_save
import core

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:gj6143585@127.0.0.1:3306/paddle'
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# 查询时会显示原始SQL语句
app.config['SQLALCHEMY_ECHO'] = True
db = SQLAlchemy(app)

CORS(app, supports_credentials=True)


@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers["Access-Control-Allow-Methods"] = "PUT,GET,POST,DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,Accept,Origin,Referer,User-Agent"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    return response

#模型切割请求
@app.route('/cut',methods=['POST'])
def cut():
    print(request.data)
    model_divide_id = db_op.add_restriction(eval(request.data))

    #模型切割模块....
    edge_model,cloud_model,edge_cloud_model = db_save.add_submodel(model_divide_id)
    edge = {'flops':edge_model.flops,'params':edge_model.params}
    cloud = {'flops':cloud_model.flops,'params':cloud_model.params}
    edge_cloud = {'flops':edge_cloud_model.flops,'params':edge_cloud_model.params}
    models = {'edge':edge,'cloud':cloud,'edge_cloud':edge_cloud}
    return jsonify({'result':'success','models':models})

#图像测试文件上传
@app.route('/image_upload',methods=['POST'])
def image_upload():
    file = request.files.get('file')
    input_dir = '../../backend/data/test/'
    filename = file.filename
    file_path = input_dir + filename
    print(file_path)
    file.save(file_path)
    flag = True
    output_dir = '../../backend/data/output'
    while flag:
        for file in os.listdir(output_dir):
            if file == filename:
                flag = False
    f = open(os.path.join(output_dir,filename),'rb')
    base64_str = base64.b64encode(f.read())
    name, edgetime, cloudtime, transmitsize, transmittime = db_op.find_result(filename=filename)
    msg = {'filename':name,
             'edgetime':edgetime,
             'cloudtime':cloudtime,
             'transmitsize':transmitsize,
             'transmittime':transmittime,
             'img_base64':str(base64_str,'utf-8')}
    #print(msg2)
    return jsonify({'msg':msg})

# 传输预处理后的待检测图片
image_list = []
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
            image_list.append(filename)
            number += 1
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

# 接收检测结果
@app.route('/receive_result',methods=['POST','GET'])
def receive_result():
    results = request.args
    # flag = 0, success; flag = 1, no object detected
    flag = draw_box(results['result'], results['filename'])
    return "success"

def draw_box(bboxes,filename):
    if bboxes == '[]':
        return 1
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
    return 0

if __name__ == '__main__':
    app.run()