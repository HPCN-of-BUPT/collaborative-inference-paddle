import os
import base64
from flask import *
from flask_cors import *
from db_utils.db_op import *
from flask_sqlalchemy import SQLAlchemy
from db_utils.db_predict import *
from db_utils.db_model import *
import pymysql
from db_utils.config import URL
pymysql.version_info = (1, 4, 13, "final", 0)
pymysql.install_as_MySQLdb()

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['SQLALCHEMY_DATABASE_URI'] = URL
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
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

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/hello')
def hello():
    return 'Hello!'

#模型切割请求（前端）
@app.route('/cut',methods=['POST'])
def cut():
    print(request.data)
    model_divide_id = add_restriction(eval(request.data))

    #模型切割模块....
    edge_model,cloud_model,edge_cloud_model = add_submodel(model_divide_id)
    edge = {'flops':edge_model.flops,'params':edge_model.params}
    cloud = {'flops':cloud_model.flops,'params':cloud_model.params}
    edge_cloud = {'flops':edge_cloud_model.flops,'params':edge_cloud_model.params}
    models = {'edge':edge,'cloud':cloud,'edge_cloud':edge_cloud}
    return jsonify({'result':'success','models':models})


#图像测试文件上传（前端）   ***1
@app.route('/image_upload',methods=['POST'])
def image_upload():
    file = request.files.get('file')
    input_dir = './data/input/'
    file.filename = 'predict.jpg'
    file_path = input_dir + file.filename
    print(file_path)
    file.save(file_path)
    #ID = add_edge_status(db)
    #print('id ',ID)
    status.IMAGE_STATUS = 1
    return jsonify({'msg':'success'})

#检测边端是否已接收图片（前端）  ****2
@app.route('/edge',methods=['POST','GET'])
def edge():
    #检测是否上传新的图片
    if status.EDGE_STATUS == 0:
        return jsonify({'msg':'false'})
    else:
        return jsonify({'msg':'true'})

#请求检测结果（前端）   ***3
@app.route('/get_result',methods=['POST','GET'])
def getResult():
    if status.EDGE_STATUS == 0:
        return jsonify({'msg':'false'})
    else:
        status.IMAGE_STATUS = 0
        status.EDGE_STATUS = 0
        status.TEST_STATUS = 0
        results = get_result(db,System,status.ID)
        return jsonify({'msg':'true','data':results})


#请求检测图片（后端）
@app.route('/getImage',methods=['POST','GET'])
def getImage():
    #检测是否上传新的图片
    if status.IMAGE_STATUS == 0:
        return 'false'
    else:
        #base64位图像编码返回
        input_dir = './data/input/'
        filename = 'predict.jpg'
        f = open(os.path.join(input_dir, filename), 'rb')
        base64_str = base64.b64encode(f.read())
        status.EDGE_STATUS = 1
        return jsonify({'data': str(base64_str, 'utf-8')})


#发送目标检测结果（后端）
@app.route('/receive_result',methods=['POST','GET'])
def receive_result():
    print(request.args)
    status.ID = add_result(db, System, request.args)
    status.TEST_STATUS = 1
    return "success"

if __name__ == '__main__':
    app.run()
