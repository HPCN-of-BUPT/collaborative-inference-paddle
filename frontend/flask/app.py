import os
import base64
from flask import *
from flask_cors import *
from db_op import *
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:192223@127.0.0.1:3308/paddle'
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

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/hello')
def hello():
    return 'Hello!'

#模型切割请求
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
    name, edgetime, cloudtime, transmitsize, transmittime = find_result(filename=filename)
    msg = {'filename':name,
             'edgetime':edgetime,
             'cloudtime':cloudtime,
             'transmitsize':transmitsize,
             'transmittime':transmittime,
             'img_base64':str(base64_str,'utf-8')}
    #print(msg2)
    return jsonify({'msg':msg})

if __name__ == '__main__':
    app.run()
