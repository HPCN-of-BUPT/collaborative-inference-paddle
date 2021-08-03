from flask import *
from flask_cors import *
from test_model import infer


app = Flask(__name__)
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
    #模型切割模块....
    return jsonify({'result':'success'})


#图像测试文件上传
@app.route('/image_upload',methods=['POST'])
def image_upload():
    file = request.files.get('file')
    directory = 'static/images/'
    file_path = directory + file.filename
    file.save(file_path)
    result = infer(file_path)
    print(result)
    return jsonify({'result':str(result)})

if __name__ == '__main__':
    app.run()
