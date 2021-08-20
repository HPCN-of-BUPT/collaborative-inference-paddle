from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import pymysql
pymysql.version_info = (1, 4, 13, "final", 0)
pymysql.install_as_MySQLdb()
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:gj6143585@127.0.0.1:3306/paddle'
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# 查询时会显示原始SQL语句
app.config['SQLALCHEMY_ECHO'] = True
db = SQLAlchemy(app)


class Model(db.Model):
    #表名
    __tablename__ = 'model'
    #列对象
    id = db.Column(db.Integer,primary_key=True)
    name = db.Column(db.String(255))

class Restriction(db.Model):
    __tablename__ = 'restriction'
    id = db.Column(db.Integer, primary_key=True)
    delay = db.Column(db.Float)
    bandwidth = db.Column(db.Float)
    cloud_computation = db.Column(db.Float)
    edge_computation = db.Column(db.Float)

class Submodel(db.Model):
    __tablename__ = 'submodel'
    id = db.Column(db.Integer, primary_key=True)
    flops = db.Column(db.Float)
    params = db.Column(db.Float)
    type = db.Column(db.Integer)#模型类型 0：云-边模型  1：云模型  2：边模型
    model_divide_id = db.Column(db.Integer)


class ModelDivide(db.Model):
    __tablename__ = 'model_divide'
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer)
    restric_id = db.Column(db.Integer)


class System(db.Model):
    __tablename__ = 'system'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255)) #检测文件名
    edge_time = db.Column(db.Float) #边推理时间
    cloud_time = db.Column(db.Float) #云推理时间
    transmit_size = db.Column(db.Float)#传输数据量
    transmit_time = db.Column(db.Float)#传输时间
    cloud_edge_ratio = db.Column(db.Float) #云边协同比
    result_path = db.Column(db.String(255)) #检测结果文件名

    time = db.Column(db.Float)
    accuracy = db.Column(db.Float)
    throughput = db.Column(db.Float)
    model_divide_id = db.Column(db.Integer)

if __name__ == '__main__':
    db.drop_all()
    db.create_all()


