import sys
sys.path.append("..")
from app import db

# from flask import Flask
# from flask_sqlalchemy import SQLAlchemy
# import pymysql
# from db_utils.config import URL
# pymysql.version_info = (1, 4, 13, "final", 0)
# pymysql.install_as_MySQLdb()
#
# app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = URL
# app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# # 查询时会显示原始SQL语句
# app.config['SQLALCHEMY_ECHO'] = True
# db = SQLAlchemy(app)

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
    flops = db.Column(db.String(255))
    params = db.Column(db.String(255))
    output_size = db.Column(db.String(255))
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
    filename = db.Column(db.String(255))  # 文件名

    transmit_size = db.Column(db.Float)  # 传输数据量
    edge_time = db.Column(db.Float)  # 边推理时间
    cloud_time = db.Column(db.Float)  # 云推理时间
    transmit_time = db.Column(db.Float)  # 传输时间
    cloud_edge_ratio = db.Column(db.Float)  # 云边协同比

    time = db.Column(db.Float)
    accuracy = db.Column(db.Float)
    throughput = db.Column(db.Float)
    model_divide_id = db.Column(db.Integer)



