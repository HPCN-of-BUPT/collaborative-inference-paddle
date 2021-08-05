from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:192223@127.0.0.1:3308/paddle'
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
    infer_time = db.Column(db.Float)#推理时间（云-边模型：云边协同比）
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
    tensorsize = db.Column(db.Float)#传输数据量
    transmit_time = db.Column(db.Float)#传输时间
    time = db.Column(db.Float)
    accuracy = db.Column(db.Float)
    throughput = db.Column(db.Float)
    model_divide_id = db.Column(db.Integer)

if __name__ == '__main__':
    db.drop_all()
    db.create_all()


