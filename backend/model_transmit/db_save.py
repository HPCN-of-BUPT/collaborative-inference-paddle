from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:192223@127.0.0.1:3308/paddle'
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# 查询时会显示原始SQL语句
app.config['SQLALCHEMY_ECHO'] = True
db = SQLAlchemy(app)

class System(db.Model):
    __tablename__ = 'system'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255)) #检测文件名
    tensorsize = db.Column(db.Float)#传输数据量
    edge_infer_time = db.Column(db.Float) #边推理时间
    cloud_infer_time = db.Column(db.Float) #云推理时间
    cloud_edge_ratio = db.Column(db.Float) #云边协同比
    transmit_time = db.Column(db.Float)#传输时间
    time = db.Column(db.Float)
    accuracy = db.Column(db.Float)
    throughput = db.Column(db.Float)
    model_divide_id = db.Column(db.Integer)

def add_system_edge(file_name,edge_infer_time,tensorsize):
    system = System(filename=file_name,edge_infer_time=edge_infer_time,tensorsize=tensorsize)
    db.session.add(system)
    db.session.commit()
    return system.id

def add_system_cloud(transmit_time,cloud_infer_time, results):
    system = System(transmit_time=transmit_time,cloud_infer_time=cloud_infer_time)
    db.session.add(system)
    db.session.commit()
    return system.id
