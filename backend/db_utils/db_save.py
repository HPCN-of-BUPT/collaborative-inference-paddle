# from flask import Flask
# from flask_sqlalchemy import SQLAlchemy
# import pymysql
# import config
#
# pymysql.version_info = (1, 4, 13, "final", 0)
# pymysql.install_as_MySQLdb()
# app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = config.URL
# app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# # 查询时会显示原始SQL语句
# app.config['SQLALCHEMY_ECHO'] = True
# db = SQLAlchemy(app)
#
# class System(db.Model):
#     __tablename__ = 'system'
#     id = db.Column(db.Integer, primary_key=True)
#     filename = db.Column(db.String(255)) #检测文件名
#     edge_time = db.Column(db.Float) #边推理时间
#     cloud_time = db.Column(db.Float) #云推理时间
#     transmit_size = db.Column(db.Float)#传输数据量
#     transmit_time = db.Column(db.Float)#传输时间
#     cloud_edge_ratio = db.Column(db.Float) #云边协同比
#     result_path = db.Column(db.String(255)) #检测结果文件名
#
#     time = db.Column(db.Float)
#     accuracy = db.Column(db.Float)
#     throughput = db.Column(db.Float)
#     model_divide_id = db.Column(db.Integer)
#
#
# def add_system_result(infos):
#     system = System(filename=infos['filename'],
#                     edge_time=infos['edgetime'],
#                     cloud_time=infos['cloudtime'],
#                     transmit_size=infos['transmitsize'],
#                     transmit_time=infos['transmittime'])
#     db.session.add(system)
#     db.session.commit()
#     return system.id
