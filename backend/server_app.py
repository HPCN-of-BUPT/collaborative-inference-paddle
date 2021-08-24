from flask import *
from flask_cors import *
from flask_sqlalchemy import SQLAlchemy
import pymysql
pymysql.version_info = (1, 4, 13, "final", 0)
pymysql.install_as_MySQLdb()

from db_utils.db_model import *
from db_utils import config

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['SQLALCHEMY_DATABASE_URI'] = config.URL
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# 查询时会显示原始SQL语句
app.config['SQLALCHEMY_ECHO'] = True
db = SQLAlchemy(app)


# 模型切割请求
@app.route('/cut', methods=['POST'])
def cut():
    print(request.data)

    return "success"