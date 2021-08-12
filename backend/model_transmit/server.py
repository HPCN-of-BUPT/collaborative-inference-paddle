import socket,os,sys,time,glob,json,struct

import core
from load_model import edge_load_model_yolo
import numpy as np
import channal_noise as cn
model_dict = [] # pdmodel发送记录
param_dict = [] # pdiparams发送记录
image_dict = [] # 发送/接收图片记录

def send_loop(type):
    if type == 'cloud':
        # 建立通信连接
        server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        server.bind((core.CLOUD_HOST, core.CLOUD_SENTTO_EDGE))
        server.listen(5)
        while True:
            conn, addr = server.accept()
            print("Cloud Server(I) {} : {} has connected to Edge client(others) {} : {}".
                  format(core.CLOUD_HOST,core.CLOUD_SENTTO_EDGE,addr[0],addr[1]))
            while True:
                # 发送pdmodel文件
                for filename in glob.glob(r'./data/send/client_infer_*.pdmodel'):
                    if(filename not in model_dict):
                        model_dict.append(filename)
                        send_file(conn, filename, "model")
                # 发送pdiparams文件
                for filename in glob.glob(r'./data/send/client_infer_*.pdiparams'):
                    if(filename not in param_dict):
                        param_dict.append(filename)
                        send_file(conn, filename, "model")
                # 发送测试图片
                for filename in glob.glob(r'./data/test/*'):
                    if(filename not in image_dict):
                        image_dict.append(filename)
                        send_file(conn, filename, "image")

    if type == 'edge':
        # 建立通信连接
        server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        server.bind((core.EDGE_HOST, core.EDGE_SENDTO_CLOUD))
        server.listen(5)
        while True:
            conn, addr = server.accept()
            print("Edge Server(I) {} : {} has connected to Cloud client(others) {} : {}".
                  format(core.EDGE_HOST,core.EDGE_SENDTO_CLOUD,addr[0],addr[1]))
            while True:
                # 轮训检测是否收到新的待检测图片
                for filename in glob.glob(r'./data/test/*'):
                    if filename not in image_dict:
                        image_dict.append(filename)
                        send_tensor(conn=conn,filename=filename.split("/")[-1],model_prefix=core.EDGE_MODEL_DIR)

# 云端发送模型或图片
def send_file(conn, filename, type):
    filesize = os.path.getsize(filename)
    dict = {
        'filename': filename,
        'filesize': filesize,
        'type':type
    }
    head_info = json.dumps(dict)
    head_info_len = struct.pack('i', len(head_info))
    # 发送头部长度
    conn.send(head_info_len)
    # 发送头部信息
    conn.send(head_info.encode('utf-8'))
    with open(filename, 'rb') as f:
        # 发送文件信息
        data = f.read()
        conn.sendall(data)
    print("\nFile {} ({} MB) send finish.".format(filename, round(filesize/1000/1000,2)))

# 边端发送中间特征
def send_tensor(conn, filename, model_prefix):
    # 计算得到中间特征
    image_shape, tensor_list, edge_infer_time = edge_load_model_yolo(model_path=model_prefix, img_dir=core.LOAD_DIR, img_name=filename)
    print("\nEdge cost {}s infer {} ".format(edge_infer_time, filename))

    tensor_size = sys.getsizeof(tensor_list[0])
    tensor_shape = get_tensor_shape(tensor_list)
    # 文件头内容
    dict = {
        'filename': filename,
        'filesize': tensor_size,
        'imageshape':image_shape.tolist(),
        'tensorshape':tensor_shape,
        'starttime':time.time(),
        'edgetime':edge_infer_time,
    }
    head_info = json.dumps(dict)
    head_info_len = struct.pack('i', len(head_info))
    # 发送头部长度
    conn.send(head_info_len)
    # 发送头部信息
    conn.send(head_info.encode('utf-8'))
    
    # 利用memoryview封装发送tensor
    for index, tensor in enumerate(tensor_list):
        # 二进制信道翻转
        # if tensor.dtype == "int8":
        #     tensor = cn.reverse_int8(tensor=tensor)
        # else:
        #     tensor = cn.reverse_float32(tensor=tensor)
        
        view = memoryview(tensor).cast("B")
        while len(view):
            nsent = conn.send(view)
            view = view[nsent:]
        print("{} mid-tensor{} ({} KB).\t Shape: {}".
            format(filename, index, round(tensor_size/1000,3), tensor.shape))
    return filename,edge_infer_time,tensor_size

def get_tensor_shape(tensor_list):
    shape_list = []
    for tensor in tensor_list:
        shape_list.append(tensor.shape)
    return shape_list


    