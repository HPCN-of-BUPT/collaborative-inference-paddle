import socket,os,sys
import time

import core
import glob
import json
import struct
from threading import Thread
from load_model import edge_load_model_yolo
import numpy as np
import channal_noise as cn
import core
model_dict = []
param_dict = []
image_dict = []

def send_loop(type):
    if type == 'cloud':
        server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        server.bind((core.CLOUD_HOST, core.CLOUD_SENTTO_EDGE))
        server.listen(5)
        while True:
            conn, addr = server.accept()
            print("Cloud Server(I) {} : {} has connected to Edge client(others) {} : {}".
                  format(core.CLOUD_HOST,core.CLOUD_SENTTO_EDGE,addr[0],addr[1]))
            while True:
                # 发送pdmodel文件
                for filename in glob.glob(r'../data/send/model/client_infer_*.pdmodel'):
                    if(filename not in model_dict):
                        model_dict.append(filename)
                        # send_file(conn, filename)
                # 发送pdiparams文件
                for filename in glob.glob(r'../data/send/model/client_infer_*.pdiparams'):
                    if(filename not in param_dict):
                        param_dict.append(filename)
                        # send_file(conn, filename)

    if type == 'edge':
        server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        server.bind((core.EDGE_HOST, core.EDGE_SENDTO_CLOUD))
        server.listen(5)
        while True:
            conn, addr = server.accept()
            print("Edge Server(I) {} : {} has connected to Cloud client(others) {} : {}".
                  format(core.EDGE_HOST,core.EDGE_SENDTO_CLOUD,addr[0],addr[1]))
            while True:
                for filename in glob.glob(os.path.join(core.LOAD_DIR, "*.jpg")):
                    if filename not in image_dict:
                        print(filename)
                        image_dict.append(filename)
                        send_tensor(conn=conn,filename=filename.split("/")[-1],model_prefix=core.EDGE_MODEL_DIR)
                # send_tensor(conn=conn,filename='dog.jpg',model_prefix="../data/send/model/split_pruned_client")
                # time.sleep(5)
                # send_tensor(conn=conn,filename='eagle.jpg',model_prefix="../data/send/model/split_pruned_client")
                # break


def send_file(conn, filename):
    filesize = os.path.getsize(filename)
    dict = {
        'filename': filename,
        'filesize': filesize,
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

def send_tensor(conn, filename, model_prefix):
    image_shape, tensor_list, edge_infer_time = edge_load_model_yolo(model_path=model_prefix, img_dir=core.LOAD_DIR, img_name=filename)
    print("Edge cost {}s infer {} ".format(edge_infer_time, filename))

    tensorsize = sys.getsizeof(tensor_list)
    tensorshape = get_tensor_shape(tensor_list)
    print("Tensor list size:" + str(tensorsize))
    # 发送文件头信息
    dict = {
        'filename': filename,
        'filesize': tensorsize,
        'imageshape':image_shape.tolist(),
        'tensorshape':tensorshape,
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
        print("Filename  {} mid-tensor ({} KB) send finish.\t Shape: {}".
            format(filename, round(tensorsize/1000,3), tensor.shape))
    return filename,edge_infer_time,tensorsize

def get_tensor_shape(tensor_list):
    shape_list = []
    for tensor in tensor_list:
        shape_list.append(tensor.shape)
    return shape_list

if __name__ == '__main__':
    edge_server = Thread(target=send_loop, args=("cloud", ))
    