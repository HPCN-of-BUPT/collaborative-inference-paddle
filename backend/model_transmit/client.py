import json, glob
import socket
import struct

import core
import json
import time
import numpy as np
from processbar import process_bar
from load_model import cloud_load_tensor_yolo

def receive_loop(type):
    flag = -1
    if type == "cloud":
        while flag != 0:
            client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            flag = client.connect_ex((core.EDGE_HOST, core.EDGE_SENDTO_CLOUD))
            if flag != 0:
                print("Edge refused to connect, please start edge process!")
            time.sleep(2)
        while True:
            infos = recv_tensor(client=client, model_prefix=core.CLOUD_MODEL_DIR)                  
    elif type == "edge":
        while flag != 0:
            client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            flag = client.connect_ex((core.CLOUD_HOST, core.CLOUD_SENTTO_EDGE))
            if flag != 0 :
                print("Cloud refused to connect, please start cloud process!")
            time.sleep(2)
        while True:
            recv_file(client)

def recv_file(client):
    # 解析头部长度
    head_struct = client.recv(4)
    head_len = struct.unpack('i', head_struct)[0]
    # 解析文件信息
    file_info = client.recv(head_len)
    file_info = json.loads(file_info.decode('utf-8'))
    filesize = file_info['filesize']
    filename = file_info['filename']
    # 接收文件
    recv_len = 0
    start_time = time.time()
    filename = filename.replace("send", "receive")
    with open(filename, 'wb') as f:
        while recv_len < filesize:
            precent = recv_len / filesize
            process_bar(precent)
            if(filesize - recv_len > core.BUFFER_SIZE):
                recv_msg = client.recv(core.BUFFER_SIZE)
                f.write(recv_msg)
                recv_len += len(recv_msg)
            else:
                recv_msg = client.recv(filesize - recv_len)
                recv_len += len(recv_msg)
                f.write(recv_msg)
            end_time = time.time()
            during_time = end_time - start_time
            filesize_mb = filesize / 1000 /1000
        print("\n{}({}MB) received correctly! Time: {}s\t Speed: {} MB/s".
              format(filename.split("/")[-1], round(filesize_mb,2), round(during_time,2), round(filesize_mb / during_time, 2)))

def recv_tensor(client, model_prefix):
    # 解析头部长度
    head_struct = client.recv(4)
    head_len = struct.unpack('i', head_struct)[0]
    
    # 解析文件信息
    file_info = client.recv(head_len)
    file_info = json.loads(file_info.decode('utf-8'))
    tensorsize,filename,tensorshape,imageshape,starttime,edgetime = file_info['filesize'],file_info['filename'],file_info['tensorshape'],file_info['imageshape'],file_info['starttime'],file_info['edgetime']
    
    # 使用memoryview接收tensor
    tensor_list = []
    for i,shape in enumerate(tensorshape):
        tensor = np.array(np.zeros(shape), dtype=core.NUMPY_TYPE)
        recv_into(tensor, client)
        tensor_list.append(tensor)
    end_time = time.time()
    tensor_transmit_time = round(end_time - starttime)
    # print("Tensor {} received correctly.\t Transmit time {}s".format(filename, tensor_transmit_time))
    # 云端计算剩余网络层
    results, cloud_infer_time = cloud_load_tensor_yolo(image_shape=np.array(imageshape, dtype=np.int32),tensor=tensor_list,model_path=model_prefix,img_dir=core.LOAD_DIR,img_name=filename)
    # print("Cloud cost {}s infer Tensor {}".format(cloud_infer_time, filename))
    # print("Tensor {}\t Result:{}".format(filename, results))

    # ACC 测试
    # core.TOTAL += 1
    # print(results[0])
    # print(filename.split('/')[-2])
    # if int(results[0]) == int(filename.split('/')[-1].split("_")[0]):
    #     core.CORRECT += 1
    # print('Acc:{:.3f}'.format(core.CORRECT / core.TOTAL))

    # 记录信息
    infos = {'filename':filename,
             'edgetime':edgetime,
             'cloudtime':cloud_infer_time,
             'transmitsize':tensorsize * len(tensor_list),
             'transmittime':tensor_transmit_time,
             'result':results}
    print("\nTransmit info of " + infos['filename'])
    print(infos)  
    return infos

def recv_into(arr, source):
    view = memoryview(arr).cast('B')
    while len(view):
        nrecv = source.recv_into(view)
        view = view[nrecv:]