import time,argparse,struct, json, sys 
import socket
import requests
import core
import numpy as np
from threading import Thread
from load_model import cloud_load_tensor_yolo
from transmit import processbar
def cloud_receive_tensor_loop():
    flag = -1
    # 建立通信连接
    while flag != 0:
        client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        flag = client.connect_ex((core.EDGE_HOST, core.CLOUD_TENSOR_PORT))
        if flag != 0:
            print("Edge refused to connect, please start edge process!")
        time.sleep(2)
    while True:
        # 云端接收中间特征并计算最终结果
        infos = recv_tensor(client=client, model_prefix=core.CLOUD_MODEL_DIR)
        # 云端保存最终结果至数据库
        # add_system_result(infos)
        r = requests.get("http://127.0.0.1:5000/receive_result", params=infos)
        print(r.text)

def cloud_receive_model_loop():
    # 建立通信连接
    flag = -1
    while flag != 0:
        client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        flag = client.connect_ex((core.BACKEND_HOST, core.CLOUD_MODEL_PORT))
        if flag != 0 :
            print("Cloud refused to connect, please start cloud process!")
        time.sleep(2)
    while True:
        # 边端接收切割模型或待检测图片
        filename = recv_file(client)

# 边端接收模型文件或待检测图片
def recv_file(client):
    # 解析头部长度
    head_struct = client.recv(4)
    head_len = struct.unpack('i', head_struct)[0]
    # 解析文件信息
    file_info = client.recv(head_len)
    file_info = json.loads(file_info.decode('utf-8'))
    filesize = file_info['filesize']
    filename = file_info['filename']
    type = file_info['type']
    # 接收文件
    recv_len = 0
    start_time = time.time()
    # 根据类型判断存储路径
    save_dir = filename.replace("send", "cloud") if type == "model" else filename
    # 存储文件
    with open(save_dir, 'wb') as f:
        while recv_len < filesize:
            precent = recv_len / filesize
            processbar.process_bar(precent)
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
              format(save_dir.split("/")[-1], round(filesize_mb,2), round(during_time,2), round(filesize_mb / during_time, 2)))
    return filename

def recv_tensor(client, model_prefix):
    # 解析头部长度
    head_struct = client.recv(4)
    head_len = struct.unpack('i', head_struct)[0]
    
    # 解析文件信息
    file_info = client.recv(head_len)
    file_info = json.loads(file_info.decode('utf-8'))
    filename,tensor_shape,image_shape,start_time,edge_infer_time = \
        file_info['filename'],file_info['tensorshape'],file_info['imageshape'],file_info['starttime'],file_info['edgetime']
    
    # 使用memoryview接收tensor
    tensor_list = []
    tensor_size = 0
    for i,shape in enumerate(tensor_shape):
        tensor = np.array(np.zeros(shape), dtype=core.NUMPY_TYPE)
        length = recv_into(tensor, client)
        tensor_list.append(tensor)
        tensor_size += length
    
    # 传输时间
    end_time = time.time()
    tensor_transmit_time = round(end_time - start_time)
    # print("Tensor {} received correctly.\t Transmit time {}s".format(filename, tensor_transmit_time))
    
    # 云端计算剩余网络层
    results, cloud_infer_time = cloud_load_tensor_yolo(image_shape=np.array(image_shape, dtype=np.int32),tensor=tensor_list,model_path=model_prefix,img_dir=core.LOAD_DIR,img_name=filename)
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
             'edgetime':edge_infer_time,
             'cloudtime':cloud_infer_time,
             'transmitsize':tensor_size,
             'transmittime':tensor_transmit_time,
             'result':results}

    print("\nTransmit info of " + infos['filename'])
    print(infos)  
    return infos
def recv_into(arr, source):
    view = memoryview(arr).cast('B')
    length = sys.getsizeof(view)
    while len(view):
        nrecv = source.recv_into(view)
        view = view[nrecv:]
    return length

def parse_args():
    parser = argparse.ArgumentParser("Cloud Threads")
    parser.add_argument('--backend_host', type=str, default='',help="host of backend")
    parser.add_argument('--edge_host', type=str, default='', help='host of edge')
    parser.add_argument('--cloud_model_port', type=int, default=0, help='port of backend send tensor to cloud')
    parser.add_argument('--cloud_tensor_port', type=int, default=0, help="port of edge send tensor to cloud")
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    # 测试配置
    args = parse_args()
    core.BACKEND_HOST = args.backend_host if args.backend_host else core.BACKEND_HOST
    core.EDGE_HOST = args.edge_host if args.edge_host else core.EDGE_HOST
    core.CLOUD_MODEL_PORT = args.cloud_model_port if args.cloud_model_port else core.CLOUD_MODEL_PORT
    core.CLOUD_TENSOR_PORT = args.cloud_tensor_port if args.cloud_tensor_port else core.CLOUD_TENSOR_PORT
   
    # 云端发送文件线程
    cloud_receive_model_thread = Thread(target=cloud_receive_model_loop,name="cloud_receive_model")
    # 云端接受中间特征线程
    cloud_receive_tensor_thread = Thread(target=cloud_receive_tensor_loop,name="cloud_receive_tensor_thread")
    cloud_receive_model_thread.start()
    cloud_receive_tensor_thread.start()
