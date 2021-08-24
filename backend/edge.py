import time,argparse, struct, json, socket, glob, os, sys
import numpy as np
import requests
import core
from threading import Thread
import paddle
from load_model import model_to_lite, edge_load_model_yolo_lite
from transmit import processbar
image_list = []
def edge_receive_loop():
    # 建立通信连接
    flag = -1
    while flag != 0:
        client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        flag = client.connect_ex((core.BACKEND_HOST, core.EDGE_MODEL_PORT))
        if flag != 0 :
            print("Cloud refused to connect, please start cloud process!")
        time.sleep(2)
    while True:
        # 边端接收切割模型
        filename = recv_file(client)

def edge_send_loop():
    server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.bind((core.EDGE_HOST, core.CLOUD_TENSOR_PORT))
    server.listen(5)
    while True:
        # 建立通信连接
        conn, addr = server.accept()
        print("Edge {} : {} has connected to Cloud {} : {}".
                format(core.EDGE_HOST,core.CLOUD_TENSOR_PORT,addr[0],addr[1]))
        while True:
            # 加载模型
            if (os.path.isfile(core.EDGE_MODEL_DIR + '.pdmodel') and os.path.isfile(core.EDGE_MODEL_DIR + '.pdiparams')):
                # 使用opt工具优化原始模型
                output_model = model_to_lite(model_path=core.EDGE_MODEL_DIR + '.pdmodel',
                    param_path=core.EDGE_MODEL_DIR + '.pdiparams')

                # paddle-inference
                # paddle.enable_static()
                # exe = paddle.static.Executor(paddle.CPUPlace())
                # [inference_program, feed_target_names, fetch_targets] = (
                #     paddle.static.load_inference_model(core.EDGE_MODEL_DIR, exe))
                
                # 循环发送请求预处理图片
                while True:
                    r = requests.get('http://127.0.0.1:5000/transmit_image')
                    results = r.json()
                    # print(results)
                    if (int(results['number']) > 0):
                        images = r.json()['file_list']
                    else:
                        images = []       
                    for index, image in enumerate(images):
                        # Windows: change / to \\
                        filename = image['filename'].split("/")[-1]
                        
                        # paddle inference
                        # start_time = time.time()
                        # results = exe.run(inference_program,
                        #         feed={feed_target_names[0]: np.array(json.loads(image['tensor']), dtype=np.float32)},
                        #         fetch_list=fetch_targets)
                        # shape = np.array(json.loads(image['shape']), dtype=np.int32)
                        # end_time = time.time()
                        # edge_infer_time = round(end_time - start_time, 3)

                        # paddle lite
                        shape, results, edge_infer_time = edge_load_model_yolo_lite(
                                                                model_path=output_model,
                                                                image_shape=np.array(json.loads(image['shape']), dtype=np.int32),
                                                                tensor_image=np.array(json.loads(image['tensor']), dtype=np.float32))
                        # 边端计算得到中间tensor
                        print("\nEdge cost {}s infer {} ".format(edge_infer_time, filename))
                        send_tensor(conn=conn, 
                                    filename= filename, 
                                    edge_infer_time=edge_infer_time, 
                                    tensor_list = results, 
                                    image_shape=shape)
                    time.sleep(1)

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
    # 接收文件
    recv_len = 0
    start_time = time.time()
    # 根据类型判断存储路径
    save_dir = filename.replace("send", "edge")
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

def send_tensor(conn, filename, edge_infer_time, tensor_list, image_shape):
    tensor_shape = get_tensor_shape(tensor_list)
    # 发送文件头信息
    dict = {
        'filename': filename,
        # 'filesize': tensor_size,
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
        # # 二进制信道翻转
        # if tensor.dtype == "int8":
        #     tensor = cn.reverse_int8(tensor=tensor)
        # else:
        #     tensor = cn.reverse_float32(tensor=tensor)
        view = memoryview(tensor).cast("B")
        tensor_size = sys.getsizeof(view)
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
def parse_args():
    parser = argparse.ArgumentParser("Cloud Threads")
    parser.add_argument('--backend_host', type=str, default='',help="host of backend")
    parser.add_argument('--edge_host', type=str, default='', help='host of edge')
    parser.add_argument('--edge_model_port', type=int, default=0, help='port of backend send model to edge')
    parser.add_argument('--cloud_tensor_port', type=int, default=0, help="port of edge send tensor to cloud")
    # parser.add_argument('--channal_error', type=float, default=0, help='channal error(0~0.25)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    core.BACKEND_HOST = args.backend_host if args.backend_host else core.BACKEND_HOST
    core.EDGE_HOST = args.edge_host if args.edge_host else core.EDGE_HOST
    core.EDGE_MODEL_PORT = args.edge_model_port if args.edge_model_port else core.EDGE_MODEL_PORT
    core.CLOUD_TENSOR_PORT = args.cloud_tensor_port if args.cloud_tensor_port else core.CLOUD_TENSOR_PORT
    # core.ERROR_RATE = args.channal_error if args.channal_error else core.ERROR_RATE

    # 边端发送中间特征线程
    edge_send_thread = Thread(target=edge_send_loop, name="edge_server_thread")
    # 边端接收切割模型/待检测图片线程
    edge_receive_thread = Thread(target=edge_receive_loop, name="edge_client_thread")
    edge_receive_thread.start()
    # time.sleep(10)
    edge_send_thread.start()
