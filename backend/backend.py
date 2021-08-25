import socket, glob, json, struct, os, argparse, requests
import core
from performance import *
from threading import Thread
cloud_model_list = []
edge_model_list = []

def send_edge_loop():
    server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.bind((core.BACKEND_HOST, core.EDGE_MODEL_PORT))
    server.listen(5)
    while True:
        conn, addr = server.accept()
        print("Backend {} : {} has connected to Edge {} : {}".
                format(core.BACKEND_HOST,core.EDGE_MODEL_PORT,addr[0],addr[1]))
        while True:
            # 发送边端模型文件
            for filename in glob.glob(r'./data/send/client_infer_*'):
                if(filename not in cloud_model_list):
                    cloud_model_list.append(filename)
                    send_file(conn, filename)
                    filepath = os.path.join('./data/send',filename)
                    infos = client_analyse(filepath)  #模型评估
                    r = requests.get("http://127.0.0.1:5000/perform_client_result", params=infos)
                    print(r.text)

def send_cloud_loop():
    server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.bind((core.BACKEND_HOST, core.CLOUD_MODEL_PORT))
    server.listen(5)
    while True:
        conn, addr = server.accept()
        print("Backend {} : {} has connected to Cloud {} : {}".
                format(core.BACKEND_HOST,core.CLOUD_MODEL_PORT,addr[0],addr[1]))
        while True:
            # 发送云端模型文件
            for filename in glob.glob(r'./data/send/server_infer_*'):
                if(filename not in edge_model_list):
                    edge_model_list.append(filename)
                    send_file(conn, filename)
                    filepath = os.path.join('./data/send', filename)
                    infos = server_analyse(filepath)  # 模型评估
                    r = requests.get("http://127.0.0.1:5000/perform_server_result", params=infos)
                    print(r.text)

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


def parse_args():
    parser = argparse.ArgumentParser("Backend Threads")
    parser.add_argument('--backend_host', type=str, default='',help="host of backend")
    parser.add_argument('--cloud_model_port', type=int, default=0, help='port of backend send model to cloud')
    parser.add_argument('--edge_model_port', type=int, default=0, help="port of backend send model to edge")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    core.BACKEND_HOST = args.backend_host if args.backend_host else core.BACKEND_HOST
    core.CLOUD_MODEL_PORT = args.cloud_model_port if args.cloud_model_port else core.CLOUD_MODEL_PORT
    core.EDGE_MODEL_PORT = args.edge_model_port if args.edge_model_port else core.EDGE_MODEL_PORT


    # 后台发送边端模型
    backend_sendto_edge_thread = Thread(target=send_edge_loop, name="edge_server_thread")
    # 后台发送云端模型
    backend_sendto_cloud_thread = Thread(target=send_cloud_loop, name="edge_client_thread")
    
    backend_sendto_edge_thread.start()
    backend_sendto_cloud_thread.start()