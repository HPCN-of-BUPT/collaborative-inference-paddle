import socket, glob, json, struct, os, argparse
import core
from threading import Thread
cloud_model_list = []
edge_model_list = []
image_list = []
def send_edge_loop():
    server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.bind((core.CLOUD_HOST, core.EDGE_PORT))
    server.listen(5)
    while True:
        conn, addr = server.accept()
        print("Cloud Server(I) {} : {} has connected to Edge client(others) {} : {}".
                format(core.CLOUD_HOST,core.EDGE_PORT,addr[0],addr[1]))
        while True:
            # 发送边端模型文件
            for filename in glob.glob(r'./data/send/client_infer_*'):
                if(filename not in cloud_model_list):
                    cloud_model_list.append(filename)
                    send_file(conn, filename, "model")
            # 发送待检测图片
            for filename in glob.glob(r'./data/test/*'):
                if(filename not in image_list):
                    image_list.append(filename)
                    send_file(conn, filename, "image")

def send_cloud_loop():
    server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.bind((core.CLOUD_HOST, core.CLOUD_MODEL_PORT))
    server.listen(5)
    while True:
        conn, addr = server.accept()
        print("Cloud Server(I) {} : {} has connected to Edge client(others) {} : {}".
                format(core.CLOUD_HOST,core.CLOUD_TENSOR_PORT,addr[0],addr[1]))
        while True:
            # 发送云端模型文件
            for filename in glob.glob(r'./data/send/server_infer_*'):
                if(filename not in edge_model_list):
                    edge_model_list.append(filename)
                    send_file(conn, filename, "model")

def send_file(conn, filename, filetype):
    filesize = os.path.getsize(filename)
    dict = {
        'filename': filename,
        'filesize': filesize,
        'type':filetype,
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
    parser = argparse.ArgumentParser("Cloud Threads")
    parser.add_argument('--backend_host', type=str, default='',help="host of cloud")
    parser.add_argument('--cloud_host', type=str, default='',help="host of cloud")
    parser.add_argument('--edge_host', type=str, default='', help='host of edge')
    parser.add_argument('--cloud_model_port', type=int, default=0, help='port of backend send model to cloud')
    parser.add_argument('--edge_port', type=int, default=0, help="port of backend send model/image to edge")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    core.CLOUD_HOST = args.cloud_host if args.cloud_host else core.CLOUD_HOST
    core.EDGE_HOST = args.edge_host if args.edge_host else core.EDGE_HOST
    core.CLOUD_MODEL_PORT = args.cloud_model_port if args.cloud_model_port else core.CLOUD_MODEL_PORT
    core.EDGE_PORT = args.edge_port if args.edge_port else core.EDGE_PORT


    # 后台发送边端模型
    edge_server_thread = Thread(target=send_edge_loop, name="edge_server_thread")
    # 后台发送云端模型
    edge_client_thread = Thread(target=send_cloud_loop, name="edge_client_thread")
    
    # edge_client_thread.start()
    # edge_server_thread.start()