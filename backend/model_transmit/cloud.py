import time,argparse
import core
from threading import Thread
from client import receive_loop
from server import send_loop

def parse_args():
    parser = argparse.ArgumentParser("Cloud Threads")
    parser.add_argument('--cloud_host', type=str, default='',help="host of cloud")
    parser.add_argument('--edge_host', type=str, default='', help='host of edge')
    parser.add_argument('--cloud_port', type=int, default=0, help='port of cloud sendto edge')
    parser.add_argument('--edge_port', type=int, default=0, help="port of edge sendto cloud")
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    core.CLOUD_HOST = args.cloud_host if args.cloud_host else core.CLOUD_HOST
    core.EDGE_HOST = args.edge_host if args.edge_host else core.EDGE_HOST
    core.CLOUD_SENTTO_EDGE = args.cloud_port if args.cloud_port else core.CLOUD_SENTTO_EDGE
    core.EDGE_SENDTO_CLOUD = args.edge_port if args.edge_port else core.EDGE_SENDTO_CLOUD
   
    cloud_server_thread = Thread(target=send_loop, args=("cloud", ),name="cloud_server_thread")
    cloud_client_thread = Thread(target=receive_loop, args=("cloud", ),name="cloud_client_thread")
    cloud_server_thread.start()
    cloud_client_thread.start()
    # print(time.time())
