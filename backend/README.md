# Backend of Project

## Quick Start
```bash
# 后端进程：建立socket连接，下发模型；
$ python3 backend.py
    --backend_host "xxx.xxx.xxx.xxx"  # 后台IP
    --cloud_model_port xxxx   # 后台发送云端模型端口
    --edge_model_port xxxx    # 后台发送边端模型端口

# 后端进程：建立http连接，发送图片&返回结果；
$ python3 app.py

# 云端进程：socket接收模型和中间特征，http返回结果
$ python3 cloud.py 
    --backend_host "xxx.xxx.xxx.xxx"  # 后台IP
    --edge_host "xxx.xxx.xxx.xxx"   # 端IP
    --cloud_model_port xxxx   # 后台发送云端模型端口
    --cloud_tensor_port xxxx    # 边端发送中间特征端口

# 边端进程：socket接收模型和发送中间特征，http接收预处理图片
$ python3 edge.py 
    --backend_host "xxx.xxx.xxx.xxx"  # 后台IP
    --edge_host "xxx.xxx.xxx.xxx"   # 端IP
    --edge_model_port xxxx   # 后台发送边端模型端口
    --cloud_tensor_port xxxx    # 边端发送中间特征端口
```

## 模块说明

### 后台模块 
**backend.py**
- `backend_sendto_edge_thread`:通过建立socket连接，发送send文件夹中的边端模型，文件前缀为`infer_client_*`,包括`pdmodel`和`pdiparams`两个文件
- `backend_sendto_cloud_thread`:通过建立socket连接，发送send文件夹中的云端模型，文件前缀为`infer_server_*`,包括`pdmodel`和`pdiparams`两个文件

**app.py**
- `/transmit_image`,本地预处理用户上传到`input`文件夹的图片后发送至边端
- `/receive_result`,接收云端最终推理结果及传输信息，由后台进行标框

### 云端模块 
**cloud.py**
- `cloud_receive_model_thread`:接收后台下发模型，存储至cloud文件夹；
- `cloud_receive_tensor_thread`:通过建立socket连接，接收边端推演得到的中间tensor，并进行剩余计算，将最终结果通过`/receive_result`返回至后台，包括文件名、边端推理时间、云端推理时间、传输数据量、传输时间、标注框（由后台进行标注）；

### 边端模块 
**edge.py**
- `edge_receive_thread`:接收后台下发模型，存储至edge文件夹；
- `edge_send_thread`:通过`/transmit_image`批量请求预处理后的待检测图片，并进行部分结果推演，将图片信息和中间特征通过socket传输至云端，包括文件名、特征边端推演时间；

## Utils
- `db_utils`:数据库；
- `transmit`:数据压缩、信道翻转；
- `model`:模型切割、裁剪、量化；
- `performance_test`:性能评估；

## Else
- `core.py`:参数配置；
- `load_model.py`:模型部署；