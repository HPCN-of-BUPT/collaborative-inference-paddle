# Model Deployment 
Main idea: deploy the client(edge) model with Paddlelite, deploy the server(cloud) model with Paddle Serving

## Edge Model
Convert the edge model with the following opt commands:
```
paddle_lite_opt \
    --model_file=./split_pruned_client_model \
    --param_file=./split_pruned_client_params\
    --optimize_out_type=naive_buffer \
    --optimize_out=client_opt \
    --valid_targets=x86 \
    --quant_model=true \
    --quant_type=QUANT_INT8
```

## Server Model

### Environmental Settings
You should install the serving-client and serving-server with different versions to match the yolo_box op used in our neural networks.

A. Paddle Serving:
```
# Setting
pip3.6 install -U paddle-serving-server==0.4.0 paddle-serving-client==0.4.0
pip3.6 install -U paddlepaddle==1.8.4
# Model convert and start serving
python3 convert.py
python3 -m paddle_serving_server.serve --model serving_server --thread 10 --port 9393
```
B. Paddle Client:
```
#setting
pip3.7 install paddlepaddle==2.1.0 paddle-serving-client==0.5.0
# start inference
python3.7 infer.py
```



