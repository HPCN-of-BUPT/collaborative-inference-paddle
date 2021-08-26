# YoLov3_edge_cloud
This project focuses on edge-cloud collaborative object detection inference, which is established based on PaddlePaddle YoLov3-DarkNet (https://github.com/PaddlePaddle/models/tree/release/1.4/PaddleCV/yolov3). 
## Preparation
1. Download the supported model parameters from: https://1drv.ms/u/s!AqK-Jk7aHxL8gUaizyJ6nxkpwMVy?e=PLo9tx
2. Put the downloaded parameters into the main branch.
3. Download the coco-2014 dataset with "cd dataset/coco"  "./download.sh"
4. Set ```export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7``` to specify 8 GPU to train.

## Training
- If you want to train the model from scratch, run train.py with ```python3 train.py```. We have the pre-trained model, which is stored in complete_model/model_final
- After having the pre-trained model, you can train your pruned model with fine-tuning operations using ```train_prune.py```. You can change the pruning ratio and choose the pruning layers. We have finished training one pruned model, which is stored in pruned_model/model_final.

## Evaluation
You can use eval.py to evaluate the completed model and use eval_prune.py to evaluate the pruned model with the "training" type.
You can use eval_infer.py to evaluate the completed model and use eval_infer_split.py to evaluate the splited model with the "inference" type.


## Model Export
- You can export the trained model for inference. Here, we provide several python files corresponding to different types of models:
- export.py:  export completed model (without split and prune)
- export_pruned.py:  export completed pruned model (without split and with prune)
- export_client.py and export_server.py:  split the completed model into the client model and the server model(with split and without prune)
- export_client_pruned.py and export_server_pruned.py:  split the completed and pruned model into the client model and the server model (with split and prune)
The exported models are stored in freezed/model
- export_quant.py:  export the true quantized model (int 8/16) and the fake quantized model (fp32)

## Model Inference
Finally, you can use infer_completed.py to infer the completed model (with prune or without prune), use infer_split.py to infer the split model (with prune or without prune).
Or you can convert the quantized model to the .nb style and infer it by the infer_split_lite.py file.

## Models
### complete_model
The pre-trained completed model
### pruned_model
The pre-trained pruned model
### freezed_model
The freezed model for inference stage
#### quantized_model
True int8/16 quantized model
#### test_model
Fake int8/16 quantized model. The model is still stored with data type fp32.
1. completed_model/completed_params : original model
2. completed_pruned_model/completed_pruned_params : pruned model
3. split_client(server)_model/split_client(server)_params : splited model
4. split_pruned_client(server)_model/split_pruned_client(server)_params : splited and pruned model

