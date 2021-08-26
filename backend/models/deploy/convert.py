import paddle_serving_client.io as serving_io
serving_io.inference_model_to_serving(dirname='cloud_model', serving_server="serving_server", serving_client="serving_client",  model_filename='server.pdmodel', params_filename='server.pdiparams')
