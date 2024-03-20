PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /home/faith/miniconda3/bin/python /home/faith/llm-export/llm_export.py --path /home/faith/.cache/huggingface/hub/models--Qwen--Qwen1.5-0.5B-Chat/snapshots/6c705984bb8b5591dd4e1a9e66e1a127965fd08d --type Qwen1_5-0_5B-Chat --export_token --onnx_path ./Qwen1.5-0.5B-Chat-onnx --export_split



PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /home/faith/miniconda3/bin/python /home/faith/llm-export/llm_export.py --path /home/faith/.cache/huggingface/hub/models--Qwen--Qwen1.5-0.5B-Chat/snapshots/6c705984bb8b5591dd4e1a9e66e1a127965fd08d --type Qwen1_5-0_5B-Chat --onnx_path ./Qwen1.5-0.5B-Chat-onnx --export

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /home/faith/miniconda3/bin/python /home/faith/llm-export/llm_export.py --path /home/faith/.cache/huggingface/hub/models--Qwen--Qwen1.5-0.5B-Chat/snapshots/6c705984bb8b5591dd4e1a9e66e1a127965fd08d --type Qwen1_5-0_5B-Chat --export_token --onnx_path ./Qwen1.5-0.5B-Chat-onnx --export_split --export_mnn --mnn_path  ./Qwen1.5-0.5B-Chat-mnn 




mnnconvert -f ONNX --modelFile block_0.onnx --MNNModel block_0.mnn --weightQuantBits 4 --bizCode biz


https://onnxruntime.ai/docs/performance/model-optimizations/float16.html



 import onnx
 from onnxconverter_common import float16

 model = onnx.load("path/to/model.onnx")
 model_fp16 = float16.convert_float_to_float16(model)
 onnx.save(model_fp16, "path/to/model_fp16.onnx")
