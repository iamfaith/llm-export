import os
try:
    import _tools as MNNTools
except:
    MNNTools = None

def onnx2mnn(onnx_path, mnn_dir, quant_bit = 4, asymmetric = True, external_data = False, bizCode : str= None):
    model_name, model_extension = os.path.splitext(os.path.basename(onnx_path))
    if model_extension != '.onnx':
        return
    mnn_name = model_name + '.mnn'
    mnn_path = os.path.join(mnn_dir, mnn_name)
    convert_args = [
        '',
        '-f',
        'ONNX',
        '--modelFile',
        str(onnx_path),
        '--MNNModel',
        str(mnn_path),
        # '--weightQuantBits',
        # str(quant_bit),
    ]
    if asymmetric:
        convert_args.append("--weightQuantAsymmetric")
    if external_data:
        convert_args.append("--saveExternalData")
    if bizCode is not None:
        convert_args.append("--bizCode")
        convert_args.append(str(bizCode))
    # MNNTools.mnnconvert(convert_args)
    convert_args = ' '.join(convert_args)
    os.system(f"/home/faith/MNN/build/MNNConvert {convert_args}")

onnx_model = "/home/faith/llm-export/Qwen1.5-0.5B-Chat-onnx-old"
mnn_path = "/home/faith/llm-export/Qwen1.5-0.5B-Chat-mnn"
from glob import glob

def convertmnn():
    for f in glob(f"{onnx_model}/*.onnx"):
        onnx2mnn(f, mnn_path, asymmetric=False)
    
    
    
import onnx
from onnxconverter_common import float16

for f in glob(f"{onnx_model}/*.onnx"):
    model = onnx.load(f)
    
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, f"/home/faith/llm-export/Qwen1.5-0.5B-Chat-onnx-fp16/{os.path.basename(f)}")