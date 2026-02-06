import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# onnx onnxsim onnxruntime onnxruntime-gpu

# 导出参数官方详解链接：https://docs.ultralytics.com/modes/export/#usage-examples

if __name__ == '__main__':
    model = YOLO('runs/train/exp-yolov11s-a+b+c/weights/best.pt')
    model.export(format='onnx', simplify=True, opset=13, imgsz=1024)