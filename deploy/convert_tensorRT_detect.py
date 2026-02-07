# rebuild_engine.py
from ultralytics import YOLO
from pathlib import Path

print("重新构建TensorRT引擎...")

# 使用原始模型
model_path = "./pt/best.pt"
if not Path(model_path).exists():
    print(f"错误: 找不到 {model_path}")
    exit(1)

# 加载模型
model = YOLO(model_path, task='detect')

# 重新导出为640x640的TensorRT引擎
print("正在导出为TensorRT引擎 (640x640)...")
model.export(
    format='engine',
    imgsz=640,      # 使用640x640，兼容性更好
    half=True,      # FP16加速
    workspace=6,    # 4GB工作空间
    simplify=True,  # 简化模型
    batch=1,        # 单批次
    device=0        # GPU 0
)

print("✓ 引擎重新构建完成!")
print("新引擎文件: ./pt/best.engine")
print("\n现在可以正常推理了!")

# 测试推理
print("\n测试推理...")
test_engine = YOLO('./pt/best.engine', task='detect')
results = test_engine.predict(
    './images/',  # 或者指定一张测试图像
    imgsz=640,
    conf=0.25,
    save=True
)

print("✓ 推理测试成功!")
