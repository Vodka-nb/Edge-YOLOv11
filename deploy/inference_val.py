# inference_test_with_classes.py
import cv2
import torch
import time
import numpy as np
from pathlib import Path
import argparse
import yaml


class InferenceTester:
    def __init__(self, weights_path: str, num_classes: int = None, 
                 img_size: int = 640, device: str = None):
        """
        
        Args:
            weights_path: 权重文件路径
            num_classes: 类别数量，如果不指定则尝试从权重文件读取
            img_size: 输入图像尺寸
            device: 推理设备
        """
        self.img_size = img_size
        self.weights_path = weights_path
        self.num_classes = num_classes
        
        # 自动选择设备
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                print("使用CPU")
        else:
            self.device = device
            
        # 加载模型
        print(f"加载模型: {weights_path}")
        self.model, self.class_names = self.load_model()
        
        # 打印类别信息
        print(f"检测类别数量: {len(self.class_names)}")
        if len(self.class_names) <= 20:  # 如果类别较少，打印出来
            print(f"类别列表: {self.class_names}")
        
        # 预热GPU
        self.warmup()
    
    def load_model(self):
        """加载模型，支持自定义类别"""
        try:
            # 方法1: 优先使用ultralytics YOLO
            try:
                from ultralytics import YOLO
                model = YOLO(self.weights_path)
                model.to(self.device)
                
                # 获取类别信息
                class_names = model.names if hasattr(model, 'names') else self.get_class_names_from_model(model)
                
                # 如果用户指定了类别数但模型不一致，发出警告
                if self.num_classes and len(class_names) != self.num_classes:
                    print(f"警告: 模型有{len(class_names)}个类别，但指定了{self.num_classes}个类别")
                
                print("使用Ultralytics YOLO接口")
                return model, class_names
            except ImportError:
                print("Ultralytics未安装，使用PyTorch直接加载")
            
            # 方法2: 直接使用PyTorch加载
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            
            # 尝试从checkpoint获取类别数
            if isinstance(checkpoint, dict):
                # 检查checkpoint中是否包含类别信息
                if 'model' in checkpoint:
                    model_dict = checkpoint['model']
                    # 尝试从模型state_dict推断类别数
                    if 'model.22.anchors' in model_dict:
                        # 这是YOLOv5格式
                        nc = self.infer_num_classes_from_state_dict(model_dict)
                    else:
                        nc = self.num_classes if self.num_classes else 80  # 默认80
                else:
                    nc = self.num_classes if self.num_classes else 80
                
                # 创建模型（需要models.py）
                try:
                    from models.yolo import Model
                    # 加载配置文件
                    cfg = '.yaml'
                    model = Model(cfg=cfg, ch=3, nc=nc)
                    model.load_state_dict(model_dict)
                    print(f"从checkpoint加载模型，类别数: {nc}")
                except ImportError:
                    print("找不到models.yolo，尝试其他方式")
                    model = checkpoint
            else:
                model = checkpoint
            
            model.to(self.device).eval()
            
            # 尝试获取类别名
            class_names = self.get_class_names()
            
            return model, class_names
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("尝试使用Torch Hub加载...")
            
            # 方法3: 使用Torch Hub
            model = torch.hub.load('ultralytics/yolov11', 'custom', 
                                   path=self.weights_path, 
                                   device=self.device)
            model.eval()
            
            # 获取类别名
            if hasattr(model, 'names'):
                class_names = model.names
            else:
                class_names = [f'class_{i}' for i in range(self.num_classes or 80)]
            
            print("使用Torch Hub加载")
            return model, class_names
    
    def infer_num_classes_from_state_dict(self, state_dict):
        """从state_dict推断类别数"""
        # 查找输出层
        for key in state_dict.keys():
            if 'm.0.weight' in key or 'm.3.weight' in key:  # YOLO输出层
                weight_shape = state_dict[key].shape
                if len(weight_shape) == 2:
                    # 全连接层
                    return weight_shape[0] - 5  # 减去4个坐标和1个置信度
                elif len(weight_shape) == 4:
                    # 卷积层
                    return weight_shape[0] // 3 - 5  # YOLO格式
        
        return self.num_classes if self.num_classes else 80
    
    def get_class_names(self):
        """获取类别名称"""
        # 尝试从labels目录读取类别
        labels_dir = Path('labels')
        if labels_dir.exists():
            # 查找类别文件
            class_files = list(labels_dir.glob('*.names')) + \
                          list(labels_dir.glob('*.txt'))
            if class_files:
                with open(class_files[0], 'r') as f:
                    class_names = [line.strip() for line in f if line.strip()]
                return class_names
        
        # 尝试从data.yaml读取
        yaml_files = list(Path('.').glob('*.yaml')) + list(Path('.').glob('*.yml'))
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'names' in data:
                        return data['names']
            except:
                continue
        
        # 使用默认类别或生成占位符
        if self.num_classes:
            return [f'class_{i}' for i in range(self.num_classes)]
        else:
            # COCO类别
            return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
                    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
                    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
                    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
                    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
                    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
                    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
                    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
                    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 
                    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
                    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
                    'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    def get_class_names_from_model(self, model):
        """从Ultralytics模型获取类别名"""
        if hasattr(model, 'names'):
            return model.names
        elif hasattr(model, 'model'):
            if hasattr(model.model, 'names'):
                return model.model.names
        return self.get_class_names()
    
    def visualize_results(self, image_path: str, outputs, original_size, 
                         confidence_threshold: float = 0.25):
        """
        可视化推理结果
        
        Args:
            image_path: 图像路径
            outputs: 模型输出
            original_size: 原始图像尺寸
            confidence_threshold: 置信度阈值
        """
        # 读取原始图像
        img = cv2.imread(image_path)
        if img is None:
            return
        
        original_w, original_h = original_size
        
        # 根据模型类型解析输出
        if hasattr(self.model, 'predict'):
            # Ultralytics输出格式
            results = outputs
            boxes = results[0].boxes if hasattr(results[0], 'boxes') else None
            if boxes is not None:
                for box in boxes:
                    conf = box.conf.item()
                    if conf < confidence_threshold:
                        continue
                    
                    cls = int(box.cls.item())
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    # 反归一化到原始尺寸
                    x1 = int(xyxy[0] * original_w / self.img_size)
                    y1 = int(xyxy[1] * original_h / self.img_size)
                    x2 = int(xyxy[2] * original_w / self.img_size)
                    y2 = int(xyxy[3] * original_h / self.img_size)
                    
                    # 绘制边界框
                    color = (0, 255, 0)  # 绿色
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # 添加标签
                    label = f"{self.class_names[cls]}: {conf:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # 标准YOLO输出格式
            if outputs is not None:
                # 这里需要根据实际输出格式调整
                # 通常是 [batch, boxes, (x1, y1, x2, y2, conf, cls1, cls2, ...)]
                pass
        
        # 显示图像
        cv2.imshow('Detection Results', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果
        output_path = Path(image_path).stem + '_result.jpg'
        cv2.imwrite(output_path, img)
        print(f"结果保存到: {output_path}")

    # 其他函数保持不变（warmup, preprocess, inference_single_image, batch_inference_test等）
    # ...

def main():
    parser = argparse.ArgumentParser(description='推理性能测试')
    parser.add_argument('--weights', type=str, default='best.pt', help='权重文件路径')
    parser.add_argument('--num-classes', type=int, default=None, help='类别数量（如果不指定则尝试自动检测）')
    parser.add_argument('--image-dir', type=str, default='images', help='测试图像目录')
    parser.add_argument('--img-size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8], help='测试的批次大小')
    parser.add_argument('--iterations', type=int, default=100, help='每个批次大小的迭代次数')
    parser.add_argument('--device', type=str, default=None, help='推理设备')
    parser.add_argument('--visualize', action='store_true', help='可视化检测结果')
    parser.add_argument('--confidence', type=float, default=0.25, help='检测置信度阈值')
    
    args = parser.parse_args()
    
    print("推理性能测试")
    print("="*60)
    
    # 创建测试器
    tester = InferenceTester(
        weights_path=args.weights,
        num_classes=args.num_classes,
        img_size=args.img_size,
        device=args.device
    )
    
    # 执行批处理推理测试
    results = tester.batch_inference_test(
        image_dir=args.image_dir,
        batch_sizes=args.batch_sizes,
        num_iterations=args.iterations
    )
    
    # 可视化示例图像
    if args.visualize:
        print("\n可视化检测结果...")
        image_files = list(Path(args.image_dir).glob("*.*"))
        if image_files:
            image_path = str(image_files[0])
            outputs, inference_time, original_size = tester.inference_single_image(image_path)
            tester.visualize_results(image_path, outputs, original_size, args.confidence)
    
    # 打印摘要
    tester.print_summary(results)
    
    # 保存结果
    tester.save_results(results)

if __name__ == "__main__":
    main()
