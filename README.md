
🌍 Code and dataset DOI will be permanently archived upon publication. 

🌱 Code archived with DOI via Zenodo:[https://doi.org/10.5281/zenodo.18514495]
    Dataset with DOI via Zenodo: [https://doi.org/10.5281/zenodo.19364014]

🌟 Why Edge-YOLOv11?  
Designed specifically for real-world orchard challenges:  
✅ Lightweight (6.35 MB) – Runs efficiently on Jetson Orin Nano (26.52 FPS, 135 MB GPU memory)  
✅ Occlusion-Aware – Spatial-enhanced attention handles clustered fruits & dense foliage  
✅ Multi-Fruit Generalization – Validated on litchi, tomato, and citrus datasets  
✅ Edge-Ready – TensorRT optimized for UAV field deployment  
✅ Reproducible – Full training/inference pipeline + pre-trained weights  

📦 Installation  
Clone repository
git clone https://github.com/Vodka-nb/Edge-YOLOv11.git
cd Edge-YOLOv11

Create environment (Python ≥3.8 required)
conda create -n edgeyolo python=3.9 -y
conda activate edgeyolo

Install dependencies
pip install -r requirements.txt
For TensorRT deployment (Jetson):
bash scripts/install_tensorrt_jetson.sh

🚀 Quick Start  
Inference on your UAV image  
from edgeyolo import EdgeYOLOv11
model = EdgeYOLOv11(weights="weights/best.pt")
results = model.predict("your_uav_image.jpg", save=True, show=True)

Evaluate on Litchi-UAV dataset  
python val.py --data litchi_uav.yaml --weights weights/best.pt --img 640

📌 Pre-trained weights:  
Dataset   mAP@0.5   F1-Score  
Litchi    90.1%     85.5%      

Tomato    88.7%     83.2%      

Citrus    86.9%     81.5%      


📚 Project Structure  
Edge-YOLOv11/  
├── Edge-YOLOv11/               # Core model implementation  
├── data/                  
├── detect/                  
├── engine/             
├── solutions/      
├── trackers/      
├── utils/      
├── deploy/      
├── v11/      
└── README.md               # You are here!  
  

📜 License  
MIT License – See LICENSE for details.  Academic use encouraged. For commercial applications, please contact authors.  

✨Empowering precision agriculture through accessible, edge-friendly computer vision ✨  

