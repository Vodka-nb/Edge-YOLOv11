🌿 Edge-YOLOv11: Lightweight & Occlusion-Aware Fruit Detection for UAV Agriculture  

  
  
  
⚠️ **Official implementation of the manuscript submitted to *The Visual Computer***  

📌 Important Notice  
🔒 This repository corresponds directly to the manuscript:  "A Lightweight and Occlusion-Aware Framework for UAV-Based Fruit Detection in Dense Canopy Environments"  
**Submitted toThe Visual Computer (2024)**  
✨ If you use this code in your research, please cite our work (see Citation below).  
🌍 Code and dataset DOI will be permanently archived upon publication.  

ℹ️ *Note: "Edge-YOLOv11" denotes our custom lightweight architecture inspired by YOLO series principles. It is not an official Ultralytics YOLO version (current public release: YOLOv10). Naming reflects our model's evolution within this research context.*  

🌟 Why Edge-YOLOv11?  
Designed specifically for real-world orchard challenges:  
✅ Lightweight (6.35 MB) – Runs efficiently on Jetson Orin Nano (26.52 FPS, 135 MB GPU memory)  
✅ Occlusion-Aware – Spatial-enhanced attention handles clustered fruits & dense foliage  
✅ Multi-Fruit Generalization – Validated on litchi, tomato, and citrus datasets  
✅ Edge-Ready – TensorRT optimized for UAV field deployment  
✅ Reproducible – Full training/inference pipeline + pre-trained weights  

📦 Installation  
Clone repository
git clone https://github.com/yourusername/Edge-YOLOv11.git
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
model = EdgeYOLOv11(weights="weights/edgeyolo_litchi.pt")
results = model.predict("your_uav_image.jpg", save=True, show=True)

Evaluate on Litchi-UAV dataset  
python val.py --data litchi_uav.yaml --weights weights/edgeyolo_litchi.pt --img 640

📌 Pre-trained weights:  
Dataset   mAP@0.5   F1-Score   Download
Litchi    90.1%     85.5%      Google Drive / Zenodo

Tomato    88.7%     83.2%      Google Drive

Citrus    86.9%     81.5%      Google Drive
(Full benchmark details in manuscript Section 4.3)  

📚 Project Structure  
Edge-YOLOv11/  
├── edgeyolo/               # Core model implementation  
│   ├── models/             # C3-MSR module, F3 fusion, Litchi-Head  
│   ├── utils/              # SEAM attention, data augmentations  
│   └── deploy/             # TensorRT export scripts for Jetson  
├── data/                   # Dataset configs (litchi_uav.yaml, etc.)  
├── weights/                # Pre-trained models (download links above)  
├── notebooks/              # Demo: inference on sample UAV images  
├── scripts/                # Environment setup, TensorRT conversion  
├── requirements.txt        # Dependencies  
└── README.md               # You are here!  

📖 Documentation Highlights  
- docs/INSTALL_JETSON.md: Step-by-step TensorRT deployment on Jetson Orin Nano  
- docs/DATASET_PREP.md: Format your fruit dataset (supports YOLO, COCO)  
- notebooks/demo_inference.ipynb: Visualize detection results interactively  
- docs/REPRODUCE_RESULTS.md: Exact commands to replicate manuscript experiments  

📝 Citation  
If this work benefits your research, please cite our manuscript:  
@article{Author2024EdgeYOLOv11,
  title={A Lightweight and Occlusion-Aware Framework for UAV-Based Fruit Detection in Dense Canopy Environments},
  author={Your Name and Co-authors},
  journal={The Visual Computer},
  year={2024},
  note={Under review},
  doi={10.5281/zenodo.xxxxxxx}  % Update upon publication
}
  
🙏Your citation supports open science and acknowledges the effort behind this work.  

🤝 Contributing & Support  
- 🐞 Report bugs via Issues  
- 💡 Suggestions welcome! Open a PR with clear description  
- ❓ Questions? Email: your.email@university.edu (mention "Edge-YOLOv11 inquiry")  

🌾 Acknowledgements  
- Dataset collected in collaboration with [Agricultural Institute Name]  
- Jetson deployment validated on NVIDIA Orin Nano Developer Kit  
- Inspired by YOLO series; built with PyTorch & MMDetection ecosystem  

📜 License  
MIT License – See LICENSE for details.  Academic use encouraged. For commercial applications, please contact authors.  

✨Empowering precision agriculture through accessible, edge-friendly computer vision ✨  
🌱 Code archived with DOI via Zenodo | 📬 Manuscript ID: TVC-2024-XXXXX
