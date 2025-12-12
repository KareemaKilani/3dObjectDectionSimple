# 3D Object Detection - Homework Two

Minimal implementation of 3D object detection using 2 models on 2 datasets.

## Quick Setup

```bash
# Navigate to project directory
cd /Users/k/Desktop/CMPE\ 249/HomeworkTwo

# Install dependencies
pip install numpy matplotlib open3d opencv-python psutil pandas tabulate

# Run all steps
python code/minimal_detection.py
python code/visualize_open3d.py
python code/create_video.py
python code/generate_metrics.py
```

## Models & Datasets

### Models
1. **PointNet++**: Hierarchical point cloud processing
2. **VoteNet**: Voting-based 3D detection

### Datasets
1. **KITTI**: Outdoor driving scenes (3 synthetic scenes)
2. **SUNRGBD**: Indoor RGB-D scenes (3 synthetic scenes)

## Reproducibility

- **Random Seed**: Fixed in numpy (42) for consistent synthetic data generation
- **Environment**: Python 3.8+, macOS
- **Dependencies**: Listed in requirements below

## Output Structure

```
results/
├── PointNet++_KITTI/
│   ├── frames/*.png           # 2D visualizations
│   ├── point_clouds/*.ply     # 3D point clouds with detections
│   └── metadata/*.json        # Detection metadata
├── PointNet++_SUNRGBD/
├── VoteNet_KITTI/
├── VoteNet_SUNRGBD/
├── screenshots/               # Open3D screenshots
├── demo_*.mp4                 # Demo videos
├── comparison_table.md        # Metrics table
├── takeaways.md              # Analysis
└── inference_summary.json    # All results
```

## Dependencies

```
numpy>=1.21.0
matplotlib>=3.4.0
open3d>=0.13.0
opencv-python>=4.5.0
psutil>=5.8.0
pandas>=1.3.0
tabulate>=0.8.9
```

## Notes

- Uses synthetic data for fast execution
- Simplified model implementations for demonstration
- Real-world usage requires trained models and annotated datasets
