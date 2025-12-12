#!/bin/bash
# Run complete 3D object detection pipeline

echo "========================================"
echo "3D Object Detection Pipeline"
echo "========================================"

# Step 1: Run inference
echo "Step 1/4: Running inference..."
python code/minimal_detection.py

# Step 2: Generate screenshots
echo "Step 2/4: Generating screenshots..."
python code/visualize_open3d.py

# Step 3: Create videos
echo "Step 3/4: Creating demo videos..."
python code/create_video.py

# Step 4: Generate metrics
echo "Step 4/4: Generating metrics..."
python code/generate_metrics.py

echo "========================================"
echo "Pipeline complete!"
echo "Check results/ folder for outputs"
echo "========================================"
