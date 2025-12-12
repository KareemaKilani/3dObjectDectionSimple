"""
Open3D Visualization Script
Loads .ply files and saves screenshots of detected objects
"""

import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_ply_points(ply_path):
    """Load points from PLY file"""
    points = []
    with open(ply_path, 'r') as f:
        in_data = False
        for line in f:
            if line.startswith('end_header'):
                in_data = True
                continue
            if in_data:
                parts = line.strip().split()
                if len(parts) >= 3:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(points)

def visualize_ply_with_screenshots(ply_path, metadata_path, screenshot_path):
    """Load PLY file and save screenshot using matplotlib"""
    print(f"Loading {ply_path}")
    
    # Load point cloud
    points = load_ply_points(ply_path)
    
    # Load metadata for detections
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points (subsample for speed)
    sample_idx = np.random.choice(len(points), min(2000, len(points)), replace=False)
    sample_points = points[sample_idx]
    ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], 
               c='lightgray', s=1, alpha=0.4)
    
    # Draw bounding boxes
    for i, det in enumerate(metadata['detections']):
        center = np.array(det['bbox_center'])
        size = np.array(det['bbox_size'])
        
        # Draw box edges
        x = [center[0] - size[0]/2, center[0] + size[0]/2]
        y = [center[1] - size[1]/2, center[1] + size[1]/2]
        z = [center[2] - size[2]/2, center[2] + size[2]/2]
        
        color = ['red', 'blue', 'green'][i % 3]
        
        # Draw wireframe box
        for xi in [0, 1]:
            for yi in [0, 1]:
                ax.plot([x[0], x[1]], [y[yi], y[yi]], [z[xi], z[xi]], color=color, linewidth=2)
                ax.plot([x[xi], x[xi]], [y[0], y[1]], [z[yi], z[yi]], color=color, linewidth=2)
                ax.plot([x[yi], x[yi]], [y[xi], y[xi]], [z[0], z[1]], color=color, linewidth=2)
        
        # Label
        ax.text(center[0], center[1], center[2], 
               f"{det['class']}\n{det['confidence']:.2f}", 
               color=color, fontsize=10, weight='bold')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f"3D Detection - {metadata['model']} on {metadata['dataset']}", fontsize=14)
    
    plt.savefig(screenshot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Screenshot saved to {screenshot_path}")

def main():
    """Main visualization function"""
    results_dir = Path('/Users/k/Desktop/CMPE 249/HomeworkTwo/results')
    
    # Find all PLY files
    ply_files = list(results_dir.glob("**/point_clouds/*.ply"))
    
    print(f"Found {len(ply_files)} PLY files to visualize\n")
    
    screenshot_dir = results_dir / "screenshots"
    screenshot_dir.mkdir(exist_ok=True)
    
    # Process each PLY file
    for ply_path in ply_files[:6]:  # Limit to 6 screenshots
        # Get corresponding metadata
        relative_path = ply_path.relative_to(results_dir)
        model_dataset = ply_path.parent.parent.name
        scene_name = ply_path.stem
        
        metadata_path = ply_path.parent.parent / "metadata" / f"{scene_name}.json"
        
        if metadata_path.exists():
            screenshot_name = f"{model_dataset}_{scene_name}.png"
            screenshot_path = screenshot_dir / screenshot_name
            
            try:
                visualize_ply_with_screenshots(ply_path, metadata_path, screenshot_path)
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"  Metadata not found for {ply_path}")
    
    print(f"\nAll screenshots saved to {screenshot_dir}")

if __name__ == "__main__":
    main()
