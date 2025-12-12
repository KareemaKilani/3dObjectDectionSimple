"""
Create demo video from PNG frames
"""

import cv2
import numpy as np
from pathlib import Path
import glob

def create_video_from_frames(frames_dir, output_path, fps=2):
    """Stitch PNG frames into a video"""
    # Get all PNG files
    frame_paths = sorted(glob.glob(str(frames_dir / "*.png")))
    
    if not frame_paths:
        print(f"No frames found in {frames_dir}")
        return
    
    print(f"Creating video from {len(frame_paths)} frames...")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Write frames
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is not None:
            video.write(frame)
    
    video.release()
    print(f"Video saved to {output_path}")

def main():
    """Create videos for each model-dataset combination"""
    results_dir = Path('/Users/k/Desktop/CMPE 249/HomeworkTwo/results')
    
    # Find all frame directories
    frame_dirs = list(results_dir.glob("*/frames"))
    
    print(f"Found {len(frame_dirs)} frame directories\n")
    
    for frame_dir in frame_dirs:
        model_dataset = frame_dir.parent.name
        output_path = results_dir / f"demo_{model_dataset}.mp4"
        
        create_video_from_frames(frame_dir, output_path, fps=2)
    
    print(f"\nAll videos created in {results_dir}")

if __name__ == "__main__":
    main()
