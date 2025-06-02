import os
from pathlib import Path
import shutil
from PIL import Image
import cv2
from rembg import remove
import numpy as np
from typing import List, Dict, Tuple
import concurrent.futures
import threading
import time

# Thread-safe printing
print_lock = threading.Lock()
def safe_print(message):
    with print_lock:
        print(message)

def ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def pil_to_cv2(pil_image):
    """Convert PIL Image to CV2 format."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convert CV2 image to PIL Image format."""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def is_frames_already_processed(output_dir: str, expected_frames: int = 21) -> bool:
    """Check if frames are already properly processed."""
    images_dir = os.path.join(output_dir, "image")
    masks_dir = os.path.join(output_dir, "mask")
    
    # Check if directories exist
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        return False
    
    # Count image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    if len(image_files) != expected_frames:
        return False
    
    # Count mask files
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
    if len(mask_files) != expected_frames:
        return False
    
    return True

def extract_frames_from_video(video_path: str, output_dir: str, expected_frames: int = 21) -> bool:
    """Extract frames from video using OpenCV."""
    # Skip if already processed
    if is_frames_already_processed(output_dir, expected_frames):
        safe_print(f"Skipping already processed video: {os.path.basename(video_path)}")
        return True
    
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames != expected_frames:
            safe_print(f"Warning: Video {video_path} has {total_frames} frames instead of expected {expected_frames}")
            return False
        
        # Create images directory
        images_dir = os.path.join(output_dir, "image")
        ensure_directory(images_dir)
        
        # Create masks directory
        masks_dir = os.path.join(output_dir, "mask")
        ensure_directory(masks_dir)
        
        frame_num = 0
        while frame_num < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert CV2 frame to PIL for saving and mask generation
            pil_frame = cv2_to_pil(frame)
            
            # Save original image
            frame_path = os.path.join(images_dir, f"{frame_num:03d}.png")
            pil_frame.save(frame_path, "PNG")
            
            # Generate and save mask
            mask = remove(pil_frame)  # rembg returns RGBA image
            # Convert RGBA to binary mask (white foreground, black background)
            mask_array = np.array(mask)
            binary_mask = ((mask_array[:, :, 3] > 0) * 255).astype(np.uint8)
            mask_img = Image.fromarray(binary_mask)
            
            mask_path = os.path.join(masks_dir, f"{frame_num:03d}.png")
            mask_img.save(mask_path, "PNG")
            
            frame_num += 1
        
        cap.release()
        return frame_num == expected_frames
        
    except Exception as e:
        safe_print(f"Error processing video {video_path}: {str(e)}")
        return False

def create_6_view_structure(src_dir: str, dst_dir: str) -> None:
    """Create 6-view structure by copying selected frames from 21-view."""
    # Skip if already processed
    if is_frames_already_processed(dst_dir, 6):
        safe_print(f"Skipping already processed 6-view structure: {os.path.basename(dst_dir)}")
        return
    
    # Create destination directories
    dst_images_dir = os.path.join(dst_dir, "image")
    dst_masks_dir = os.path.join(dst_dir, "mask")
    ensure_directory(dst_images_dir)
    ensure_directory(dst_masks_dir)
    
    # Source directories
    src_images_dir = os.path.join(src_dir, "image")
    src_masks_dir = os.path.join(src_dir, "mask")
    
    # Get every third frame until we have 6 frames
    src_files = sorted([f for f in os.listdir(src_images_dir) if f.endswith('.png')])
    selected_frames = src_files[::3][:6]
    
    # Copy selected frames and their masks
    for i, frame in enumerate(selected_frames):
        # Copy image
        src_image_path = os.path.join(src_images_dir, frame)
        dst_image_path = os.path.join(dst_images_dir, f"{i:03d}.png")
        shutil.copy2(src_image_path, dst_image_path)
        
        # Copy mask
        src_mask_path = os.path.join(src_masks_dir, frame)
        dst_mask_path = os.path.join(dst_masks_dir, f"{i:03d}.png")
        shutil.copy2(src_mask_path, dst_mask_path)

def build_name_mappings(input_dir: str) -> Dict[str, Tuple[str, str, str]]:
    """
    Build mapping from object names to their files.
    Returns dict with object name as key and tuple of (index, image_file, video_file) as value.
    """
    # Read the mapping file
    mapping_file = "./00_data/object_id_and_name.txt"
    if not os.path.exists(mapping_file):
        safe_print(f"Error: Mapping file '{mapping_file}' not found!")
        return {}
        
    with open(mapping_file, "r") as f:
        id_name_pairs = [line.strip().split(" ", 1) for line in f]
    
    # Create mapping from names to their files
    mappings = {}
    all_files = os.listdir(input_dir)
    
    for obj_id, obj_name in id_name_pairs:
        # Find matching image and video files
        image_file = next((f for f in all_files if f == obj_id), None)
        if image_file:
            image_path = os.path.join(image_file, "000000.jpg")
            video_path = os.path.join(image_file, "000000.mp4")
            
            # Check if files exist
            full_image_path = os.path.join(input_dir, image_path)
            full_video_path = os.path.join(input_dir, video_path)
            
            if os.path.exists(full_image_path) and os.path.exists(full_video_path):
                mappings[obj_name] = (obj_id, image_path, video_path)
            else:
                if not os.path.exists(full_image_path):
                    safe_print(f"Warning: Missing image file for {obj_name}: {image_path}")
                if not os.path.exists(full_video_path):
                    safe_print(f"Warning: Missing video file for {obj_name}: {video_path}")
        else:
            safe_print(f"Warning: Could not find directory for object {obj_name} (ID: {obj_id})")
    
    return mappings

def is_object_already_processed(obj_dir: str) -> bool:
    """Check if an object has already been fully processed."""
    input_image = os.path.join(obj_dir, "input_image.png")
    views_21_dir = os.path.join(obj_dir, "21_views")
    
    # Check if input image exists
    if not os.path.exists(input_image):
        return False
    
    # Check if 21 views are processed
    if not is_frames_already_processed(views_21_dir):
        return False
    
    return True

def process_object(input_dir: str, output_base: str, obj_name: str, file_info: Tuple[str, str, str]) -> None:
    """Process a single object's images and videos."""
    obj_id, image_file, video_file = file_info
    
    # Create object directory
    obj_dir = os.path.join(output_base, obj_id)
    ensure_directory(obj_dir)
    
    # Skip if already processed
    if is_object_already_processed(obj_dir):
        safe_print(f"Skipping already processed object {obj_name} (ID: {obj_id})")
        return
    
    try:
        start_time = time.time()
        safe_print(f"Processing object {obj_name} (ID: {obj_id})")
        
        # Convert and copy input image (from jpg to png)
        input_image_path = os.path.join(obj_dir, "input_image.png")
        if not os.path.exists(input_image_path):
            input_image = Image.open(os.path.join(input_dir, image_file))
            input_image.save(input_image_path, "PNG")
        
        # Create view directories
        views_21_dir = os.path.join(obj_dir, "21_views")
        ensure_directory(views_21_dir)
        
        # Process video
        success = extract_frames_from_video(os.path.join(input_dir, video_file), views_21_dir)
        
        if not success:
            safe_print(f"Warning: Failed to process video for object {obj_name} (ID: {obj_id})")
            
        processing_time = time.time() - start_time
        safe_print(f"Completed processing object {obj_name} in {processing_time:.2f} seconds")
            
    except Exception as e:
        safe_print(f"Error processing object {obj_name} (ID: {obj_id}): {str(e)}")

def main():
    # Get current directory for output
    base_dir = "./06_results"
    
    # Get input directory path from user
    input_dir = "06_results/generated_videos"
    input_dir = os.path.abspath(input_dir)  # Convert to absolute path
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return
    
    # Get number of threads from user (default to CPU count)
    try:
        max_threads = input(f"Enter number of threads to use (default: {os.cpu_count()}): ")
        max_threads = int(max_threads) if max_threads.strip() else os.cpu_count()
        if max_threads < 1:
            max_threads = os.cpu_count()
    except ValueError:
        max_threads = os.cpu_count()
    
    print(f"Using {max_threads} threads for processing")
    
    # Create main output directory
    output_base = os.path.join(base_dir, "generated_videos")
    ensure_directory(output_base)
    
    try:
        # Build name mappings
        name_mappings = build_name_mappings(input_dir)
        
        if not name_mappings:
            print("No valid objects found to process!")
            return
        
        print(f"Found {len(name_mappings)} objects to process")
        
        # Process objects in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            # Submit tasks
            futures = {executor.submit(process_object, input_dir, output_base, obj_name, file_info): obj_name 
                      for obj_name, file_info in name_mappings.items()}
            
            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                obj_name = futures[future]
                try:
                    future.result()  # This will raise any exceptions from the thread
                except Exception as e:
                    safe_print(f"Error processing {obj_name}: {str(e)}")
        
        print("All processing completed!")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()