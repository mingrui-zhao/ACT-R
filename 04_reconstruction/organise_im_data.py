import os
import numpy as np
from PIL import Image
import shutil
import glob

def process_model(model_id, input_base_path, output_base_path):
    model_id_str = f"{model_id:05d}"
    
    input_image_path = glob.glob(os.path.join(input_base_path, model_id_str, "*", 'image'))[0]
    # print(input_image_path)
    input_camera_path = glob.glob(os.path.join(input_base_path, model_id_str, "*", 'cameras_sphere.npz'))[0]
    output_image_path = os.path.join(output_base_path, 'images')
    output_camera_path = os.path.join(output_base_path, 'camera')
    
    os.makedirs(output_image_path, exist_ok=True)
    os.makedirs(output_camera_path, exist_ok=True)
    
    # Selected view indices
    view_indices = [0, 3, 7, 10, 13, 17]
    
    images = []
    for i in view_indices:
        img_path = os.path.join(input_image_path, f"{i:03d}.png")
        img = Image.open(img_path)
        img_resized = img.resize((320, 320), Image.Resampling.LANCZOS)
        images.append(img_resized)
    
    combined_img = Image.new('RGB', (640, 960))
    
    positions = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
    for idx, pos in enumerate(positions):
        x, y = pos
        combined_img.paste(images[idx], (y * 320, x * 320))
    
    combined_img.save(os.path.join(output_image_path, f"{model_id_str}.png"))
    shutil.copy2(input_camera_path, os.path.join(output_camera_path, f"{model_id_str}.npz"))

def main():
    input_base_path = "./06_results/generated_videos/"
    output_base_path = "./06_results/generated_videos_im_style"

    for model_id in range(0, 1030):
        try:
            process_model(model_id, input_base_path, output_base_path)
            print(f"Processed model {model_id:05d}")
        except Exception as e:
            print(f"Error processing model {model_id:05d}: {str(e)}")

if __name__ == "__main__":
    main()