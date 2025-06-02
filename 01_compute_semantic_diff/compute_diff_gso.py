import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import argparse

class ModelConfig:
    """Configuration class for different models"""
    def __init__(self, model_type='dino'):
        self.model_type = model_type.lower()
        self.feature_size = 16 if self.model_type == 'dino' else 7
        self.model = None
        self.preprocess = None
        self._setup_model_and_preprocessing()
    
    def _setup_model_and_preprocessing(self):
        """Setup model and preprocessing based on model type"""
        if self.model_type == 'dino':
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').cuda()
            self.model.eval()
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif self.model_type == 'vgg':
            self.model = models.vgg16(pretrained=True).features.eval().cuda()
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            raise ValueError("Model type must be 'dino' or 'vgg'")

def extract_features(image_path, config):
    """Extract features from an image using the specified model configuration"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = config.preprocess(image).unsqueeze(0).cuda()
    
    with torch.no_grad():
        if config.model_type == 'dino':
            features = config.model.get_intermediate_layers(image_tensor, n=1)[0]
            features = features.permute(0, 2, 1).reshape(1, -1, 16, 16)
        elif config.model_type == 'vgg':
            features = config.model(image_tensor)
    
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features

def compute_difference(feat_obj, feat_obj_slice, metric='cosine'):
    """Compute the semantic difference between two feature maps"""
    if metric == 'cosine':
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        diff_map = 1 - cos(feat_obj, feat_obj_slice)
    elif metric == 'euclidean':
        diff_map = torch.norm(feat_obj - feat_obj_slice, p=2, dim=1)
    return diff_map.squeeze().cpu().numpy()

def upsample_diff_map(diff_map, target_size=(224, 224)):
    """Upsample the difference map back to the input resolution"""
    diff_map_tensor = torch.tensor(diff_map).unsqueeze(0).unsqueeze(0)
    upsampled_diff_map = F.interpolate(diff_map_tensor, size=target_size, mode='bilinear', align_corners=False)
    return upsampled_diff_map.squeeze().cpu().numpy()

def downsample_masks(masks, target_size=(7, 7)):
    """Downsample masks to target size"""
    masks_tensor = torch.tensor(masks.astype(float)).unsqueeze(0).unsqueeze(0)
    downsampled_masks = F.interpolate(masks_tensor, size=target_size, mode='bilinear', align_corners=False)
    return downsampled_masks.squeeze().cpu().numpy()

def create_non_black_mask(image_path, target_size=(224, 224)):
    """Create a mask where img_obj_slice is not black"""
    image = Image.open(image_path)
    image = image.resize(target_size)

    if image.mode == 'RGBA':
        image_alpha = np.array(image)[:, :, 3]
        non_black_mask = (image_alpha != 0)
    else:
        image = image.convert("RGB")
        image_np = np.array(image)
        non_black_mask = np.any(image_np != [0, 0, 0], axis=-1)
    return non_black_mask

def overlay_heatmap_with_slice_mask(path_obj_slice_img, heatmap, save_path, image_alpha=0.5, heatmap_alpha=0.7, show_colorbar=False):
    """Overlay masked heatmap on object_img, showing only where object_slice_img is not black"""
    img_obj_slice = Image.open(path_obj_slice_img).convert("RGB")
    img_obj_slice = img_obj_slice.resize((224, 224))

    non_black_mask = create_non_black_mask(path_obj_slice_img)
    masked_heatmap = np.ma.masked_where(~non_black_mask, heatmap)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_obj_slice, alpha=image_alpha)
    heatmap_plot = plt.imshow(masked_heatmap, cmap='hot', interpolation='nearest', alpha=heatmap_alpha)

    if show_colorbar:
        plt.colorbar(heatmap_plot, orientation='vertical', fraction=0.046, pad=0.04)

    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def visualize_3d(array):
    """Visualize 3D array as subcubes"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    subcube_size = 1
    
    def on_surface(i, j, k):
        return i == 0 or i == array.shape[0] - 1 or j == 0 or j == array.shape[1] - 1 or k == 0 or k == array.shape[2] - 1

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                x = k * subcube_size
                y = i * subcube_size
                z = (array.shape[1] - 1 - j) * subcube_size
                if array[i, j, k] > 0.78 and on_surface(i, j, k):
                    print(f'No.{str(i)} slice, pos({str(j)}, {str(k)})')
                    print(x, y, z)
                    ax.bar3d(x, y, z, subcube_size, subcube_size, subcube_size, color='yellow', alpha=0.9)
                else:
                    ax.bar3d(x, y, z, subcube_size, subcube_size, subcube_size, color='red', alpha=0.03)

    ax.set_xlim(0, array.shape[1])
    ax.set_ylim(0, array.shape[0])
    z_ticks = np.arange(0, 5, 1)
    plt.yticks(z_ticks)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis')
    ax.set_zlabel('Y axis')
    ax.set_title('3D Visualization of Subcubes')
    plt.savefig('3d_grid_visualization.png')
    plt.show()

def calculate_bounding_box(image_path):
    """Calculate bounding box of non-transparent pixels"""
    image = Image.open(image_path).convert("RGBA")
    data = np.array(image)
    alpha_channel = data[:, :, 3]
    mask = alpha_channel > 0
    non_zero_indices = np.argwhere(mask)

    if non_zero_indices.size == 0:
        return None

    y_min, x_min = non_zero_indices.min(axis=0)
    y_max, x_max = non_zero_indices.max(axis=0)
    bounding_box = (x_min, y_min, x_max + 1, y_max + 1)
    return bounding_box

def main(model_type='vgg'):
    """Main function with model type selection"""
    print(f"Using {model_type.upper()} model with {16 if model_type == 'dino' else 7}x{16 if model_type == 'dino' else 7} features")
    
    # Initialize model configuration
    config = ModelConfig(model_type)
    
    # Directory setup
    dir_root = './00_data'
    dir_objs = f'{dir_root}/GSO_rm_bg'
    dir_objs_slices = f'{dir_root}/img_slices'
    dir_ret = f'{dir_root}/diff_blocks_{model_type}_normalized'

    obj_fnames = os.listdir(dir_objs)
    obj_fnames.sort()

    for obj_id, obj_fname in enumerate(obj_fnames):
        case_uid = "%05d" % obj_id
        print(f"Processing {case_uid} with {model_type.upper()}")

        path_obj_img = f"{dir_objs}/{obj_fname}"
        bounding_box = calculate_bounding_box(path_obj_img)
        os.makedirs(f"{dir_ret}/{case_uid}", exist_ok=True)

        diff_block_list = []

        for slice_z_idx in range(1, 5):
            path_obj_slice_img = f"{dir_objs_slices}/{case_uid}/Z_{str(slice_z_idx)}.png"
            overlay_save_path = f"{dir_ret}/{case_uid}/image{str(slice_z_idx)}.png"

            # Extract features using the configured model
            feat_obj = extract_features(path_obj_img, config)
            feat_obj_slice = extract_features(path_obj_slice_img, config)

            # Compute semantic difference
            semantic_diff = compute_difference(feat_obj, feat_obj_slice, metric='cosine')

            # Create non-black mask with appropriate size
            mask_size = (config.feature_size, config.feature_size)
            non_black_mask = create_non_black_mask(path_obj_slice_img, mask_size).astype(float)

            # Mask the semantic difference
            semantic_diff_masked = non_black_mask * semantic_diff
            diff_block_list.append(semantic_diff_masked[None, ...])

            # 2D visualization
            show_2d = True
            if show_2d:
                upsampled_diff_map = upsample_diff_map(semantic_diff, target_size=(224, 224))
                overlay_heatmap_with_slice_mask(path_obj_slice_img, upsampled_diff_map, overlay_save_path, 
                                              image_alpha=0.5, heatmap_alpha=0.7)

        # Process 3D block
        diff_3d_block = np.concatenate(diff_block_list, axis=0)
        
        # Calculate downsampling factor based on feature size
        downsampling_factor = Image.open(path_obj_img).size[0] / config.feature_size

        # Clamp coordinates to fit within feature map dimensions
        x_min = max(0, int(bounding_box[0] / downsampling_factor))
        y_min = max(0, int(bounding_box[1] / downsampling_factor))
        x_max = min(config.feature_size, int(bounding_box[2] / downsampling_factor))
        y_max = min(config.feature_size, int(bounding_box[3] / downsampling_factor))

        # Crop feature maps
        cropped_feature_maps = diff_3d_block[:, y_min:y_max+1, x_min:x_max+1]
        np.save(f"{dir_ret}/{case_uid}/diff_blocks.npy", cropped_feature_maps)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Feature extraction with DINO or VGG models')
    parser.add_argument('--model', type=str, default='vgg', choices=['dino', 'vgg'],
                        help='Choose model type: dino (16x16 features) or vgg (7x7 features). Default: vgg')
    
    args = parser.parse_args()
    
    print(f"Starting feature extraction with {args.model.upper()} model")
    main(args.model)
    print(f"Completed feature extraction with {args.model.upper()} model")