import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import argparse
import clip

class CLIPConfig:
    """Configuration class for CLIP model"""
    def __init__(self, model_name='ViT-B/32'):
        self.model_name = model_name
        self.feature_size = self._get_feature_size(model_name)
        self.model = None
        self.preprocess = None
        self._setup_model_and_preprocessing()
    
    def _get_feature_size(self, model_name):
        """Get feature map size based on CLIP model variant"""
        if 'ViT-B/32' in model_name:
            return 7  # 7x7 patch grid for ViT-B/32
        elif 'ViT-B/16' in model_name:
            return 14  # 14x14 patch grid for ViT-B/16
        elif 'ViT-L/14' in model_name:
            return 16  # 16x16 patch grid for ViT-L/14
        elif 'RN50' in model_name or 'RN101' in model_name:
            return 7   # ResNet models typically give 7x7 features
        else:
            return 7   # Default fallback
    
    def _setup_model_and_preprocessing(self):
        """Setup CLIP model and preprocessing"""
        self.model, self.preprocess = clip.load(self.model_name, device='cuda')
        self.model.eval()
        # Convert model to float32 to avoid precision issues
        self.model.float()

def extract_clip_features(image_path, config):
    """Extract visual features from an image using CLIP"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = config.preprocess(image).unsqueeze(0).cuda().float()
    
    with torch.no_grad():
        # Get image features from CLIP
        image_features = config.model.encode_image(image_tensor)
        
        # For patch-based models (ViT), we need to extract patch features
        if 'ViT' in config.model_name:
            # Access the vision transformer directly
            vision_model = config.model.visual
            
            # Extract patch embeddings before pooling
            x = vision_model.conv1(image_tensor)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            
            # Add class token and positional encoding
            x = torch.cat([vision_model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
            x = x + vision_model.positional_embedding.to(x.dtype)
            x = vision_model.ln_pre(x)
            
            # Pass through transformer layers
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = vision_model.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            
            # Remove class token and reshape to spatial grid
            patch_features = x[:, 1:, :]  # Remove class token
            grid_size = int(patch_features.shape[1] ** 0.5)
            patch_features = patch_features.reshape(1, grid_size, grid_size, -1)
            patch_features = patch_features.permute(0, 3, 1, 2)  # [1, dim, H, W]
            
        else:
            # For ResNet models, extract features from intermediate layers
            vision_model = config.model.visual
            
            def forward_resnet_features(x):
                x = vision_model.relu1(vision_model.bn1(vision_model.conv1(x)))
                x = vision_model.relu2(vision_model.bn2(vision_model.conv2(x)))
                x = vision_model.relu3(vision_model.bn3(vision_model.conv3(x)))
                x = vision_model.avgpool1(x)
                
                x = vision_model.layer1(x)
                x = vision_model.layer2(x)
                x = vision_model.layer3(x)
                x = vision_model.layer4(x)
                
                return x
            
            patch_features = forward_resnet_features(image_tensor)
    
    # Normalize features
    patch_features = torch.nn.functional.normalize(patch_features, p=2, dim=1)
    return patch_features

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

def main(model_name='ViT-B/32'):
    """Main function with CLIP model selection"""
    print(f"Using CLIP model: {model_name}")
    
    # Initialize CLIP configuration
    config = CLIPConfig(model_name)
    print(f"Feature map size: {config.feature_size}x{config.feature_size}")
    
    # Directory setup
    dir_root = './00_data'
    dir_objs = f'{dir_root}/GSO_rm_bg'
    dir_objs_slices = f'{dir_root}/img_slices'
    model_safe_name = model_name.replace('/', '_')
    dir_ret = f'{dir_root}/diff_blocks_clip_{model_safe_name}_normalized'

    obj_fnames = os.listdir(dir_objs)
    obj_fnames.sort()

    for obj_id, obj_fname in enumerate(obj_fnames):
        case_uid = "%05d" % obj_id
        print(f"Processing {case_uid} with CLIP {model_name}")

        path_obj_img = f"{dir_objs}/{obj_fname}"
        bounding_box = calculate_bounding_box(path_obj_img)
        os.makedirs(f"{dir_ret}/{case_uid}", exist_ok=True)

        diff_block_list = []

        for slice_z_idx in range(1, 5):
            path_obj_slice_img = f"{dir_objs_slices}/{case_uid}/Z_{str(slice_z_idx)}.png"
            overlay_save_path = f"{dir_ret}/{case_uid}/image{str(slice_z_idx)}.png"

            # Extract features using CLIP
            feat_obj = extract_clip_features(path_obj_img, config)
            feat_obj_slice = extract_clip_features(path_obj_slice_img, config)

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
    parser = argparse.ArgumentParser(description='Feature extraction with CLIP models')
    parser.add_argument('--model', type=str, default='ViT-B/32', 
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64'],
                        help='Choose CLIP model variant. Default: ViT-B/32')
    
    args = parser.parse_args()
    
    print(f"Starting feature extraction with CLIP {args.model}")
    main(args.model)
    print(f"Completed feature extraction with CLIP {args.model}") 