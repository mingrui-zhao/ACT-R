#!/usr/bin/env python3
"""
Camera Parameters Generator

This script generates camera projection matrices and transformation parameters
from trajectory data. It processes predicted trajectories and converts them
into camera parameters suitable for 3D rendering and view synthesis.
"""

import os
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class CameraSettings:
    """Camera intrinsic and configuration parameters."""
    focal_length_mm: float = 59.24
    sensor_size_mm: float = 36.0
    pixel_aspect_ratio: float = 1.0
    resolution_percent: float = 100.0
    skew: float = 0.0
    image_width: int = 576
    image_height: int = 576


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters."""
    radius_scale: float = np.sqrt(3) / 2
    default_distance: float = 2.0
    rotation_angle: float = -np.pi / 2


class BlenderProjectionCalculator:
    """Calculates 3D to 2D projection matrices using Blender-compatible parameters."""
    
    def __init__(self, camera_settings: CameraSettings):
        self.settings = camera_settings
        
        # Blender camera rotation matrix (fixed)
        self.cam_rotation = np.asarray([
            [1.910685676922942e-15, 4.371138828673793e-08, 1.0],
            [1.0, -4.371138828673793e-08, -0.0],
            [4.371138828673793e-08, 1.0, -4.371138828673793e-08]
        ])

    def calculate_intrinsic_matrix(self) -> np.ndarray:
        """Calculate camera intrinsic matrix K."""
        scale = self.settings.resolution_percent / 100
        
        # Calculate focal lengths in pixels
        f_u = (self.settings.focal_length_mm * self.settings.image_width * scale 
               / self.settings.sensor_size_mm)
        f_v = (self.settings.focal_length_mm * self.settings.image_height * scale 
               * self.settings.pixel_aspect_ratio / self.settings.sensor_size_mm)
        
        # Calculate principal points
        u_0 = self.settings.image_width * scale / 2
        v_0 = self.settings.image_height * scale / 2
        
        # Construct intrinsic matrix
        K = np.matrix([
            [f_u, self.settings.skew, u_0],
            [0, f_v, v_0],
            [0, 0, 1]
        ])
        
        return K

    def calculate_extrinsic_matrix(self, azimuth: float, elevation: float, distance: float) -> np.ndarray:
        """Calculate camera extrinsic matrix [R|t]."""
        # Calculate rotation matrices
        sa, ca = np.sin(-azimuth), np.cos(-azimuth)
        se, ce = np.sin(-elevation), np.cos(-elevation)
        
        # World to object coordinate transformation
        R_world2obj = np.transpose(np.matrix([
            [ca * ce, -sa, ca * se],
            [sa * ce, ca, sa * se],
            [-se, 0, ce]
        ]))

        # Object to camera coordinate transformation
        R_obj2cam = np.transpose(np.matrix(self.cam_rotation))
        R_world2cam = R_obj2cam * R_world2obj
        
        # Calculate translation
        cam_location = np.transpose(np.matrix([distance, 0, 0]))
        T_world2cam = -1 * R_obj2cam * cam_location

        # Fix Blender's coordinate system (flip Y and Z axes)
        R_camfix = np.matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        R_world2cam = R_camfix * R_world2cam
        T_world2cam = R_camfix * T_world2cam

        # Combine rotation and translation
        RT = np.hstack((R_world2cam, T_world2cam))
        
        return RT

    def get_projection_matrices(self, azimuth: float, elevation: float, distance: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate complete projection matrices.
        
        Args:
            azimuth: Azimuth angle in radians
            elevation: Elevation angle in radians
            distance: Distance from camera to object center
            
        Returns:
            Tuple of (intrinsic_matrix, extrinsic_matrix)
        """
        K = self.calculate_intrinsic_matrix()
        RT = self.calculate_extrinsic_matrix(azimuth, elevation, distance)
        
        return K, RT


class TransformationMatrixGenerator:
    """Generates various transformation matrices for 3D rendering."""
    
    @staticmethod
    def get_rotation_matrix(rotation_angle: float) -> np.ndarray:
        """Generate combined rotation matrix."""
        cos_val = np.cos(rotation_angle)
        sin_val = np.sin(rotation_angle)

        # Individual rotation matrices
        rotation_x = np.array([
            [1, 0, 0, 0],
            [0, cos_val, -sin_val, 0],
            [0, sin_val, cos_val, 0],
            [0, 0, 0, 1]
        ])

        rotation_y = np.array([
            [cos_val, 0, sin_val, 0],
            [0, 1, 0, 0],
            [-sin_val, 0, cos_val, 0],
            [0, 0, 0, 1]
        ])

        rotation_z = np.array([
            [cos_val, -sin_val, 0, 0],
            [sin_val, cos_val, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Scale and negation matrices
        scale_y_neg = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        neg_matrix = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        # Combine transformations
        return np.linalg.multi_dot([
            neg_matrix, rotation_z, rotation_z, scale_y_neg, rotation_x
        ])

    @staticmethod
    def get_world_to_object_matrix(offset: Tuple[float, float, float]) -> np.ndarray:
        """Generate world to object transformation matrix."""
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = offset
        return matrix


class TrajectoryLoader:
    """Handles loading and processing of trajectory data."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug

    def load_trajectory_from_file(self, trajectory_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load trajectory data from saved NPY file.
        
        Args:
            trajectory_path: Path to trajectory file
            
        Returns:
            Tuple of (elevations_rad, azimuths_rad) or (None, None) if failed
        """
        try:
            if self.debug:
                print(f"  Loading trajectory from: {trajectory_path}")
                
            trajectory_data = np.load(trajectory_path)
            
            if trajectory_data.shape[0] != 2:
                raise ValueError(f"Expected trajectory shape (2, N), got {trajectory_data.shape}")
                
            # Extract elevation and azimuth angles in degrees
            elevations_deg = trajectory_data[0]
            azimuths_deg = trajectory_data[1]
            
            # Convert to radians
            elevations_rad = np.radians(elevations_deg)
            azimuths_rad = np.radians(azimuths_deg)
            
            if self.debug:
                print(f"  Loaded {len(elevations_rad)} trajectory points")
                print(f"  Elevation range: {np.degrees(elevations_rad.min()):.1f}° to {np.degrees(elevations_rad.max()):.1f}°")
                print(f"  Azimuth range: {np.degrees(azimuths_rad.min()):.1f}° to {np.degrees(azimuths_rad.max()):.1f}°")
            
            return elevations_rad, azimuths_rad
            
        except Exception as e:
            print(f"  Error loading trajectory from {trajectory_path}: {e}")
            return None, None


class CameraParametersGenerator:
    """Main class for generating camera parameters from trajectories."""
    
    def __init__(self, camera_settings: CameraSettings, processing_config: ProcessingConfig, debug: bool = False):
        self.camera_settings = camera_settings
        self.processing_config = processing_config
        self.debug = debug
        
        self.projection_calc = BlenderProjectionCalculator(camera_settings)
        self.transform_gen = TransformationMatrixGenerator()
        self.trajectory_loader = TrajectoryLoader(debug)

    def generate_camera_parameters(self, elevations: np.ndarray, azimuths: np.ndarray, 
                                 distance: float) -> Dict[str, np.ndarray]:
        """
        Generate camera parameters for a trajectory.
        
        Args:
            elevations: Elevation angles in radians
            azimuths: Azimuth angles in radians
            distance: Camera distance from object center
            
        Returns:
            Dictionary containing camera matrices for all views
        """
        cam_dict = {}
        
        if self.debug:
            print(f"  Generating parameters for {len(elevations)} camera positions")
            print(f"  Camera distance: {distance:.3f}")
        
        # Pre-calculate transformation matrices
        rotation_matrix = self.transform_gen.get_rotation_matrix(self.processing_config.rotation_angle)
        world_to_object_matrix = self.transform_gen.get_world_to_object_matrix((0, 0, 0))
        scale_matrix = np.diag([self.processing_config.radius_scale] * 3 + [1.0]).astype(np.float32)
        scale_matrix_inv = np.linalg.inv(scale_matrix)
        
        for i, (elevation, azimuth) in enumerate(zip(elevations, azimuths)):
            # Calculate projection matrices
            K, RT = self.projection_calc.get_projection_matrices(-azimuth, elevation, distance)
            
            # Compute complete transformation matrix
            transformation_matrix = np.linalg.multi_dot([K, RT, rotation_matrix, world_to_object_matrix])
            transformation_matrix = np.vstack([transformation_matrix, [0, 0, 0, 1]])
            
            # Store matrices for this view
            cam_dict[f'world_mat_{i}'] = transformation_matrix
            cam_dict[f'world_mat_inv_{i}'] = np.linalg.inv(transformation_matrix)
            cam_dict[f'scale_mat_{i}'] = scale_matrix
            cam_dict[f'scale_mat_inv_{i}'] = scale_matrix_inv
        
        if self.debug:
            print(f"  Generated {len(cam_dict) // 4} camera parameter sets")
        
        return cam_dict

    def extract_subset_views(self, cam_dict_full: Dict[str, np.ndarray], 
                           indices: List[int]) -> Dict[str, np.ndarray]:
        """
        Extract a subset of views from full camera parameters.
        
        Args:
            cam_dict_full: Full camera parameters dictionary
            indices: List of view indices to extract
            
        Returns:
            Dictionary containing subset of camera parameters
        """
        subset_dict = {}
        
        for new_idx, original_idx in enumerate(indices):
            for matrix_type in ['world_mat', 'world_mat_inv', 'scale_mat', 'scale_mat_inv']:
                original_key = f'{matrix_type}_{original_idx}'
                new_key = f'{matrix_type}_{new_idx}'
                
                if original_key in cam_dict_full:
                    subset_dict[new_key] = cam_dict_full[original_key]
                else:
                    print(f"Warning: Missing key {original_key} in camera dictionary")
        
        return subset_dict

    def process_single_object(self, case_uid: str, gso_params_dir: Path, 
                            trajectory_dir: Path, output_dir: Path,
                            trajectory_filename: str = "predicted_trajectory.npy") -> bool:
        """
        Process camera parameters for a single object.
        
        Args:
            case_uid: Object case identifier
            gso_params_dir: Directory containing GSO camera parameters
            trajectory_dir: Directory containing trajectory files
            output_dir: Output directory for camera parameters
            trajectory_filename: Name of trajectory file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.debug:
                print(f"Processing object {case_uid}")
            
            # Load camera parameters from GSO
            cam_params_path = gso_params_dir / case_uid / 'cam_param_est.npy'
            if not cam_params_path.exists():
                print(f"  Camera parameters not found: {cam_params_path}")
                return False
                
            cam_params = np.load(cam_params_path)
            distance = cam_params[-2]  # Second to last entry is distance
            
            if self.debug:
                print(f"  Loaded camera parameters, distance: {distance:.3f}")
            
            # Load trajectory
            trajectory_path = trajectory_dir / case_uid / trajectory_filename
            if not trajectory_path.exists():
                print(f"  Trajectory file not found: {trajectory_path}")
                return False
                
            elevations_rad, azimuths_rad = self.trajectory_loader.load_trajectory_from_file(trajectory_path)
            
            if elevations_rad is None or azimuths_rad is None:
                return False
            
            # Generate camera parameters
            cam_dict = self.generate_camera_parameters(elevations_rad, azimuths_rad, distance)
            
            # Create output directory and save
            output_case_dir = output_dir / case_uid / "21_views"
            output_case_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_case_dir / "cameras_sphere.npz"
            np.savez(output_file, **cam_dict)
            
            if self.debug:
                print(f"  Saved camera parameters to: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"  Error processing object {case_uid}: {e}")
            return False

    def process_batch(self, gso_params_dir: Path, trajectory_dir: Path, output_dir: Path,
                     max_cases: Optional[int] = None, trajectory_filename: str = "predicted_trajectory.npy") -> dict:
        """
        Process camera parameters for multiple objects.
        
        Args:
            gso_params_dir: Directory containing GSO camera parameters
            trajectory_dir: Directory containing trajectory files
            output_dir: Output directory for camera parameters
            max_cases: Maximum number of cases to process
            trajectory_filename: Name of trajectory files
            
        Returns:
            Dictionary with processing statistics
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped_no_params': 0,
            'skipped_no_trajectory': 0
        }
        
        case_id = 0
        while True:
            if max_cases and stats['processed'] >= max_cases:
                break
                
            case_uid = f"{case_id:05d}"
            
            # Check for camera parameters
            cam_params_path = gso_params_dir / case_uid / 'cam_param_est.npy'
            if not cam_params_path.exists():
                case_id += 1
                if case_id > 99999:
                    break
                stats['skipped_no_params'] += 1
                continue
            
            # Check for trajectory file
            trajectory_path = trajectory_dir / case_uid / trajectory_filename
            if not trajectory_path.exists():
                case_id += 1
                stats['skipped_no_trajectory'] += 1
                continue
            
            print(f"Processing case {case_uid}")
            
            # Process the case
            success = self.process_single_object(
                case_uid, gso_params_dir, trajectory_dir, output_dir, trajectory_filename
            )
            
            if success:
                stats['successful'] += 1
                print(f"  ✓ Successfully generated camera parameters")
            else:
                stats['failed'] += 1
                print(f"  ✗ Failed to generate camera parameters")
            
            stats['processed'] += 1
            case_id += 1
            
            if case_id > 99999:
                break
        
        return stats


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate camera parameters from predicted trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output paths
    parser.add_argument('--gso-params-dir', type=str, default='./00_data/cmt_est/gso',
                       help='Directory containing GSO camera parameters')
    parser.add_argument('--trajectory-dir', type=str, default='./06_results/predicted_path',
                       help='Directory containing trajectory files')
    parser.add_argument('--output-dir', type=str, default='./06_results/generated_videos',
                       help='Output directory for camera parameters')
    parser.add_argument('--trajectory-filename', type=str, default='predicted_trajectory.npy',
                       help='Name of trajectory files to process')
    
    # Camera settings
    parser.add_argument('--focal-length', type=float, default=59.24,
                       help='Camera focal length in mm')
    parser.add_argument('--sensor-size', type=float, default=36.0,
                       help='Camera sensor size in mm')
    parser.add_argument('--image-width', type=int, default=576,
                       help='Output image width in pixels')
    parser.add_argument('--image-height', type=int, default=576,
                       help='Output image height in pixels')
    
    # Processing options
    parser.add_argument('--max-cases', type=int, default=None,
                       help='Maximum number of cases to process')
    parser.add_argument('--case-range', type=str, default=None,
                       help='Process specific case range (e.g., "0-100")')
    parser.add_argument('--radius-scale', type=float, default=None,
                       help=f'Radius scale factor (default: {np.sqrt(3)/2:.3f})')
    
    # Debug and output
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress non-essential output')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without generating files')
    
    return parser.parse_args()


def print_statistics(stats: dict, quiet: bool = False):
    """Print processing statistics."""
    if quiet:
        return
        
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total processed:        {stats['processed']}")
    print(f"Successful:             {stats['successful']}")
    print(f"Failed:                 {stats['failed']}")
    print(f"Skipped (no params):    {stats['skipped_no_params']}")
    print(f"Skipped (no trajectory): {stats['skipped_no_trajectory']}")
    
    if stats['processed'] > 0:
        success_rate = (stats['successful'] / stats['processed']) * 100
        print(f"Success rate:           {success_rate:.1f}%")
    
    print("="*50)


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup paths
    gso_params_dir = Path(args.gso_params_dir)
    trajectory_dir = Path(args.trajectory_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input paths
    if not gso_params_dir.exists():
        print(f"Error: GSO parameters directory does not exist: {gso_params_dir}")
        return 1
    
    if not trajectory_dir.exists():
        print(f"Error: Trajectory directory does not exist: {trajectory_dir}")
        return 1
    
    # Setup configurations
    camera_settings = CameraSettings(
        focal_length_mm=args.focal_length,
        sensor_size_mm=args.sensor_size,
        image_width=args.image_width,
        image_height=args.image_height
    )
    
    processing_config = ProcessingConfig()
    if args.radius_scale is not None:
        processing_config.radius_scale = args.radius_scale
    
    # Initialize generator
    generator = CameraParametersGenerator(
        camera_settings=camera_settings,
        processing_config=processing_config,
        debug=args.debug
    )
    
    if not args.quiet:
        print("Camera Parameters Generator")
        print(f"GSO parameters: {gso_params_dir}")
        print(f"Trajectories: {trajectory_dir}")
        print(f"Output: {output_dir}")
        print(f"Camera settings: {camera_settings}")
        print("-" * 50)
    
    if args.dry_run:
        print("DRY RUN: Would process camera parameters but not generate files")
        return 0
    
    # Process camera parameters
    try:
        stats = generator.process_batch(
            gso_params_dir=gso_params_dir,
            trajectory_dir=trajectory_dir,
            output_dir=output_dir,
            max_cases=args.max_cases,
            trajectory_filename=args.trajectory_filename
        )
        
        print_statistics(stats, args.quiet)
        
        if stats['failed'] > 0:
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())