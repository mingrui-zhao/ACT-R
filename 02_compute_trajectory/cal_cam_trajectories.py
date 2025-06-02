#!/usr/bin/env python3
"""
Camera Trajectory Generator

This script generates camera trajectories from optimization results using mirrored 
segments to form closed orbits. It processes incremental path data and converts
it into smooth trajectory sequences for camera movement.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from utils import cal_sphere_coord


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation parameters."""
    azimuth_step_deg: float = 18.0
    path_steps: int = 21
    segment_length: int = 5


class CoordinateTransformer:
    """Handles coordinate transformations between world and camera spaces."""
    
    @staticmethod
    def world_to_camera(coordinate: np.ndarray, inverse_world_matrix: np.ndarray) -> np.ndarray:
        """
        Transform world coordinates to camera coordinates.
        
        Args:
            coordinate: 3D coordinate in world space
            inverse_world_matrix: Inverse transformation matrix
            
        Returns:
            3D coordinate in camera space
        """
        def transform_coordinate(matrix: np.ndarray, coord: np.ndarray) -> np.ndarray:
            # Convert to homogeneous coordinates
            coord_homogeneous = np.array([*coord, 1])
            # Perform transformation
            transformed_coord = matrix @ coord_homogeneous
            # Convert back to 3D by dividing by homogeneous component
            return transformed_coord[:3] / transformed_coord[3]
        
        return transform_coordinate(inverse_world_matrix, coordinate)


class TrajectoryGenerator:
    """Generates camera trajectories from optimization increments."""
    
    def __init__(self, config: TrajectoryConfig, debug: bool = False):
        self.config = config
        self.debug = debug
        self.transformer = CoordinateTransformer()

    def generate_trajectory_from_increments(self, ele_init: float, increments: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Generate a trajectory with mirrored segments to form a closed orbit.

        Args:
            ele_init: Initial elevation angle in degrees
            increments: Array of two values [increment1, increment2] for segments

        Returns:
            Tuple of (elevations_deg, azimuths_deg)
        """
        if len(increments) != 2:
            raise ValueError(f"Expected 2 increments, got {len(increments)}")
            
        increment1, increment2 = increments
        
        if self.debug:
            print(f"Generating trajectory with increments: [{increment1}, {increment2}]")
            print(f"Initial elevation: {ele_init}°")

        # Generate elevation changes for each segment
        segment1_changes = [increment1] * self.config.segment_length
        segment2_changes = [increment2] * self.config.segment_length
        
        # Build elevation trajectory
        elevations_deg = self._build_elevation_sequence(
            ele_init, segment1_changes, segment2_changes
        )
        
        # Generate evenly spaced azimuth angles (0° to 360°)
        azimuths_deg = self._generate_azimuth_sequence()
        
        if self.debug:
            print(f"Generated {len(elevations_deg)} elevation points")
            print(f"Elevation range: {min(elevations_deg):.1f}° to {max(elevations_deg):.1f}°")
            print(f"Generated {len(azimuths_deg)} azimuth points")
        
        return elevations_deg, azimuths_deg

    def _build_elevation_sequence(self, ele_init: float, segment1_changes: List[float], 
                                 segment2_changes: List[float]) -> List[float]:
        """Build the complete elevation sequence with mirrored segments."""
        elevations = [ele_init]
        
        # First segment (first quarter)
        for change in segment1_changes:
            elevations.append(elevations[-1] + change)
        
        # Second segment (second quarter)
        for change in segment2_changes:
            elevations.append(elevations[-1] + change)
        
        # Third segment (mirrors second segment in reverse with negated changes)
        for change in reversed(segment2_changes):
            elevations.append(elevations[-1] - change)
        
        # Fourth segment (mirrors first segment in reverse with negated changes)
        for change in reversed(segment1_changes):
            elevations.append(elevations[-1] - change)
        
        return elevations

    def _generate_azimuth_sequence(self) -> List[float]:
        """Generate evenly spaced azimuth angles for a complete orbit."""
        return [(self.config.azimuth_step_deg * i) % 360 
                for i in range(self.config.path_steps)]

    def calculate_world_coordinates(self, elevations_deg: List[float], azimuths_deg: List[float], 
                                  radius: float, inv_world_matrix: Optional[np.ndarray] = None) -> Tuple[List, List, List, List]:
        """
        Calculate camera trajectory coordinates in world space.
        
        Args:
            elevations_deg: Elevation angles in degrees
            azimuths_deg: Azimuth angles in degrees
            radius: Sphere radius for camera movement
            inv_world_matrix: Optional inverse world transformation matrix
            
        Returns:
            Tuple of (trajectory_list, x_coords, y_coords, z_coords)
        """
        # Calculate spherical coordinates
        x_trajectory, y_trajectory, z_trajectory = cal_sphere_coord(
            [0, 0, 0], radius, elevations_deg, azimuths_deg
        )
        
        trajectory_list = []
        ret_x, ret_y, ret_z = [], [], []
        
        for idx in range(len(elevations_deg)):
            world_coord = [x_trajectory[idx], y_trajectory[idx], z_trajectory[idx]]
            
            if inv_world_matrix is not None:
                # Transform to camera coordinates
                coord_transformed = self.transformer.world_to_camera(world_coord, inv_world_matrix)
                # Reorder coordinates for visualization
                coord_final = [coord_transformed[0], coord_transformed[2], coord_transformed[1]]
                trajectory_list.append(coord_final)
                ret_x.append(coord_transformed[0])
                ret_y.append(coord_transformed[2])
                ret_z.append(coord_transformed[1])
            else:
                # Use world coordinates directly
                trajectory_list.append(world_coord)
                ret_x.append(world_coord[0])
                ret_y.append(world_coord[1])
                ret_z.append(world_coord[2])
        
        return trajectory_list, ret_x, ret_y, ret_z


class TrajectoryProcessor:
    """Processes trajectory generation for multiple cases."""
    
    def __init__(self, config: TrajectoryConfig, debug: bool = False):
        self.config = config
        self.debug = debug
        self.generator = TrajectoryGenerator(config, debug)

    def process_single_case(self, case_uid: str, increments: np.ndarray, 
                           arr_cam_params: np.ndarray, output_dir: Path,
                           output_filename: str = "trajectory.npy") -> bool:
        """
        Process trajectory generation for a single case.
        
        Args:
            case_uid: Case identifier
            increments: Path optimization increments
            arr_cam_params: Camera parameters array
            output_dir: Output directory for trajectory files
            output_filename: Name of output file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract camera parameters
            distance = arr_cam_params[-2]  # Distance from camera to center
            ele_init_rad = arr_cam_params[-1]  # Initial elevation in radians
            ele_init = np.degrees(ele_init_rad)
            
            if self.debug:
                print(f"  Camera distance: {distance:.2f}")
                print(f"  Initial elevation: {ele_init:.2f}°")
            
            # Generate trajectory
            elevations_deg, azimuths_deg = self.generator.generate_trajectory_from_increments(
                ele_init, increments
            )
            
            # Create output directory
            case_output_dir = output_dir / case_uid
            case_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save trajectory data
            trajectory_data = np.array([elevations_deg, azimuths_deg])
            output_path = case_output_dir / output_filename
            np.save(output_path, trajectory_data)
            
            if self.debug:
                print(f"  Saved trajectory to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"  Error generating trajectory for {case_uid}: {e}")
            return False

    def process_batch(self, input_dir: Path, cam_params_dir: Path, output_dir: Path,
                     max_cases: Optional[int] = None, output_filename: str = "trajectory.npy") -> dict:
        """
        Process trajectory generation for multiple cases.
        
        Args:
            input_dir: Directory containing best_path.npy files
            cam_params_dir: Directory containing camera parameter files
            output_dir: Output directory for trajectory files
            max_cases: Maximum number of cases to process
            output_filename: Name of output files
            
        Returns:
            Dictionary with processing statistics
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped_no_path': 0,
            'skipped_no_params': 0
        }
        
        case_id = 0
        while True:
            if max_cases and stats['processed'] >= max_cases:
                break
                
            case_uid = f"{case_id:05d}"
            
            # Check for best_path.npy file
            best_path_file = input_dir / case_uid / 'best_path.npy'
            if not best_path_file.exists():
                case_id += 1
                if case_id > 99999:  # Reasonable upper limit
                    break
                stats['skipped_no_path'] += 1
                continue
            
            print(f"Processing case {case_uid}")
            
            # Check for camera parameters
            cam_params_file = cam_params_dir / case_uid / 'cam_param_est.npy'
            if not cam_params_file.exists():
                print(f"  Camera parameters not found, skipping")
                stats['skipped_no_params'] += 1
                case_id += 1
                continue
            
            try:
                # Load data
                increments = np.load(best_path_file)
                arr_cam_params = np.load(cam_params_file)
                
                if self.debug:
                    print(f"  Loaded increments: {increments}")
                
                # Process case
                success = self.process_single_case(
                    case_uid, increments, arr_cam_params, output_dir, output_filename
                )
                
                if success:
                    stats['successful'] += 1
                    print(f"  ✓ Successfully generated trajectory")
                else:
                    stats['failed'] += 1
                
                stats['processed'] += 1
                
            except Exception as e:
                print(f"  ✗ Error processing case {case_uid}: {e}")
                stats['failed'] += 1
                stats['processed'] += 1
            
            case_id += 1
            if case_id > 99999:
                break
        
        return stats


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate camera trajectories from optimization results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output paths
    parser.add_argument('--data-root', type=str, default='./00_data',
                       help='Root directory for input data')
    parser.add_argument('--cam-params-dir', type=str, default='cmt_est/gso',
                       help='Camera parameters directory (relative to data-root)')
    parser.add_argument('--input-dir', type=str, default='./06_results/predicted_path',
                       help='Input directory containing best_path.npy files')
    parser.add_argument('--output-dir', type=str, default='./06_results/predicted_path',
                       help='Output directory for generated trajectories')
    parser.add_argument('--output-filename', type=str, default='predicted_trajectory.npy',
                       help='Name of output trajectory files')
    
    # Trajectory generation parameters
    parser.add_argument('--azimuth-step', type=float, default=18.0,
                       help='Azimuth step size in degrees')
    parser.add_argument('--path-steps', type=int, default=21,
                       help='Number of steps in trajectory path')
    parser.add_argument('--segment-length', type=int, default=5,
                       help='Length of each trajectory segment')
    
    # Processing options
    parser.add_argument('--max-cases', type=int, default=1030,
                       help='Maximum number of cases to process (None for all)')
    parser.add_argument('--case-range', type=str, default=None,
                       help='Process specific case range (e.g., "0-100" or "500-600")')
    
    # Debugging and output
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress non-essential output')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually generating files')
    
    return parser.parse_args()


def print_statistics(stats: dict, quiet: bool = False):
    """Print processing statistics."""
    if quiet:
        return
        
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total processed:     {stats['processed']}")
    print(f"Successful:          {stats['successful']}")
    print(f"Failed:              {stats['failed']}")
    print(f"Skipped (no path):   {stats['skipped_no_path']}")
    print(f"Skipped (no params): {stats['skipped_no_params']}")
    
    if stats['processed'] > 0:
        success_rate = (stats['successful'] / stats['processed']) * 100
        print(f"Success rate:        {success_rate:.1f}%")
    
    print("="*50)


def parse_case_range(case_range: str) -> Tuple[int, int]:
    """Parse case range string into start and end values."""
    try:
        if '-' in case_range:
            start, end = case_range.split('-', 1)
            return int(start), int(end)
        else:
            # Single case
            case_num = int(case_range)
            return case_num, case_num + 1
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid case range format: {case_range}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup paths
    data_root = Path(args.data_root)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    cam_params_dir = data_root / args.cam_params_dir
    
    # Validate input paths
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1
    
    if not cam_params_dir.exists():
        print(f"Error: Camera parameters directory does not exist: {cam_params_dir}")
        return 1
    
    # Setup configuration
    config = TrajectoryConfig(
        azimuth_step_deg=args.azimuth_step,
        path_steps=args.path_steps,
        segment_length=args.segment_length
    )
    
    # Initialize processor
    processor = TrajectoryProcessor(config, args.debug)
    
    if not args.quiet:
        print("Camera Trajectory Generator")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Camera params: {cam_params_dir}")
        print(f"Configuration: {config}")
        print("-" * 50)
    
    # Handle case range if specified
    max_cases = args.max_cases
    if args.case_range:
        start_case, end_case = parse_case_range(args.case_range)
        max_cases = end_case - start_case
        if not args.quiet:
            print(f"Processing case range: {start_case}-{end_case-1}")
    
    if args.dry_run:
        print("DRY RUN: Would process trajectories but not generate files")
        return 0
    
    # Process trajectories
    try:
        stats = processor.process_batch(
            input_dir=input_dir,
            cam_params_dir=cam_params_dir,
            output_dir=output_dir,
            max_cases=max_cases,
            output_filename=args.output_filename
        )
        
        print_statistics(stats, args.quiet)
        
        if stats['failed'] > 0:
            return 1  # Exit with error code if some cases failed
        
        return 0
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())