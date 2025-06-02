#!/usr/bin/env python3
"""
Camera Path Optimization for 3D Object Visibility Analysis

This script optimizes camera trajectories to maximize visibility of 3D cubes
representing object features, using sphere-based camera movement patterns.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from dataclasses import dataclass

from utils import (
    draw_cube, draw_ray, draw_arrow, draw_sphere,
    read_pickle, plot_orbit_trajectory, world_to_camera_transform, cal_sphere_coord,
    does_segment_pass_through_cube
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass
class CameraConfig:
    """Configuration for camera parameters."""
    position: np.ndarray
    field_of_view_deg: float = 33.6
    
    @property
    def field_of_view_rad(self) -> float:
        return np.radians(self.field_of_view_deg)


@dataclass
class PathConfig:
    """Configuration for path optimization."""
    interval_range: Tuple[int, int] = (-5, 6)
    path_steps: int = 21
    azimuth_step_deg: float = 18.0


class Cube:
    """Represents a 3D cube with position, dimensions, and visibility properties."""
    
    def __init__(self, cube_id: str, x: float, y: float, z: float, 
                 dx: float, dy: float, dz: float, value: float):
        self.id = cube_id
        self.position = np.array([x, y, z])
        self.dimensions = np.array([dx, dy, dz])
        self.value = value
        self.faces = self._get_faces()
        self.center = self.position + self.dimensions / 2.0

    def _get_faces(self) -> List[dict]:
        """Generate face definitions for the cube."""
        x, y, z = self.position
        dx, dy, dz = self.dimensions
        
        faces = [
            # Right face (+X)
            {
                "name": "right_face", 
                "vl": self.value, 
                "normal": np.array([1, 0, 0]), 
                "corners": [
                    np.array([x+dx, y, z]), np.array([x+dx, y+dy, z]), 
                    np.array([x+dx, y, z+dz]), np.array([x+dx, y+dy, z+dz])
                ]
            },
            # Left face (-X)
            {
                "name": "left_face", 
                "vl": self.value, 
                "normal": np.array([-1, 0, 0]), 
                "corners": [
                    np.array([x, y, z]), np.array([x, y+dy, z]), 
                    np.array([x, y, z+dz]), np.array([x, y+dy, z+dz])
                ]
            },
            # Up face (+Y)
            {
                "name": "up_face", 
                "vl": self.value, 
                "normal": np.array([0, 1, 0]), 
                "corners": [
                    np.array([x, y+dy, z]), np.array([x+dx, y+dy, z]), 
                    np.array([x, y+dy, z+dz]), np.array([x+dx, y+dy, z+dz])
                ]
            },
            # Down face (-Y)
            {
                "name": "down_face", 
                "vl": self.value, 
                "normal": np.array([0, -1, 0]), 
                "corners": [
                    np.array([x, y, z]), np.array([x+dx, y, z]), 
                    np.array([x, y, z+dz]), np.array([x+dx, y, z+dz])
                ]
            },
            # Back face (+Z)
            {
                "name": "back_face", 
                "vl": self.value, 
                "normal": np.array([0, 0, 1]), 
                "corners": [
                    np.array([x, y, z+dz]), np.array([x+dx, y, z+dz]), 
                    np.array([x, y+dy, z+dz]), np.array([x+dx, y+dy, z+dz])
                ]
            },
            # Front face (-Z)
            {
                "name": "front_face", 
                "vl": self.value, 
                "normal": np.array([0, 0, -1]), 
                "corners": [
                    np.array([x, y, z]), np.array([x+dx, y, z]), 
                    np.array([x, y+dy, z]), np.array([x+dx, y+dy, z])
                ]
            }
        ]
        return faces


class VisibilityAnalyzer:
    """Handles visibility analysis for cubes from camera positions."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug

    def world_to_camera(self, coordinate: np.ndarray, inverse_world_matrix: np.ndarray) -> np.ndarray:
        """Transform world coordinates to camera coordinates."""
        def transform_coordinate(matrix: np.ndarray, coord: np.ndarray) -> np.ndarray:
            # Convert to homogeneous coordinates
            coord_homogeneous = np.array([*coord, 1])
            # Perform transformation
            transformed_coord = matrix @ coord_homogeneous
            # Convert back to 3D
            return transformed_coord[:3] / transformed_coord[3]
        
        return transform_coordinate(inverse_world_matrix, coordinate)

    def is_cube_visible(self, camera_position: np.ndarray, camera_direction: np.ndarray, 
                       other_cubes: List[Cube], current_cube: Cube, field_of_view: float) -> bool:
        """Check if a cube is visible from the camera position."""
        # Calculate vector from camera to cube center
        cube_center = current_cube.center
        to_cube = cube_center - camera_position
        to_cube_unit = to_cube / np.linalg.norm(to_cube)

        # Check field of view
        angle_to_cube = np.arccos(np.dot(to_cube_unit, camera_direction / np.linalg.norm(camera_direction)))
        if angle_to_cube > field_of_view:
            return False

        # Check occlusion by other cubes
        for other_cube in other_cubes:
            if self._is_ray_blocked(camera_position, cube_center, other_cube):
                return False

        return True

    def _is_ray_blocked(self, camera_position: np.ndarray, target_point: np.ndarray, other_cube: Cube) -> bool:
        """Check if a ray from camera to target is blocked by another cube."""
        if self.debug:
            print(f"Checking ray blockage by cube {other_cube.id}")
            print(f"Camera: {camera_position}, Target: {target_point}")
            
        return does_segment_pass_through_cube(
            camera_position, target_point, other_cube.position, other_cube.dimensions
        )

    def count_visible_cubes(self, cubes: List[Cube], camera_position: np.ndarray, 
                           camera_direction: np.ndarray, field_of_view: float) -> Tuple[int, List[Cube]]:
        """Count and return visible cubes from a camera position."""
        visible_cubes = []
        
        for i, cube in enumerate(cubes):
            if self.debug:
                print(f'Checking cube {i}')
                
            other_cubes = cubes[:i] + cubes[i+1:]
            if self.is_cube_visible(camera_position, camera_direction, other_cubes, cube, field_of_view):
                visible_cubes.append(cube)
                
        return len(visible_cubes), visible_cubes


class GridConstructor:
    """Handles construction of 3D grid cubes from data arrays."""
    
    def __init__(self, draw_fig: bool = False):
        self.draw_fig = draw_fig

    def construct_grids(self, arr_bbox: np.ndarray, arr_diff: np.ndarray, ax=None) -> List[Cube]:
        """Construct cubes from 3D difference array within bounding box."""
        cubes = []
        cube_count = 0
        
        # Extract bounding box dimensions
        min_x, max_x = arr_bbox[0][0], arr_bbox[0][1]
        min_y, max_y = arr_bbox[2][0], arr_bbox[2][1]
        min_z, max_z = arr_bbox[1][0], arr_bbox[1][1]

        # Calculate subcube dimensions
        subcube_size_i = (max_y - min_y) / arr_diff.shape[0]
        subcube_size_j = (max_z - min_z) / arr_diff.shape[1]
        subcube_size_k = (max_x - min_x) / arr_diff.shape[2]
        
        def is_on_surface(i: int, j: int, k: int) -> bool:
            """Check if position is on the surface of the 3D array."""
            return (i == 0 or i == arr_diff.shape[0] - 1 or 
                   j == 0 or j == arr_diff.shape[1] - 1 or 
                   k == 0 or k == arr_diff.shape[2] - 1)

        # Generate cubes for surface voxels with positive values
        for i in range(arr_diff.shape[0]):
            for j in range(arr_diff.shape[1]):
                for k in range(arr_diff.shape[2]):
                    if arr_diff[i, j, k] > 0 and is_on_surface(i, j, k):
                        # Calculate cube position
                        x = min_x + (arr_diff.shape[2] - 1 - k) * subcube_size_k
                        y = min_y + (arr_diff.shape[0] - 1 - i) * subcube_size_i
                        z = min_z + (arr_diff.shape[1] - 1 - j) * subcube_size_j
                        
                        # Create cube
                        cube = Cube(
                            str(cube_count), x, y, z, 
                            subcube_size_k, subcube_size_i, subcube_size_j, 
                            arr_diff[i, j, k]
                        )
                        cubes.append(cube)
                        
                        # Visualize if requested
                        if self.draw_fig and ax is not None:
                            color = [arr_diff[i, j, k], arr_diff[i, j, k], 0]
                            ax.bar3d(x, y, z, subcube_size_k, subcube_size_i, subcube_size_j, 
                                   color=color, alpha=0.5)
                        
                        cube_count += 1

        return cubes


class CameraPathOptimizer:
    """Optimizes camera paths for maximum cube visibility."""
    
    def __init__(self, camera_config: CameraConfig, path_config: PathConfig, 
                 debug: bool = False, draw_fig: bool = False):
        self.camera_config = camera_config
        self.path_config = path_config
        self.debug = debug
        self.draw_fig = draw_fig
        self.visibility_analyzer = VisibilityAnalyzer(debug)
        self.grid_constructor = GridConstructor(draw_fig)

    def calculate_camera_trajectory(self, elevations_deg: List[float], azimuths_deg: List[float], 
                                  radius: float, inv_world_matrix: np.ndarray) -> Tuple[List, List, List, List]:
        """Calculate camera trajectory in world coordinates."""
        x_traj, y_traj, z_traj = cal_sphere_coord([0, 0, 0], radius, elevations_deg, azimuths_deg)

        trajectory = []
        ret_x, ret_y, ret_z = [], [], []
        
        for idx in range(len(elevations_deg)):
            coord_world = [x_traj[idx], y_traj[idx], z_traj[idx]]
            coord_camera = self.visibility_analyzer.world_to_camera(coord_world, inv_world_matrix)
            
            # Reorder coordinates for visualization
            coord_reordered = [coord_camera[0], coord_camera[2], coord_camera[1]]
            trajectory.append(coord_reordered)
            ret_x.append(coord_camera[0])
            ret_y.append(coord_camera[2])
            ret_z.append(coord_camera[1])
            
        return trajectory, ret_x, ret_y, ret_z

    def generate_path_elevations(self, ele_init: float, interval_1st: int, interval_2nd: int) -> List[float]:
        """Generate elevation angles for the camera path."""
        elevations = [ele_init + interval_1st * i for i in range(6)]
        elevations += [elevations[-1] + interval_2nd * (i - 5) for i in range(6, 11)]
        elevations += [elevations[-1] - interval_2nd * (i - 10) for i in range(11, 16)]
        elevations += [elevations[-1] - interval_1st * (i - 15) for i in range(16, 21)]
        return elevations

    def generate_path_azimuths(self, azimuth_init: float) -> List[float]:
        """Generate azimuth angles for the camera path."""
        return [(azimuth_init + self.path_config.azimuth_step_deg * i) % 360 
                for i in range(self.path_config.path_steps)]

    def evaluate_path(self, cubes: List[Cube], camera_trajectory: List, sphere_center: List[float]) -> float:
        """Evaluate the visibility score for a camera path."""
        total_score = 0.0
        
        for cam_pos in camera_trajectory:
            cam_pos_array = np.array(cam_pos)
            camera_direction = [a - b for a, b in zip(sphere_center, cam_pos)]
            
            _, visible_cubes = self.visibility_analyzer.count_visible_cubes(
                cubes, cam_pos_array, np.array(camera_direction), self.camera_config.field_of_view_rad
            )
            
            for cube in visible_cubes:
                total_score += cube.value
                
        return total_score

    def optimize_single_case(self, case_uid: str, arr_cam_params: np.ndarray, 
                           arr_diff: np.ndarray, ax=None) -> Tuple[int, int]:
        """Optimize camera path for a single case."""
        # Extract camera parameters
        bbox_x_extent, bbox_y_extent, bbox_z_extent = arr_cam_params[0:3]
        distance = arr_cam_params[-2]
        ele_init_rad = arr_cam_params[-1]
        
        # Setup coordinate system
        arr_bbox = np.array([
            [0. - bbox_x_extent / 2., 0. + bbox_x_extent / 2.],
            [0. - bbox_y_extent / 2., 0. + bbox_y_extent / 2.],
            [-distance - bbox_z_extent / 2., -distance + bbox_z_extent / 2.]
        ])
        
        sphere_center = [0, -distance, 0]
        ele_init = np.degrees(ele_init_rad)
        azimuth_init = 0  # Set initial azimuth to 0

        # Construct cubes
        cubes = self.grid_constructor.construct_grids(arr_bbox, arr_diff, ax)
        
        # Setup camera transformation
        camera_init_loc = cal_sphere_coord([0, 0, 0], distance, ele_init, 0)
        camera_target = np.array([0, 0, 0])
        inv_world_matrix = world_to_camera_transform(camera_init_loc, camera_target)

        # Visualization setup
        if self.draw_fig and ax is not None:
            self._setup_visualization(ax, sphere_center, inv_world_matrix)

        # Optimize path
        best_score = 0.0
        best_path = (-5, -5)  # Default fallback
        
        interval_range = range(*self.path_config.interval_range)
        for interval_1st in interval_range:
            for interval_2nd in interval_range:
                # Generate path
                elevations = self.generate_path_elevations(ele_init, interval_1st, interval_2nd)
                azimuths = self.generate_path_azimuths(azimuth_init)
                
                # Calculate trajectory
                cam_traj, ret_x, ret_y, ret_z = self.calculate_camera_trajectory(
                    elevations, azimuths, distance, inv_world_matrix
                )
                
                # Visualize trajectory
                if self.draw_fig and ax is not None:
                    ax.plot(ret_x, ret_y, ret_z, color='green', linewidth=1.0, 
                           label='Camera Orbit (Adaptive)')
                    draw_sphere(ax, sphere_center, distance)
                    
                # Evaluate path
                score = self.evaluate_path(cubes, cam_traj, sphere_center)
                
                if score > best_score:
                    best_score = score
                    best_path = (interval_1st, interval_2nd)

        if self.debug:
            print(f"Case {case_uid}: Best path {best_path} with score {best_score}")
            
        return best_path

    def _setup_visualization(self, ax, sphere_center: List[float], inv_world_matrix: np.ndarray):
        """Setup visualization elements."""
        obj_standing_vec = (self.visibility_analyzer.world_to_camera([0, 0, 1], inv_world_matrix) - 
                           self.visibility_analyzer.world_to_camera([0, 0, 0], inv_world_matrix))
        obj_standing_vec[1], obj_standing_vec[2] = obj_standing_vec[2], obj_standing_vec[1]
        
        draw_arrow(ax, start=sphere_center, direction=obj_standing_vec, 
                  color='m', label='Obj Standing Direction')
        ax.scatter(*self.camera_config.position, color='g')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize camera paths for 3D object visibility analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output paths
    parser.add_argument('--data-root', type=str, default='./00_data',
                       help='Root directory for input data')
    parser.add_argument('--objects-dir', type=str, default='GSO_rm_bg',
                       help='Directory name for object files (relative to data-root)')
    parser.add_argument('--diff-blocks-dir', type=str, default='diff_blocks_vgg_normalized',
                       help='Directory name for difference blocks (relative to data-root)')
    parser.add_argument('--cam-params-dir', type=str, default='cmt_est/gso',
                       help='Directory name for camera parameters (relative to data-root)')
    parser.add_argument('--output-dir', type=str, default='./06_results/predicted_path',
                       help='Output directory for results')
    
    # Camera configuration
    parser.add_argument('--camera-fov', type=float, default=33.6,
                       help='Camera field of view in degrees')
    
    # Path optimization parameters
    parser.add_argument('--interval-min', type=int, default=-5,
                       help='Minimum interval for path optimization')
    parser.add_argument('--interval-max', type=int, default=6,
                       help='Maximum interval for path optimization (exclusive)')
    parser.add_argument('--azimuth-step', type=float, default=18.0,
                       help='Azimuth step size in degrees')
    parser.add_argument('--path-steps', type=int, default=21,
                       help='Number of steps in camera path')
    
    # Debugging and visualization
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--draw-fig', action='store_true',
                       help='Enable figure drawing and visualization')
    parser.add_argument('--case-limit', type=int, default=None,
                       help='Limit processing to first N cases (for testing)')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup configurations
    camera_config = CameraConfig(
        position=np.array([0, 0, 0]),
        field_of_view_deg=args.camera_fov
    )
    
    path_config = PathConfig(
        interval_range=(args.interval_min, args.interval_max),
        path_steps=args.path_steps,
        azimuth_step_deg=args.azimuth_step
    )
    
    # Initialize optimizer
    optimizer = CameraPathOptimizer(
        camera_config=camera_config,
        path_config=path_config,
        debug=args.debug,
        draw_fig=args.draw_fig
    )
    
    # Setup paths
    dir_objs = os.path.join(args.data_root, args.objects_dir)
    dir_diff_blocks = os.path.join(args.data_root, args.diff_blocks_dir)
    dir_cam_params = os.path.join(args.data_root, args.cam_params_dir)
    
    # Get object files
    obj_fnames = sorted(os.listdir(dir_objs))
    if args.case_limit:
        obj_fnames = obj_fnames[:args.case_limit]
    
    print(f"Processing {len(obj_fnames)} cases...")
    
    # Process each case
    for obj_id, obj_fname in enumerate(obj_fnames):
        case_uid = f"{obj_id:05d}"
        print(f"Processing case {case_uid}")
        
        try:
            # Load data
            arr_cam_params = np.load(os.path.join(dir_cam_params, case_uid, 'cam_param_est.npy'))
            arr_diff = np.load(os.path.join(dir_diff_blocks, case_uid, 'diff_blocks.npy'))
            
            # Setup visualization if requested
            ax = None
            if args.draw_fig:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
            
            # Optimize path
            best_path = optimizer.optimize_single_case(case_uid, arr_cam_params, arr_diff, ax)
            
            # Save results
            output_case_dir = os.path.join(args.output_dir, case_uid)
            os.makedirs(output_case_dir, exist_ok=True)
            np.save(os.path.join(output_case_dir, 'best_path.npy'), np.array(best_path))
            
            if args.draw_fig:
                plt.show()
                
        except Exception as e:
            print(f"Error processing case {case_uid}: {e}")
            continue
    
    print("Processing complete!")


if __name__ == "__main__":
    main()