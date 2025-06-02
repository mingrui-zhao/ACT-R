import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

def draw_sphere(ax, center, r):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + r * np.outer(np.cos(u), np.sin(v))
    y = center[1] + r * np.outer(np.sin(u), np.sin(v))
    z = center[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='c', alpha=0.1)

def draw_cube(ax, min_x, max_x, min_y, max_y, min_z, max_z, color):
    x = [min_x, min_x, max_x, max_x, min_x, min_x, max_x, max_x]
    y = [min_y, max_y, max_y, min_y, min_y, max_y, max_y, min_y]
    z = [min_z, min_z, min_z, min_z, max_z, max_z, max_z, max_z]
    
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    for edge in edges:
        ax.plot3D(*zip((x[edge[0]], y[edge[0]], z[edge[0]]), (x[edge[1]], y[edge[1]], z[edge[1]])), color=color)

def draw_ray(ax, src, dst, color='k', label='view direction'):
    ray_x = [src[0], dst[0]]
    ray_y = [src[1], dst[1]]
    ray_z = [src[2], dst[2]]
    ax.plot(ray_x, ray_y, ray_z, color=color, linewidth=2, label=label)


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def plot_orbit_trajectory(ax, center, radius, elevations_deg, azimuths_deg, color='coral'):
    coord_new = world_to_cmr([0, 0, 0], radius)
    elevations = np.radians(elevations_deg)
    azimuths = np.radians(azimuths_deg)
    x_trajectory = center[0] + radius * np.cos(elevations) * np.cos(azimuths)
    y_trajectory = center[1] + radius * np.cos(elevations) * np.sin(azimuths)
    z_trajectory = center[2] + radius * np.sin(elevations)    

    for idx in range(len(elevations_deg)):
        coord_new = world_to_cmr([x_trajectory[idx], y_trajectory[idx], z_trajectory[idx]], radius)
        x_trajectory[idx] = coord_new[0]
        y_trajectory[idx] = coord_new[2]
        z_trajectory[idx] = coord_new[1]
    ax.plot(x_trajectory, y_trajectory, z_trajectory, color=color, linewidth=1.0, label='Camera Orbit (Adaptive)') #marker='o', 


def draw_arrow(ax, start, direction, color='orange', label='Obj Standing Direction'):
    # Create an arrow using the quiver function
    ax.quiver(start[0], start[1], start[2], 
              direction[0], direction[1], direction[2],
              color=color, arrow_length_ratio=0.1, label=label)

def cal_sphere_coord(center, radius, elevations_deg, azimuths_deg):
    elevations = np.radians(elevations_deg)
    azimuths = np.radians(azimuths_deg)
    x = center[0] + radius * np.cos(elevations) * np.cos(azimuths)
    y = center[1] + radius * np.cos(elevations) * np.sin(azimuths)
    z = center[2] + radius * np.sin(elevations)
    return x, y, z


def calculate_rotation_matrix(camera_forward, world_up=np.array([0, 1, 0])):
    """
    Calculates the rotation matrix based on the camera's pointing (forward) direction.
    
    Parameters:
    - camera_forward: A 3D numpy array representing the camera's forward direction in world coordinates.
    - world_up: A 3D numpy array representing the world's up direction (default is [0, 1, 0]).
    
    Returns:
    - rotation_matrix: A 3x3 rotation matrix as a numpy array.
    """
    # Normalize the forward vector to get the Z-axis
    z_axis = camera_forward / np.linalg.norm(camera_forward)
    
    # Calculate the right vector (X-axis) as the cross product of world_up and forward (Z-axis)
    x_axis = np.cross(world_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)  # Normalize
    
    # Calculate the true up vector (Y-axis) as the cross product of forward (Z-axis) and right (X-axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)  # Normalize
    
    # Create the rotation matrix
    rotation_matrix = np.row_stack((y_axis, x_axis, -z_axis))

    return rotation_matrix


def world_to_camera_transform(camera_position, camera_target, up_vector=np.array([0, 1, 0])):
    """
    Compute the world-to-camera transformation matrix (4x4) given the camera's position, target, and up vector.
    
    Args:
        camera_position (np.array): Position of the camera in world coordinates, shape (3,).
        camera_target (np.array): Point the camera is looking at in world coordinates, shape (3,).
        up_vector (np.array): Up direction vector in the world (optional), default is [0, 1, 0].
    
    Returns:
        np.array: World-to-camera transformation matrix of shape (4, 4).
    """
    # Calculate the forward vector (direction from the camera position to the target)
    forward = camera_target - camera_position
    forward /= np.linalg.norm(forward)  # Normalize the forward vector

    # Calculate the right vector (perpendicular to both forward and up vector)
    right = np.cross(up_vector, forward)
    right /= np.linalg.norm(right)  # Normalize the right vector

    # Recompute the up vector to ensure orthogonality
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)  # Normalize the up vector

    # Create the rotation matrix
    rotation_matrix = np.array([
        [right[0], right[1], right[2], 0],
        [up[0], up[1], up[2], 0],
        [-forward[0], -forward[1], -forward[2], 0],
        [0, 0, 0, 1]
    ])

    # Create the translation matrix
    translation_matrix = np.array([
        [1, 0, 0, -camera_position[0]],
        [0, 1, 0, -camera_position[1]],
        [0, 0, 1, -camera_position[2]],
        [0, 0, 0, 1]
    ])

    # Combine rotation and translation to form the world-to-camera transformation matrix
    world_to_camera_matrix = rotation_matrix @ translation_matrix
    world_to_camera_matrix[[0, 1]] = world_to_camera_matrix[[1, 0]]
    # world_to_camera_matrix[0], world_to_camera_matrix[1] = world_to_camera_matrix[1], world_to_camera_matrix[0]
    return world_to_camera_matrix


def segment_box_intersect(seg_start, seg_end, box_min, box_max):
    """
    Determine if a segment intersects an axis-aligned bounding box (AABB).
    """
    t_min, t_max = 0, 1  # Parameters for the segment

    # Define the segment direction
    direction = np.subtract(seg_end, seg_start)

    for i in range(3):
        if direction[i] == 0:
            # Segment is parallel to the current axis
            if seg_start[i] < box_min[i] or seg_start[i] > box_max[i]:
                return False  # Segment is outside the box on this axis
        else:
            # Compute intersection t-values for the planes on this axis
            t1 = (box_min[i] - seg_start[i]) / direction[i]
            t2 = (box_max[i] - seg_start[i]) / direction[i]

            # Order t1 and t2 to simplify the next steps
            t1, t2 = min(t1, t2), max(t1, t2)

            # Update t_min and t_max to narrow down the intersection range
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

            if t_min > t_max:
                return False  # No intersection exists if t_min > t_max

    return True  # The segment intersects the box if we reach here

def does_segment_pass_through_cube(seg_start, seg_end, cube_start, cube_size):
    """
    Determine if a segment passes through a given cube.
    """
    # Define the cube's minimum and maximum corners
    box_min = cube_start
    box_max = cube_start + cube_size

    # Check if the segment intersects the cube's AABB
    return segment_box_intersect(seg_start, seg_end, box_min, box_max)