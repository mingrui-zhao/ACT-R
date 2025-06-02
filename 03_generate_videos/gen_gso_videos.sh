#!/bin/bash

# Get the absolute path of the script's directory and go up to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Directory paths (absolute, relative to project root)
TRAJ_DIR="${PROJECT_ROOT}/06_results/predicted_path"
INPUT_DIR="${PROJECT_ROOT}/00_data/gso_images" 
OUTPUT_DIR="${PROJECT_ROOT}/06_results/generated_videos" 
PYTHON_SCRIPT="${PROJECT_ROOT}/03_generate_videos/sv3d_video_sample.py"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Verify that the Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "Error: Python script not found at ${PYTHON_SCRIPT}"
    exit 1
fi

echo "Script location: ${SCRIPT_DIR}"
echo "Project root: ${PROJECT_ROOT}"
echo "Using Python script: ${PYTHON_SCRIPT}"
echo "Trajectory directory: ${TRAJ_DIR}"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Loop through all objects (0 to 1029, for 1030 total)
for i in {0..1029}
do
    # Format object ID with leading zeros
    OBJ_ID=$(printf "%05d" $i)
    echo "Processing object ${OBJ_ID}"
    
    # Create object output directory
    mkdir -p "${OUTPUT_DIR}/${OBJ_ID}"
    
    # File paths (absolute)
    INPUT_IMAGE="${INPUT_DIR}/${OBJ_ID}.jpg"
    TRAJ_FILE="${TRAJ_DIR}/${OBJ_ID}/predicted_trajectory.npy"
    
    # Check if input image exists
    if [ ! -f "${INPUT_IMAGE}" ]; then
        echo "Warning: Input image not found: ${INPUT_IMAGE}"
        continue
    fi
    
    # Check if trajectory file exists
    if [ ! -f "${TRAJ_FILE}" ]; then
        echo "Warning: Trajectory file not found: ${TRAJ_FILE}"
        continue
    fi
    
    # Extract elevation and azimuth values using Python with absolute path
    ANGLES=$(python -c "
import numpy as np
traj = np.load('${TRAJ_FILE}')
elevations = traj[0]
azimuths = traj[1]
elev_str = '[' + ','.join([str(e) for e in elevations]) + ']'
azi_str = '[' + ','.join([str(a) for a in azimuths]) + ']'
print(elev_str + ' ' + azi_str)
")
    
    # Check if Python command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: Failed to extract angles from trajectory file for object ${OBJ_ID}"
        continue
    fi
    
    # Split the angles into elevation and azimuth arrays
    read -r ELEVATIONS AZIMUTHS <<< "${ANGLES}"
    
    # Run the sampling command with absolute paths
    python "${PYTHON_SCRIPT}" \
        --input_path "${INPUT_IMAGE}" \
        --version sv3d_p \
        --elevations_deg "${ELEVATIONS}" \
        --azimuths_deg "${AZIMUTHS}" \
        --output_folder "${OUTPUT_DIR}/${OBJ_ID}"
    
    # Check if the sampling command succeeded
    if [ $? -eq 0 ]; then
        echo "✓ Completed object ${OBJ_ID}"
    else
        echo "✗ Failed to process object ${OBJ_ID}"
    fi
done

echo ""
echo "All processing complete!"
echo "Results saved to: ${OUTPUT_DIR}"