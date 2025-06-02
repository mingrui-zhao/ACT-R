#!/bin/bash

# Get the absolute path of the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NEUS_DIR="${PROJECT_ROOT}/05_externals/neus"

# Store original directory
ORIGINAL_DIR=$(pwd)

# Function to process a single case and view configuration
process_case() {
    local case_id=$1
    local views=$2
    local padded_id=$(printf "%05d" $case_id)
    local case_path="${padded_id}/${views}"
    
    echo "Processing case: ${case_path}"
    
    # Change to NeuS directory
    cd "${NEUS_DIR}"
    echo "Changed to NeuS directory: $(pwd)"
    
    # Training phase
    python exp_runner.py \
        --mode train \
        --conf "${PROJECT_ROOT}/04_reconstruction/neus.conf" \
        --case ${case_path} \
        --gpu 0
    
    local train_status=$?
    
    if [ $train_status -ne 0 ]; then
        echo "Error: Training failed for ${case_path}"
        cd "${ORIGINAL_DIR}"
        return 1
    fi
    
    # Validation phase
    python exp_runner.py \
        --mode validate_mesh \
        --conf "${PROJECT_ROOT}/04_reconstruction/neus.conf" \
        --case ${case_path} \
        --is_continue \
        --gpu 0

    local val_status=$?
    
    if [ $val_status -ne 0 ]; then
        echo "Error: Validation failed for ${case_path}"
        cd "${ORIGINAL_DIR}"
        return 1
    fi

    # Change back to original directory for file operations
    cd "${ORIGINAL_DIR}"

    # Create target directory if it doesn't exist
    mkdir -p "${PROJECT_ROOT}/06_results/neus_recon/${padded_id}/${views}"

    # Move the mesh file to target location
    local mesh_source="${PROJECT_ROOT}/06_results/neus_exp/${case_path}/meshes/00010000.ply"
    local mesh_target="${PROJECT_ROOT}/06_results/neus_recon/${padded_id}/${views}/mesh.ply"
    
    if [ -f "${mesh_source}" ]; then
        mv "${mesh_source}" "${mesh_target}"
        echo "✓ Mesh file moved successfully for ${case_path}"
    else
        echo "✗ Warning: Mesh file not found at ${mesh_source}"
        return 1
    fi
    
    # Remove the experiment directory
    rm -rf "${PROJECT_ROOT}/06_results/neus_exp/${case_path}"
    echo "✓ Cleaned up experiment directory for ${case_path}"

    echo "✓ Completed processing ${case_path}"
    echo "----------------------------------------"
    return 0
}

# Print configuration info
echo "Script directory: ${SCRIPT_DIR}"
echo "Project root: ${PROJECT_ROOT}"
echo "NeuS directory: ${NEUS_DIR}"
echo "Original directory: ${ORIGINAL_DIR}"
echo "Configuration file: ${PROJECT_ROOT}/04_reconstruction/neus.conf"
echo ""

# Verify NeuS installation
if [ ! -d "${NEUS_DIR}" ]; then
    echo "Error: NeuS directory not found at ${NEUS_DIR}"
    exit 1
fi

if [ ! -f "${NEUS_DIR}/exp_runner.py" ]; then
    echo "Error: NeuS exp_runner.py not found at ${NEUS_DIR}/exp_runner.py"
    exit 1
fi

if [ ! -f "${PROJECT_ROOT}/04_reconstruction/neus.conf" ]; then
    echo "Error: Configuration file not found at ${PROJECT_ROOT}/04_reconstruction/neus.conf"
    exit 1
fi

echo "Starting NeuS reconstruction processing..."
echo "=========================================="

# Statistics tracking
total_cases=0
successful_cases=0
failed_cases=0

# Main processing loop
for ((i=0; i<=1030; i++)); do
    echo "Starting case $i..."
    total_cases=$((total_cases + 1))
    
    # Process 21 views
    if process_case $i "21_views"; then
        successful_cases=$((successful_cases + 1))
    else
        failed_cases=$((failed_cases + 1))
        echo "✗ Failed to process case $i"
    fi
done

# Ensure we're back in the original directory
cd "${ORIGINAL_DIR}"

echo ""
echo "=========================================="
echo "PROCESSING SUMMARY"
echo "=========================================="
echo "Total cases processed: ${total_cases}"
echo "Successful: ${successful_cases}"
echo "Failed: ${failed_cases}"
if [ $total_cases -gt 0 ]; then
    success_rate=$(( (successful_cases * 100) / total_cases ))
    echo "Success rate: ${success_rate}%"
fi
echo "Results saved to: ${PROJECT_ROOT}/06_results/neus_recon/"
echo "All processing completed"