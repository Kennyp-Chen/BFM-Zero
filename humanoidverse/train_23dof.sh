#!/bin/bash

# G1 23-DOF Training Script
# Usage: ./train_23dof.sh

echo "Starting G1 23-DOF training..."
echo "Configuration: g1_23dof_hard_waist"
echo "Work Directory: results/23dof-bfmzero-isaac"
echo ""

# Check if required files exist
if [ ! -f "config/robot/g1/g1_23dof_hard_waist.yaml" ]; then
    echo "Error: g1_23dof_hard_waist.yaml not found!"
    echo "Please run convert_g1_dof.py first to generate the configuration."
    exit 1
fi

if [ ! -f "data/robots/g1/g1_23dof.xml" ]; then
    echo "Error: g1_23dof.xml not found!"
    exit 1
fi

# Run training
python train_low_23dof.py

echo "Training completed!"
