#!/bin/bash
# Setup script for viz-streamlit-ahu environment
# Run this to set PYTHONPATH before running the app

export PYTHONPATH=/Users/putra/ahu-backend-server:$PYTHONPATH

echo "PYTHONPATH set to: $PYTHONPATH"
echo "You can now run: streamlit run app2.py"
