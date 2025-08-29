#!/bin/bash
#
# This script automates the download and extraction of the Volleyball Group
# Activity Recognition dataset from Kaggle.

# Exit immediately if a command exits with a non-zero status.
set -e

# The Kaggle dataset to download. Format: <username>/<dataset-name>
KAGGLE_DATASET="sherif31/group-activity-recognition-volleyball"


# --- Script Logic ---

echo "===== Kaggle Volleyball Dataset Downloader ====="

# 1. Check for dependencies
echo "Checking for the 'kaggle' command..."
if ! command -v kaggle &> /dev/null; then
    echo "Error: The Kaggle CLI is not installed or not in your PATH." >&2
    echo "Please install it by running: pip install kaggle" >&2
    exit 1
fi
echo "Kaggle CLI found."

# 2. Define paths
# Get the absolute path to the directory where this script is located.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Set the target data directory to be a 'data' folder in the parent directory.
DATA_DIR="$(dirname "$SCRIPT_DIR")/data"

# 3. Create the target directory if it doesn't exist
echo "Ensuring data directory exists at: $DATA_DIR"
mkdir -p "$DATA_DIR"

# 4. Download and unzip the dataset using the Kaggle API
echo "Downloading and extracting '$KAGGLE_DATASET'..."
kaggle datasets download -d "$KAGGLE_DATASET" -p "$DATA_DIR" --unzip

