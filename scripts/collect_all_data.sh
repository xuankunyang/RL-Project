#!/bin/bash

# Configuration
OUTPUT_FILE="rl_project_all_data.tar.gz"
SEARCH_ROOT="results"

echo "=== Collecting All RL Training Data ==="
echo "Search Root: $SEARCH_ROOT"
echo "Output File: $OUTPUT_FILE"

if [ ! -d "$SEARCH_ROOT" ]; then
    echo "Error: Directory $SEARCH_ROOT not found!"
    exit 1
fi

# Create a temporary list of files to archive
# We want:
# 1. TensorBoard logs (events.out.tfevents*)
# 2. Text logs (log.txt)
# 3. Training Scripts (scripts/*.sh) - useful for reproducibility reference
# We DO NOT want:
# 1. Model checkpoints (*.pth) - too large, unless specifically requested for a few best ones
# 2. Videos (*.mp4) - usually large

echo "Scanning for log files..."
find "$SEARCH_ROOT" -name "events.out.tfevents*" > file_list.txt
find "$SEARCH_ROOT" -name "log.txt" >> file_list.txt

# Count files
FILE_COUNT=$(wc -l < file_list.txt)
echo "Found $FILE_COUNT log files to archive."

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "No files found. Exiting."
    rm file_list.txt
    exit 0
fi

# Create Tarball
# We use -h to follow symlinks if any, though unlikely in results
tar -czf "$OUTPUT_FILE" -T file_list.txt

echo "=== Data Collection Complete ==="
echo "Archive created at: $(pwd)/$OUTPUT_FILE"
echo "Size: $(du -h $OUTPUT_FILE | cut -f1)"
echo "You can now download this file to your local machine."

# Cleanup
rm file_list.txt