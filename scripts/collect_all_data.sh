#!/bin/bash

# 1. Robustly find project root
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Assume script is in project/scripts/, so root is project/
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SEARCH_ROOT="$PROJECT_ROOT/results"
OUTPUT_FILE="$PROJECT_ROOT/rl_project_all_data.tar.gz"

echo "=== Collecting All RL Training Data ==="
echo "Script Location: $SCRIPT_DIR"
echo "Project Root:    $PROJECT_ROOT"
echo "Search Root:     $SEARCH_ROOT"
echo "Output File:     $OUTPUT_FILE"

# 2. Check if results directory exists
if [ ! -d "$SEARCH_ROOT" ]; then
    echo "Error: Directory $SEARCH_ROOT not found!"
    echo "Current directory structure:"
    ls -F "$PROJECT_ROOT"
    exit 1
fi

# 3. Debug: Show what's inside results (first 2 levels)
echo "--- Debug: Checking directory structure (max depth 3) ---"
find "$SEARCH_ROOT" -maxdepth 3 | head -n 10
echo "---------------------------------------------------------"

# 4. Create file list
echo "Scanning for log files..."
# We look for tfevents AND log.txt
# Using -type f to ensure we match files
find "$SEARCH_ROOT" -type f -name "*tfevents*" > "$PROJECT_ROOT/file_list.txt"
find "$SEARCH_ROOT" -type f -name "log.txt" >> "$PROJECT_ROOT/file_list.txt"

# 5. Check and Compress
FILE_COUNT=$(wc -l < "$PROJECT_ROOT/file_list.txt")
echo "Found $FILE_COUNT log files to archive."

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "No files found. Please check the 'Debug' output above to see if your logs are in a different folder."
    rm "$PROJECT_ROOT/file_list.txt"
    exit 1
fi

# Tarball
# -C changes directory to PROJECT_ROOT so paths in tar are relative to it
tar -czf "$OUTPUT_FILE" -T "$PROJECT_ROOT/file_list.txt"

echo "=== Data Collection Complete ==="
echo "Archive created at: $OUTPUT_FILE"
echo "Size: $(du -h "$OUTPUT_FILE" | cut -f1)"

# Cleanup
rm "$PROJECT_ROOT/file_list.txt"