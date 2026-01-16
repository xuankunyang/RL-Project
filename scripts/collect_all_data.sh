#!/bin/bash

# 获取脚本所在目录的上一级作为项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SEARCH_ROOT="$PROJECT_ROOT/results"
OUTPUT_FILE="$PROJECT_ROOT/rl_project_all_data.tar.gz"

echo "=== Collecting All RL Training Data (Follow Symlinks) ==="
echo "Project Root: $PROJECT_ROOT"
echo "Search Root:  $SEARCH_ROOT"

# 1. 检查 results 文件夹
if [ ! -d "$SEARCH_ROOT" ]; then
    echo "ERROR: Directory $SEARCH_ROOT does not exist!"
    ls -F "$PROJECT_ROOT"
    exit 1
fi

# 2. 查找文件 (使用 -L 选项跟随符号链接)
echo "Scanning for log files (tfevents) following symlinks..."
# -L: Follow symbolic links
# We search for *tfevents* AND log.txt
find -L "$SEARCH_ROOT" -type f \( -name "*tfevents*" -o -name "log.txt" \) > "$PROJECT_ROOT/file_list.txt"

FILE_COUNT=$(wc -l < "$PROJECT_ROOT/file_list.txt")
echo "Found $FILE_COUNT log files."

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "ERROR: No files found even with -L. Please check permissions or directory content."
    ls -L -R "$SEARCH_ROOT" | head -n 20
    rm "$PROJECT_ROOT/file_list.txt"
    exit 1
fi

# 3. 打包 (使用 -h 选项解引用符号链接，将实际文件打包)
echo "Archiving files..."
# -h: dereference symlinks (store the file, not the link)
# -T: read file list from file
tar -czh -f "$OUTPUT_FILE" -T "$PROJECT_ROOT/file_list.txt"

echo "=== Success! ==="
echo "Archive: $OUTPUT_FILE"
echo "Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
rm "$PROJECT_ROOT/file_list.txt"
