#!/bin/bash

# Check if the file with directory paths is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <file_with_dir_paths>"
    exit 1
fi

# Read the file with directory paths
file_with_dir_paths=$1

# Check if the file exists
if [ ! -f "$file_with_dir_paths" ]; then
    echo "File not found: $file_with_dir_paths"
    exit 1
fi

# Iterate through each line in the file and create the directory if it doesn't exist
while IFS= read -r dir_path; do
    if [ ! -d "$dir_path" ]; then
        mkdir -p "$dir_path"
        echo "Created directory: $dir_path"
    else
        echo "Directory already exists: $dir_path"
    fi
done < "$file_with_dir_paths"
