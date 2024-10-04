#!/bin/bash

# Check if the directory is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Assign the provided directory to a variable
TARGET_DIR=$1

# Check if the provided directory exists
if [ ! -d "$TARGET_DIR" ]; then
  echo "The directory $TARGET_DIR does not exist."
  exit 1
fi

# Find and delete subdirectories starting with 'checkpoint'
find "$TARGET_DIR" -type d -name 'checkpoint*' -exec echo {} + # rm -rf {} +

echo "All subdirectories starting with 'checkpoint' have been deleted from $TARGET_DIR."
