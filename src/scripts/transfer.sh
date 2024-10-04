#!/bin/bash

# Variables
REMOTE_HOST='n12halki.cs.ntua.gr'
REMOTE_USER='dbek'
REMOTE_PASS='dbek_cvsp'
REMOTE_BASE_DIR='/gpu-data4/dbek/thesis'
LOCAL_BASE_PREFIX='/home/dbek/thesis'
LOCAL_DIR="$1"
FILE_LIST='files_to_transfer.txt'


# Check if the directory path is provided
if [ -z "$LOCAL_DIR" ]; then
  echo "Usage: $0 /path/to/local/directory"
  exit 1
fi

# Create the file list excluding files in directories named 'checkpoint-*'
find "$LOCAL_DIR" -type f ! -path '*/checkpoint-*/*' > "$FILE_LIST"

DIR_LIST='dirs_to_create.txt'
awk -v base="$REMOTE_BASE_DIR" -v prefix="$LOCAL_BASE_PREFIX" '{sub(prefix, "", $0); sub(/\/[^/]*$/, "", $0); print base$0}' "$FILE_LIST" | sort -u > "$DIR_LIST"

# Debug output
echo "Directories to be created:"
cat "$DIR_LIST"

# Ensure all directories are created before proceeding
if [ $? -eq 0 ]; then
  echo "All directories created successfully."
else
  echo "Error creating directories."
  exit 1
fi
Transfer files using SFTP and preserve directory structure
while IFS= read -r file; do
  # Determine the remote path
  relative_path="${file#$LOCAL_BASE_PREFIX/}"
  remote_path="$REMOTE_BASE_DIR/$relative_path"
  
  echo "Transferring file: $file to $remote_path"
  
  sshpass -p "$REMOTE_PASS" sftp -oBatchMode=no -b - $REMOTE_USER@$REMOTE_HOST << EOF
put "$file" "$remote_path"
bye
EOF

done < "$FILE_LIST"



echo "File transfer complete."
