#!/bin/bash

# Default folder path
FOLDER_PATH="./config"

# Check if directory exists
if [ ! -d "$FOLDER_PATH" ]; then
    echo "Error: Directory '$FOLDER_PATH' does not exist"
    exit 1
fi

# Process each JSON file
for json_file in "$FOLDER_PATH"/*.json; do
    # Check if any JSON files exist
    [ -e "$json_file" ] || { echo "No JSON files found in $FOLDER_PATH"; exit 1; }
    
    # Extract filename without path and extension
    filename=$(basename "$json_file" .json)
    
    echo "Processing: $filename"
    
    # Run first script
    echo "  Running loop_inner.py..."
    python loop_inner.py -f "$filename"
    
    # Run second script
    echo "  Running report_annex.py..."
    python report_annex.py -f "$filename"
    
    echo "  Completed: $filename"
    echo "---"
done

echo "All individual files processed!"
echo ""
echo "Running report merger..."
python report_merger.py

echo "✅ All processing complete - reports merged!"
