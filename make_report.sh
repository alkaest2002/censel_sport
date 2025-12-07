#!/bin/bash

# Default folder path
FOLDER_PATH="./config"

# Flag to control whether to run loop_inner.py
RUN_LOOP_INNER=true

# Parse command line arguments
while getopts "s" opt; do
    case $opt in
        s)
            RUN_LOOP_INNER=false
            echo "Skipping loop_inner.py execution"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            echo "Usage: $0 [-s]"
            echo "  -s: Skip loop_inner.py execution"
            exit 1
            ;;
    esac
done

# Process each JSON file
for json_file in "$FOLDER_PATH"/*.json; do
    # Check if any JSON files exist
    [ -e "$json_file" ] || { echo "No JSON files found in $FOLDER_PATH"; exit 1; }
    
    # Extract filename without path and extension
    filename=$(basename "$json_file" .json)
    
    # Conditionally run first script
    if [ "$RUN_LOOP_INNER" = true ]; then
        python loop_inner.py -f "$filename"
    fi
    
    # Run second script
    python report_annex.py -f "$filename"
    
done

# Process TOC
echo "Running report_toc.py..."
python report_toc.py -n 9

# Proceess DB stats
echo "Running report_db.py..."
python report_db_stats.py -n 10

# Process DB norms
echo "Running report_db_norms.py..."
python report_db_norms.py -l L -n 51

echo "All individual files processed!"
echo ""
echo "Running report merger..."
python report_merger.py

echo "✅ All processing complete - reports merged!"
