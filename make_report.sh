#!/bin/bash

# Default folder path
FOLDER_PATH="./config"

# Flags to control execution flow
RUN_LOOP_INNER=false
RUN_RENDER=false
RUN_MERGE=false

# Parse command line arguments
while getopts "mrx" opt; do
    case $opt in
        m)
            RUN_MERGE=true
            echo "Mode: Merge only"
            ;;
        r)
            RUN_RENDER=true
            RUN_MERGE=true
            echo "Mode: Render and merge"
            ;;
        x)
            RUN_LOOP_INNER=true
            RUN_RENDER=true
            RUN_MERGE=true
            echo "Mode: Full execution (inner loop + render + merge)"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            echo "Usage: $0 [-m|-r|-x]"
            echo "  -m: Merge only"
            echo "  -r: Render and merge"
            echo "  -x: Inner loop + render + merge (full execution)"
            exit 1
            ;;
    esac
done

# Check if at least one option was provided
if [ "$RUN_MERGE" = false ]; then
    echo "Error: Please specify one of the execution modes"
    echo "Usage: $0 [-m|-r|-x]"
    echo "  -m: Merge only"
    echo "  -r: Render and merge"
    echo "  -x: Inner loop + render + merge (full execution)"
    exit 1
fi

# Process each JSON file (only if rendering is enabled)
if [ "$RUN_RENDER" = true ]; then
    for json_file in "$FOLDER_PATH"/*.json; do
        # Check if any JSON files exist
        [ -e "$json_file" ] || { echo "No JSON files found in $FOLDER_PATH"; exit 1; }
        
        # Extract filename without path and extension
        filename=$(basename "$json_file" .json)
        
        # Conditionally run first script
        if [ "$RUN_LOOP_INNER" = true ]; then
            echo "Running loop_inner.py for $filename..."
            python loop_inner.py -f "$filename"
        fi
        
        # Run second script
        echo "Running report_annex.py for $filename..."
        python report_annex.py -f "$filename"
        
    done

    # Process TOC
    python report_toc.py -n 9

    # Process DB stats
    python report_db_recap.py -l A -n 10

    # Process DB stats
    python report_db_stats.py -l L -n 51

    # Process DB norms
    python report_db_norms.py -l M -n 52

    echo "All individual files processed!"
    echo ""
fi

# Run merger (always runs if any mode is selected)
if [ "$RUN_MERGE" = true ]; then
    echo "Running report merger..."
    python report_merger.py
    echo "✅ All processing complete - reports merged!"
fi
