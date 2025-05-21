import csv
import json
import os
import argparse
from collections import defaultdict


def parse_csv(csv_file):
    """Parse a CSV file and extract the necessary data."""
    data = []
    file_id = os.path.basename(csv_file).split(".")[
        0
    ]  # Use filename without extension as ID
    min_time = float("inf")
    max_time = 0

    # Extract prefix from file_id
    prefix = file_id.split("_")[0] if "_" in file_id else file_id

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        # Get the first row to check column names
        rows = list(reader)
        if not rows:
            return [], file_id, 0, 0  # Empty file

        for row in rows:
            # Create identifier based on prefix
            if prefix == "hbm":
                # For hbm files, use the current identifier logic with error handling
                try:
                    if "num_elems" in row:
                        identifier = f"{row.get('outer', 'N/A')},{row.get('m', 'N/A')},{row.get('n', 'N/A')},{row.get('k', 'N/A')} ({row['num_elems']})"
                    else:
                        identifier = f"{row.get('outer', 'N/A')},{row.get('m', 'N/A')},{row.get('n', 'N/A')},{row.get('k', 'N/A')}"
                except KeyError:
                    # If any required keys are missing, use a simpler identifier
                    identifier = f"hbm_{file_id}"
            else:
                # For other prefixes, use the counter column if available, otherwise blank
                identifier = row.get("counter", "")

            # Check if start_ns and end_ns columns exist
            if "start_ns" in row and "end_ns" in row:
                start_time = float(row["start_ns"])
                end_time = float(row["end_ns"])
            # Fallback to old column names if necessary
            elif "start(ms)" in row and "end(ms)" in row:
                start_time = float(row["start(ms)"])
                end_time = float(row["end(ms)"])
            elif "start" in row and "end" in row:
                start_time = float(row["start"])
                end_time = float(row["end"])
            else:
                print(
                    f"Warning: File {file_id} missing timing columns. Available columns: {list(row.keys())}"
                )
                continue  # Skip this row

            # Store data
            item = {
                "file_id": file_id,
                "prefix": prefix,
                "identifier": identifier,
                "start": start_time,
                "end": end_time,
                "output_available": (
                    row.get("output_tile_available") == "True"
                    if "output_tile_available" in row
                    else False
                ),
            }
            data.append(item)

            # Track min and max times for the timeline
            min_time = min(min_time, start_time)
            max_time = max(max_time, end_time)

    # Check if we actually added any data
    if not data:
        print(f"Warning: No data extracted from {file_id}")
        return [], file_id, 0, 0

    return data, file_id, min_time, max_time


def process_multiple_files(csv_files):
    """Process multiple CSV files and combine their data."""
    all_data = []
    all_file_ids = []
    global_min_time = float("inf")
    global_max_time = 0

    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        data, file_id, min_time, max_time = parse_csv(csv_file)

        if data:  # Only add non-empty results
            all_data.extend(data)
            all_file_ids.append(file_id)

            if min_time < global_min_time:
                global_min_time = min_time
            if max_time > global_max_time:
                global_max_time = max_time

    # Check if we have any data
    if not all_data:
        print("Error: No data could be extracted from any of the CSV files.")
        return [], [], 0, 0

    return all_data, all_file_ids, global_min_time, global_max_time


def generate_html(data, file_ids, min_time, max_time):
    """Generate HTML for the Gantt chart visualization."""
    # Group data by file_id
    file_data = defaultdict(list)
    for item in data:
        file_data[item["file_id"]].append(item)

    # Generate JSON data for the visualization
    visualization_data = []
    for file_id in file_ids:
        items = file_data[file_id]
        # Sort by start time
        items.sort(key=lambda x: x["start"])

        for item in items:
            visualization_data.append(
                {
                    "file_id": file_id,
                    "prefix": item["prefix"],
                    "identifier": item["identifier"],
                    "start": item["start"],
                    "end": item["end"],
                    "output_available": item["output_available"],
                }
            )

    html_content = (
        """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSV Gantt Chart Visualization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            overflow: hidden;
        }
        #container {
            width: 100%;
            height: calc(100vh - 100px);
            overflow: auto;
            position: relative;
        }
        #timeline {
            position: relative;
            margin-top: 40px;
        }
        .file-row {
            height: 50px;
            margin-bottom: 5px;
            position: relative;
            border-bottom: 1px solid #eee;
        }
        .file-label {
            position: absolute;
            left: 0;
            top: 15px;
            width: 150px;
            font-weight: bold;
            text-align: right;
            padding-right: 10px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .timeline-container {
            margin-left: 170px;
            position: relative;
            height: 100%;
        }
        .block {
            position: absolute;
            height: 30px;
            top: 10px;
            border-radius: 3px;
            text-align: center;
            font-size: 10px;
            overflow: hidden;
            white-space: nowrap;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        .block:hover {
            opacity: 0.8;
        }
        .prefix-comp {
            background-color: #FF5733; /* Red-orange for compute */
        }
        .prefix-hbm {
            background-color: #33A8FF; /* Blue for HBM */
        }
        .prefix-load {
            background-color: #33FF57; /* Green for load */
        }
        .prefix-store {
            background-color: #A633FF; /* Purple for store */
        }
        .prefix-default {
            background-color: #999999; /* Gray for unknown prefixes */
        }
        .timeline-marker {
            position: absolute;
            width: 1px;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.1);
            top: 0;
        }
        .timeline-label {
            position: absolute;
            font-size: 10px;
            color: #666;
            top: -20px;
            transform: translateX(-50%);
        }
        .controls {
            margin-bottom: 20px;
        }
        .tooltip {
            position: absolute;
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 100;
            pointer-events: none;
            display: none;
        }
        button {
            margin-right: 10px;
            padding: 5px 10px;
            cursor: pointer;
        }
        #scale-slider {
            width: 200px;
            display: inline-block;
            vertical-align: middle;
        }
        .legend {
            margin-top: 10px;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin-right: 20px;
            margin-bottom: 5px;
        }
        .legend-color {
            width: 15px;
            height: 15px;
            border-radius: 3px;
            margin-right: 5px;
        }
    </style>
</head>
<body>
    <h1>CSV Gantt Chart Visualization</h1>
    <div class="controls">
        <button id="zoom-in">Zoom In</button>
        <button id="zoom-out">Zoom Out</button>
        <button id="reset">Reset</button>
        <label for="scale-slider">Scale: </label>
        <input type="range" id="scale-slider" min="1" max="100" value="10">
        <span id="scale-value">1x</span>
    </div>
    <div class="legend">
        <div class="legend-item">
            <div class="legend-color prefix-comp"></div>
            <span>Compute</span>
        </div>
        <div class="legend-item">
            <div class="legend-color prefix-hbm"></div>
            <span>HBM</span>
        </div>
        <div class="legend-item">
            <div class="legend-color prefix-load"></div>
            <span>Load</span>
        </div>
        <div class="legend-item">
            <div class="legend-color prefix-store"></div>
            <span>Store</span>
        </div>
        <div class="legend-item">
            <div class="legend-color prefix-default"></div>
            <span>Other</span>
        </div>
    </div>
    <div id="container">
        <div id="timeline"></div>
    </div>
    <div id="tooltip" class="tooltip"></div>

    <script>
        // Data from Python
        const data = """
        + json.dumps(visualization_data)
        + """;
        const fileIds = """
        + json.dumps(file_ids)
        + """;
        const minTime = """
        + str(min_time)
        + """;
        const maxTime = """
        + str(max_time)
        + """;
        
        // Visualization variables
        let scale = 0.1;
        let timelineEl = document.getElementById('timeline');
        let containerEl = document.getElementById('container');
        let tooltipEl = document.getElementById('tooltip');
        let scaleSlider = document.getElementById('scale-slider');
        let scaleValue = document.getElementById('scale-value');
        
        // Function to render the timeline
        function renderTimeline() {
            timelineEl.innerHTML = '';
            const timelineWidth = (maxTime - minTime) * scale;
            
            // Create a row for each file
            fileIds.forEach(fileId => {
                const fileRow = document.createElement('div');
                fileRow.className = 'file-row';
                
                const fileLabel = document.createElement('div');
                fileLabel.className = 'file-label';
                fileLabel.textContent = fileId;
                fileLabel.title = fileId; // Add tooltip for long filenames
                
                const timelineContainer = document.createElement('div');
                timelineContainer.className = 'timeline-container';
                timelineContainer.style.width = `${timelineWidth}px`;
                
                fileRow.appendChild(fileLabel);
                fileRow.appendChild(timelineContainer);
                timelineEl.appendChild(fileRow);
                
                // Add blocks for this file
                const fileDataItems = data.filter(item => item.file_id === fileId);
                fileDataItems.forEach(item => {
                    const block = document.createElement('div');
                    
                    // Set color class based on prefix
                    const prefix = item.prefix;
                    if (['comp', 'hbm', 'load', 'store'].includes(prefix)) {
                        block.className = `block prefix-${prefix}`;
                    } else {
                        block.className = 'block prefix-default';
                    }
                    
                    // Position and size based on time values
                    const left = (item.start - minTime) * scale;
                    const width = (item.end - item.start) * scale;
                    
                    block.style.left = `${left}px`;
                    block.style.width = `${Math.max(width, 1)}px`;
                    
                    // Only show text if there's enough space and we have an identifier
                    if (width > 40 && item.identifier) {
                        block.textContent = item.identifier;
                    }
                    
                    // Add tooltip data
                    block.dataset.prefix = item.prefix;
                    block.dataset.identifier = item.identifier || 'No identifier';
                    block.dataset.start = item.start;
                    block.dataset.end = item.end;
                    block.dataset.outputAvailable = item.output_available;
                    
                    // Add event listeners for tooltip
                    block.addEventListener('mouseover', showTooltip);
                    block.addEventListener('mousemove', moveTooltip);
                    block.addEventListener('mouseout', hideTooltip);
                    
                    timelineContainer.appendChild(block);
                });
            });
            
            // Add time markers
            const stepSize = calculateStepSize(maxTime - minTime);
            for (let t = minTime; t <= maxTime; t += stepSize) {
                const marker = document.createElement('div');
                marker.className = 'timeline-marker';
                marker.style.left = `${(t - minTime) * scale + 170}px`;
                
                const label = document.createElement('div');
                label.className = 'timeline-label';
                label.textContent = t.toFixed(2) + 'ns';
                
                marker.appendChild(label);
                timelineEl.appendChild(marker);
            }
        }
        
        // Calculate appropriate step size for timeline markers
        function calculateStepSize(range) {
            const targetSteps = 10;
            const roughStep = range / targetSteps;
            
            // Round to a nice number
            const magnitude = Math.pow(10, Math.floor(Math.log10(roughStep)));
            const normalized = roughStep / magnitude;
            
            if (normalized < 1.5) return magnitude;
            if (normalized < 3.5) return 2 * magnitude;
            if (normalized < 7.5) return 5 * magnitude;
            return 10 * magnitude;
        }
        
        // Tooltip functions
        function showTooltip(e) {
            const block = e.target;
            tooltipEl.innerHTML = `
                Type: ${block.dataset.prefix}<br>
                ${block.dataset.identifier !== 'No identifier' ? `Identifier: ${block.dataset.identifier}<br>` : ''}
                Start: ${parseFloat(block.dataset.start).toFixed(2)} ns<br>
                End: ${parseFloat(block.dataset.end).toFixed(2)} ns<br>
                Duration: ${(parseFloat(block.dataset.end) - parseFloat(block.dataset.start)).toFixed(2)} ns<br>
                Output Available: ${block.dataset.outputAvailable}
            `;
            tooltipEl.style.display = 'block';
            moveTooltip(e);
        }
        
        function moveTooltip(e) {
            tooltipEl.style.left = `${e.pageX + 10}px`;
            tooltipEl.style.top = `${e.pageY + 10}px`;
        }
        
        function hideTooltip() {
            tooltipEl.style.display = 'none';
        }
        
        // Initialize controls
        document.getElementById('zoom-in').addEventListener('click', () => {
            scale *= 1.5;
            updateScale();
            renderTimeline();
        });
        
        document.getElementById('zoom-out').addEventListener('click', () => {
            scale /= 1.5;
            updateScale();
            renderTimeline();
        });
        
        document.getElementById('reset').addEventListener('click', () => {
            scale = 0.1;
            updateScale();
            renderTimeline();
            containerEl.scrollLeft = 0;
        });
        
        scaleSlider.addEventListener('input', () => {
            scale = scaleSlider.value / 100;
            updateScale();
            renderTimeline();
        });
        
        function updateScale() {
            scaleValue.textContent = `${scale.toFixed(2)}x`;
            scaleSlider.value = scale * 100;
        }
        
        // Initial render
        updateScale();
        renderTimeline();
    </script>
</body>
</html>
    """
    )

    return html_content


def main():
    parser = argparse.ArgumentParser(
        description="Generate Gantt chart HTML from CSV files."
    )
    parser.add_argument("--csv_files", nargs="+", help="Path to the input CSV files.")
    parser.add_argument("--output_file", help="Path to the output HTML file.")

    args = parser.parse_args()

    # Check if CSV files were provided
    if not args.csv_files:
        print(
            "Error: No CSV files provided. Please use --csv_files to specify input files."
        )
        return

    # Process the CSV files
    data, file_ids, min_time, max_time = process_multiple_files(args.csv_files)

    # Check if we have data
    if not data:
        print("Error: No valid data found in any of the provided CSV files.")
        return

    # Generate HTML content
    html_content = generate_html(data, file_ids, min_time, max_time)

    # Write to output file
    with open(args.output_file, "w") as f:
        f.write(html_content)

    print(f"HTML Gantt chart generated at {args.output_file}")


if __name__ == "__main__":
    main()
