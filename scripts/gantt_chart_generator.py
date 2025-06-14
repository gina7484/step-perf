import csv
import json
import os
import argparse
from collections import defaultdict


def parse_csv(csv_file):
    """Parse a CSV file and extract the necessary data grouped by name_id combination."""
    data_by_name_id = defaultdict(list)
    global_min_time = float("inf")
    global_max_time = 0
    unique_name_ids = set()

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return {}, [], 0, 0  # Empty file

        # Process each row and group by name_id combination
        for row_idx, row in enumerate(rows, 1):
            # Check if required columns exist
            if (
                "start_ns" not in row
                or "end_ns" not in row
                or "id" not in row
                or "name" not in row
            ):
                print(
                    f"Warning: Row {row_idx} missing required columns (id, name, start_ns, end_ns)."
                )
                continue

            # Parse values
            try:
                event_id = str(row["id"])  # Convert to string for consistency
                event_name = str(
                    row["name"]
                ).strip()  # Convert to string and strip whitespace
                start_time = float(row["start_ns"])
                end_time = float(row["end_ns"])
                is_stop = row.get("is_stop", "").lower() == "true"
            except ValueError as e:
                print(f"Warning: Row {row_idx} has invalid numeric values: {e}")
                continue

            # Create composite key using name and id
            name_id_key = f"{event_name}_{event_id}"
            unique_name_ids.add(name_id_key)

            # Create identifier for this specific event
            event_count = len(data_by_name_id[name_id_key]) + 1
            identifier = f"event_{event_name}_{event_id}_{event_count}"

            # Store data
            item = {
                "file_id": name_id_key,  # Using name_id combination as file_id for compatibility with existing HTML
                "name": event_name,
                "id": event_id,
                "prefix": "event",
                "identifier": identifier,
                "start": start_time,
                "end": end_time,
                "is_stop": is_stop,
            }
            data_by_name_id[name_id_key].append(item)

            # Track min and max times for the timeline
            global_min_time = min(global_min_time, start_time)
            global_max_time = max(global_max_time, end_time)

    # Convert to list format expected by the HTML generator
    all_data = []
    # Sort by id in ascending order
    sorted_name_ids = sorted(
        unique_name_ids,
        key=lambda x: (
            int(x.rsplit("_", 1)[1])
            if x.rsplit("_", 1)[1].isdigit()
            else x.rsplit("_", 1)[1]
        ),
    )

    for name_id_key in sorted_name_ids:
        all_data.extend(data_by_name_id[name_id_key])

    # Check if we actually added any data
    if not all_data:
        print(f"Warning: No valid data extracted from {csv_file}")
        return {}, [], 0, 0

    return all_data, sorted_name_ids, global_min_time, global_max_time


def generate_html(data, name_id_list, min_time, max_time):
    """Generate HTML for the Gantt chart visualization."""
    # Group data by file_id (which is now the name_id combination)
    file_data = defaultdict(list)
    for item in data:
        file_data[item["file_id"]].append(item)

    # Generate JSON data for the visualization
    visualization_data = []
    for name_id_key in name_id_list:
        items = file_data[name_id_key]
        # Sort by start time
        items.sort(key=lambda x: x["start"])

        for item in items:
            visualization_data.append(
                {
                    "file_id": name_id_key,
                    "name": item["name"],
                    "id": item["id"],
                    "prefix": item["prefix"],
                    "identifier": item["identifier"],
                    "start": item["start"],
                    "end": item["end"],
                    "is_stop": item["is_stop"],
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
            top: 10px;
            width: 300px;
            font-weight: bold;
            text-align: right;
            padding-right: 10px;
            overflow: visible;
            word-wrap: break-word;
            line-height: 1.2;
        }
        .timeline-container {
            margin-left: 320px;
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
        .event-normal {
            background-color: #33A8FF; /* Blue for normal events */
        }
        .event-stop {
            background-color: #FF5733; /* Red-orange for stop events */
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
            <div class="legend-color event-normal"></div>
            <span>Normal Event</span>
        </div>
        <div class="legend-item">
            <div class="legend-color event-stop"></div>
            <span>Stop Event</span>
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
        const nameIdList = """
        + json.dumps(name_id_list)
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
            
            // Create a row for each name_id combination
            nameIdList.forEach(nameId => {
                const fileRow = document.createElement('div');
                fileRow.className = 'file-row';
                
                const fileLabel = document.createElement('div');
                fileLabel.className = 'file-label';
                fileLabel.textContent = nameId;  // Display as name_id
                fileLabel.title = nameId; // Add tooltip for long labels
                
                const timelineContainer = document.createElement('div');
                timelineContainer.className = 'timeline-container';
                timelineContainer.style.width = `${timelineWidth}px`;
                
                fileRow.appendChild(fileLabel);
                fileRow.appendChild(timelineContainer);
                timelineEl.appendChild(fileRow);
                
                // Add blocks for this name_id combination
                const fileDataItems = data.filter(item => item.file_id === nameId);
                fileDataItems.forEach(item => {
                    const block = document.createElement('div');
                    
                    // Set color class based on is_stop flag
                    block.className = item.is_stop ? 'block event-stop' : 'block event-normal';
                    
                    // Position and size based on time values
                    const left = (item.start - minTime) * scale;
                    const width = (item.end - item.start) * scale;
                    
                    block.style.left = `${left}px`;
                    block.style.width = `${Math.max(width, 1)}px`;
                    
                    // Only show text if there's enough space
                    if (width > 40) {
                        block.textContent = item.identifier;
                    }
                    
                    // Add tooltip data
                    block.dataset.identifier = item.identifier;
                    block.dataset.name = item.name;
                    block.dataset.id = item.id;
                    block.dataset.start = item.start;
                    block.dataset.end = item.end;
                    block.dataset.isStop = item.is_stop;
                    
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
                marker.style.left = `${(t - minTime) * scale + 320}px`;
                
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
                Name: ${block.dataset.name}<br>
                ID: ${block.dataset.id}<br>
                Identifier: ${block.dataset.identifier}<br>
                Start: ${parseFloat(block.dataset.start).toFixed(2)} ns<br>
                End: ${parseFloat(block.dataset.end).toFixed(2)} ns<br>
                Duration: ${(parseFloat(block.dataset.end) - parseFloat(block.dataset.start)).toFixed(2)} ns<br>
                Type: ${block.dataset.isStop === 'true' ? 'Stop Event' : 'Normal Event'}
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
        description="Generate Gantt chart HTML from a single CSV file with name_id grouping, sorted by ID."
    )
    parser.add_argument("--csv_file", required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--output_file", required=True, help="Path to the output HTML file."
    )

    args = parser.parse_args()

    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found.")
        return

    # Process the CSV file
    print(f"Processing {args.csv_file}...")
    data, name_id_list, min_time, max_time = parse_csv(args.csv_file)

    # Check if we have data
    if not data:
        print("Error: No valid data found in the CSV file.")
        return

    print(
        f"Found {len(name_id_list)} unique name_id combinations (sorted by ID): {name_id_list}"
    )
    print(f"Time range: {min_time} - {max_time} ns")
    print(f"Total events: {len(data)}")

    # Generate HTML content
    html_content = generate_html(data, name_id_list, min_time, max_time)

    # Write to output file
    with open(args.output_file, "w") as f:
        f.write(html_content)

    print(f"HTML Gantt chart generated at {args.output_file}")


if __name__ == "__main__":
    main()
