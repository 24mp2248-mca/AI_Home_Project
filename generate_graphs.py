import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def get_latest_benchmark_file():
    """Find the most recent benchmark JSON report."""
    files = glob.glob("benchmark_report_*.json")
    if not files:
        return None
    return max(files, key=os.path.getctime)

def generate_benchmark_graphs(json_path=None):
    """Generate visualization from benchmark data."""
    if json_path is None:
        json_path = get_latest_benchmark_file()
    
    if not json_path or not os.path.exists(json_path):
        print(f"❌ Error: Benchmark file not found: {json_path}")
        return False

    print(f"📊 Generating graphs from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    stats = data.get('statistics', {})
    raw_results = data.get('raw_results', [])
    
    if not stats or not raw_results:
        print("❌ Error: Invalid benchmark data format.")
        return False

    # Prepare data for plotting
    components = {
        "cv_preprocessing": "CV Preprocessing",
        "dl_inference": "DL Inference",
        "layout_extraction": "Layout Extraction",
        "extraction_3d": "3D Extraction",
        "total_latency": "Total Pipeline"
    }

    labels = []
    averages = []
    mins = []
    maxs = []
    
    for key, label in components.items():
        if key in stats:
            labels.append(label)
            averages.append(stats[key]['average'] / 1000)
            mins.append(stats[key]['min'] / 1000)
            maxs.append(stats[key]['max'] / 1000)

    # 1. Component Latency Breakdown (Bar Chart)
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x, averages, width, color='skyblue', label='Average')
    plt.errorbar(x, averages, yerr=[np.array(averages)-np.array(mins), np.array(maxs)-np.array(averages)], 
                 fmt='none', ecolor='gray', capsize=5, label='Min-Max Range')
    
    plt.ylabel('Time (seconds)')
    plt.title('Component Latency Breakdown')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Latency over Iterations (Line Chart)
    plt.subplot(2, 2, 2)
    iterations = [r['iteration'] for r in raw_results if r['status'] == 'Success']
    total_latencies = [r['total_latency_ms'] / 1000 for r in raw_results if r['status'] == 'Success']
    
    plt.plot(iterations, total_latencies, marker='o', linestyle='-', color='coral')
    plt.axhline(y=np.mean(total_latencies), color='r', linestyle='--', label=f'Avg: {np.mean(total_latencies):.3f}s')
    
    plt.xlabel('Iteration')
    plt.ylabel('Total Latency (seconds)')
    plt.title('Performance Stability over Iterations')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. Component Distribution (Box Plot)
    plt.subplot(2, 2, 3)
    dist_data = []
    dist_labels = []
    
    iteration_data = data.get('all_iterations_data', {})
    for key, label in components.items():
        if key + "_ms" in iteration_data:
            dist_data.append([t / 1000 for t in iteration_data[key + "_ms"]])
            dist_labels.append(label)
        elif key in iteration_data: # Fallback for total_latency
            dist_data.append([t / 1000 for t in iteration_data[key]])
            dist_labels.append(label)

    plt.boxplot(dist_data, labels=dist_labels)
    plt.ylabel('Time (seconds)')
    plt.title('Component Latency Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Latency Composition (Pie Chart of Averages)
    plt.subplot(2, 2, 4)
    # Exclude total_latency from pie chart
    pie_labels = labels[:-1]
    pie_values = averages[:-1]
    
    plt.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Average Latency Composition')

    plt.tight_layout()
    output_path = "benchmark_analysis.png"
    plt.savefig(output_path)
    print(f"✅ Benchmark Analysis Graph saved to: {output_path}")
    return True

if __name__ == "__main__":
    generate_benchmark_graphs()
