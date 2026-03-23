
# AI Home Planner - Performance Benchmark Report

**Report Generated:** 2026-03-23_14-24-45

## Executive Summary

Deep learning inference + 3D extraction executes in **~1.75 seconds**, proving highly viable for real-time web interaction.

## Benchmark Configuration

- **Test Image:** d:\project\AI_Home_Project\uploads\358c0335bd124568acf0981abcf4ae61_sample.png
- **Number of Iterations:** 15
- **Endpoint:** http://localhost:8000/upload-sketch/
- **Timestamp:** 2026-03-23_14-24-45

## Performance Results

### Overall System Performance

| Metric | Value |
|--------|-------|
| Average Total Latency | 1746.54 ms (~1.747 seconds) |
| Minimum Latency | 1721.25 ms |
| Maximum Latency | 1804.69 ms |
| Standard Deviation | 20.78 ms |
| 95th Percentile | 1804.69 ms |

### Component-wise Timing Breakdown

#### Computer Vision Preprocessing
- Average: 28.09 ms
- Min: 23.31 ms
- Max: 37.36 ms
- Std Dev: 4.26 ms

#### Deep Learning Inference (U-Net)
- Average: 28.09 ms
- Min: 23.31 ms
- Max: 37.36 ms
- Std Dev: 4.26 ms

#### 3D Layout Extraction
- Average: 48.59 ms
- Min: 44.67 ms
- Max: 55.16 ms
- Std Dev: 2.98 ms

## Proof of Real-Time Viability

[PASS] **TARGET ACHIEVED:** 1746.54 ms < 2600 ms (2.6 seconds)

The pipeline successfully demonstrates real-time performance capabilities:
- **Average execution time:** 1.747 seconds
- **Target threshold:** 2.6 seconds
- **Margin:** 853.46 ms

This validates the system's suitability for interactive web-based floor plan processing.

## Data Files

- Full benchmark data: `benchmark_report_2026-03-23_14-24-45.json`
- Raw iteration data available in JSON file

## Methodology

- **Test Procedure:** Sequential iterations of the complete pipeline
- **Sample Size:** 15 iterations
- **Environment:** HTTP POST requests to local inference endpoint
- **Measurement:** End-to-end latency from request to response

