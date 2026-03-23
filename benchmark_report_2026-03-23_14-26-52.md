
# AI Home Planner - Performance Benchmark Report

**Report Generated:** 2026-03-23_14-26-52

## Executive Summary

Deep learning inference + 3D extraction executes in **~1.79 seconds**, proving highly viable for real-time web interaction.

## Benchmark Configuration

- **Test Image:** d:\project\AI_Home_Project\uploads\358c0335bd124568acf0981abcf4ae61_sample.png
- **Number of Iterations:** 10
- **Endpoint:** http://localhost:8000/upload-sketch/
- **Timestamp:** 2026-03-23_14-26-52

## Performance Results

### Overall System Performance

| Metric | Value |
|--------|-------|
| Average Total Latency | 1787.25 ms (~1.787 seconds) |
| Minimum Latency | 1724.28 ms |
| Maximum Latency | 1893.99 ms |
| Standard Deviation | 49.29 ms |
| 95th Percentile | 1893.99 ms |

### Component-wise Timing Breakdown

#### Computer Vision Preprocessing
- Average: 28.00 ms
- Min: 23.40 ms
- Max: 33.13 ms
- Std Dev: 3.81 ms

#### Deep Learning Inference (U-Net)
- Average: 28.00 ms
- Min: 23.40 ms
- Max: 33.13 ms
- Std Dev: 3.81 ms

#### 3D Layout Extraction
- Average: 49.72 ms
- Min: 46.44 ms
- Max: 56.31 ms
- Std Dev: 3.45 ms

## Proof of Real-Time Viability

[PASS] **TARGET ACHIEVED:** 1787.25 ms < 2600 ms (2.6 seconds)

The pipeline successfully demonstrates real-time performance capabilities:
- **Average execution time:** 1.787 seconds
- **Target threshold:** 2.6 seconds
- **Margin:** 812.75 ms

This validates the system's suitability for interactive web-based floor plan processing.

## Data Files

- Full benchmark data: `benchmark_report_2026-03-23_14-26-52.json`
- Raw iteration data available in JSON file

## Methodology

- **Test Procedure:** Sequential iterations of the complete pipeline
- **Sample Size:** 10 iterations
- **Environment:** HTTP POST requests to local inference endpoint
- **Measurement:** End-to-end latency from request to response

