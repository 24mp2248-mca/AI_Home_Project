
# AI Home Planner - Performance Benchmark Report

**Report Generated:** 2026-03-23_14-34-33

## Executive Summary

Deep learning inference + 3D extraction executes in **~1.90 seconds**, proving highly viable for real-time web interaction.

## Benchmark Configuration

- **Test Image:** d:\project\AI_Home_Project\uploads\358c0335bd124568acf0981abcf4ae61_sample.png
- **Number of Iterations:** 8
- **Endpoint:** http://localhost:8000/upload-sketch/
- **Timestamp:** 2026-03-23_14-34-33

## Performance Results

### Overall System Performance

| Metric | Value |
|--------|-------|
| Average Total Latency | 1901.38 ms (~1.901 seconds) |
| Minimum Latency | 1719.41 ms |
| Maximum Latency | 2496.90 ms |
| Standard Deviation | 296.07 ms |
| 95th Percentile | 2496.90 ms |

### Component-wise Timing Breakdown

#### Computer Vision Preprocessing
- Average: 26.29 ms
- Min: 23.16 ms
- Max: 31.63 ms
- Std Dev: 2.43 ms

#### Deep Learning Inference (U-Net)
- Average: 26.29 ms
- Min: 23.16 ms
- Max: 31.63 ms
- Std Dev: 2.43 ms

#### 3D Layout Extraction
- Average: 52.10 ms
- Min: 44.00 ms
- Max: 66.24 ms
- Std Dev: 8.81 ms

## Data Files

- Full benchmark data: `benchmark_report_2026-03-23_14-34-33.json`
- Raw iteration data available in JSON file

## Methodology

- **Test Procedure:** Sequential iterations of the complete pipeline
- **Sample Size:** 8 iterations
- **Environment:** HTTP POST requests to local inference endpoint
- **Measurement:** End-to-end latency from request to response

