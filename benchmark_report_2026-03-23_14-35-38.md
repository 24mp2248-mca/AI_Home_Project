
# AI Home Planner - Performance Benchmark Report

**Report Generated:** 2026-03-23_14-35-38

## Executive Summary

Deep learning inference + 3D extraction executes in **~1.79 seconds**, proving highly viable for real-time web interaction.

## Benchmark Configuration

- **Test Image:** d:\project\AI_Home_Project\uploads\358c0335bd124568acf0981abcf4ae61_sample.png
- **Number of Iterations:** 7
- **Endpoint:** http://localhost:8000/upload-sketch/
- **Timestamp:** 2026-03-23_14-35-38

## Performance Results

### Overall System Performance

| Metric | Value |
|--------|-------|
| Average Total Latency | 1787.91 ms (~1.788 seconds) |
| Minimum Latency | 1690.59 ms |
| Maximum Latency | 1925.67 ms |
| Standard Deviation | 87.87 ms |
| 95th Percentile | 1925.67 ms |

### Component-wise Timing Breakdown

#### Computer Vision Preprocessing
- Average: 27.58 ms
- Min: 25.09 ms
- Max: 31.35 ms
- Std Dev: 2.35 ms

#### Deep Learning Inference (U-Net)
- Average: 27.58 ms
- Min: 25.09 ms
- Max: 31.35 ms
- Std Dev: 2.35 ms

#### 3D Layout Extraction
- Average: 48.68 ms
- Min: 44.36 ms
- Max: 51.56 ms
- Std Dev: 3.28 ms

## Data Files

- Full benchmark data: `benchmark_report_2026-03-23_14-35-38.json`
- Raw iteration data available in JSON file

## Methodology

- **Test Procedure:** Sequential iterations of the complete pipeline
- **Sample Size:** 7 iterations
- **Environment:** HTTP POST requests to local inference endpoint
- **Measurement:** End-to-end latency from request to response

