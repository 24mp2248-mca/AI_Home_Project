
# AI Home Planner - Performance Benchmark Report

**Report Generated:** 2026-03-23_15-11-13

## Executive Summary

Deep learning inference + 3D extraction executes in **~1.75 seconds**, proving highly viable for real-time web interaction.

## Benchmark Configuration

- **Test Image:** d:\project\AI_Home_Project\uploads\00b5a4c56ff94a5b87136c6d24acde45_358c0335bd124568acf0981abcf4ae61_sample.png
- **Number of Iterations:** 3
- **Endpoint:** http://localhost:8000/upload-sketch/
- **Timestamp:** 2026-03-23_15-11-13

## Performance Results

### Overall System Performance

| Metric | Value |
|--------|-------|
| Average Total Latency | 1748.47 ms (~1.748 seconds) |
| Minimum Latency | 1709.96 ms |
| Maximum Latency | 1810.06 ms |
| Standard Deviation | 53.89 ms |
| 95th Percentile | 1810.06 ms |

### Component-wise Timing Breakdown

#### Computer Vision Preprocessing
- Average: 25.61 ms
- Min: 24.08 ms
- Max: 27.69 ms
- Std Dev: 1.87 ms

#### Deep Learning Inference (U-Net)
- Average: 25.61 ms
- Min: 24.08 ms
- Max: 27.69 ms
- Std Dev: 1.87 ms

#### 3D Layout Extraction
- Average: 50.26 ms
- Min: 48.24 ms
- Max: 51.74 ms
- Std Dev: 1.81 ms

## Data Files

- Full benchmark data: `benchmark_report_2026-03-23_15-11-13.json`
- Raw iteration data available in JSON file

## Methodology

- **Test Procedure:** Sequential iterations of the complete pipeline
- **Sample Size:** 3 iterations
- **Environment:** HTTP POST requests to local inference endpoint
- **Measurement:** End-to-end latency from request to response

