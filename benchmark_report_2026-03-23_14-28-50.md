
# AI Home Planner - Performance Benchmark Report

**Report Generated:** 2026-03-23_14-28-50

## Executive Summary

Deep learning inference + 3D extraction executes in **~1.74 seconds**, proving highly viable for real-time web interaction.

## Benchmark Configuration

- **Test Image:** d:\project\AI_Home_Project\uploads\358c0335bd124568acf0981abcf4ae61_sample.png
- **Number of Iterations:** 5
- **Endpoint:** http://localhost:8000/upload-sketch/
- **Timestamp:** 2026-03-23_14-28-50

## Performance Results

### Overall System Performance

| Metric | Value |
|--------|-------|
| Average Total Latency | 1743.02 ms (~1.743 seconds) |
| Minimum Latency | 1649.66 ms |
| Maximum Latency | 1856.94 ms |
| Standard Deviation | 79.29 ms |
| 95th Percentile | 1856.94 ms |

### Component-wise Timing Breakdown

#### Computer Vision Preprocessing
- Average: 27.01 ms
- Min: 24.28 ms
- Max: 31.15 ms
- Std Dev: 2.69 ms

#### Deep Learning Inference (U-Net)
- Average: 27.01 ms
- Min: 24.28 ms
- Max: 31.15 ms
- Std Dev: 2.69 ms

#### 3D Layout Extraction
- Average: 49.74 ms
- Min: 43.41 ms
- Max: 55.26 ms
- Std Dev: 4.28 ms

## Proof of Real-Time Viability

[PASS] **TARGET ACHIEVED:** 1743.02 ms < 2600 ms (2.6 seconds)

The pipeline successfully demonstrates real-time performance capabilities:
- **Average execution time:** 1.743 seconds
- **Target threshold:** 2.6 seconds
- **Margin:** 856.98 ms

This validates the system's suitability for interactive web-based floor plan processing.

## Data Files

- Full benchmark data: `benchmark_report_2026-03-23_14-28-50.json`
- Raw iteration data available in JSON file

## Methodology

- **Test Procedure:** Sequential iterations of the complete pipeline
- **Sample Size:** 5 iterations
- **Environment:** HTTP POST requests to local inference endpoint
- **Measurement:** End-to-end latency from request to response

