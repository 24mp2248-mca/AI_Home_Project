
# AI Home Planner - Performance Benchmark Report

**Report Generated:** 2026-03-23_14-31-56

## Executive Summary

Deep learning inference + 3D extraction executes in **~1.75 seconds**, proving highly viable for real-time web interaction.

## Benchmark Configuration

- **Test Image:** d:\project\AI_Home_Project\uploads\358c0335bd124568acf0981abcf4ae61_sample.png
- **Number of Iterations:** 5
- **Endpoint:** http://localhost:8000/upload-sketch/
- **Timestamp:** 2026-03-23_14-31-56

## Performance Results

### Overall System Performance

| Metric | Value |
|--------|-------|
| Average Total Latency | 1750.94 ms (~1.751 seconds) |
| Minimum Latency | 1721.54 ms |
| Maximum Latency | 1775.02 ms |
| Standard Deviation | 19.61 ms |
| 95th Percentile | 1775.02 ms |

### Component-wise Timing Breakdown

#### Computer Vision Preprocessing
- Average: 29.16 ms
- Min: 24.34 ms
- Max: 34.71 ms
- Std Dev: 3.72 ms

#### Deep Learning Inference (U-Net)
- Average: 29.16 ms
- Min: 24.34 ms
- Max: 34.71 ms
- Std Dev: 3.72 ms

#### 3D Layout Extraction
- Average: 50.29 ms
- Min: 46.40 ms
- Max: 53.61 ms
- Std Dev: 2.82 ms

## Proof of Real-Time Viability

[PASS] **TARGET ACHIEVED:** 1750.94 ms < 2600 ms (2.6 seconds)

The pipeline successfully demonstrates real-time performance capabilities:
- **Average execution time:** 1.751 seconds
- **Target threshold:** 2.6 seconds
- **Margin:** 849.06 ms

This validates the system's suitability for interactive web-based floor plan processing.

## Data Files

- Full benchmark data: `benchmark_report_2026-03-23_14-31-56.json`
- Raw iteration data available in JSON file

## Methodology

- **Test Procedure:** Sequential iterations of the complete pipeline
- **Sample Size:** 5 iterations
- **Environment:** HTTP POST requests to local inference endpoint
- **Measurement:** End-to-end latency from request to response

