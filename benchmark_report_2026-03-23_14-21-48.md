
# AI Home Planner - Performance Benchmark Report

**Report Generated:** 2026-03-23_14-21-48

## Executive Summary

Deep learning inference + 3D extraction executes in **~1.75 seconds**, proving highly viable for real-time web interaction.

## Benchmark Configuration

- **Test Image:** d:\project\AI_Home_Project\uploads\358c0335bd124568acf0981abcf4ae61_sample.png
- **Number of Iterations:** 20
- **Endpoint:** http://localhost:8000/upload-sketch/
- **Timestamp:** 2026-03-23_14-21-48

## Performance Results

### Overall System Performance

| Metric | Value |
|--------|-------|
| Average Total Latency | 1751.24 ms (~1.751 seconds) |
| Minimum Latency | 1711.60 ms |
| Maximum Latency | 1827.57 ms |
| Standard Deviation | 28.56 ms |
| 95th Percentile | 1827.57 ms |

### Component-wise Timing Breakdown

#### Computer Vision Preprocessing
- Average: 28.38 ms
- Min: 24.35 ms
- Max: 36.21 ms
- Std Dev: 3.08 ms

#### Deep Learning Inference (U-Net)
- Average: 28.38 ms
- Min: 24.35 ms
- Max: 36.21 ms
- Std Dev: 3.08 ms

#### 3D Layout Extraction
- Average: 47.41 ms
- Min: 45.00 ms
- Max: 53.52 ms
- Std Dev: 2.03 ms

## Proof of Real-Time Viability

[PASS] **TARGET ACHIEVED:** 1751.24 ms < 2600 ms (2.6 seconds)

The pipeline successfully demonstrates real-time performance capabilities:
- **Average execution time:** 1.751 seconds
- **Target threshold:** 2.6 seconds
- **Margin:** 848.76 ms

This validates the system's suitability for interactive web-based floor plan processing.

## Data Files

- Full benchmark data: `benchmark_report_2026-03-23_14-21-48.json`
- Raw iteration data available in JSON file

## Methodology

- **Test Procedure:** Sequential iterations of the complete pipeline
- **Sample Size:** 20 iterations
- **Environment:** HTTP POST requests to local inference endpoint
- **Measurement:** End-to-end latency from request to response

