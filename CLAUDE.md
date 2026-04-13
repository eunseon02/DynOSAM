# Project context

This project is an existing DynoSAM-based pipeline.
The system is already mostly implemented.

## Goal
Improve odometry accuracy by reducing trajectory error.

## Evaluation
Run the pipeline with:
`run_dynosam.sh`

Evaluate the output trajectory using:
`evo_ape tum /root/dataset/tum-rgbd/SEQ_NAME/groundtruth.txt /root/result/OUTPUT_TRAJECTORY -vap`
`evo_rpe tum /root/dataset/tum-rgbd/SEQ_NAME/groundtruth.txt /root/result/OUTPUT_TRAJECTORY -vap`

Use RMSE as the primary metric to reduce.

## Constraints
- Do not rewrite the entire pipeline.
- Keep the existing local map structure unless explicitly requested.
- Prefer minimal, local, testable changes.
- Prioritize changes that can be evaluated with ablation.

## Current research directions
1. Improve structural edge extraction.
2. Add a global map on top of the current local map to reduce drift.

## Important notes
- Structural edges are currently inspected through `viz_edge_depth_discontinuity`.
- The current issue is that desired structural edges are not extracted reliably.
- Focus on performance improvement, not codebase refactoring.