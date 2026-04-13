#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_dynosam.sh fr2_xyz
# If no argument is given, it will prompt for the sequence name (e.g. fr2_xyz).

SEQ_NAME="${1:-}"
if [[ -z "${SEQ_NAME}" ]]; then
  read -rp "Sequence name (e.g. fr2_xyz): " SEQ_NAME
fi
EXTRA_ARGS=("${@:2}")

TUM_ROOT="/root/data/tum-rgbd"
RESULTS_ROOT="/root/results"

TUM_DIR="${TUM_ROOT}/${SEQ_NAME}/"
TUM_ASSOC="${TUM_ROOT}/${SEQ_NAME}/${SEQ_NAME}_associated.txt"
TUM_DETECTIONS_JSON="${TUM_ROOT}/${SEQ_NAME}/detections_yolov8x_seg_${SEQ_NAME}_with_ellipse.json"
OUTPUT_TRAJECTORY="${RESULTS_ROOT}/${SEQ_NAME}.txt"

BIN="/home/user/dev_ws/install/dynosam/lib/dynosam/dyno_sam"

export LD_LIBRARY_PATH="/home/user/dev_ws/install/dynosam/lib:/home/user/dev_ws/install/dynosam_common/lib:/home/user/dev_ws/install/dynosam_cv/lib:/home/user/dev_ws/install/dynosam_nn/lib:/home/user/dev_ws/install/dynosam_opt/lib:/home/user/dev_ws/install/dynosam_ros/lib:/home/user/dev_ws/install/dynosam_utils/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -x "${BIN}" ]]; then
  echo "Error: dyno_sam binary not found or not executable: ${BIN}" >&2
  exit 1
fi

echo "Running dyno_sam for seq: ${SEQ_NAME}"
echo "  TUM_DIR: ${TUM_DIR}"
echo "  OUTPUT_TRAJECTORY: ${OUTPUT_TRAJECTORY}"

exec "${BIN}" \
  --use_tum \
  --path_to_tum="${TUM_DIR}" \
  --tum_association="${TUM_ASSOC}" \
  --tum_detections_json="${TUM_DETECTIONS_JSON}" \
  --use_pipeline=true \
  --output_trajectory="${OUTPUT_TRAJECTORY}" \
  --use_dynamic_track=false \
  --v=1 \
  "${EXTRA_ARGS[@]}"

