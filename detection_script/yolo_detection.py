import argparse
import glob
import json
import math
import os

import cv2
import numpy as np
from ultralytics import YOLO


def resolve_path(path_value, base_dir):
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(base_dir, path_value))


def ensure_parent_dir(file_path):
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)


def write_association_txt(correspondence, output_path):
    """Write TUM-style RGB-D association txt:
    <t_rgb> rgb/<rgb_file> <t_depth> depth/<depth_file>
    """
    ensure_parent_dir(output_path)
    with open(output_path, "w") as f:
        for rgb_path, depth_path in zip(correspondence["rgb"], correspondence["depth"]):
            rgb_name = os.path.basename(rgb_path)
            depth_name = os.path.basename(depth_path)
            t_rgb = os.path.splitext(rgb_name)[0]
            t_depth = os.path.splitext(depth_name)[0]
            line = f"{t_rgb} rgb/{rgb_name} {t_depth} depth/{depth_name}\n"
            f.write(line)
    print(f"Association TXT file created: {output_path}")


def estimate_mask_contour(mask_box):
    mask_box = (mask_box * 255).astype(np.uint8)
    canny = cv2.Canny(mask_box, 100, 150)
    contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    return contour


def read_timestamps_from_pngs(directory):
    files = sorted(filter(os.path.isfile, glob.glob(os.path.join(directory, "*.png"))))
    timestamps = []
    for file_path in files:
        _, filename = os.path.split(file_path)
        timestamps.append(float(filename[:-4]))
    return files, timestamps


def nearest_match_indices(source_times, target_times):
    indices = []
    j = 0
    for source in source_times:
        best_delta = float("inf")
        while j < len(target_times):
            delta = abs(source - target_times[j])
            if delta <= best_delta:
                best_delta = delta
                j += 1
            else:
                break
        indices.append(max(0, j - 1))
    return indices


def build_tum_correspondence(data_path):
    rgb_dir = os.path.join(data_path, "rgb")
    depth_dir = os.path.join(data_path, "depth")
    gt_file = os.path.join(data_path, "groundtruth.txt")

    if not os.path.isdir(rgb_dir):
        raise ValueError(f"RGB directory not found: {rgb_dir}")
    if not os.path.isdir(depth_dir):
        raise ValueError(f"Depth directory not found: {depth_dir}")
    if not os.path.isfile(gt_file):
        raise ValueError(f"Ground truth file not found: {gt_file}")

    rgb_files, rgb_times = read_timestamps_from_pngs(rgb_dir)
    depth_files, depth_times = read_timestamps_from_pngs(depth_dir)

    if len(rgb_times) == 0:
        raise ValueError(f"No rgb .png files found under: {rgb_dir}")
    if len(depth_times) == 0:
        raise ValueError(f"No depth .png files found under: {depth_dir}")

    depth_indices = nearest_match_indices(rgb_times, depth_times)
    corres_depth = [depth_files[idx] for idx in depth_indices]

    gt = np.loadtxt(gt_file, delimiter=" ")
    timestamp_gt = list(np.array(gt)[:, 0])
    gt_indices = nearest_match_indices(rgb_times, timestamp_gt)
    corres_gt = [list(gt[idx]) for idx in gt_indices]

    return {"rgb": rgb_files, "depth": corres_depth, "gt": corres_gt}


def run_detection(correspondence, predictor, mask_dir, detection_output, no_ellipse: bool):
    list_to_save = []

    for rgb_path in correspondence["rgb"]:
        _, filename = os.path.split(rgb_path)
        dict_per_im = {"file_name": filename, "detections": []}

        im_rgb = cv2.imread(rgb_path)
        if im_rgb is None:
            print(f"[WARN] Failed to read image: {rgb_path}")
            list_to_save.append(dict_per_im)
            continue

        if mask_dir:
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)
            h, w = im_rgb.shape[:2]
            seg_id = np.zeros((h, w), dtype=np.uint16)
            seg_color = np.zeros((h, w, 3), dtype=np.uint8)

        results = predictor.predict(
            im_rgb, save=False, conf=0.1, device=0, visualize=False, show=False
        )
        boxes = results[0].boxes.to("cpu").numpy() if results[0].boxes is not None else None
        masks = results[0].masks.to("cpu").numpy() if results[0].masks is not None else None

        if boxes is not None and len(boxes.xyxy) > 0:
            if masks is not None:
                for box_, cls, conf, mask in zip(boxes.xyxy, boxes.cls, boxes.conf, masks.data):
                    y1, x1, y2, x2 = box_
                    box = np.array([y1, x1, y2, x2], dtype=np.float64)
                    category_id = int(cls)

                    if mask_dir:
                        mask_bin = mask > 0.5
                        seg_id[mask_bin] = category_id + 1
                        class_idx = category_id + 1
                        color = (
                            (class_idx * 37) % 256,
                            (class_idx * 17) % 256,
                            (class_idx * 97) % 256,
                        )
                        seg_color[mask_bin] = color

                    detection = {
                        "category_id": category_id,
                        "detection_score": np.float64(conf),
                        "bbox": list(box),
                    }

                    if not no_ellipse:
                        contour = estimate_mask_contour(mask)
                        if contour is None or len(contour) < 10:
                            continue
                        ellipse = cv2.fitEllipse(contour)
                        theta = ellipse[2] * math.pi / 180
                        ellipse_data = np.array(
                            [
                                ellipse[0][0],
                                ellipse[0][1],
                                ellipse[1][0],
                                ellipse[1][1],
                                theta,
                            ],
                            dtype=np.float64,
                        )
                        detection["ellipse"] = list(ellipse_data)

                    dict_per_im["detections"].append(detection)
            else:
                # Segmentation masks not available: store bbox-only.
                for box_, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                    y1, x1, y2, x2 = box_
                    box = np.array([y1, x1, y2, x2], dtype=np.float64)
                    category_id = int(cls)
                    dict_per_im["detections"].append(
                        {
                            "category_id": category_id,
                            "detection_score": np.float64(conf),
                            "bbox": list(box),
                        }
                    )

        if mask_dir:
            base_name, _ = os.path.splitext(filename)
            seg_id_path = os.path.join(mask_dir, f"{base_name}_id.png")
            seg_color_path = os.path.join(mask_dir, filename)
            cv2.imwrite(seg_id_path, seg_id)
            cv2.imwrite(seg_color_path, seg_color)

        list_to_save.append(dict_per_im)

    ensure_parent_dir(detection_output)
    with open(detection_output, "w") as outfile:
        json.dump(list_to_save, outfile)
    print(f"Detection file created: {detection_output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build TUM correspondences and generate YOLOv8 segmentation detections in one script."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help="Base filename (without extension) for input JSON in support_files/ (legacy mode).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/root/data/tum-rgbd/fr1_rpy",
        help="Path to TUM sequence directory. If set, correspondence JSON is built in-script.",
    )
    parser.add_argument(
        "--corr_output",
        type=str,
        default=None,
        help="Optional path to save generated correspondence JSON when --data_path is used.",
    )
    parser.add_argument(
        "--assoc_txt_output",
        type=str,
        default=None,
        help="Optional path to save generated RGB-D association TXT.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/root/data/weights/yolo26x-seg.pt",
        help="Path to YOLOv8 segmentation model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Detection output JSON path (default: support_files/detections_yolov8x_seg_{filename}_with_ellipse.json)",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="masks",
        help="Directory to save segmentation mask images",
    )
    parser.add_argument(
        "--no_ellipse",
        action="store_true",
        help="Do not fit/save ellipse; store bbox-only detections.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.data_path:
        data_path = resolve_path(args.data_path, script_dir)
        if args.mask_dir:
            if os.path.isabs(args.mask_dir):
                mask_dir = args.mask_dir
            else:
                mask_dir = os.path.normpath(os.path.join(data_path, args.mask_dir))
        else:
            mask_dir = None

        correspondence = build_tum_correspondence(data_path)
        if args.corr_output:
            corr_output = resolve_path(args.corr_output, script_dir)
        else:
            dataset_name = os.path.basename(data_path.rstrip("/"))
            corr_output = os.path.join(data_path, f"{dataset_name}_associated.json")
        ensure_parent_dir(corr_output)
        with open(corr_output, "w") as outfile:
            json.dump(correspondence, outfile)
        print(f"Correspondence file created: {corr_output}")

        if args.assoc_txt_output:
            assoc_txt_output = resolve_path(args.assoc_txt_output, script_dir)
        else:
            dataset_name = os.path.basename(data_path.rstrip("/"))
            assoc_txt_output = os.path.join(data_path, f"{dataset_name}_associated.txt")
        write_association_txt(correspondence, assoc_txt_output)

        if args.output:
            detection_output = resolve_path(args.output, script_dir)
        else:
            dataset_name = os.path.basename(data_path.rstrip("/"))
            suffix = "no_ellipse" if args.no_ellipse else "with_ellipse"
            detection_output = os.path.join(
                data_path, f"detections_yolov8x_seg_{dataset_name}_{suffix}.json"
            )
    else:
        if args.mask_dir:
            mask_dir = resolve_path(args.mask_dir, script_dir)
        else:
            mask_dir = None
        if not args.filename:
            raise ValueError("Provide either positional filename or --data_path.")
        input_file = resolve_path(f"support_files/{args.filename}.json", script_dir)
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        with open(input_file) as infile:
            correspondence = json.load(infile)
        if args.assoc_txt_output:
            assoc_txt_output = resolve_path(args.assoc_txt_output, script_dir)
        else:
            assoc_txt_output = resolve_path(
                f"support_files/{args.filename}_associated.txt", script_dir
            )
        write_association_txt(correspondence, assoc_txt_output)
        if args.output:
            detection_output = resolve_path(args.output, script_dir)
        else:
            suffix = "no_ellipse" if args.no_ellipse else "with_ellipse"
            detection_output = resolve_path(
                f"support_files/detections_yolov8x_seg_{args.filename}_{suffix}.json",
                script_dir,
            )

    predictor = YOLO(args.model)
    run_detection(correspondence, predictor, mask_dir, detection_output, args.no_ellipse)


if __name__ == "__main__":
    main()