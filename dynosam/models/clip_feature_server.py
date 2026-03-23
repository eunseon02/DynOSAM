#!/usr/bin/env python3
"""
CLIP Feature Extraction Server (ZMQ-based).

Receives bbox-cropped images via ZMQ, returns CLIP feature vectors.
Runs as a standalone process; C++ frontend connects as a client.

Usage:
    python3 clip_feature_server.py [--port 5555] [--clip_model ViT-B-32] [--pretrained laion2b_s34b_b79k]
"""

import argparse
import struct
import time

import cv2
import numpy as np
import open_clip
import torch
import zmq


class CLIPFeatureServer:
    def __init__(self, model_name: str, pretrained: str, port: int):
        self.port = port
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[CLIPServer] Loading model {model_name} ({pretrained}) on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.feat_dim = self.model.visual.output_dim
        print(f"[CLIPServer] Model loaded. Feature dim = {self.feat_dim}")

        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        print(f"[CLIPServer] Listening on tcp://*:{port}")

    @torch.no_grad()
    def extract_feature(self, bgr_image: np.ndarray) -> np.ndarray:
        """Extract CLIP feature from a BGR image (OpenCV format)."""
        from PIL import Image

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(img_tensor)
        feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat.cpu().numpy().flatten().astype(np.float32)

    def run(self):
        """Main server loop."""
        print("[CLIPServer] Ready to receive requests.")
        while True:
            try:
                msg = self.socket.recv()

                # Protocol: [4 bytes rows][4 bytes cols][4 bytes type][raw pixels]
                if len(msg) < 12:
                    self.socket.send(b"ERR")
                    continue

                rows = struct.unpack("<i", msg[0:4])[0]
                cols = struct.unpack("<i", msg[4:8])[0]
                cv_type = struct.unpack("<i", msg[8:12])[0]
                pixel_data = msg[12:]

                if cv_type == 16:  # CV_8UC3
                    img = np.frombuffer(pixel_data, dtype=np.uint8).reshape(rows, cols, 3)
                elif cv_type == 0:  # CV_8UC1
                    img = np.frombuffer(pixel_data, dtype=np.uint8).reshape(rows, cols)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    self.socket.send(b"ERR")
                    continue

                feat = self.extract_feature(img)

                # Reply: [4 bytes dim][dim * 4 bytes float32 features]
                reply = struct.pack("<i", len(feat)) + feat.tobytes()
                self.socket.send(reply)

            except KeyboardInterrupt:
                print("\n[CLIPServer] Shutting down.")
                break
            except Exception as e:
                print(f"[CLIPServer] Error: {e}")
                try:
                    self.socket.send(b"ERR")
                except:
                    pass


def main():
    parser = argparse.ArgumentParser(description="CLIP Feature Extraction Server")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32",
                        help="OpenCLIP model name")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k",
                        help="OpenCLIP pretrained weights")
    args = parser.parse_args()

    server = CLIPFeatureServer(args.clip_model, args.pretrained, args.port)
    server.run()


if __name__ == "__main__":
    main()
