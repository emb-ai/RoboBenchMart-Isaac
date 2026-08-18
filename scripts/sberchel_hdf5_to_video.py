"""Convert an HDF5 demo file to one MP4 per trajectory with all cameras tiled side by side.

Usage:
    python scripts/sberchel_hdf5_to_video.py --input_file datasets/dex_20260817_175624cam.hdf5

Output goes to <output_root>/<hdf5 stem>/demo_N.mp4 (default output_root: videos/).
Camera keys are auto-detected (any uint8 image dataset under obs/) unless --input_keys is given.
Demos are converted in parallel processes for speed.
"""

import argparse
import os

# Datasets may still be open by a recording session; don't fight over the lock.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np

DEFAULT_FRAMERATE = 30
# Frames decompressed per read; multiple of the typical 45-frame gzip chunk.
SLAB_SIZE = 90


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input HDF5 file.")
    parser.add_argument(
        "--output_root",
        type=str,
        default="videos",
        help="Root directory; videos go to <output_root>/<hdf5 stem>/.",
    )
    parser.add_argument(
        "--input_keys",
        type=str,
        nargs="+",
        default=None,
        help="Camera keys under obs/ to include (default: auto-detect all image datasets).",
    )
    parser.add_argument("--framerate", type=int, default=DEFAULT_FRAMERATE, help="Output video FPS.")
    parser.add_argument("--panel_size", type=int, default=None, help="Resize each camera panel to this square size.")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers (default: min(#demos, #cpus)).")
    return parser.parse_args()


def detect_image_keys(hdf5_file, demo):
    with h5py.File(hdf5_file, "r") as f:
        obs = f[f"data/{demo}/obs"]
        return [k for k, d in obs.items() if isinstance(d, h5py.Dataset) and d.ndim == 4 and d.dtype == np.uint8]


def convert_demo(hdf5_file, demo, keys, output_path, framerate, panel_size):
    with h5py.File(hdf5_file, "r") as f:
        dsets = [f[f"data/{demo}/obs/{k}"] for k in keys]
        num_frames = min(d.shape[0] for d in dsets)
        sizes = [panel_size or d.shape[1] for d in dsets]
        height = max(sizes)
        width = sum(round(d.shape[2] * h / d.shape[1]) for d, h in zip(dsets, sizes))

        video = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), framerate, (width, height))
        if not video.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        for start in range(0, num_frames, SLAB_SIZE):
            end = min(start + SLAB_SIZE, num_frames)
            slabs = [d[start:end] for d in dsets]
            for i in range(end - start):
                panels = []
                for key, slab, size in zip(keys, slabs, sizes):
                    panel = cv2.cvtColor(slab[i], cv2.COLOR_RGB2BGR)
                    if panel.shape[0] != size:
                        panel = cv2.resize(
                            panel, (round(panel.shape[1] * size / panel.shape[0]), size), interpolation=cv2.INTER_AREA
                        )
                    if panel.shape[0] < height:
                        panel = cv2.copyMakeBorder(panel, 0, height - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT)
                    cv2.putText(panel, key, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(panel, key, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    panels.append(panel)
                video.write(np.hstack(panels))

        video.release()
    return demo, num_frames


def main():
    args = parse_args()
    input_file = Path(args.input_file)
    output_dir = Path(args.output_root) / input_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_file, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda s: int(s.rsplit("_", 1)[-1]))
    if not demos:
        raise SystemExit(f"No demos found in {input_file}")

    keys = args.input_keys or detect_image_keys(input_file, demos[0])
    if not keys:
        raise SystemExit("No image datasets found under obs/; pass --input_keys explicitly.")
    print(f"{len(demos)} demos, cameras: {keys} -> {output_dir}/")

    workers = args.workers or min(len(demos), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                convert_demo, input_file, demo, keys, output_dir / f"{demo}.mp4", args.framerate, args.panel_size
            )
            for demo in demos
        ]
        for fut in as_completed(futures):
            demo, num_frames = fut.result()
            print(f"  {demo}: {num_frames} frames -> {output_dir / f'{demo}.mp4'}")


if __name__ == "__main__":
    main()
