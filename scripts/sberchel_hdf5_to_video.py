"""Convert an HDF5 demo file to one MP4 per trajectory with all cameras tiled side by side.

Usage:
    python scripts/sberchel_hdf5_to_video.py --input_file datasets/dex_20260817_175624cam.hdf5

Output goes to <output_root>/<hdf5 stem>/demo_N.mp4 (default output_root: videos/).
Camera keys are auto-detected (any uint8 image dataset under obs/) unless --input_keys is given.
Demos are converted in parallel processes for speed.

Frames are piped to ffmpeg (H.264 by default). To shrink the output further, raise --crf
(28 is roughly half of 23) and/or downscale the panels with --panel_size.
"""

import argparse
import os

# Datasets may still be open by a recording session; don't fight over the lock.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np

DEFAULT_FRAMERATE = 30
# Frames decompressed per read; multiple of the typical 45-frame gzip chunk.
SLAB_SIZE = 90
DEFAULT_CRF = 23


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
    parser.add_argument(
        "--codec",
        type=str,
        default="libx264",
        help="ffmpeg video encoder (libx264, libx265, h264_nvenc, ...).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=DEFAULT_CRF,
        help="Quality: lower is bigger/better. 18 near-lossless, 23 default, 28 small.",
    )
    parser.add_argument("--preset", type=str, default="medium", help="Encoder preset (slower = smaller file).")
    return parser.parse_args()


def detect_image_keys(hdf5_file, demo):
    with h5py.File(hdf5_file, "r") as f:
        obs = f[f"data/{demo}/obs"]
        return [k for k, d in obs.items() if isinstance(d, h5py.Dataset) and d.ndim == 4 and d.dtype == np.uint8]


def open_writer(output_path, width, height, framerate, encode):
    """Start an ffmpeg process that reads raw BGR frames on stdin and writes an MP4."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(framerate),
        "-i", "pipe:0", "-an",
        "-c:v", encode["codec"], "-preset", encode["preset"],
        # yuv420p needs even dimensions; pad rather than reject an odd panel layout.
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
        "-threads", str(encode["threads"]), "-movflags", "+faststart",
    ]
    # NVENC spells constant quality differently and needs the bitrate cap lifted.
    cmd += ["-cq", str(encode["crf"]), "-b:v", "0"] if "nvenc" in encode["codec"] else ["-crf", str(encode["crf"])]
    return subprocess.Popen(cmd + [str(output_path)], stdin=subprocess.PIPE)


def convert_demo(hdf5_file, demo, keys, output_path, framerate, panel_size, encode):
    with h5py.File(hdf5_file, "r") as f:
        dsets = [f[f"data/{demo}/obs/{k}"] for k in keys]
        num_frames = min(d.shape[0] for d in dsets)
        sizes = [panel_size or d.shape[1] for d in dsets]
        height = max(sizes)
        width = sum(round(d.shape[2] * h / d.shape[1]) for d, h in zip(dsets, sizes))

        video = open_writer(output_path, width, height, framerate, encode)

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
                video.stdin.write(np.ascontiguousarray(np.hstack(panels)).tobytes())

        video.stdin.close()
        if video.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {output_path}")
    return demo, num_frames, output_path.stat().st_size


def main():
    args = parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH; install it (apt install ffmpeg) to encode videos.")
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
    encode = {
        "codec": args.codec,
        "crf": args.crf,
        "preset": args.preset,
        # Each demo owns an ffmpeg process; split the cores so they don't oversubscribe.
        "threads": max(1, (os.cpu_count() or 1) // workers),
    }
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                convert_demo,
                input_file,
                demo,
                keys,
                output_dir / f"{demo}.mp4",
                args.framerate,
                args.panel_size,
                encode,
            )
            for demo in demos
        ]
        for fut in as_completed(futures):
            demo, num_frames, num_bytes = fut.result()
            print(f"  {demo}: {num_frames} frames, {num_bytes / 1e6:.1f} MB -> {output_dir / f'{demo}.mp4'}")


if __name__ == "__main__":
    main()
