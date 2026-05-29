#!/usr/bin/env python3
"""Standalone FRC worker for two image .npy inputs.

Kept independent from camstim GUI imports so matplotlib can display reliably.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _plot_frc_curve(frc, frc_curve, img_size, figure_title):
    xs_pix = np.arange(len(frc_curve)) / float(img_size)
    xs_freq = xs_pix * 1.0

    frc_res = None
    threshold_fn = None
    try:
        frc_res, _, threshold_fn = frc.frc_res(xs_freq, frc_curve, img_size)
    except Exception as exc:
        print(f"FRC resolution intersection not found for {figure_title}: {exc}", flush=True)

    import matplotlib.pyplot as plt

    plt.figure(figure_title)
    plt.clf()
    plt.plot(xs_freq, frc_curve, label="FRC")
    if threshold_fn is not None:
        plt.plot(xs_freq, threshold_fn(xs_freq), label="Threshold")
    if frc_res is not None:
        plt.axvline(float(frc_res), color="r", linestyle="--", label=f"Resolution {float(frc_res):.4f}")
    plt.xlabel("Spatial Frequency (pixel^-1)")
    plt.ylabel("FRC")
    plt.title(figure_title)
    plt.legend()
    plt.tight_layout()


def run_frc_worker(img1_path: Path, img2_path: Path) -> int:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    import frc
    import matplotlib.pyplot as plt

    img1 = np.load(img1_path)
    img2 = np.load(img2_path)

    img1 = frc.util.square_image(np.array(img1), add_padding=False)
    img2 = frc.util.square_image(np.array(img2), add_padding=False)
    img1 = frc.util.apply_tukey(img1)
    img2 = frc.util.apply_tukey(img2)

    two_frc_curve = frc.two_frc(img1, img2)
    one_frc_curve = frc.one_frc(img1)
    img_size = img1.shape[0]

    _plot_frc_curve(
        frc=frc,
        frc_curve=two_frc_curve,
        img_size=img_size,
        figure_title="Fourier Ring Correlation (two consecutive frames)",
    )
    _plot_frc_curve(
        frc=frc,
        frc_curve=one_frc_curve,
        img_size=img_size,
        figure_title="Fourier Ring Correlation (single frame)",
    )

    # Block until the user closes the plot window.
    plt.show()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FRC from two .npy image paths.")
    parser.add_argument("img1_path", type=Path)
    parser.add_argument("img2_path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_frc_worker(args.img1_path, args.img2_path)
    finally:
        for p in (args.img1_path, args.img2_path):
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())