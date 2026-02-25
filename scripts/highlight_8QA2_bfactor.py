#!/usr/bin/env python3
"""Highlight chains A and G by B-factor in PyMOL and save a restorable session.

Examples:
  pymol -cq highlight_8QA2_bfactor.py
  python highlight_8QA2_bfactor.py \
    --pdb /path/to/8QA2.pdb \
    --out-session /path/to/8QA2_highlight.pse \
    --out-image /path/to/8QA2_highlight.png
"""

from __future__ import annotations

import argparse
import os
import sys


DEFAULT_PDB = "/home/loci/main/tandem_website_dev/tandem/data/GJB2/structures/8QA2.pdb"
DEFAULT_SESSION = "/home/loci/main/tandem_website_dev/tandem/scripts/8QA2_highlight.pse"
DEFAULT_IMAGE = "/home/loci/main/tandem_website_dev/tandem/scripts/8QA2_highlight.png"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", default=DEFAULT_PDB, help="Input PDB file path")
    parser.add_argument(
        "--out-session",
        default=DEFAULT_SESSION,
        help="Output PyMOL session (.pse) path",
    )
    parser.add_argument(
        "--out-image",
        default=DEFAULT_IMAGE,
        help="Output snapshot image (.png) path",
    )
    parser.add_argument(
        "--image-width", type=int, default=1800, help="PNG width in pixels"
    )
    parser.add_argument(
        "--image-height", type=int, default=1400, help="PNG height in pixels"
    )
    parser.add_argument(
        "--ray", type=int, default=1, choices=[0, 1], help="Ray trace image"
    )
    return parser.parse_args()


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _style_and_save(cmd, pdb_path: str, out_session: str, out_image: str, w: int, h: int, ray: int) -> None:
    cmd.reinitialize()
    cmd.load(pdb_path, "prot")

    cmd.hide("everything", "all")
    cmd.show("cartoon", "prot")
    cmd.bg_color("white")

    cmd.select("focus_chains", "prot and chain A+G")
    cmd.select("other_chains", "prot and not chain A+G")

    cmd.color("gray85", "other_chains")
    cmd.set("cartoon_transparency", 0.80, "other_chains")

    cmd.show("sticks", "focus_chains")
    cmd.set("stick_radius", 0.20, "focus_chains")
    cmd.spectrum("b", "blue_white_red", "focus_chains")

    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("antialias", 2)
    cmd.set("ray_opaque_background", 0)
    cmd.orient("focus_chains")
    cmd.zoom("focus_chains", 8)

    _ensure_parent(out_image)
    _ensure_parent(out_session)
    cmd.png(out_image, width=w, height=h, dpi=300, ray=ray)
    cmd.save(out_session)


def main() -> int:
    args = _parse_args()

    if not os.path.isfile(args.pdb):
        print(f"ERROR: PDB file not found: {args.pdb}", file=sys.stderr)
        return 1

    # Prefer pymol2 (pure Python embedding), fallback to pymol cmd in CLI mode.
    try:
        import pymol2  # type: ignore

        with pymol2.PyMOL() as pm:
            _style_and_save(
                pm.cmd,
                args.pdb,
                args.out_session,
                args.out_image,
                args.image_width,
                args.image_height,
                args.ray,
            )
    except Exception:
        from pymol import cmd, finish_launching  # type: ignore

        finish_launching(["pymol", "-cq"])
        _style_and_save(
            cmd,
            args.pdb,
            args.out_session,
            args.out_image,
            args.image_width,
            args.image_height,
            args.ray,
        )
        cmd.quit()

    print(f"Saved session: {args.out_session}")
    print(f"Saved image:   {args.out_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
