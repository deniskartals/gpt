"""Command line interface: ``python -m model3d <command> ...``."""

from __future__ import annotations

import argparse
import math
import sys

from . import io, primitives
from .render import Camera, save_png, save_svg
from .vector import Vec3

_SHAPES = {
    "box": primitives.box,
    "cube": primitives.cube,
    "plane": primitives.plane,
    "sphere": primitives.uv_sphere,
    "icosphere": primitives.icosphere,
    "cylinder": primitives.cylinder,
    "cone": primitives.cone,
    "torus": primitives.torus,
    "pyramid": primitives.pyramid,
}


def _info(args) -> None:
    mesh = io.load(args.file)
    for key, value in mesh.stats().items():
        if isinstance(value, tuple):
            value = "(" + ", ".join(f"{v:.6g}" for v in value) + ")"
        elif isinstance(value, float):
            value = f"{value:.6g}"
        print(f"{key:>14}: {value}")


def _convert(args) -> None:
    mesh = io.load(args.input)
    if args.scale != 1.0:
        mesh = mesh.scaled(args.scale)
    if args.center:
        mesh = mesh.centered()
    io.save(mesh, args.output)
    print(f"wrote {args.output} ({mesh.face_count} faces)")


def _render(args) -> None:
    mesh = io.load(args.input)
    direction = Vec3(*args.direction)
    camera = Camera.framing(mesh, direction=direction, fov=math.radians(args.fov), orthographic=args.ortho)
    if args.output.lower().endswith(".png"):
        if args.wireframe:
            raise ValueError("--wireframe is only supported for SVG output")
        save_png(mesh, args.output, width=args.width, height=args.height, camera=camera, supersample=args.supersample)
    else:
        save_svg(
            mesh,
            args.output,
            width=args.width,
            height=args.height,
            camera=camera,
            stroke="#1b1b1b" if args.wireframe else None,
        )
    print(f"wrote {args.output}")


def _create(args) -> None:
    mesh = _SHAPES[args.shape]()
    if args.scale != 1.0:
        mesh = mesh.scaled(args.scale)
    io.save(mesh, args.output)
    print(f"wrote {args.output} ({mesh.face_count} faces)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="model3d", description="3D model utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="print mesh statistics")
    p.add_argument("file")
    p.set_defaults(func=_info)

    p = sub.add_parser("convert", help="convert between OBJ, STL and PLY")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--center", action="store_true", help="center the mesh at the origin")
    p.set_defaults(func=_convert)

    p = sub.add_parser("render", help="render a mesh to SVG or PNG (by extension)")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--fov", type=float, default=40.0, help="vertical field of view in degrees")
    p.add_argument("--direction", type=float, nargs=3, default=(1.0, -1.4, 0.9), metavar=("X", "Y", "Z"))
    p.add_argument("--ortho", action="store_true", help="orthographic projection")
    p.add_argument("--wireframe", action="store_true", help="draw triangle edges (SVG only)")
    p.add_argument("--supersample", type=int, default=2, help="anti-aliasing factor (PNG only)")
    p.set_defaults(func=_render)

    p = sub.add_parser("create", help="generate a primitive shape")
    p.add_argument("shape", choices=sorted(_SHAPES))
    p.add_argument("output")
    p.add_argument("--scale", type=float, default=1.0)
    p.set_defaults(func=_create)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
