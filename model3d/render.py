"""Minimal flat-shaded renderers: SVG (painter's algorithm) and PNG (z-buffer)."""

from __future__ import annotations

import math
import os
import struct
import zlib
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple, Union

from .matrix import Mat4
from .mesh import Mesh
from .vector import Vec3

Color = Tuple[int, int, int]


def _parse_color(color: Union[str, Sequence[int]]) -> Color:
    if isinstance(color, str):
        c = color.lstrip("#")
        if len(c) == 3:
            c = "".join(ch * 2 for ch in c)
        if len(c) != 6:
            raise ValueError(f"invalid color: {color!r}")
        return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
    r, g, b = color
    return (int(r), int(g), int(b))


@dataclass
class Camera:
    """A camera described by its position, target and lens."""

    eye: Vec3
    target: Vec3 = field(default_factory=Vec3)
    up: Vec3 = field(default_factory=lambda: Vec3(0, 0, 1))
    fov: float = math.radians(40)
    orthographic: bool = False
    ortho_height: float = 2.0

    @classmethod
    def framing(
        cls,
        mesh: Mesh,
        direction: Vec3 = Vec3(1.0, -1.4, 0.9),
        fov: float = math.radians(40),
        orthographic: bool = False,
    ) -> "Camera":
        """Place a camera along ``direction`` so the whole mesh is in view."""
        box = mesh.bounds()
        radius = max(box.diagonal / 2, 1e-9)
        distance = radius / math.sin(fov / 2) * 1.05
        eye = box.center + Vec3.of(direction).normalized() * distance
        up = Vec3(0, 0, 1)
        if abs(Vec3.of(direction).normalized().dot(up)) > 0.999:
            up = Vec3(0, 1, 0)
        return cls(eye, box.center, up, fov, orthographic, radius * 2.1)

    def view_matrix(self) -> Mat4:
        return Mat4.look_at(self.eye, self.target, self.up)

    def projection_matrix(self, aspect: float) -> Mat4:
        far = (self.eye - self.target).length() * 4 + 1.0
        if self.orthographic:
            h = self.ortho_height / 2
            return Mat4.orthographic(-h * aspect, h * aspect, -h, h, 0.01, far)
        return Mat4.perspective(self.fov, aspect, 0.01, far)


@dataclass
class Model:
    """A mesh plus its appearance in a scene."""

    mesh: Mesh
    color: Union[str, Sequence[int]] = "#4f8fd6"
    transform: Mat4 = field(default_factory=Mat4.identity)


@dataclass
class _ScreenTriangle:
    points: Tuple[Tuple[float, float, float], ...]  # (x, y, ndc depth) in pixels
    color: Color
    view_depth: float


def _project(
    models: Union[Mesh, Model, Iterable[Union[Mesh, Model]]],
    width: int,
    height: int,
    camera: Union[Camera, None],
    light_dir: Vec3,
    ambient: float,
    backface_culling: bool,
) -> List[_ScreenTriangle]:
    """Transform, cull, shade and project every triangle to screen space."""
    if isinstance(models, (Mesh, Model)):
        models = [models]
    items = [m if isinstance(m, Model) else Model(m) for m in models]
    if not items:
        raise ValueError("nothing to render")

    world = [(it.mesh.transformed(it.transform), _parse_color(it.color)) for it in items]
    if camera is None:
        camera = Camera.framing(Mesh.merge(m for m, _ in world))

    view = camera.view_matrix()
    proj = camera.projection_matrix(width / height)
    light = Vec3.of(light_dir).normalized()

    out: List[_ScreenTriangle] = []
    for mesh, base in world:
        view_pts = [view.transform_point(p) for p in mesh.vertices]
        for (ia, ib, ic), (a, b, c) in zip(mesh.faces, mesh.triangles()):
            va, vb, vc = view_pts[ia], view_pts[ib], view_pts[ic]
            # Skip triangles that cross the near plane (camera looks down -Z).
            if max(va.z, vb.z, vc.z) > -0.01:
                continue
            n_view = (vb - va).cross(vc - va)
            to_face = Vec3(0, 0, -1) if camera.orthographic else va
            facing = n_view.dot(to_face) < 0
            if backface_culling and not facing:
                continue
            n_world = (b - a).cross(c - a).normalized()
            if not facing:
                n_world = -n_world
            shade = ambient + (1 - ambient) * max(0.0, n_world.dot(light))
            color = tuple(min(255, int(ch * shade)) for ch in base)
            pts = []
            for v in (va, vb, vc):
                ndc = proj.transform_point(v)
                pts.append(((ndc.x + 1) * width / 2, (1 - ndc.y) * height / 2, ndc.z))
            out.append(_ScreenTriangle(tuple(pts), color, (va.z + vb.z + vc.z) / 3))
    return out


def render_svg(
    models: Union[Mesh, Model, Iterable[Union[Mesh, Model]]],
    width: int = 800,
    height: int = 600,
    camera: Camera = None,
    light_dir: Vec3 = Vec3(0.4, -0.5, 1.0),
    ambient: float = 0.25,
    background: str = "#ffffff",
    stroke: Union[str, None] = None,
    backface_culling: bool = True,
) -> str:
    """Render one or more meshes to an SVG document string.

    Triangles are flat-shaded with a single directional light and drawn back
    to front (painter's algorithm), so very large triangles next to small ones
    can overlap incorrectly; use :func:`render_png` for exact visibility.
    Pass ``stroke`` (e.g. ``"#000"``) to draw wireframe edges.
    """
    tris = _project(models, width, height, camera, light_dir, ambient, backface_culling)
    tris.sort(key=lambda t: t.view_depth)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{background}"/>',
    ]
    for t in tris:
        fill = "#%02x%02x%02x" % t.color
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in t.points)
        out.append(
            f'<polygon points="{pts}" fill="{fill}" stroke="{stroke or fill}" '
            'stroke-width="0.6" stroke-linejoin="round"/>'
        )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_image(
    models: Union[Mesh, Model, Iterable[Union[Mesh, Model]]],
    width: int = 800,
    height: int = 600,
    camera: Camera = None,
    light_dir: Vec3 = Vec3(0.4, -0.5, 1.0),
    ambient: float = 0.25,
    background: Union[str, Sequence[int]] = "#ffffff",
    backface_culling: bool = True,
    supersample: int = 1,
) -> bytearray:
    """Rasterize meshes with a depth buffer; returns packed RGB bytes.

    ``supersample`` renders at N times the resolution and averages down for
    anti-aliasing (cost grows with N squared).
    """
    if supersample < 1:
        raise ValueError("supersample must be >= 1")
    s = supersample
    w, h = width * s, height * s
    bg = _parse_color(background)
    pixels = bytearray(bytes(bg) * (w * h))
    depth = [math.inf] * (w * h)

    for t in _project(models, w, h, camera, light_dir, ambient, backface_culling):
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = t.points
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area == 0:
            continue
        inv_area = 1.0 / area
        min_x = max(int(math.floor(min(x0, x1, x2))), 0)
        max_x = min(int(math.ceil(max(x0, x1, x2))), w - 1)
        min_y = max(int(math.floor(min(y0, y1, y2))), 0)
        max_y = min(int(math.ceil(max(y0, y1, y2))), h - 1)
        color = bytes(t.color)
        for py in range(min_y, max_y + 1):
            cy = py + 0.5
            row = py * w
            for px in range(min_x, max_x + 1):
                cx = px + 0.5
                # Barycentric weights; all share the sign of ``area`` inside.
                w0 = ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)) * inv_area
                if w0 < 0:
                    continue
                w1 = ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)) * inv_area
                if w1 < 0:
                    continue
                w2 = 1.0 - w0 - w1
                if w2 < 0:
                    continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                i = row + px
                if z < depth[i]:
                    depth[i] = z
                    pixels[i * 3 : i * 3 + 3] = color

    if s == 1:
        return pixels
    out = bytearray(width * height * 3)
    n = s * s
    for y in range(height):
        for x in range(width):
            r = g = b = 0
            for sy in range(y * s, y * s + s):
                base = (sy * w + x * s) * 3
                for k in range(base, base + s * 3, 3):
                    r += pixels[k]
                    g += pixels[k + 1]
                    b += pixels[k + 2]
            o = (y * width + x) * 3
            out[o] = r // n
            out[o + 1] = g // n
            out[o + 2] = b // n
    return out


def encode_png(pixels: bytes, width: int, height: int) -> bytes:
    """Encode packed 8-bit RGB pixels as a PNG file."""
    if len(pixels) != width * height * 3:
        raise ValueError("pixel buffer size does not match width * height * 3")

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    stride = width * 3
    raw = b"".join(b"\x00" + bytes(pixels[y * stride : (y + 1) * stride]) for y in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def save_png(models, path: Union[str, os.PathLike], width: int = 800, height: int = 600, **options) -> None:
    """Render with :func:`render_image` and write a PNG file to ``path``."""
    pixels = render_image(models, width=width, height=height, **options)
    with open(path, "wb") as f:
        f.write(encode_png(pixels, width, height))


def save_svg(models, path: Union[str, os.PathLike], **options) -> None:
    """Render with :func:`render_svg` and write the result to ``path``."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_svg(models, **options))
