"""Procedural primitive shapes.

All shapes are centered at the origin with Z as the "up" axis, and their
triangles wind counter-clockwise when seen from outside. Closed shapes are
watertight.
"""

from __future__ import annotations

import math

from .mesh import Mesh
from .vector import Vec3


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def box(width: float = 1.0, depth: float = None, height: float = None) -> Mesh:
    """Axis-aligned box; with one argument it is a cube."""
    depth = width if depth is None else depth
    height = width if height is None else height
    _require(width > 0 and depth > 0 and height > 0, "box dimensions must be positive")
    x, y, z = width / 2, depth / 2, height / 2
    vertices = [
        (-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
        (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z),
    ]
    quads = [
        (0, 3, 2, 1),  # -Z
        (4, 5, 6, 7),  # +Z
        (0, 1, 5, 4),  # -Y
        (3, 7, 6, 2),  # +Y
        (0, 4, 7, 3),  # -X
        (1, 2, 6, 5),  # +X
    ]
    return Mesh(vertices, quads, name="box")


def cube(size: float = 1.0) -> Mesh:
    return box(size, size, size).copy(name="cube")


def plane(width: float = 1.0, depth: float = None, segments_x: int = 1, segments_y: int = None) -> Mesh:
    """Flat grid in the XY plane facing +Z."""
    depth = width if depth is None else depth
    segments_y = segments_x if segments_y is None else segments_y
    _require(width > 0 and depth > 0, "plane dimensions must be positive")
    _require(segments_x >= 1 and segments_y >= 1, "plane needs at least one segment")
    vertices = []
    for j in range(segments_y + 1):
        for i in range(segments_x + 1):
            vertices.append(
                (width * (i / segments_x - 0.5), depth * (j / segments_y - 0.5), 0.0)
            )
    row = segments_x + 1
    faces = []
    for j in range(segments_y):
        for i in range(segments_x):
            a = j * row + i
            faces.append((a, a + 1, a + row + 1, a + row))
    return Mesh(vertices, faces, name="plane")


def uv_sphere(radius: float = 1.0, segments: int = 32, rings: int = 16) -> Mesh:
    """Latitude/longitude sphere."""
    _require(radius > 0, "radius must be positive")
    _require(segments >= 3 and rings >= 2, "sphere needs segments >= 3 and rings >= 2")
    vertices = [(0.0, 0.0, radius)]
    for i in range(1, rings):
        theta = math.pi * i / rings
        st, ct = math.sin(theta), math.cos(theta)
        for j in range(segments):
            phi = 2 * math.pi * j / segments
            vertices.append((radius * st * math.cos(phi), radius * st * math.sin(phi), radius * ct))
    vertices.append((0.0, 0.0, -radius))
    bottom = len(vertices) - 1

    def ring(i: int, j: int) -> int:
        return 1 + (i - 1) * segments + j % segments

    faces = []
    for j in range(segments):
        faces.append((0, ring(1, j), ring(1, j + 1)))
    for i in range(1, rings - 1):
        for j in range(segments):
            faces.append((ring(i, j), ring(i + 1, j), ring(i + 1, j + 1), ring(i, j + 1)))
    for j in range(segments):
        faces.append((bottom, ring(rings - 1, j + 1), ring(rings - 1, j)))
    return Mesh(vertices, faces, name="uv_sphere")


sphere = uv_sphere


def icosphere(radius: float = 1.0, subdivisions: int = 2) -> Mesh:
    """Geodesic sphere built by subdividing an icosahedron."""
    _require(radius > 0, "radius must be positive")
    _require(subdivisions >= 0, "subdivisions must be >= 0")
    mesh = icosahedron(1.0).subdivided(subdivisions)
    mesh.vertices = [p.normalized() * radius for p in mesh.vertices]
    mesh.name = "icosphere"
    return mesh


def icosahedron(radius: float = 1.0) -> Mesh:
    _require(radius > 0, "radius must be positive")
    t = (1 + math.sqrt(5)) / 2
    raw = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    vertices = [Vec3.of(p).normalized() * radius for p in raw]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    # Orient every face outward (the shape is convex and centered).
    oriented = []
    for a, b, c in faces:
        pa, pb, pc = vertices[a], vertices[b], vertices[c]
        if (pb - pa).cross(pc - pa).dot(pa + pb + pc) < 0:
            b, c = c, b
        oriented.append((a, b, c))
    return Mesh(vertices, oriented, name="icosahedron")


def cylinder(radius: float = 0.5, height: float = 1.0, segments: int = 32, capped: bool = True) -> Mesh:
    """Cylinder along the Z axis."""
    _require(radius > 0 and height > 0, "cylinder dimensions must be positive")
    _require(segments >= 3, "cylinder needs at least 3 segments")
    h = height / 2
    vertices = []
    for z in (-h, h):
        for j in range(segments):
            phi = 2 * math.pi * j / segments
            vertices.append((radius * math.cos(phi), radius * math.sin(phi), z))
    faces = []
    for j in range(segments):
        k = (j + 1) % segments
        faces.append((j, k, segments + k, segments + j))
    if capped:
        bottom_c = len(vertices)
        vertices.append((0.0, 0.0, -h))
        top_c = len(vertices)
        vertices.append((0.0, 0.0, h))
        for j in range(segments):
            k = (j + 1) % segments
            faces.append((bottom_c, k, j))
            faces.append((top_c, segments + j, segments + k))
    return Mesh(vertices, faces, name="cylinder")


def cone(radius: float = 0.5, height: float = 1.0, segments: int = 32, capped: bool = True) -> Mesh:
    """Cone along the Z axis with its apex at +height/2."""
    _require(radius > 0 and height > 0, "cone dimensions must be positive")
    _require(segments >= 3, "cone needs at least 3 segments")
    h = height / 2
    vertices = []
    for j in range(segments):
        phi = 2 * math.pi * j / segments
        vertices.append((radius * math.cos(phi), radius * math.sin(phi), -h))
    apex = len(vertices)
    vertices.append((0.0, 0.0, h))
    faces = []
    for j in range(segments):
        faces.append((j, (j + 1) % segments, apex))
    if capped:
        center = len(vertices)
        vertices.append((0.0, 0.0, -h))
        for j in range(segments):
            faces.append((center, (j + 1) % segments, j))
    return Mesh(vertices, faces, name="cone")


def torus(major_radius: float = 1.0, minor_radius: float = 0.25, major_segments: int = 48, minor_segments: int = 16) -> Mesh:
    """Torus lying in the XY plane."""
    _require(major_radius > minor_radius > 0, "torus requires major_radius > minor_radius > 0")
    _require(major_segments >= 3 and minor_segments >= 3, "torus needs at least 3 segments")
    vertices = []
    for i in range(major_segments):
        u = 2 * math.pi * i / major_segments
        for j in range(minor_segments):
            v = 2 * math.pi * j / minor_segments
            r = major_radius + minor_radius * math.cos(v)
            vertices.append((r * math.cos(u), r * math.sin(u), minor_radius * math.sin(v)))

    def idx(i: int, j: int) -> int:
        return (i % major_segments) * minor_segments + j % minor_segments

    faces = []
    for i in range(major_segments):
        for j in range(minor_segments):
            faces.append((idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)))
    return Mesh(vertices, faces, name="torus")


def pyramid(base: float = 1.0, height: float = 1.0) -> Mesh:
    """Square pyramid with its apex at +height/2."""
    _require(base > 0 and height > 0, "pyramid dimensions must be positive")
    b, h = base / 2, height / 2
    vertices = [(-b, -b, -h), (b, -b, -h), (b, b, -h), (-b, b, -h), (0.0, 0.0, h)]
    faces = [(0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
    return Mesh(vertices, faces, name="pyramid")
