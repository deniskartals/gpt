"""model3d - a small, dependency-free 3D model library.

Quick start::

    import model3d as m3

    mesh = m3.torus(1.0, 0.3).translated(0, 0, 1) + m3.box(3, 3, 0.2)
    print(mesh.stats())
    m3.save(mesh, "scene.stl")
    m3.save_png(mesh, "scene.png")
"""

from .io import load, load_obj, load_ply, load_stl, save, save_obj, save_ply, save_stl
from .matrix import Mat4
from .mesh import BoundingBox, Mesh
from .primitives import (
    box,
    cone,
    cube,
    cylinder,
    icosahedron,
    icosphere,
    plane,
    pyramid,
    sphere,
    torus,
    uv_sphere,
)
from .render import Camera, Model, encode_png, render_image, render_svg, save_png, save_svg
from .vector import UNIT_X, UNIT_Y, UNIT_Z, ZERO, Vec3

__version__ = "0.1.0"

__all__ = [
    "Vec3", "Mat4", "Mesh", "BoundingBox", "ZERO", "UNIT_X", "UNIT_Y", "UNIT_Z",
    "box", "cube", "plane", "sphere", "uv_sphere", "icosphere", "icosahedron",
    "cylinder", "cone", "torus", "pyramid",
    "load", "save", "load_obj", "save_obj", "load_stl", "save_stl", "load_ply", "save_ply",
    "Camera", "Model", "render_svg", "save_svg", "render_image", "save_png", "encode_png",
]
