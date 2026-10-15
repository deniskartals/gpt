"""Build a small scene, export it in every format and render it to SVG and PNG.

Run from the repository root:  python examples/demo.py
"""

import math
import os
import sys

# Allow running the example without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import model3d as m3  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

# Large triangles sort poorly with the painter's algorithm, so subdivide the base.
base = m3.box(4, 4, 0.2).translated(0, 0, -0.1).subdivided(3)
ring = m3.torus(0.8, 0.22).rotated(m3.UNIT_X, 90, degrees=True).translated(-1, 1, 1.02)
ball = m3.icosphere(0.6, 3).translated(1, -1, 0.6)
tower = m3.cylinder(0.35, 1.6).translated(1, 1, 0.8)
roof = m3.cone(0.5, 0.7).translated(1, 1, 1.95)
pyramid = m3.pyramid(1.0, 1.0).rotated(m3.UNIT_Z, math.pi / 4).translated(-1, -1, 0.5)

scene = m3.Mesh.merge([base, ring, ball, tower, roof, pyramid], name="demo_scene")

print("Scene statistics:")
for key, value in scene.stats().items():
    print(f"  {key}: {value}")

for ext in ("obj", "stl", "ply"):
    path = os.path.join(OUT, f"scene.{ext}")
    m3.save(scene, path)
    print("wrote", path)

models = [
    m3.Model(base, color="#d9d4c7"),
    m3.Model(ring, color="#e0a030"),
    m3.Model(ball, color="#3f88c5"),
    m3.Model(tower, color="#b5513b"),
    m3.Model(roof, color="#5d3a6b"),
    m3.Model(pyramid, color="#4c9f70"),
]
camera = m3.Camera.framing(scene)
svg_path = os.path.join(OUT, "scene.svg")
m3.save_svg(models, svg_path, width=900, height=650, camera=camera)
print("wrote", svg_path)

png_path = os.path.join(OUT, "scene.png")
m3.save_png(models, png_path, width=900, height=650, camera=camera, supersample=2)
print("wrote", png_path)
