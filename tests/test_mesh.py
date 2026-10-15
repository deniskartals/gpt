import math
import unittest

import model3d as m3
from model3d import Mat4, Mesh, Vec3

CLOSED_SHAPES = {
    "box": lambda: m3.box(1, 2, 3),
    "uv_sphere": lambda: m3.uv_sphere(1.0, 24, 12),
    "icosphere": lambda: m3.icosphere(1.0, 2),
    "icosahedron": lambda: m3.icosahedron(1.0),
    "cylinder": lambda: m3.cylinder(0.5, 2.0, 24),
    "cone": lambda: m3.cone(0.5, 2.0, 24),
    "torus": lambda: m3.torus(1.0, 0.25, 32, 12),
    "pyramid": lambda: m3.pyramid(2.0, 3.0),
}


class PrimitiveTest(unittest.TestCase):
    def test_closed_shapes_are_watertight_and_outward(self):
        for name, make in CLOSED_SHAPES.items():
            with self.subTest(shape=name):
                mesh = make()
                self.assertTrue(mesh.is_watertight())
                self.assertTrue(mesh.is_consistently_oriented())
                self.assertGreater(mesh.signed_volume(), 0)

    def test_volumes_approach_analytic_values(self):
        cases = [
            (m3.box(1, 2, 3), 6.0, 1e-12),
            (m3.pyramid(2.0, 3.0), 4.0, 1e-12),
            (m3.uv_sphere(1.0, 128, 64), 4 / 3 * math.pi, 0.01),
            (m3.icosphere(1.0, 4), 4 / 3 * math.pi, 0.01),
            (m3.cylinder(0.5, 2.0, 256), math.pi * 0.25 * 2, 0.01),
            (m3.cone(0.5, 2.0, 256), math.pi * 0.25 * 2 / 3, 0.01),
            (m3.torus(1.0, 0.25, 128, 64), 2 * math.pi ** 2 * 0.25 ** 2, 0.01),
        ]
        for mesh, expected, rel in cases:
            with self.subTest(shape=mesh.name):
                self.assertAlmostEqual(mesh.volume() / expected, 1.0, delta=rel)

    def test_box_surface_area(self):
        self.assertAlmostEqual(m3.box(1, 2, 3).surface_area(), 22.0)

    def test_plane_is_open_and_faces_up(self):
        p = m3.plane(2, 4, 3, 5)
        self.assertEqual(p.face_count, 3 * 5 * 2)
        self.assertFalse(p.is_watertight())
        self.assertAlmostEqual(p.surface_area(), 8.0)
        for n in p.face_normals():
            self.assertTrue(n.is_close(Vec3(0, 0, 1)))

    def test_icosphere_vertices_on_sphere(self):
        for p in m3.icosphere(2.5, 2).vertices:
            self.assertAlmostEqual(p.length(), 2.5)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            m3.box(-1)
        with self.assertRaises(ValueError):
            m3.torus(0.2, 0.5)
        with self.assertRaises(ValueError):
            m3.cylinder(segments=2)


class MeshTest(unittest.TestCase):
    def test_polygon_faces_are_triangulated(self):
        mesh = Mesh([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [(0, 1, 2, 3)])
        self.assertEqual(mesh.faces, [(0, 1, 2), (0, 2, 3)])

    def test_bad_face_index(self):
        with self.assertRaises(IndexError):
            Mesh([(0, 0, 0)], [(0, 1, 2)])

    def test_bounds(self):
        box = m3.box(2, 4, 6).translated(1, 1, 1).bounds()
        self.assertEqual(box.min, Vec3(0, -1, -2))
        self.assertEqual(box.max, Vec3(2, 3, 4))
        self.assertEqual(box.center, Vec3(1, 1, 1))
        self.assertTrue(box.contains(Vec3(1, 1, 1)))

    def test_transforms_preserve_or_scale_volume(self):
        cube = m3.cube(1)
        self.assertAlmostEqual(cube.rotated(Vec3(1, 2, 3), 37, degrees=True).volume(), 1.0)
        self.assertAlmostEqual(cube.scaled(2, 3, 4).volume(), 24.0)
        self.assertAlmostEqual(cube.translated(5, 5, 5).signed_volume(), 1.0)

    def test_mirror_keeps_normals_outward(self):
        mirrored = m3.cone().transformed(Mat4.scaling(-1, 1, 1))
        self.assertGreater(mirrored.signed_volume(), 0)

    def test_flipped(self):
        self.assertLess(m3.cube().flipped().signed_volume(), 0)

    def test_transform_does_not_mutate(self):
        cube = m3.cube()
        before = list(cube.vertices)
        cube.translated(1, 2, 3)
        self.assertEqual(cube.vertices, before)

    def test_merge(self):
        a, b = m3.cube(), m3.cube().translated(3, 0, 0)
        merged = a + b
        self.assertEqual(merged.vertex_count, 16)
        self.assertEqual(merged.face_count, 24)
        self.assertAlmostEqual(merged.volume(), 2.0)
        self.assertTrue(merged.is_watertight())

    def test_weld(self):
        # Two triangles sharing an edge but stored with duplicated vertices.
        mesh = Mesh(
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 0), (1, 1, 0), (0, 1, 0)],
            [(0, 1, 2), (3, 4, 5)],
        )
        welded = mesh.welded()
        self.assertEqual(welded.vertex_count, 4)
        self.assertEqual(welded.face_count, 2)

    def test_weld_drops_degenerate_faces(self):
        mesh = Mesh([(0, 0, 0), (1e-12, 0, 0), (1, 1, 0)], [(0, 1, 2)])
        self.assertEqual(mesh.welded(1e-9).face_count, 0)

    def test_subdivide(self):
        cube = m3.cube()
        sub = cube.subdivided(2)
        self.assertEqual(sub.face_count, cube.face_count * 16)
        self.assertTrue(sub.is_watertight())
        self.assertAlmostEqual(sub.volume(), 1.0)

    def test_vertex_normals_of_sphere_point_outward(self):
        sphere = m3.icosphere(1.0, 1)
        for p, n in zip(sphere.vertices, sphere.vertex_normals()):
            self.assertGreater(n.dot(p), 0.95)

    def test_without_unused_vertices(self):
        mesh = Mesh([(9, 9, 9), (0, 0, 0), (1, 0, 0), (0, 1, 0)], [(1, 2, 3)])
        clean = mesh.without_unused_vertices()
        self.assertEqual(clean.vertex_count, 3)
        self.assertEqual(clean.faces, [(0, 1, 2)])

    def test_normalized_to_unit(self):
        m = m3.box(4, 2, 1).translated(10, 0, 0).normalized_to_unit()
        size = m.bounds().size
        self.assertAlmostEqual(max(size), 1.0)
        self.assertTrue(m.bounds().center.is_close(Vec3()))

    def test_stats(self):
        stats = m3.cube(2).stats()
        self.assertEqual(stats["faces"], 12)
        self.assertTrue(stats["watertight"])
        self.assertAlmostEqual(stats["volume"], 8.0)


if __name__ == "__main__":
    unittest.main()
