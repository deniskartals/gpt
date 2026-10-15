import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

import model3d as m3
from model3d.__main__ import main


class IoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.mesh = m3.torus(1.0, 0.3, 16, 8).translated(0.5, -0.25, 1.0)

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.dir, name)

    def assertSameMesh(self, a, b, tol=1e-6):
        self.assertEqual(a.vertex_count, b.vertex_count)
        self.assertEqual(a.face_count, b.face_count)
        self.assertAlmostEqual(a.volume(), b.volume(), delta=tol * 100)
        self.assertTrue(b.is_watertight())
        self.assertGreater(b.signed_volume(), 0)

    def test_round_trips(self):
        cases = [
            ("a.obj", {}),
            ("b.obj", {"normals": True}),
            ("c.stl", {"binary": True}),
            ("d.stl", {"binary": False}),
            ("e.ply", {"binary": False}),
            ("f.ply", {"binary": True}),
        ]
        for name, opts in cases:
            with self.subTest(file=name, **opts):
                m3.save(self.mesh, self.path(name), **opts)
                self.assertSameMesh(self.mesh, m3.load(self.path(name)))

    def test_obj_negative_indices_and_polygons(self):
        with open(self.path("q.obj"), "w") as f:
            f.write("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf -4/1/1 -3/2/1 -2/3/1 -1/4/1\n")
        mesh = m3.load(self.path("q.obj"))
        self.assertEqual(mesh.faces, [(0, 1, 2), (0, 2, 3)])
        self.assertAlmostEqual(mesh.surface_area(), 1.0)

    def test_binary_stl_with_solid_header(self):
        # Some exporters start binary headers with "solid"; size check must win.
        m3.save_stl(m3.cube(), self.path("s.stl"))
        with open(self.path("s.stl"), "r+b") as f:
            f.write(b"solid trick")
        self.assertEqual(m3.load(self.path("s.stl")).face_count, 12)

    def test_ply_big_endian_and_extra_properties(self):
        import struct

        header = (
            "ply\nformat binary_big_endian 1.0\n"
            "element vertex 3\nproperty double x\nproperty double y\nproperty double z\n"
            "property uchar red\n"
            "element face 1\nproperty list uchar uint vertex_index\nend_header\n"
        ).encode()
        body = b"".join(struct.pack(">3dB", *v, 7) for v in [(0, 0, 0), (2, 0, 0), (0, 2, 0)])
        body += struct.pack(">B3I", 3, 0, 1, 2)
        with open(self.path("be.ply"), "wb") as f:
            f.write(header + body)
        mesh = m3.load(self.path("be.ply"))
        self.assertEqual(mesh.faces, [(0, 1, 2)])
        self.assertAlmostEqual(mesh.surface_area(), 2.0)

    def test_unknown_extension(self):
        with self.assertRaises(ValueError):
            m3.save(self.mesh, self.path("x.fbx"))
        with self.assertRaises(ValueError):
            m3.load(self.path("x.fbx"))


class RenderTest(unittest.TestCase):
    def test_render_svg(self):
        svg = m3.render_svg(m3.cube(), width=200, height=100)
        self.assertTrue(svg.startswith("<svg"))
        # A cube viewed from a corner shows exactly three faces (six triangles).
        self.assertEqual(svg.count("<polygon"), 6)

    def test_render_without_culling_draws_everything(self):
        svg = m3.render_svg(m3.cube(), backface_culling=False)
        self.assertEqual(svg.count("<polygon"), 12)

    def test_orthographic_and_models(self):
        scene = [
            m3.Model(m3.cube(), color="#c33"),
            m3.Model(m3.uv_sphere(0.5, 12, 6), color=(40, 160, 90), transform=m3.Mat4.translation(2, 0, 0)),
        ]
        mesh = m3.Mesh.merge([m3.cube(), m3.uv_sphere(0.5, 12, 6).translated(2, 0, 0)])
        camera = m3.Camera.framing(mesh, orthographic=True)
        svg = m3.render_svg(scene, camera=camera, stroke="#000")
        self.assertGreater(svg.count("<polygon"), 6)
        self.assertIn('stroke="#000"', svg)

    def test_png_depth_buffer(self):
        import struct
        import zlib

        # A red cube in front of a large green box: the cube must win at the center.
        scene = [
            m3.Model(m3.cube(1), color="#ff0000"),
            m3.Model(m3.box(10, 0.1, 10), color="#00ff00", transform=m3.Mat4.translation(0, 3, 0)),
        ]
        camera = m3.Camera(eye=m3.Vec3(0, -5, 0), up=m3.Vec3(0, 0, 1))
        pixels = m3.render_image(scene, width=40, height=40, camera=camera, ambient=1.0)
        center = (20 * 40 + 20) * 3
        self.assertEqual(tuple(pixels[center : center + 3]), (255, 0, 0))
        corner = (2 * 40 + 2) * 3
        self.assertEqual(tuple(pixels[corner : corner + 3]), (0, 255, 0))

        png = m3.encode_png(pixels, 40, 40)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        width, height = struct.unpack(">II", png[16:24])
        self.assertEqual((width, height), (40, 40))
        idat_len = struct.unpack(">I", png[33:37])[0]
        raw = zlib.decompress(png[41 : 41 + idat_len])
        self.assertEqual(len(raw), 40 * (40 * 3 + 1))

    def test_png_supersample(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.png")
            m3.save_png(m3.icosphere(1, 1), path, width=32, height=24, supersample=2)
            with open(path, "rb") as f:
                self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")
        with self.assertRaises(ValueError):
            m3.render_image(m3.cube(), supersample=0)

    def test_bad_color(self):
        with self.assertRaises(ValueError):
            m3.render_svg(m3.Model(m3.cube(), color="#12"))


class CliTest(unittest.TestCase):
    def run_cli(self, *argv):
        out = StringIO()
        with redirect_stdout(out):
            code = main(list(argv))
        return code, out.getvalue()

    def test_create_info_convert_render(self):
        with tempfile.TemporaryDirectory() as d:
            stl = os.path.join(d, "t.stl")
            obj = os.path.join(d, "t.obj")
            svg = os.path.join(d, "t.svg")
            self.assertEqual(self.run_cli("create", "torus", stl)[0], 0)
            self.assertEqual(self.run_cli("convert", stl, obj, "--scale", "2", "--center")[0], 0)
            code, out = self.run_cli("info", obj)
            self.assertEqual(code, 0)
            self.assertIn("watertight: True", out)
            self.assertEqual(self.run_cli("render", obj, svg, "--wireframe")[0], 0)
            self.assertTrue(os.path.getsize(svg) > 0)
            png = os.path.join(d, "t.png")
            args = ("render", obj, png, "--width", "64", "--height", "48", "--supersample", "1")
            self.assertEqual(self.run_cli(*args)[0], 0)
            self.assertTrue(os.path.getsize(png) > 0)

    def test_error_exit_code(self):
        from contextlib import redirect_stderr

        with redirect_stderr(StringIO()):
            self.assertEqual(main(["info", "/nonexistent/file.obj"]), 1)


if __name__ == "__main__":
    unittest.main()
