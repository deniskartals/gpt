import math
import unittest

from model3d import Mat4, Vec3


class Vec3Test(unittest.TestCase):
    def test_arithmetic(self):
        a, b = Vec3(1, 2, 3), Vec3(4, 5, 6)
        self.assertEqual(a + b, Vec3(5, 7, 9))
        self.assertEqual(b - a, Vec3(3, 3, 3))
        self.assertEqual(a * 2, Vec3(2, 4, 6))
        self.assertEqual(2 * a, Vec3(2, 4, 6))
        self.assertEqual(-a, Vec3(-1, -2, -3))
        self.assertEqual(a.dot(b), 32)

    def test_cross_is_right_handed(self):
        self.assertEqual(Vec3(1, 0, 0).cross(Vec3(0, 1, 0)), Vec3(0, 0, 1))

    def test_normalized(self):
        self.assertAlmostEqual(Vec3(3, 4, 0).normalized().length(), 1.0)
        self.assertEqual(Vec3().normalized(), Vec3())

    def test_immutable(self):
        with self.assertRaises(AttributeError):
            Vec3().x = 1

    def test_of(self):
        self.assertEqual(Vec3.of((1, 2, 3)), Vec3(1, 2, 3))
        with self.assertRaises(ValueError):
            Vec3.of((1, 2))


class Mat4Test(unittest.TestCase):
    def test_translation_and_scale(self):
        p = Vec3(1, 1, 1)
        self.assertEqual(Mat4.translation(1, 2, 3).transform_point(p), Vec3(2, 3, 4))
        self.assertEqual(Mat4.scaling(2).transform_point(p), Vec3(2, 2, 2))
        self.assertEqual(Mat4.translation(1, 2, 3).transform_direction(p), p)

    def test_rotations(self):
        r = Mat4.rotation_z(math.pi / 2)
        self.assertTrue(r.transform_point(Vec3(1, 0, 0)).is_close(Vec3(0, 1, 0)))
        r = Mat4.rotation_x(math.pi / 2)
        self.assertTrue(r.transform_point(Vec3(0, 1, 0)).is_close(Vec3(0, 0, 1)))
        r = Mat4.rotation_y(math.pi / 2)
        self.assertTrue(r.transform_point(Vec3(0, 0, 1)).is_close(Vec3(1, 0, 0)))

    def test_axis_rotation_matches_basic(self):
        for angle in (0.3, 1.2, -2.0):
            self.assertTrue(Mat4.rotation_axis(Vec3(0, 0, 1), angle).is_close(Mat4.rotation_z(angle)))
            self.assertTrue(Mat4.rotation_axis(Vec3(1, 0, 0), angle).is_close(Mat4.rotation_x(angle)))
            self.assertTrue(Mat4.rotation_axis(Vec3(0, 1, 0), angle).is_close(Mat4.rotation_y(angle)))

    def test_composition_order(self):
        m = Mat4.translation(5, 0, 0) @ Mat4.rotation_z(math.pi / 2)
        self.assertTrue(m.transform_point(Vec3(1, 0, 0)).is_close(Vec3(5, 1, 0)))

    def test_inverse_and_determinant(self):
        m = Mat4.translation(1, -2, 3) @ Mat4.rotation_axis(Vec3(1, 1, 0), 0.7) @ Mat4.scaling(2, 3, 4)
        self.assertTrue((m @ m.inverse()).is_close(Mat4.identity()))
        self.assertAlmostEqual(m.determinant(), 24.0)
        with self.assertRaises(ValueError):
            Mat4.scaling(0).inverse()

    def test_look_at_puts_target_on_negative_z(self):
        view = Mat4.look_at(Vec3(0, -5, 0), Vec3(), Vec3(0, 0, 1))
        self.assertTrue(view.transform_point(Vec3()).is_close(Vec3(0, 0, -5)))

    def test_perspective_maps_near_and_far(self):
        p = Mat4.perspective(math.radians(90), 1.0, 1.0, 10.0)
        self.assertAlmostEqual(p.transform_point(Vec3(0, 0, -1)).z, -1.0)
        self.assertAlmostEqual(p.transform_point(Vec3(0, 0, -10)).z, 1.0)


if __name__ == "__main__":
    unittest.main()
