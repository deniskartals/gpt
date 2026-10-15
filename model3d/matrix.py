"""4x4 transformation matrices (row-major, column vectors: p' = M @ p)."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from .vector import Vec3


class Mat4:
    """An immutable 4x4 matrix stored row-major."""

    __slots__ = ("rows",)

    def __init__(self, rows: Iterable[Sequence[float]]) -> None:
        data = tuple(tuple(float(v) for v in row) for row in rows)
        if len(data) != 4 or any(len(r) != 4 for r in data):
            raise ValueError("Mat4 requires 4 rows of 4 values")
        object.__setattr__(self, "rows", data)

    def __setattr__(self, name, value):
        raise AttributeError("Mat4 is immutable")

    def __getitem__(self, index):
        r, c = index
        return self.rows[r][c]

    def __eq__(self, other) -> bool:
        if not isinstance(other, Mat4):
            return NotImplemented
        return self.rows == other.rows

    def __hash__(self) -> int:
        return hash(self.rows)

    def __repr__(self) -> str:
        body = ",\n     ".join(
            "[" + ", ".join(f"{v:g}" for v in row) + "]" for row in self.rows
        )
        return f"Mat4([{body}])"

    # Constructors ------------------------------------------------------
    @classmethod
    def identity(cls) -> "Mat4":
        return cls([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    @classmethod
    def translation(cls, x: float, y: float = 0.0, z: float = 0.0) -> "Mat4":
        if isinstance(x, Vec3):
            x, y, z = x
        return cls([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

    @classmethod
    def scaling(cls, x: float, y: float = None, z: float = None) -> "Mat4":
        """Scale matrix; a single argument scales uniformly."""
        if isinstance(x, Vec3):
            x, y, z = x
        if y is None:
            y = x
        if z is None:
            z = x
        return cls([[x, 0, 0, 0], [0, y, 0, 0], [0, 0, z, 0], [0, 0, 0, 1]])

    @classmethod
    def rotation_x(cls, angle: float) -> "Mat4":
        c, s = math.cos(angle), math.sin(angle)
        return cls([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

    @classmethod
    def rotation_y(cls, angle: float) -> "Mat4":
        c, s = math.cos(angle), math.sin(angle)
        return cls([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

    @classmethod
    def rotation_z(cls, angle: float) -> "Mat4":
        c, s = math.cos(angle), math.sin(angle)
        return cls([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    @classmethod
    def rotation_axis(cls, axis: Vec3, angle: float) -> "Mat4":
        """Right-handed rotation of ``angle`` radians about ``axis``."""
        a = Vec3.of(axis).normalized()
        if a.length_squared() == 0.0:
            raise ValueError("rotation axis must be non-zero")
        x, y, z = a
        c, s = math.cos(angle), math.sin(angle)
        t = 1.0 - c
        return cls(
            [
                [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0],
                [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0],
                [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0],
                [0, 0, 0, 1],
            ]
        )

    @classmethod
    def look_at(cls, eye: Vec3, target: Vec3, up: Vec3) -> "Mat4":
        """View matrix placing the camera at ``eye`` looking down -Z toward ``target``."""
        eye, target, up = Vec3.of(eye), Vec3.of(target), Vec3.of(up)
        f = (target - eye).normalized()
        s = f.cross(up).normalized()
        if s.length_squared() == 0.0:
            raise ValueError("up vector is parallel to the view direction")
        u = s.cross(f)
        return cls(
            [
                [s.x, s.y, s.z, -s.dot(eye)],
                [u.x, u.y, u.z, -u.dot(eye)],
                [-f.x, -f.y, -f.z, f.dot(eye)],
                [0, 0, 0, 1],
            ]
        )

    @classmethod
    def perspective(cls, fov_y: float, aspect: float, near: float, far: float) -> "Mat4":
        """OpenGL-style perspective projection (``fov_y`` in radians)."""
        f = 1.0 / math.tan(fov_y / 2.0)
        return cls(
            [
                [f / aspect, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                [0, 0, -1, 0],
            ]
        )

    @classmethod
    def orthographic(
        cls, left: float, right: float, bottom: float, top: float, near: float, far: float
    ) -> "Mat4":
        return cls(
            [
                [2 / (right - left), 0, 0, -(right + left) / (right - left)],
                [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
                [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                [0, 0, 0, 1],
            ]
        )

    # Operations --------------------------------------------------------
    def __matmul__(self, other):
        if isinstance(other, Mat4):
            a, b = self.rows, other.rows
            return Mat4(
                [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
            )
        if isinstance(other, Vec3):
            return self.transform_point(other)
        return NotImplemented

    def transpose(self) -> "Mat4":
        return Mat4([[self.rows[j][i] for j in range(4)] for i in range(4)])

    def transform_point(self, p: Vec3) -> Vec3:
        """Transform a point (w=1), applying the perspective divide if needed."""
        r = self.rows
        x, y, z = p.x, p.y, p.z
        tx = r[0][0] * x + r[0][1] * y + r[0][2] * z + r[0][3]
        ty = r[1][0] * x + r[1][1] * y + r[1][2] * z + r[1][3]
        tz = r[2][0] * x + r[2][1] * y + r[2][2] * z + r[2][3]
        w = r[3][0] * x + r[3][1] * y + r[3][2] * z + r[3][3]
        if w != 1.0 and w != 0.0:
            return Vec3(tx / w, ty / w, tz / w)
        return Vec3(tx, ty, tz)

    def transform_direction(self, d: Vec3) -> Vec3:
        """Transform a direction (w=0); translation is ignored."""
        r = self.rows
        return Vec3(
            r[0][0] * d.x + r[0][1] * d.y + r[0][2] * d.z,
            r[1][0] * d.x + r[1][1] * d.y + r[1][2] * d.z,
            r[2][0] * d.x + r[2][1] * d.y + r[2][2] * d.z,
        )

    def determinant(self) -> float:
        m = [list(r) for r in self.rows]
        det = 1.0
        for col in range(4):
            pivot = max(range(col, 4), key=lambda r: abs(m[r][col]))
            if m[pivot][col] == 0.0:
                return 0.0
            if pivot != col:
                m[col], m[pivot] = m[pivot], m[col]
                det = -det
            det *= m[col][col]
            for r in range(col + 1, 4):
                factor = m[r][col] / m[col][col]
                for c in range(col, 4):
                    m[r][c] -= factor * m[col][c]
        return det

    def inverse(self) -> "Mat4":
        """Gauss-Jordan inverse; raises ValueError for singular matrices."""
        m = [list(r) + [1.0 if i == j else 0.0 for j in range(4)] for i, r in enumerate(self.rows)]
        for col in range(4):
            pivot = max(range(col, 4), key=lambda r: abs(m[r][col]))
            if abs(m[pivot][col]) < 1e-12:
                raise ValueError("matrix is singular")
            m[col], m[pivot] = m[pivot], m[col]
            p = m[col][col]
            m[col] = [v / p for v in m[col]]
            for r in range(4):
                if r != col and m[r][col] != 0.0:
                    factor = m[r][col]
                    m[r] = [a - factor * b for a, b in zip(m[r], m[col])]
        return Mat4([row[4:] for row in m])

    def normal_matrix(self) -> "Mat4":
        """Inverse-transpose of the upper 3x3, for transforming normals."""
        return self.inverse().transpose()

    def is_close(self, other: "Mat4", tol: float = 1e-9) -> bool:
        return all(
            math.isclose(a, b, rel_tol=tol, abs_tol=tol)
            for ra, rb in zip(self.rows, other.rows)
            for a, b in zip(ra, rb)
        )
