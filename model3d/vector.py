"""Immutable 3D vector type."""

from __future__ import annotations

import math
from typing import Iterator, Sequence, Union

Number = Union[int, float]


class Vec3:
    """An immutable three-component vector of floats."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: Number = 0.0, y: Number = 0.0, z: Number = 0.0) -> None:
        object.__setattr__(self, "x", float(x))
        object.__setattr__(self, "y", float(y))
        object.__setattr__(self, "z", float(z))

    def __setattr__(self, name, value):
        raise AttributeError("Vec3 is immutable")

    @classmethod
    def of(cls, value: Union["Vec3", Sequence[Number]]) -> "Vec3":
        """Convert a Vec3 or any 3-element sequence to a Vec3."""
        if isinstance(value, Vec3):
            return value
        if len(value) != 3:
            raise ValueError(f"expected 3 components, got {len(value)}")
        return cls(value[0], value[1], value[2])

    # Sequence protocol -------------------------------------------------
    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> float:
        return (self.x, self.y, self.z)[index]

    def __repr__(self) -> str:
        return f"Vec3({self.x:g}, {self.y:g}, {self.z:g})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vec3):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

    # Arithmetic --------------------------------------------------------
    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: Number) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: Number) -> "Vec3":
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    # Geometry ----------------------------------------------------------
    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length_squared(self) -> float:
        return self.dot(self)

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> "Vec3":
        """Return a unit vector; the zero vector is returned unchanged."""
        n = self.length()
        if n == 0.0:
            return self
        return self / n

    def distance(self, other: "Vec3") -> float:
        return (self - other).length()

    def lerp(self, other: "Vec3", t: float) -> "Vec3":
        return self + (other - self) * t

    def component_mul(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)

    def min(self, other: "Vec3") -> "Vec3":
        return Vec3(min(self.x, other.x), min(self.y, other.y), min(self.z, other.z))

    def max(self, other: "Vec3") -> "Vec3":
        return Vec3(max(self.x, other.x), max(self.y, other.y), max(self.z, other.z))

    def is_close(self, other: "Vec3", tol: float = 1e-9) -> bool:
        return (
            math.isclose(self.x, other.x, rel_tol=tol, abs_tol=tol)
            and math.isclose(self.y, other.y, rel_tol=tol, abs_tol=tol)
            and math.isclose(self.z, other.z, rel_tol=tol, abs_tol=tol)
        )

    def to_tuple(self) -> tuple:
        return (self.x, self.y, self.z)


ZERO = Vec3(0, 0, 0)
UNIT_X = Vec3(1, 0, 0)
UNIT_Y = Vec3(0, 1, 0)
UNIT_Z = Vec3(0, 0, 1)
