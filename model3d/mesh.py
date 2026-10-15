"""Triangle mesh data structure and geometry operations."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .matrix import Mat4
from .vector import Vec3

Triangle = Tuple[int, int, int]


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box."""

    min: Vec3
    max: Vec3

    @property
    def size(self) -> Vec3:
        return self.max - self.min

    @property
    def center(self) -> Vec3:
        return (self.min + self.max) * 0.5

    @property
    def diagonal(self) -> float:
        return self.size.length()

    def contains(self, p: Vec3, tol: float = 0.0) -> bool:
        return all(lo - tol <= v <= hi + tol for lo, v, hi in zip(self.min, p, self.max))


class Mesh:
    """An indexed triangle mesh.

    Faces with more than three vertices are fan-triangulated on construction.
    Triangles are expected to be wound counter-clockwise when viewed from
    outside, so that ``(b - a) x (c - a)`` points outward.

    Transform methods return new meshes; a Mesh is never modified in place.
    """

    __slots__ = ("vertices", "faces", "name")

    def __init__(
        self,
        vertices: Iterable[Sequence[float]] = (),
        faces: Iterable[Sequence[int]] = (),
        name: str = "mesh",
    ) -> None:
        self.vertices: List[Vec3] = [Vec3.of(v) for v in vertices]
        self.faces: List[Triangle] = []
        self.name = name
        n = len(self.vertices)
        for face in faces:
            idx = [int(i) for i in face]
            if len(idx) < 3:
                raise ValueError(f"face needs at least 3 vertices: {face!r}")
            for i in idx:
                if not 0 <= i < n:
                    raise IndexError(f"face index {i} out of range (0..{n - 1})")
            for k in range(1, len(idx) - 1):
                self.faces.append((idx[0], idx[k], idx[k + 1]))

    def __repr__(self) -> str:
        return (
            f"Mesh(name={self.name!r}, vertices={len(self.vertices)}, "
            f"faces={len(self.faces)})"
        )

    def __add__(self, other: "Mesh") -> "Mesh":
        return Mesh.merge([self, other], name=self.name)

    # Construction helpers -----------------------------------------------
    def copy(self, name: str = None) -> "Mesh":
        m = Mesh(name=self.name if name is None else name)
        m.vertices = list(self.vertices)
        m.faces = list(self.faces)
        return m

    @classmethod
    def merge(cls, meshes: Iterable["Mesh"], name: str = "merged") -> "Mesh":
        """Concatenate several meshes into one (vertices are not welded)."""
        out = cls(name=name)
        for mesh in meshes:
            offset = len(out.vertices)
            out.vertices.extend(mesh.vertices)
            out.faces.extend((a + offset, b + offset, c + offset) for a, b, c in mesh.faces)
        return out

    # Basic queries -------------------------------------------------------
    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    def triangles(self) -> Iterable[Tuple[Vec3, Vec3, Vec3]]:
        v = self.vertices
        for a, b, c in self.faces:
            yield v[a], v[b], v[c]

    def bounds(self) -> BoundingBox:
        if not self.vertices:
            raise ValueError("empty mesh has no bounds")
        lo = hi = self.vertices[0]
        for p in self.vertices[1:]:
            lo = lo.min(p)
            hi = hi.max(p)
        return BoundingBox(lo, hi)

    def centroid(self) -> Vec3:
        """Average of all vertex positions."""
        if not self.vertices:
            raise ValueError("empty mesh has no centroid")
        total = Vec3()
        for p in self.vertices:
            total = total + p
        return total / len(self.vertices)

    def edges(self) -> Counter:
        """Count of each undirected edge ``(min, max)``."""
        counts: Counter = Counter()
        for a, b, c in self.faces:
            for u, v in ((a, b), (b, c), (c, a)):
                counts[(u, v) if u < v else (v, u)] += 1
        return counts

    def is_watertight(self) -> bool:
        """True if every edge is shared by exactly two faces."""
        edges = self.edges()
        return bool(edges) and all(n == 2 for n in edges.values())

    def is_consistently_oriented(self) -> bool:
        """True if no directed edge is used twice (neighbours agree on winding)."""
        seen = set()
        for a, b, c in self.faces:
            for e in ((a, b), (b, c), (c, a)):
                if e in seen:
                    return False
                seen.add(e)
        return True

    # Normals, area, volume ------------------------------------------------
    def face_normals(self) -> List[Vec3]:
        return [(b - a).cross(c - a).normalized() for a, b, c in self.triangles()]

    def vertex_normals(self) -> List[Vec3]:
        """Area-weighted average of adjacent face normals."""
        acc = [Vec3()] * len(self.vertices)
        for (ia, ib, ic), (a, b, c) in zip(self.faces, self.triangles()):
            n = (b - a).cross(c - a)
            acc[ia] = acc[ia] + n
            acc[ib] = acc[ib] + n
            acc[ic] = acc[ic] + n
        return [n.normalized() for n in acc]

    def face_areas(self) -> List[float]:
        return [(b - a).cross(c - a).length() * 0.5 for a, b, c in self.triangles()]

    def surface_area(self) -> float:
        return math.fsum(self.face_areas())

    def signed_volume(self) -> float:
        """Signed enclosed volume (positive when faces point outward)."""
        return math.fsum(a.dot(b.cross(c)) for a, b, c in self.triangles()) / 6.0

    def volume(self) -> float:
        return abs(self.signed_volume())

    # Transforms -----------------------------------------------------------
    def transformed(self, matrix: Mat4) -> "Mesh":
        m = self.copy()
        m.vertices = [matrix.transform_point(p) for p in self.vertices]
        if matrix.determinant() < 0:
            # Mirroring flips the winding; restore outward-facing normals.
            m.faces = [(a, c, b) for a, b, c in m.faces]
        return m

    def translated(self, x, y: float = 0.0, z: float = 0.0) -> "Mesh":
        return self.transformed(Mat4.translation(x, y, z))

    def scaled(self, x, y: float = None, z: float = None) -> "Mesh":
        return self.transformed(Mat4.scaling(x, y, z))

    def rotated(self, axis: Vec3, angle: float, degrees: bool = False) -> "Mesh":
        if degrees:
            angle = math.radians(angle)
        return self.transformed(Mat4.rotation_axis(axis, angle))

    def centered(self) -> "Mesh":
        """Move the mesh so its bounding-box center is at the origin."""
        return self.translated(-self.bounds().center)

    def normalized_to_unit(self) -> "Mesh":
        """Center the mesh and scale it so its largest dimension is 1."""
        m = self.centered()
        size = m.bounds().size
        extent = max(size.x, size.y, size.z)
        return m if extent == 0 else m.scaled(1.0 / extent)

    def flipped(self) -> "Mesh":
        """Reverse the winding of every face (inverts normals)."""
        m = self.copy()
        m.faces = [(a, c, b) for a, b, c in self.faces]
        return m

    # Topology ---------------------------------------------------------------
    def welded(self, tolerance: float = 1e-9) -> "Mesh":
        """Merge vertices closer than ``tolerance`` and drop degenerate faces."""
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        inv = 1.0 / tolerance
        lookup: Dict[Tuple[int, int, int], int] = {}
        remap: List[int] = []
        vertices: List[Vec3] = []
        for p in self.vertices:
            key = (round(p.x * inv), round(p.y * inv), round(p.z * inv))
            idx = lookup.get(key)
            if idx is None:
                idx = lookup[key] = len(vertices)
                vertices.append(p)
            remap.append(idx)
        m = Mesh(name=self.name)
        m.vertices = vertices
        for a, b, c in self.faces:
            a, b, c = remap[a], remap[b], remap[c]
            if a != b and b != c and a != c:
                m.faces.append((a, b, c))
        return m

    def without_unused_vertices(self) -> "Mesh":
        used = sorted({i for f in self.faces for i in f})
        remap = {old: new for new, old in enumerate(used)}
        m = Mesh(name=self.name)
        m.vertices = [self.vertices[i] for i in used]
        m.faces = [(remap[a], remap[b], remap[c]) for a, b, c in self.faces]
        return m

    def subdivided(self, iterations: int = 1) -> "Mesh":
        """Split each triangle into four using edge midpoints."""
        mesh = self
        for _ in range(iterations):
            vertices = list(mesh.vertices)
            cache: Dict[Tuple[int, int], int] = {}

            def midpoint(u: int, v: int) -> int:
                key = (u, v) if u < v else (v, u)
                idx = cache.get(key)
                if idx is None:
                    idx = cache[key] = len(vertices)
                    vertices.append((vertices[u] + vertices[v]) * 0.5)
                return idx

            faces = []
            for a, b, c in mesh.faces:
                ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
                faces += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
            mesh = Mesh(name=mesh.name)
            mesh.vertices = vertices
            mesh.faces = faces
        return mesh

    def stats(self) -> Dict[str, object]:
        """Summary of the mesh's main properties."""
        info: Dict[str, object] = {
            "name": self.name,
            "vertices": self.vertex_count,
            "faces": self.face_count,
        }
        if self.vertices:
            box = self.bounds()
            info["bounds_min"] = box.min.to_tuple()
            info["bounds_max"] = box.max.to_tuple()
            info["size"] = box.size.to_tuple()
        if self.faces:
            info["surface_area"] = self.surface_area()
            watertight = self.is_watertight()
            info["watertight"] = watertight
            if watertight:
                info["volume"] = self.volume()
        return info
