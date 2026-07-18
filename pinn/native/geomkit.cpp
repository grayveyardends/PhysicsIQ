/*
 * geomkit.cpp - the two hot loops the python side is too slow for:
 *
 *   points_in_mesh(): batch inside/outside test against a triangle soup,
 *                     ray parity along +x with a yz-grid over triangles.
 *                     replaces thousands of OCCT isInside() round trips.
 *   nearest_index(): for each query point, index of the nearest sample
 *                    point. uniform grid + expanding shell search,
 *                    replaces the O(N*M) numpy broadcast in the overlay.
 *
 * plain C ABI, loaded via ctypes from both the JAX venv and freecadcmd
 * (the .so has no python dependency at all). build: see build.sh.
 *
 * Copyright (c) 2026 Gabrial Alex. MIT license, see LICENSE.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>

namespace {

struct Grid2 {
	double y0, z0, cy, cz;
	int ny, nz;
	std::vector<std::vector<int>> cells;

	int idx(int iy, int iz) const { return iy * nz + iz; }
};

/* Moller-Trumbore, ray fixed along +x. returns t or -1. */
static double ray_x_tri(const double *p, const double *a,
                        const double *b, const double *c)
{
	const double dir[3] = {1.0, 0.0, 0.0};
	double e1[3], e2[3];
	for (int i = 0; i < 3; i++) {
		e1[i] = b[i] - a[i];
		e2[i] = c[i] - a[i];
	}
	/* h = dir x e2 */
	double h[3] = {0.0, -e2[2], e2[1]};
	double det = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
	if (std::fabs(det) < 1e-14)
		return -1.0;
	double inv = 1.0 / det;
	double s[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
	double u = (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]) * inv;
	if (u < 0.0 || u > 1.0)
		return -1.0;
	double q[3] = {s[1] * e1[2] - s[2] * e1[1],
	               s[2] * e1[0] - s[0] * e1[2],
	               s[0] * e1[1] - s[1] * e1[0]};
	double v = (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]) * inv;
	if (v < 0.0 || u + v > 1.0)
		return -1.0;
	double t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv;
	return t > 1e-12 ? t : -1.0;
}

} /* namespace */

extern "C" {

/*
 * tris:   ntri * 9 doubles (a,b,c per triangle)
 * pts:    npts * 3 doubles
 * out:    npts uint8, 1 = inside
 */
void points_in_mesh(const double *tris, int ntri,
                    const double *pts, int npts, uint8_t *out)
{
	double ymin = 1e300, ymax = -1e300, zmin = 1e300, zmax = -1e300;
	for (int i = 0; i < ntri * 3; i++) {
		ymin = std::min(ymin, tris[i * 3 + 1]);
		ymax = std::max(ymax, tris[i * 3 + 1]);
		zmin = std::min(zmin, tris[i * 3 + 2]);
		zmax = std::max(zmax, tris[i * 3 + 2]);
	}

	Grid2 g;
	int n = std::max(1, (int)std::sqrt((double)ntri));
	g.ny = std::min(n, 256);
	g.nz = std::min(n, 256);
	g.y0 = ymin;
	g.z0 = zmin;
	g.cy = (ymax - ymin) / g.ny + 1e-12;
	g.cz = (zmax - zmin) / g.nz + 1e-12;
	g.cells.resize((size_t)g.ny * g.nz);

	for (int t = 0; t < ntri; t++) {
		const double *tri = tris + t * 9;
		double ty0 = std::min({tri[1], tri[4], tri[7]});
		double ty1 = std::max({tri[1], tri[4], tri[7]});
		double tz0 = std::min({tri[2], tri[5], tri[8]});
		double tz1 = std::max({tri[2], tri[5], tri[8]});
		int iy0 = std::max(0, (int)((ty0 - g.y0) / g.cy));
		int iy1 = std::min(g.ny - 1, (int)((ty1 - g.y0) / g.cy));
		int iz0 = std::max(0, (int)((tz0 - g.z0) / g.cz));
		int iz1 = std::min(g.nz - 1, (int)((tz1 - g.z0) / g.cz));
		for (int iy = iy0; iy <= iy1; iy++)
			for (int iz = iz0; iz <= iz1; iz++)
				g.cells[g.idx(iy, iz)].push_back(t);
	}

	for (int i = 0; i < npts; i++) {
		const double *p = pts + i * 3;
		int iy = (int)((p[1] - g.y0) / g.cy);
		int iz = (int)((p[2] - g.z0) / g.cz);
		if (iy < 0 || iy >= g.ny || iz < 0 || iz >= g.nz) {
			out[i] = 0;
			continue;
		}
		int hits = 0;
		for (int t : g.cells[g.idx(iy, iz)]) {
			const double *tri = tris + t * 9;
			if (ray_x_tri(p, tri, tri + 3, tri + 6) > 0.0)
				hits++;
		}
		out[i] = (uint8_t)(hits & 1);
	}
}

/*
 * samples: nsamp * 3 doubles, queries: nq * 3 doubles
 * out:     nq int32, index into samples of the nearest one
 */
void nearest_index(const double *samples, int nsamp,
                   const double *queries, int nq, int32_t *out)
{
	double lo[3] = {1e300, 1e300, 1e300};
	double hi[3] = {-1e300, -1e300, -1e300};
	for (int i = 0; i < nsamp; i++)
		for (int k = 0; k < 3; k++) {
			lo[k] = std::min(lo[k], samples[i * 3 + k]);
			hi[k] = std::max(hi[k], samples[i * 3 + k]);
		}

	int n = std::max(1, (int)std::cbrt((double)nsamp / 2.0));
	n = std::min(n, 64);
	double cell[3];
	for (int k = 0; k < 3; k++)
		cell[k] = (hi[k] - lo[k]) / n + 1e-12;

	std::vector<std::vector<int>> cells((size_t)n * n * n);
	auto cid = [&](const double *p, int *c) {
		for (int k = 0; k < 3; k++) {
			c[k] = (int)((p[k] - lo[k]) / cell[k]);
			c[k] = std::max(0, std::min(n - 1, c[k]));
		}
	};
	for (int i = 0; i < nsamp; i++) {
		int c[3];
		cid(samples + i * 3, c);
		cells[(size_t)(c[0] * n + c[1]) * n + c[2]].push_back(i);
	}

	for (int q = 0; q < nq; q++) {
		const double *p = queries + q * 3;
		int c[3];
		cid(p, c);
		int best = -1;
		double bd = 1e300;
		/* expand shells until a hit, then one extra ring to be exact */
		int found_r = -1;
		for (int r = 0; r < n; r++) {
			if (found_r >= 0 && r > found_r + 1)
				break;
			for (int x = std::max(0, c[0] - r); x <= std::min(n - 1, c[0] + r); x++)
				for (int y = std::max(0, c[1] - r); y <= std::min(n - 1, c[1] + r); y++)
					for (int z = std::max(0, c[2] - r); z <= std::min(n - 1, c[2] + r); z++) {
						if (std::max({std::abs(x - c[0]), std::abs(y - c[1]),
						              std::abs(z - c[2])}) != r)
							continue;
						for (int i : cells[(size_t)(x * n + y) * n + z]) {
							const double *s = samples + i * 3;
							double d = 0;
							for (int k = 0; k < 3; k++)
								d += (p[k] - s[k]) * (p[k] - s[k]);
							if (d < bd) {
								bd = d;
								best = i;
							}
						}
					}
			if (best >= 0 && found_r < 0)
				found_r = r;
		}
		out[q] = best;
	}
}

} /* extern "C" */
