/*
 * Original work Copyright (c) 2015, Danish Geodata Agency <gst@gst.dk>
 * Modified work Copyright (c) 2015-2016, Geoboxers <info@geoboxers.com>
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 * 
 */
 
#if defined(_MSC_VER) && defined(_EXPORTING)
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT  
#endif

struct index{
	int ncols,npoints,ntri,ncells;
	double extent[4];
	double cs;
	int **index_arr;
};

typedef struct index spatial_index;

DLL_EXPORT void inspect_index(spatial_index *ind, char *buf, int buf_len);
DLL_EXPORT int line_intersection(double *p1,double *p2, double *p3, double *p4, double *out);
DLL_EXPORT spatial_index *build_index(double *pts, int *tri, double cs, int n, int m);
DLL_EXPORT void find_triangle(double *pts, int *out, double *base_pts,int *tri, spatial_index *ind, char *mask, int np);
DLL_EXPORT void interpolate(double *pts, double *base_pts, double *base_z, double *out, double nd_val, int *tri, spatial_index *ind, char *mask, int np);
DLL_EXPORT void optimize_index(spatial_index *ind);
DLL_EXPORT void free_index(spatial_index *ind);
DLL_EXPORT void make_grid_low(double *base_pts,double *base_z, int *tri, float *grid,  float nd_val, 
                              int ncols, int nrows, double cx, double cy, double xl,
                              double yu, double cut_off, spatial_index *ind);
DLL_EXPORT void make_grid(double *base_pts,double *base_z, int *tri, float *grid, float *tgrid, float nd_val,
                          int ncols, int nrows,
                          double cx, double cy, double xl, double yu, spatial_index *ind);
