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
DLL_EXPORT int p_in_poly(double *p_in, char *mout, double *verts, unsigned int np, unsigned int  *nv, unsigned int n_rings);
DLL_EXPORT void p_in_buf(double *p_in, char *mout, double *verts, unsigned long np, unsigned long nv, double d);
DLL_EXPORT unsigned long simplify_linestring(double *xy_in, double *xy_out, double dist_tol, unsigned long n_pts);
DLL_EXPORT void get_triangle_geometry(double *xy, double *z, int *triangles, float *out , int ntriangles);
DLL_EXPORT void get_normals(double *xy, double *z, int *triangles, double *out, int ntriangles);
DLL_EXPORT void fill_it_up(unsigned char *out, unsigned int *hmap, int rows, int cols, int stacks);
DLL_EXPORT void find_floating_voxels(int *lab,  int *out, int gcomp, int rows, int cols, int stacks);
DLL_EXPORT int fill_spatial_index(int *sorted_flat_indices, int *index, int npoints, int max_index);
DLL_EXPORT typedef double(*FILTER_FUNC)(double *, double , int*, double* , double* , double, double, void*);
DLL_EXPORT void apply_filter(double *xy, 
                  double *z, 
                  double *pc_xy, 
                  double *pc_z, 
                  double *vals_out, 
                  int *spatial_index, 
                  double *header,  
                  int npoints, 
                  FILTER_FUNC filter_func,  
                  double filter_rad, 
                  double nd_val, 
                  void *opt_params);
                  
/* declare all the builtin filter functions*/
DLL_EXPORT double min_filter(double *, double, int *, double *, double *, double, double, void *);
DLL_EXPORT double max_filter(double *, double, int *, double *, double *, double, double, void *);
DLL_EXPORT double var_filter(double *, double, int *, double *, double *, double, double, void *);
DLL_EXPORT double mean_filter(double *, double, int *, double *, double *, double, double, void *);
DLL_EXPORT double adaptive_gaussian_filter(double *, double, int *, double *, double *, double, double, void *);
DLL_EXPORT double median_filter(double *, double, int *, double *, double *, double, double, void *);
DLL_EXPORT double spike_filter(double *, double, int *, double *, double *, double, double, void *);
DLL_EXPORT double density_filter(double *, double, int *, double *, double *, double, double, void *);
DLL_EXPORT double distance_filter(double *, double, int *, double *, double *, double, double, void *);
DLL_EXPORT double nearest_filter(double *, double, int *, double *, double *, double, double, void *);
DLL_EXPORT double ballcount_filter(double *, double, int *, double *, double *, double, double, void *);
DLL_EXPORT double ray_mean_dist_filter(double *, double, int *, double *, double *, double, double, void *);
DLL_EXPORT double mean_3d_filter(double *, double, int *, double *, double *, double, double, void *);
DLL_EXPORT double idw_filter(double *, double, int *, double *, double *, double, double, void *);
/* end filter functions */

DLL_EXPORT void moving_bins(double *z, int *nout, double rad, int n);
DLL_EXPORT void binary_fill_gaps(char *M, char *out, int nrows, int ncols);
