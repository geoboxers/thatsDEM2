/*
 * Copyright (c) 2016, Geoboxers <info@geoboxers.com>
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
DLL_EXPORT void wrap_bilin(double *grid, double *xy, double *out, double *geo_ref, double nd_val, int nrows, int ncols, int npoints);
DLL_EXPORT void resample_grid(double *grid, double *out, double *geo_ref, 
                   double *geo_ref_out, double nd_val, 
                   int nrows, int ncols, int nrows_out, int ncols_out);
DLL_EXPORT void grid_most_frequent_value(int *sorted_indices, int *values, int *out, int vmin,int vmax,int nd_val, int n);
DLL_EXPORT void masked_mean_filter(float *dem, float *out, char *mask, int filter_rad, int nrows, int ncols);
DLL_EXPORT int flood_cells(float *dem, float cut_off, char *mask, char *mask_out, int nrows, int ncols);
DLL_EXPORT unsigned long walk_mask(char *M, int *start, int *end, int *path, unsigned long buf_size, int nrows, int ncols);
DLL_EXPORT void binary_fill_gaps(char *M, char *out, int nrows, int ncols);
