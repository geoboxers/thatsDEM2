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

DLL_EXPORT int *use_triangle(double *xy, int np, int *nt);
DLL_EXPORT int *use_triangle_pslg(double *xy, int *segments, double *holes, int np, int nseg, int nholes, int *nt);
DLL_EXPORT void get_triangles(int *verts, int *indices, int *out,  int n_indices, int n_trigs);
DLL_EXPORT void get_triangle_centers(double *xy, int *triangles, double *out, int n_trigs);
DLL_EXPORT void free_vertices(int *verts);
