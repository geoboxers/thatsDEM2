/*
 * Original work Copyright (c) 2015, Danish Geodata Agency <gst@gst.dk>
 * Modified work Copyright (c) 2015, Geoboxers <info@geoboxers.com>
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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef _MSC_VER
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT  
#endif
#define MIN(x,y)  ((x<y) ? x:y)
#define MAX(a,b) (a>b ? a: b)
#define ABS(x) ((x) > 0 ? (x) : -(x))
/*simple bilnear interpolation here:
* If geo_ref is NULL, we assume that xy is already in array coordinates,
* If geo_ref is NOT NULL it should be an array of len 4: x1,cx,y2,cy - with (x1,y2) being the center of the upper left 'pixel', i.e. the location of the upper left grid point, i.e.:
* x1=xul_center, y2=yul_center
* This is unlike the usual GDAL-style convention (pixel corner), since we primarily focus on interpolation in 'geoid' grids... 
* We consider the values of the grid array as representing the grid values at the 'centers of the pixels'.
* Simplistic and predictable approach: If upper left corner is no-data, output will be no-data. Otherwise no-data is filled clock-wise.
* This means that no-data will 'spread' in the south-west direction... However exact cell centers should be interpolateable with no no-data spreading ;-)
* Grid should always have a nd_val. If there is none - it's up to the caller to supply one, which is not a regular grid_val, e.g. min(grid)-999...
* CONSIDER making this just use thokns esrigrid.h....
*/
static double simple_bilin(double *grid, double x, double y, double *geo_ref, double nd_val, int nrows, int ncols);

static double simple_bilin(double *grid, double x, double y, double *geo_ref, double nd_val, int nrows, int ncols){
	int i,j;
	double dx,dy,grid_vals[4],*g;
	if (geo_ref){
		x=(x-geo_ref[0])/geo_ref[1];
		y=(geo_ref[2]-y)/geo_ref[3];
	}
	i=(int) y;
	j=(int) x;
	/*ok - so the lower and right boundary is not included... too bad man*/
	if (i<0 ||  j<0 || i>(nrows-2) || j>(ncols-2)){
			return nd_val;
	}
	dx=x-j;
	dy=y-i;
	/*clock-wise filling of values and no_data check*/
	g=grid+(ncols*i+j);
	if (nd_val==(grid_vals[0]=*g)){
			return nd_val;
	}
	if (nd_val==(grid_vals[1]=*(g+1))){
		grid_vals[1]=grid_vals[0];
	}
	if (nd_val==(grid_vals[2]=*(g+1+ncols))){
		grid_vals[2]=grid_vals[1];
	}
	if (nd_val==(grid_vals[3]=*(g+ncols))){
		grid_vals[3]=grid_vals[2];
	}
	/*possibly the compiler will be able to optimize this expression...*/
	return (grid_vals[0]+dx*(grid_vals[1]-grid_vals[0])+dy*(grid_vals[3]-grid_vals[0])+dx*dy*(grid_vals[0]-grid_vals[1]-grid_vals[3]+grid_vals[2]));
}

DLL_EXPORT void wrap_bilin(double *grid, double *xy, double *out, double *geo_ref, double nd_val, int nrows, int ncols, int npoints){
	int k;
	for(k=0; k<npoints; k++){
		/*find the 4 centers that we need*/
		out[k]=simple_bilin(grid,xy[2*k],xy[2*k+1],geo_ref,nd_val,nrows,ncols);
		
	}		
}

/*both grid1 and grid2 must be georeferenced like described above*/
DLL_EXPORT void resample_grid(double *grid, double *out, double *geo_ref, double *geo_ref_out, double nd_val, int nrows, int ncols, int nrows_out, int ncols_out){
	int i,j;
	double x,y;
	for(i=0;i<nrows_out; i++){
		for(j=0;j<ncols_out; j++){
			x=geo_ref_out[0]+j*geo_ref_out[1]; /* geo_ref[0] refers to pixel 'center' - POINT interpretation...*/
			y=geo_ref_out[2]-i*geo_ref_out[3];
			out[i*ncols_out+j]=simple_bilin(grid,x,y,geo_ref,nd_val,nrows,ncols);
		}
	}
}


/* Assign most frequent value in each cell to output grid */
/* Very important that max sorted_indices is less than size ofout */
/* Precheck that range vmax-vmin+1 is not too large*/
DLL_EXPORT void grid_most_frequent_value(int *sorted_indices, int *values, int *out, int vmin,int vmax,int nd_val, int n){
	int i,j,*count,range,cell,current_cell,val;
    int most_frequent, max_count;
	range=vmax-vmin+1;
	count=calloc(range,sizeof(int));
	current_cell=sorted_indices[0];
	for(i=0; i<n ; i++){
		cell=sorted_indices[i];
		if (cell>current_cell){
			/*assign value and move on*/
			most_frequent=nd_val;
            max_count=-1; /*-1 will be a no-data value*/
			for(j=0; j<range; j++){
				if (count[j]>max_count){
					most_frequent=j;
					max_count=count[j];
				}
				count[j]=0; /*reset*/
			}
			out[current_cell]=most_frequent+vmin;
			current_cell=cell;
		}
		else{
			val=values[i]-vmin;
			if (val>=0 && val<range)
				count[val]++;
		}
		
	}
    /*assign last value*/
    most_frequent = nd_val;
    max_count = -1; /*-1 will be a no-data value*/
    for(j=0; j<range; j++){
        if (count[j] > max_count){
            most_frequent = j;
            max_count = count[j];
        }
    }
    out[current_cell] = most_frequent+vmin;
    free(count);
}

/* MASK based raster filters*/

DLL_EXPORT void masked_mean_filter(float *dem, float *out, char *mask, int filter_rad, int nrows, int ncols){
	/*consider what to do about nd_vals - should probably be handled by caller */
	int i,j,i1,i2,j1,j2,m,n,ind1,ind2,used;
	double v;
	for(i=0 ; i<nrows ; i++){
		for(j=0; j<ncols ; j++){
			ind1=i*ncols+j;
			if (!mask[ind1])
				continue;
			used=0;
			i1=MAX((i-filter_rad),0);
			i2=MIN((i+filter_rad),(nrows-1));
			j1=MAX((j-filter_rad),0);
			j2=MIN((j+filter_rad),(ncols-1));
			used=0;
			v=0.0;
			for(m=i1; m<=i2; m++){
				for(n=j1; n<=j2; n++){
					ind2=m*ncols+n;
					if (mask[ind2]){
						used++;
						v+=(double) dem[ind2];
					}
				}
			}
			/*must be at least one used - well check anyways!*/
			if (used>0)
				out[ind1]=(float) (v/used);
		}
	}
}	

/* Wander around along a water mask and expand flood cells - we can make "channels" along large triangles by setting dem-values low there...*/
DLL_EXPORT int flood_cells(float *dem, float cut_off, char *mask, char *mask_out, int nrows, int ncols){
	int i,j,m,n,i1,j1,n_set=0;
	float v,w;
	size_t ind1,ind2;
	for(i=0; i<nrows; i++){
		for(j=0; j<ncols; j++){
			ind1=i*ncols+j;
			if (mask[ind1]){
				/* a window value of one will ensure connectedness*/
				v=dem[ind1];
				for(m=-1; m<=1; m++){
					for(n=-1;n<=1; n++){
						if ((m+n)!=1 && (m+n)!=-1)
							continue;
						i1=(i+n);
						j1=(j+m);
						if (i1<0 || i1>(nrows-1) || j1<0 || j1>(ncols-1))
							continue;
						ind2=i1*ncols+j1;
						w=dem[ind2];
						
						if ((w-v)<=cut_off && !mask[ind2]){
							mask_out[ind2]=1;
							n_set++;
						}
					}
				}
			}
		}
	}
	return n_set;
}

/* path finding in mask from start to end - if possible 
 * will modify mask in place.
 * will return number of vertices */
DLL_EXPORT unsigned long walk_mask(char *M, int *start, int *end, int *path, unsigned long buf_size, int nrows, int ncols){
    /* path must have allocated 2 * bufsize * sizeof(int) bytes */
    /* will include rows like (i,j) */
    int i, j, k1, k2;
    int steps[3] = {-1, 0 , 1}; 
    unsigned long path_size = 0;
    if (!M[start[0] * ncols + start[1]] || !M[end[0] * ncols + end[1]]){
        fprintf(stderr, "walk_mask: endpoints not ok.");
        return 0;
    }
    /* now walk*/
    i = start[0];
    j = start[1];
    path[0] = i;
    path[1] = j;
    path_size = 1;
    /*puts("lets go!");*/
    while (path_size < buf_size && (i != end[0] || j != end[1])){
        int found = 0, diri = 1, dirj = 1, ni, nj, di, dj;
        /* remove current stepping stone*/
        /*printf("Cur pos: (%d, %d)\n", i,j);*/
        M[i * ncols + j] = 0;
        /* search */
        di = ( end[0] > i)? 2 : ((end[0] == i) ? 1 : 0 );
        dj = ( end[1] > j)? 2 : ((end[1] == j) ? 1 : 0 );
        diri = (di == 2) ? -1 : 1;
        dirj = (dj == 2) ? -1 : 1; 
        /*printf("d: %d, %d\n", di -1, dj -1);*/
        for(k1 = 0; k1 < 3 && !found; k1++){
            ni = i + steps[(diri*k1 + di) % 3];
            if (ni < 0 || ni > (ncols -1))
                continue;
            for(k2 = 0; k2 < 3 && !found; k2++){
                nj = j + steps[(dirj*k2 + dj) % 3];
                /* current pos has been deleted so (0, 0) is not a valid step */
                if (0 <= nj  && nj < ncols && M[ni * ncols + nj]){
                    /* append next position*/
                    j = nj;
                    i = ni;
                    path[2 * path_size] = i;
                    path[2 * path_size + 1] = j;
                    path_size++;
                    found = 1;
                }
            } /* end col step */
        }/* end row step*/
        /*printf("Found: %d, psize: %lu, bufsize: %lu\n", found, path_size, buf_size);*/
        /*printf("After: (%d, %d)\n", i, j);*/
        if (!found){
            /*puts("backwards");*/
            if (path_size > 1){
                /*try to rewind*/
                path_size --;
                i = path[2 * path_size];
                j = path[2 * path_size + 1];
            }
            else{
                break;
            }
        }
    }
    /*printf("Finally: (%d,%d), psize: %lu, bufsize: %lu\n", i, j, path_size, buf_size);*/
    if (i != end[0] || j != end[1]){
        path_size = 0;
    }
    else if (path_size > 2){
        int *new_path;
        unsigned long new_path_size = 0, pos1 = 0, pos2 = 0;
        new_path = malloc(sizeof(int) * 2 * path_size);
        if (!new_path){
            fprintf(stderr, "walk_mask: Allocation error.");
            return path_size;
        }
        while( pos1 < path_size){
            /*append*/
            memcpy(new_path + 2 * new_path_size, path + 2 * pos1, 2 * sizeof(int));
            new_path_size++;
            for(pos2 = path_size -1; pos2 > pos1; pos2--){
                if (ABS(path[2 * pos1] - path[2 * pos2]) < 2 && ABS(path[2 * pos1 + 1] - path[2 * pos2 +1]) < 2){
                    pos1 = pos2 -1;
                    break;
                }
            }
            pos1++;
        }
        memcpy(path, new_path, 2 * new_path_size * sizeof(int));
        path_size = new_path_size;
        free(new_path); 
    }
    return path_size;  
}

/* Fill gaps in order to connect close components*/
DLL_EXPORT void binary_fill_gaps(char *M, char *out, int nrows, int ncols){
	int i,j;
	size_t ind;
	for (i=0; i<nrows; i++){
		for(j=0; j<ncols; j++){
			ind=i*ncols+j;
			out[ind]=M[ind];
			if (M[ind])
				continue;
			if (j>0 && j<(ncols-1) && M[ind-1] && M[ind+1]){
				out[ind]=1;
				continue;
			}
			if (i>0 && i<(nrows-1) && M[ind-ncols] && M[ind+ncols]){
				out[ind]=1;
				continue;
			}
			if (i==0 || i==(nrows-1) || j==0 || j==(ncols-1))
				continue;
			if ((M[ind-ncols-1] && M[ind+ncols+1]) || (M[ind-ncols+1] && M[ind+ncols-1])){ /* ul && lr or  ur && ll*/
				out[ind-1]=1;
				out[ind]=1;
				out[ind+1]=1;
			}
			
		}
	}
}
