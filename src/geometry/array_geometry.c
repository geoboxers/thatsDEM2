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
/*
* Super simple "is point in buffer around line string implementation"
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "array_geometry.h"
#define DOT(x,y) (x[0]*y[0]+x[1]*y[1])
#define DOT3(x,y) (x[0]*y[0]+x[1]*y[1]+x[2]*y[2])
#define MIN(x,y)  ((x) < (y) ? (x) : (y))
#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define ABS(x) ((x) < 0 ? -(x) : (x))
#define MEPS (-1e-9)
#define PEPS (1+1e-9)
#define DET(x,y)  (x[0]*y[1]-x[1]*y[0])
#define SQUARE(x) (x)*(x)
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

/* declare some internal static functions */
static double d_p_line(double *p1,double *p2, double *p3);
static double d_p_line_string(double *p, double *verts, unsigned long nv);
static int do_lines_intersect(double *p1,double *p2, double *p3, double *p4);
static int compar (const void* a, const void* b);


/*almost copy from trig_index.c*/
static int do_lines_intersect(double *p1,double *p2, double *p3, double *p4){
    double v1[2],v2[2],v3[3],st[2],D;
    int i;
    for(i=0; i<2; i++){
        v1[i]=p2[i]-p1[i];
        v2[i]=p3[i]-p4[i];
        v3[i]=p3[i]-p1[i];
    }
    D=DET(v1,v2); 
    if (ABS(D)<1e-10){
        /* lines are almost parallel*/
        return 0;
    }
    st[0]=(v2[1]*v3[0]-v2[0]*v3[1])/D;
    st[1]=(-v1[1]*v3[0]+v1[0]*v3[1])/D;
    
    if (st[0]>MEPS && st[0]<PEPS && st[1]>MEPS && st[1]<PEPS)
        return 1;
    
    return 0;
}
    



int p_in_poly(double *p_in, char *mout, double *verts, unsigned int np, unsigned int *nv, unsigned int n_rings){
    unsigned int i,j,k,n=0, n_hits;
    double bounds[4]; /*x1,x2,y1,y2*/
    double p_end[2],*pv;
    bounds[0]=verts[0];
    bounds[1]=verts[0];
    bounds[2]=verts[1];
    bounds[3]=verts[1];
    /* printf("Npoints: %d ,Nrings: %d\n",np,n_rings); */
    /*loop over outer ring*/
    for(i=0; i<nv[0]; i++){
        bounds[0]=MIN(bounds[0],verts[2*i]);
        bounds[1]=MAX(bounds[1],verts[2*i]);
        bounds[2]=MIN(bounds[2],verts[2*i+1]);
        bounds[3]=MAX(bounds[3],verts[2*i+1]);
    }
    /* printf("Bounds %.3f %.3f %.3f %.3f\n",bounds[0],bounds[1],bounds[2],bounds[3]);*/
    
    for(i=0; i< np; i++){
        mout[i]=0;
        if (p_in[2*i]<bounds[0] || p_in[2*i]>bounds[1] || p_in[2*i+1]<bounds[2] || p_in[2*i+1]>bounds[3]){
            /* printf("out of bounds: %.3f %.3f\n",p_in[2*i],p_in[2*i+1]);*/
            continue;
        }
        /*avoid parallel lines!*/
        p_end[1]=p_in[2*i+1]+8.1234;
        p_end[0]=bounds[1]+10; /*almost an infinite ray :-) */
        n_hits=0;
        /*printf("p_in: %.2f %.2f\n",p_in[2*i],p_in[2*i+1]);*/
        pv=verts;
        for(j=0; j<n_rings; j++){
            /*printf("Ring: %d, nv: %d\n",j,nv[j]);*/
            for (k=0; k<nv[j]-1; k++){
                n_hits+=do_lines_intersect(p_in+2*i,p_end,pv,pv+2);
                /*printf("Point: %d, line: %d, (%.2f %.2f, %.2f %.2f), nhits: %d\n",i,k,*pv,*(pv+1),*(pv+2),*(pv+3),n_hits);*/
                pv+=2;
            }
            pv+=2; 
        }
        
        if (n_hits % 2 ==1){ 
            mout[i]=1;
            n+=1;
        }
    }
        
    return (n>0) ? 1 : 0;
}




/*returns squared distance*/
static double d_p_line(double *p1,double *p2, double *p3){
    /* p1 is the point, p2, p3 - the line*/
    double p[2],v[2],dot1,dot2;
    p[0]=p1[0]-p2[0];
    p[1]=p1[1]-p2[1];
    v[0]=p3[0]-p2[0];
    v[1]=p3[1]-p2[1];
    dot1=DOT(p,v);
    if (dot1<0){
        /*puts("In da start!");*/
        return DOT(p,p);
    }
    dot2=DOT(v,v);
    if (dot1<dot2){
        /*puts("Yep in da middle!");*/
        dot1/=dot2;
        v[0]=p[0]-v[0]*dot1;
        v[1]=p[1]-v[1]*dot1;
        return DOT(v,v);
    }
    /*puts("in da end");*/
    v[0]=p3[0]-p1[0];
    v[1]=p3[1]-p1[1];
    return DOT(v,v);
    /*<x,v> v /<v,v> ,, |<x,v> v /<v,v>|^2 < <v,v> <-> <x,y> < <v,v> */
}

/*returns squared distance*/
static double d_p_line_string(double *p, double *verts, unsigned long nv){
    unsigned long i;
    double d0, d=d_p_line(p,verts,verts+2);
    for (i=1; i<nv-1; i++){
        /*printf("d is: %.4f, vertex: %d\n",d,i);*/
        d0=d_p_line(p,verts+2*i,verts+2*(i+1));
        /*printf("d0 is: %.4f\n",d0);*/
        d=MIN(d,d0);
        
    }
    return d;
}

void p_in_buf(double *p_in, char *mout, double *verts, unsigned long np, unsigned long nv, double d){
    unsigned long i;
    double d2=d*d;
    for(i=0; i< np; i++)
        mout[i]=(d_p_line_string(p_in+2*i,verts,nv)<d2) ? 1 :0;
    return;
}


unsigned long simplify_linestring(double *xy_in, double *xy_out, double dist_tol, unsigned long n_pts){
    /* simplify a linestring. xy_in and and xy_out must be of size 2*n_pts doubles */
    /* will return the number of output pts. */
    unsigned long pos1=0, pos2, pos3, new_size=0;
    double *p1, *p2, *p3, dtol2, dmax;
    dtol2 = SQUARE(dist_tol);
    while (pos1 < n_pts){
        p1 = xy_in + 2*pos1;
        xy_out[2*new_size] = *p1;
        xy_out[2*new_size + 1] = *(p1 + 1);
        new_size++;
        for(pos2 = n_pts -1; pos2 > pos1 +1; pos2--){
            dmax = 0;
            p2 = xy_in + 2*pos2;
            for(pos3 = pos1 +1; pos3 < pos2 && dmax <= dist_tol;  pos3++){
                p3 = xy_in + 2*pos3;
                dmax = MAX(d_p_line(p3, p1, p2), dmax);
            }
            if (dmax < dtol2){
                pos1 = pos2-1;
                break;
            }
            
        }
        pos1++;
    }
    return new_size;
}


static void compute_normal(double *p1, double *p2, double *p3,double z1, double z2, double z3, double *n){
    double v1[3],v2[3];
    int i;
    /* compute two 3d vectors*/
    for(i=0;i<2;i++){
        v1[i]=p2[i]-p1[i];
        v2[i]=p3[i]-p1[i];
    }
    v1[2]=z2-z1;
    v2[2]=z3-z1;
    n[0]=v1[1]*v2[2]-v1[2]*v2[1];
    n[1]=-(v1[0]*v2[2]-v1[2]*v2[0]);
    n[2]=v1[0]*v2[1]-v1[1]*v2[0];
}

void get_normals(double *xy, double *z, int *triangles, double *out, int ntriangles){
    int i;
    double *p1,*p2,*p3,z1,z2,z3;
    for(i=0;i<ntriangles;i++){
        p1=xy+2*triangles[3*i];
        p2=xy+2*triangles[3*i+1];
        p3=xy+2*triangles[3*i+2];
        z1=z[triangles[3*i]];
        z2=z[triangles[3*i+1]];
        z3=z[triangles[3*i+2]];
        compute_normal(p1,p2,p3,z1,z2,z3,out+3*i);
    }
}

void get_triangle_geometry(double *xy, double *z, int *triangles, float *out , int ntriangles){
    int i;
    double n[3],*p1,*p2,*p3,z1,z2,z3,x1,x2,y1,y2,zmax,zmin;
    for(i=0;i<ntriangles;i++){
        p1=xy+2*triangles[3*i];
        p2=xy+2*triangles[3*i+1];
        p3=xy+2*triangles[3*i+2];
        z1=z[triangles[3*i]];
        z2=z[triangles[3*i+1]];
        z3=z[triangles[3*i+2]];
        compute_normal(p1,p2,p3,z1,z2,z3,n);
        /*compute bbox and tanv2 - angle between z axis and normal - thus large values are critical, 1 correponds to 45 deg*/
        out[3*i]=(float) ((n[0]*n[0]+n[1]*n[1])/(n[2]*n[2])); /*hmm could be inf*/
        x1=MIN(MIN(p1[0],p2[0]),p3[0]);
        x2=MAX(MAX(p1[0],p2[0]),p3[0]);
        y1=MIN(MIN(p1[1],p2[1]),p3[1]);
        y2=MAX(MAX(p1[1],p2[1]),p3[1]);
        zmax=MAX(MAX(z1,z2),z3);
        zmin=MIN(MIN(z1,z2),z3);
        out[3*i+1]=(float) MAX(x2-x1,y2-y1);
        out[3*i+2]=(float) (zmax-zmin);
    }
    return;
}

/***************************
** Fast filling of ground below DTM
** just fill ones below dtm
****************************/

void fill_it_up(unsigned char *out, unsigned int *hmap, int rows, int cols, int stacks){
    int z,i,j,k;
    for(i=0; i<rows; i++){
        for(j=0; j<cols; j++){
            z=hmap[i*cols+j];
            /*voxel (i,j,k) filling height k->k+1, e.g. h=0.5 lies at stack index 0*/
            for(k=0; k<z && k<stacks; k++){
                out[i*cols*stacks+j*stacks+k]=1;
            }
            
        }
    }
}

/* mark things thats not connected to  ground - return the vertical distance to top of ground component - can be negative*/
void find_floating_voxels(int *lab, int *out, int gcomp, int rows, int cols, int stacks){
    int i,j,k,z,L;
    for(i=0; i<rows; i++){
        for(j=0; j<cols; j++){
            /*loop up at (i,j) - find max height of ground component*/
            z=0;
            for(k=0; k<stacks; k++){
                L=lab[i*cols*stacks+j*stacks+k];
                if (L==gcomp)
                    z=k;
                
            }
            for(k=0; k<stacks; k++){
                L=lab[i*cols*stacks+j*stacks+k];
                if (L==0)
                    continue;
                if (L!=gcomp)
                    out[i*cols*stacks+j*stacks+k]=(k-z);
                
            }
        }
    }
}   



/***************************************
*  
*   Filtering stuff below here
*
*
*
*
***************************************/

    

/*fill a spatial index for a pointcloud*/
int fill_spatial_index(int *sorted_flat_indices, int *index, int npoints, int max_index){
    int i,j, ind, current_index=sorted_flat_indices[0];
    for(i=0;i<2*current_index;i++){
        index[i]=0;  /*empty slices here*/
    }
    index[2*current_index]=0;
    
    for(i=1; i<npoints; i++){
        ind=sorted_flat_indices[i];
        if (ind>(max_index-1))
            return 1;
        if (ind>current_index){
            for(j=2*current_index+1; j<2*ind; j++){
                index[j]=i;
            }
            index[2*ind]=i;
            current_index=ind;
        }
    }
    /*printf("Current_index: %d, max: %d\n",current_index,max_index);*/
    for(i=(2*current_index+1); i<2*max_index; i++)
        index[i]=(npoints); /* right endpt is NOT include, this fixes fuckup for last point*/
    return 0;
}



/* this will simply give us slices to boxes around the box of each pt. - the finer details are left to the filter func-*/
void apply_filter(double *xy, 
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
                  void *opt_params){
    int i,j, ind1,ind2, nfound, slices[6],r,c,r1,c1,c2,ncols,nrows;
    double x1,y2,cs,zz, frad2;
    ncols=(int) header[0];
    nrows=(int) header[1];
    x1=header[2];
    y2=header[3];
    cs=header[4];
    frad2=SQUARE(filter_rad);
    /*unsigned long mf=0;*/
    for(i=0; i<npoints; i++){
        /*if (i%2==0)
            printf("Done %d\n", i);*/
        vals_out[i]=nd_val;
        c=(int) ((xy[2*i]-x1)/cs);
        r=(int) ((y2-xy[2*i+1])/cs);
        /*should ensure that c-1 can never be larger than ncols-1, etc*/
        if (c<-1 || c>ncols || r<-1 || r>nrows)
            continue;
        /*perhaps do something if we fall suficciently outside region*/
        nfound=0;
        for(j=-1;j<2;j++){
            r1=r+j;
            if (r1<0 || r1>=nrows){ /*empty slice*/
                slices[2*j+2]=0;
                slices[2*j+3]=0;
                continue;
            }
            
            c1=MAX((c-1),0);
            c2=MIN((c+1),(ncols-1));
            ind1=r1*ncols+c1;
            ind2=r1*ncols+c2;
            slices[2*j+2]=spatial_index[2*ind1]; /*start of left cell*/
            slices[2*j+3]=spatial_index[2*ind2+1]; /*end of right included cell*/
            nfound+=slices[2*j+3]-slices[2*j+2];
        }
        if (nfound>0){
            if (z)
                zz=z[i];
            else 
                zz=-1;
            vals_out[i]=filter_func(xy+2*i,zz,slices,pc_xy,pc_z,frad2,nd_val,opt_params); /*the filter func should know how many params there are - or we can terminate list by something...*/
        }
        /*DEBUG*/
        /*if (nfound==0 || vals_out[i]==nd_val){
            printf("Oh no: %.2f %.2f %d %d, ind: %d, pn: %d, all: %d\n",xy[2*i],xy[2*i+1],r,c,r*ncols+c,i,npoints);
            for(j=0;j<3;j++)
                printf("i1: %d i2: %d\n",slices[2*j],slices[2*j+1]);
        }*/
        
        
    } /*end rows*/
}


double min_filter(double *xy, double z, int *indices, double *pc_xy, double *pc_z, double frad2, double nd_val, void *opt_params){
    int i,i1,i2,j, n=0;
    double m=HUGE_VAL,d;
    for(i=0; i<3; i++){
        i1=indices[2*i];
        i2=indices[2*i+1];
        for(j=i1; j<i2; j++){ /*possibly empty slice*/
            d=SQUARE((pc_xy[2*j]-xy[0]))+SQUARE((pc_xy[2*j+1]-xy[1]));
            if (d<=frad2){
                m=MIN(m,pc_z[j]);
                n++;
            }
        }
    }
    return (n>0)? m : nd_val;
}

double max_filter(double *xy, double z, int *indices, double *pc_xy, double *pc_z, double frad2, double nd_val, void *opt_params){
    int i,i1,i2,j, n=0;
    double m=-HUGE_VAL,d;
    for(i=0; i<3; i++){
        i1=indices[2*i];
        i2=indices[2*i+1];
        for(j=i1; j<i2; j++){ /*possibly empty slice*/
            d=SQUARE((pc_xy[2*j]-xy[0]))+SQUARE((pc_xy[2*j+1]-xy[1]));
            if (d<=frad2){
                m=MAX(m,pc_z[j]);
                n++;
            }
        }
    }
    return (n>0)? m : nd_val;
}







/* A spike is a point, which is a local extrama, and where there are steep edges in all four quadrants. An edge is steep if its slope is above a certain limit and its delta z likewise*/
/* all edges must be steep unless it is smaller than filter_radius*0.2 - so filter_radius is significant here!*/
/* paarams are: tanv2 and  delta-z*/
double spike_filter(double *xy,
                    double z,
                    int *indices,
                    double *pc_xy,
                    double *pc_z,
                    double frad2,
                    double nd_val,
                    void *params)
{
    int i,i1,i2,j, n_q1=0, n_q2=0, n_q3=0, n_q4=0, n_all_plus=0, n_all_minus=0, n_used=0, could_be_spike=1;
    double d,dz,dx,dy,mean_dz=0, abs_dz,x=xy[0],y=xy[1],d_lim,slope,tanv2,zlim,*dparams;
    dparams=(double*) params;
    d_lim=dparams[0]; /* speed up by saving as param...*/
    tanv2=dparams[1];
    zlim=dparams[2];
    /*we must ensure that there are points further away than the limit - and that they spread out nicely*/
    for(i=0; i<3 && could_be_spike; i++){
        i1=indices[2*i];
        i2=indices[2*i+1];
        for(j=i1; j<i2; j++){
            dx=pc_xy[2*j]-x;
            dy=pc_xy[2*j+1]-y;
            d=SQUARE(dx)+SQUARE(dy);
            if (d>frad2)
                continue;
            dz=pc_z[j]-z;
            n_all_plus+=(dz>1e-6);
            n_all_minus+=(dz<-1e-6);
            slope=SQUARE(dz)/d;
            abs_dz=ABS(dz);
            /* not a local max min*/
            if (n_all_plus>0 && n_all_minus>0){
                could_be_spike=0;
                break;
            }
            /*we must have steep edges in all quadrants*/
            if (slope>tanv2 && abs_dz>zlim){
                n_q1+=(dx>=0 && dy>=0);
                n_q2+=(dx>=0 && dy<0);
                n_q3+=(dx<0   && dy<0);
                n_q4+=(dx<0 && dy>=0);
            }
            else if (d>d_lim){ /*if not steep, must be close!*/
                could_be_spike=0;
                break;
            }
            mean_dz+=dz;
            n_used++;
        }
    }
    if (!could_be_spike || n_used<4)
        return 0;
    /*OK so we know: local max/min and everything further away than limit is steep*/ 
    if (n_q1>0 && n_q2>0 && n_q3>0 && n_q4>0)
        return mean_dz/n_used;
    return 0;
    
}


double mean_filter(double *xy, double z, int *indices, double *pc_xy, double *pc_z, double frad2, double nd_val, void *opt_params){
    int i,i1,i2,j,n=0;
    double m=0,d;
    for(i=0; i<3; i++){
        i1=indices[2*i];
        i2=indices[2*i+1];
        for(j=i1;j<i2;j++){
            d=SQUARE((pc_xy[2*j]-xy[0]))+SQUARE((pc_xy[2*j+1]-xy[1]));
            if (d<=frad2 && !isnan(pc_z[j])){
                m+=pc_z[j];
                n+=1;
            }
            
        }
    }
    if (n>0){
        m/=n;
        return m;
    }
    return nd_val;
}

double var_filter(double *xy, double z, int *indices, double *pc_xy, double *pc_z, double frad2, double nd_val, void *opt_params){
    int i,i1,i2,j,n=0;
    double m=0,m2=0,d;
    for(i=0; i<3; i++){
        i1=indices[2*i];
        i2=indices[2*i+1];
        for(j=i1;j<i2;j++){
            d=SQUARE((pc_xy[2*j]-xy[0]))+SQUARE((pc_xy[2*j+1]-xy[1]));
            if (d<=frad2 && !isnan(pc_z[j])){
                m+=pc_z[j];
                m2+=SQUARE(pc_z[j]);
                n+=1;
            }
            
        }
    }
    if (n>1){
        return (m2/n-SQUARE((m/n)));
    }
    return nd_val;
}


double density_filter(double *xy, double z, int *indices, double *pc_xy, double *pc_z, double frad2, double nd_val, void *opt_params){
    int i,i1,i2,j,n=0;
    double d;
    for(i=0; i<3; i++){
        i1=indices[2*i];
        i2=indices[2*i+1];
        for(j=i1;j<i2;j++){
            d=SQUARE((pc_xy[2*j]-xy[0]))+SQUARE((pc_xy[2*j+1]-xy[1]));
            if (d<=frad2){
                n+=1;
            }
            
        }
    }
    
    return ((double) n)/(M_PI*frad2);
}

/* Count number of points within a ball of radius frad2 */
double ballcount_filter(double *xy,
                               double z,
                               int *indices,
                               double *pc_xy,
                               double *pc_z,
                               double frad2,
                               double nd_val,
                               void *opt_params){
    int i,i1,i2,j,n=0;
    double d;
    for(i=0; i<3; i++){
        i1=indices[2*i];
        i2=indices[2*i+1];
        for(j=i1;j<i2;j++){
            d=SQUARE((pc_xy[2*j]-xy[0]))+SQUARE((pc_xy[2*j+1]-xy[1]))+SQUARE((pc_z[j]-z));
            if (d<=frad2){
                n+=1;
            }
            
        }
    }
    return (double) n;
}

/* Count number of points within a ball of radius frad2 */
double mean_3d_filter(double *xy,
                      double z,
                      int *indices,
                      double *pc_xy,
                      double *pc_z,
                      double frad2,
                      double nd_val,
                      void *opt_params){
    int i,i1,i2,j,n=0;
    double d, m=0, *vals;
    /*if opt_params is supplied, this MUST be a pointer to a double array*/
    vals = (opt_params) ? (double*) opt_params : pc_z;
    for(i=0; i<3; i++){
        i1=indices[2*i];
        i2=indices[2*i+1];
        for(j=i1;j<i2;j++){
            d=SQUARE((pc_xy[2*j]-xy[0]))+SQUARE((pc_xy[2*j+1]-xy[1]))+SQUARE((pc_z[j]-z));
            if (d<=frad2){
                n+=1;
                m+=vals[j];
            }
            
        }
    }
    return (n>0)? (m/n) : nd_val;
}


double idw_filter(double *xy, double z, int *indices, double *pc_xy, double *pc_z, double frad2, double nd_val, void *opt_params){
    int i,i1,i2,j,n=0;
    double m=0,d,w=0,ww;
    for(i=0; i<3; i++){
        i1=indices[2*i];
        i2=indices[2*i+1];
        for(j=i1;j<i2;j++){
            d=SQUARE((pc_xy[2*j]-xy[0]))+SQUARE((pc_xy[2*j+1]-xy[1]));
            if (d<=frad2 && !isnan(pc_z[j])){
                ww=1/MAX(d,1e-8);
                m+=pc_z[j]*ww;
                w+=ww;
                n+=1;
            }
            
        }
    }
    if (n>0){
        return m/w;
    }
    return nd_val;
}

double distance_filter(double *xy, double z, int *indices, double *pc_xy, double *pc_z, double frad2, double nd_val, void *opt_params){
    int i,i1,i2,j;
    double dmin=HUGE_VAL,d;
    for(i=0; i<3; i++){
        i1=indices[2*i];
        i2=indices[2*i+1];
        for(j=i1;j<i2;j++){
            d=SQUARE((pc_xy[2*j]-xy[0]))+SQUARE((pc_xy[2*j+1]-xy[1]));
            if (d<dmin)
                dmin=d;
            
        }
    }
    return (dmin<HUGE_VAL)?sqrt(dmin):nd_val;
}

/* returns index to nearest - TODO: extend so that index can be stored in opt_params.*/
double nearest_filter(double *xy, 
                             double z, 
                             int *indices, 
                             double *pc_xy, 
                             double *pc_z,  
                             double frad2,
                             double nd_val, 
                             void *opt_params){
    int i,i1,i2,j;
    double dmin=HUGE_VAL,d, idx;
    idx=nd_val;
    for(i=0; i<3; i++){
        i1=indices[2*i];
        i2=indices[2*i+1];
        for(j=i1;j<i2;j++){
            d=SQUARE((pc_xy[2*j]-xy[0]))+SQUARE((pc_xy[2*j+1]-xy[1]));
            if (d<dmin){
                dmin=d;
                idx = j; /*store index*/
            }
            
        }
    }
    return idx;
}
                             
/* Calcs. mean 'distance' in RP2 of rays emanating from a point*/
double ray_mean_dist_filter(double *xy, 
                            double z, 
                            int *indices, 
                            double *pc_xy, 
                            double *pc_z,  
                            double frad2,
                            double nd_val, 
                            void *opt_params){
    int i1,i2,j,k,h, count=0, n_all;
    double n[3],n1, n2, *vects, md=0;
    n_all = indices[1]-indices[0]+indices[3]-indices[2]+indices[5]-indices[4];
    vects = malloc(sizeof(double)*3*(n_all)*(n_all-1)/2);
    if (!vects){
        printf("Out of memory!");
        return 0;
    }
    for(i1=0; i1<3; i1++){
        for(j=indices[2*i1];j<indices[2*i1+1];j++){
            /*consider making this more '3dish' by using z in selection!!*/
            n1 = SQUARE((xy[0]-pc_xy[2*j]))+SQUARE((xy[1]-pc_xy[2*j+1]));
            if (n1>frad2) 
                continue;
            /*same row*/
            for(k=j+1; k<indices[2*i1+1]; k++){
                n1 = SQUARE((xy[0]-pc_xy[2*k]))+SQUARE((xy[1]-pc_xy[2*k+1]));
                if (n1>frad2) 
                    continue;
                compute_normal(xy,pc_xy+2*k,pc_xy+2*j,z,pc_z[k],pc_z[j],n);
                n2 = SQUARE(n[0])+SQUARE(n[1])+SQUARE(n[2]);
                if (n2>0.1*frad2){
                    n1=sqrt(n2);
                    /*printf("NORM: %.2f\n",nn);
                    printf("j: %d, k: %d\n",j,k);*/
                    for(h=0; h<3; h++){
                        vects[3*count+h]=n[h]/n1;
                    }
                    count++;
                }
            }
            /*next rows*/
            for(i2=i1+1; i2<3; i2++){
                for(k=indices[2*i2]; k<indices[2*i2+1]; k++){
                    n1 = SQUARE((xy[0]-pc_xy[2*k]))+SQUARE((xy[1]-pc_xy[2*k+1]));
                    if (n1>frad2) 
                        continue;
                     compute_normal(xy,pc_xy+2*k,pc_xy+2*j,z,pc_z[k],pc_z[j],n);
                     n2 = SQUARE(n[0])+SQUARE(n[1])+SQUARE(n[2]);
                     if (n2>0.1*frad2){
                        n1=sqrt(n2);
                        /*printf("NORM: %.2f\n",nn);
                        printf("j: %d, k: %d\n",j,k);*/
                        for(h=0; h<3; h++){
                            vects[3*count+h]=n[h]/n1;
                            
                        }
                        count++;
                     }
                }
            }
        }
    }
    if (count>1){
        double dot;
        for(j=0;j<count; j++){
            /*printf("vects: %2f, %2f, %2f\n", vects[3*j],vects[3*j+1],vects[3*j+2]);*/
            for(k=j+1;k<count;k++){
                /* project to RP2 by taking ABS*/
                dot=ABS(DOT3((vects+3*j),(vects+3*k)));
                /*printf("dot: %.3f\n",dot);*/
                md+=(1-dot);
            }
        }
        md/=count*(count-1)/2;
    }
    free(vects);
    return (count>1)?md:-1;
    
}

static int compar (const void* a, const void* b){
    if ( *(double*)a <  *(double*)b ) return -1;
    if ( *(double*)a == *(double*)b ) return 0;
    return 1;
}


/*todo - to fractile_filter and implement faster sorting...*/
double median_filter(double *xy, double z, int *indices, double *pc_xy, double *pc_z, double frad2, double nd_val, void *nothing){
    int i,i1,i2,j,n=0,n_all=0;
    double *zs, m=nd_val,d;
    for(i=0; i<3; i++)
        n_all+=indices[2*i+1]-indices[2*i];
    zs=malloc(sizeof(double)*n_all);
    /*rather core dump than return nd_val?*/
    for(i=0; i<3; i++){
        i1=indices[2*i];
        i2=indices[2*i+1];
        for(j=i1;j<i2;j++){
            d=SQUARE((pc_xy[2*j]-xy[0]))+SQUARE((pc_xy[2*j+1]-xy[1]));
            if (d<=frad2 && ! isnan(pc_z[j])){
                zs[n]=pc_z[j];
                n+=1;
            }
            
        }
    }
    if (n>0){
        if (n>1)
            qsort(zs,n,sizeof(double),compar);
        i=n/2;
        if (n%2==0)
            m=(zs[i-1]+zs[i])*0.5;
        else
            m=zs[i];
        
    }
    free(zs);
    return m;
}
    
/* A Gaussian filter with dynamic width according to distance distribution */
double adaptive_gaussian_filter(double *xy, double z,
                                 int *indices, double *pc_xy, 
                                 double *pc_z, double frad2, 
                                 double nd_val, void *opt_params){
    int i, i1, i2, j, n=0, n_all=0, *params;
    double *ds, m=nd_val, d, dm=0;
    params= (int*) opt_params;
    for(i=0; i<3; i++){
        n_all += indices[2*i + 1] - indices[2*i];
    }
    ds = malloc(sizeof(double)*n_all);
    for(i=0; i<3; i++){
        i1=indices[2*i];
        i2=indices[2*i + 1]; /*possibly empty slice*/
        for(j=i1; j<i2; j++){
            d=SQUARE((pc_xy[2*j]-xy[0]))+SQUARE((pc_xy[2*j + 1]-xy[1]));
            if (d<=frad2 && ! isnan(pc_z[j])){
                ds[n++] = d;
                dm += d;
                m = pc_z[j]; /* store last one for now */
            }
        } 
    }
    /*printf("m: %.2f, n: %d\n", m, n);*/
    if (n > 1){
        double w, tw=0, sd;
        int n_nearest;
        n_nearest = params[0];
        m = 0; /* reset m */
        if (n >= n_nearest){
            qsort(ds, n, sizeof(double), compar);
            i = n_nearest / 2;
            /*printf("I is: %d, n_nearest is: %d\n", i, n_nearest);*/
            sd = ds[i];
        }
        else{
            sd = (dm / n) * 0.5;
        }
        /*printf("sd: %.2f\n", sqrt(sd));*/
        for(i=0; i<3; i++){
            i1=indices[2*i];
            i2=indices[2*i+1]; /*possibly empty slice*/
            for(j=i1; j<i2; j++){
                d=SQUARE((pc_xy[2*j]-xy[0]))+SQUARE((pc_xy[2*j+1]-xy[1]));
                if (d<=frad2 && d < (4 * sd) && ! isnan(pc_z[j])){
                    w = exp(-d / sd);
                    tw += w;
                    m += w * pc_z[j];
                }
            }
        }
        /*printf("tw: %.4f\n", tw);*/
        m /= tw;
        /* DEBUG - turn off eventually */
        if (isnan(m)){
            printf("n: %d, tw: %.5f, sd: %.4f, dm: %.4f\n", n, tw, sd, dm/n);
        }
    }
    free(ds);
    return m;
}


/* tanv2 is tangens of steepnes angle squared 
void pc_spike_filter(double *xy, double *z, double *pc_xy, double *pc_z, double *z_out, double filter_rad, double tanv2, double zlim, int *spatial_index, double *header, int npoints){
    double params[3];
    params[0]=SQUARE(filter_rad*0.2);
    params[1]=tanv2;
    params[2]=zlim;
    apply_filter(xy,z,pc_xy,pc_z, z_out, spatial_index, header, npoints, spike_filter, filter_rad, 0, params);
    
}
*/

/* A triangle based 'filter' - on  input zout should be a copy of z */

void tri_filter_low(double *z, double *zout, int *tri, double cut_off, int ntri){
    int i,j,m,I[3];
    double zt[3];
    for(i=0;i<ntri;i++){
        for(j=0;j<3;j++){
            I[j]=tri[3*i+j];
            zt[j]=z[I[j]];
        }
        for(j=0;j<3;j++){
            m=(j+1)%3;
            if ((zt[j]-zt[m])>cut_off){
                zt[j]=zt[m];
                zout[I[j]]=MIN(zt[m],zout[I[j]]);
            }
        }
    }
}


void mark_bd_vertices(char *bd_candidates_mask, char *poly_mask, int *triangles, char *bd_mask_out, int ntriangles, int np){
    int i,j,v;
    for(i=0; i<np; i++) bd_mask_out[i]=0;
    for(i=0; i<ntriangles; i++){
        if (bd_candidates_mask[i]){ /*this triangle is long, or steep, or something...*/
            /* check if any of the vertices are inside the 'polygon'*/
            for(j=0; j<3; j++){
                v=triangles[3*i+j];
                if (poly_mask[v])
                    bd_mask_out[v]=1;
            }
        }
    }
    return;
}
        
/* rolling z binning for radon transform 
* input a sorted list of values - spool through and report the number of pts rad below and rad ahead
*/
void moving_bins(double *z, int *nout, double rad, int n){
    int i=0,j=0,nhere=0;
    double z0;
    /*two locaters - one should always be size ahead of the other*/
    for(i=0;i<n;i++){
        z0=z[i]; /*the pt.*/
        /*spool ahead*/
        j=i;
        nhere=0;
        while((z[j]-z0)<rad && j<n){
            j++;
        }
        nhere+=(j-i); /* counting the point also*/
        j=i;
        /*spool back*/
        while ((z0-z[j])<rad && j>=0){
            j--;
        }
        nhere+=(i-j);
        nout[i]=nhere-1; /*subtract one of the two extra counts*/
            
    }
}

    


#ifdef SVEND_BENT
int main(int argc, char **argv){
    double verts[20]={0,0,1,0,1,1,0,1,0,0,0.3,0.3,0.6,0.3,0.6,0.6,0.3,0.6,0.3,0.3};
    unsigned int nv[2]={5,5};
    double xy[2],d;
    char mask[1];
    int i,n;
    if (argc<3){
        puts("Two coords, please!");
        return 1;
    }
    xy[0]=atof(argv[1]);
    xy[1]=atof(argv[2]);
    printf("Distance from (%.2f,%.2f) to line is: %.4f\n",xy[0],xy[1],d_p_line_string(xy,verts,4));
    n=p_in_poly(xy,mask,verts,1,nv,2);
    printf("Return code %d, point in poly: %d\n",n,mask[0]);
    return 0;
}
#endif      
