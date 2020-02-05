/*

    Copyright (C) 2010-2020 PyTRiP98 Developers.

    This file is part of PyTRiP98.

    PyTRiP98 is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyTRiP98 is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PyTRiP98.  If not, see <http://www.gnu.org/licenses/>.

*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>
#include "numpy/arrayobject.h"
#include "structmember.h"

// Visual Studio 2010 doesn't support C99, so all the code below should follow C89 standard
// it means first we declare ALL variables, then we assign them values and use them

double max_list(double * list, int len)
{
    int i = 0;
    double max_value = 0;
    for(i = 0; i < len; i++)
    {
        if(i == 0 || max_value < fabs(list[i]))
            max_value = fabs(list[i]);
    }
    return max_value;
}

double dot(double * a,double * b,int len)
{
    double dot_val = 0.0;
    int i;
    for(i = 0; i < len; i++)
    {
        dot_val += a[i]*b[i];
    }
    return dot_val;
}

float *** vec_to_cube_float(PyArrayObject *arrayin)
{
    int i,j,k,l = 0;
    int dimz = arrayin->dimensions[0];
    int dimy = arrayin->dimensions[1];
    int dimx = arrayin->dimensions[2];
    float * array = (float *) arrayin->data;
    float *** out = (float ***)malloc(sizeof(float **)*dimz);
    for(i = 0; i < dimz; i++)
    {
        out[i] = (float **)malloc(sizeof(float *)*dimy);
        for(j = 0; j < dimy; j++)
        {
            out[i][j] = (float *)malloc(sizeof(float)*dimx);
            for(k = 0; k < dimx; k++)
            {
                out[i][j][k] = array[l++];
            }
        }
    }
    return out;
}

double *** vec_to_cube_double(PyArrayObject *arrayin)
{
    int i,j,k,l = 0;
    int dimz = arrayin->dimensions[0];
    int dimy = arrayin->dimensions[1];
    int dimx = arrayin->dimensions[2];

    double * array = (double *) arrayin->data;
    double *** out = (double ***)malloc(sizeof(double **)*dimz);
    for(i = 0; i < dimz; i++)
    {

        out[i] = (double **)malloc(sizeof(double *)*dimy);
        for(j = 0; j < dimy; j++)
        {

            out[i][j] = (double *)malloc(sizeof(double)*dimx);
            for(k = 0; k < dimx; k++)
            {
                out[i][j][k] = array[l++];
            }
        }
    }
    return out;
}

double ** vec_to_matrix(PyArrayObject *arrayin)
{
    int i,j,l = 0;
    int dimy = arrayin->dimensions[0];
    int dimx = arrayin->dimensions[1];
    double * array = (double *) arrayin->data;
    double ** out = (double **)malloc(sizeof(double *)*dimy);
    for(i = 0; i < dimy; i++)
    {
        out[i] = (double *)malloc(sizeof(double)*dimx);
        for(j = 0; j < dimx; j++)
        {
            out[i][j] = array[l++];
        }
    }
    return out;
}

float ** vec_to_matrix_float(PyArrayObject *arrayin)
{
    int i,j,l = 0;
    int dimy = arrayin->dimensions[0];
    int dimx = arrayin->dimensions[1];
    float * array = (float *) arrayin->data;
    float ** out = (float **)malloc(sizeof(float *)*dimy);
    for(i = 0; i < dimy; i++)
    {
        out[i] = (float *)malloc(sizeof(float)*dimx);
        for(j = 0; j < dimx; j++)
        {
            out[i][j] = array[l++];
        }
    }
    return out;
}

float get_element(float *** cube,int * dims,int * element)
{
    if(element[0] >= 0 && element[0] < dims[0] && element[1] >= 0 && element[1] < dims[1] && element[2] >= 0 && element[2] < dims[2])
    {
        return cube[element[0]][element[1]][element[2]];
    }
    return -1.0;
}

float calculate_path_length(float *** cube,float *** rho_cube,int * dimensions,int * point,int * step,double * field,double * weight)
{
    float element;
    int point_tmp[3] = {point[0],point[1],point[2]};
    double path = 0.0;

    double element_d;
    int point2[3];
    double point3[3];

    element = get_element(cube,dimensions,point);

    if(element == -1.0)
        return 0.0;

    if(element == 0.0)
    {

        element_d = element;
        point3[0]=point[0];
        point3[1]=point[1];
        point3[2]=point[2];

        while(1)
        {
            path += rho_cube[point[0]][point[1]][point[2]];

            point2[0] = point[0]+step[2];
            point2[1] = point[1];
            point2[2] = point[2];
            element_d = get_element(rho_cube,dimensions,point2);
            if(element_d > 0)
                path += element_d*weight[2];

            point2[0] = point[0];
            point2[1] = point[1]+step[1];
            point2[2] = point[2];
            element_d = get_element(rho_cube,dimensions,point2);
            if(element_d > 0)
                path += (element_d)*weight[1];
            point2[0] = point[0];
            point2[1] = point[1];
            point2[2] = point[2]+step[0];
            element_d = get_element(rho_cube,dimensions,point2);
            if(element_d > 0)
                path += (element_d)*weight[0];

            point3[0] -= field[2];
            point3[1] -= field[1];
            point3[2] -= field[0];

            point[0] = (int)(point3[0]);
            point[1] = (int)(point3[1]);
            point[2] = (int)(point3[2]);
            //~ printf("%d,%d,%d\n",point[0],point[1],point[2]);
            //~ printf("%d,%d,%d\n",dimensions[0],dimensions[1],dimensions[2]);
            if (point[0] < 0 || point[1] < 0 || point[2] < 0 || point[0] >= dimensions[0] || point[1] >= dimensions[1] || point[2] >= dimensions[2])
            {
                //~ printf("test1\n");
                break;
            }
            //~ ;

            if(fabs(point3[0]-point[0]) < 0.10 && fabs(point3[1]-point[1]) < 0.10 && fabs(point3[2]-point[2]) < 0.10)
            {
                //~ printf("test2\n");
                path += calculate_path_length(cube,rho_cube,dimensions,point,step,field,weight);
                break;
            }
        }

        cube[point_tmp[0]][point_tmp[1]][point_tmp[2]] = (float)path;
        //~ printf("%d,%f\n\n",point[1],path);
        return (float)path;
    }
    return element;
}

static PyObject * rhocube_to_water(PyObject *self, PyObject *args)
{
    int i,j,k;
    PyArrayObject *vec_rho,*vec_field,*vec_cube_size,*vec_out;
    float *** rho_cube,***cout;
    double *field,*cube_size;
    int dims[3];
    double field2[3];
    double length, field_max;
    int step[3];
    int point[3];
    double weight[3];
    double w_sum = 0;
    int l = 0;
    float * out;

    if (!PyArg_ParseTuple(args, "OOO",&vec_rho,&vec_field,&vec_cube_size))
        return NULL;
    field = (double *)vec_field->data;
    cube_size = (double *)vec_cube_size->data;

    rho_cube = vec_to_cube_float(vec_rho);
    dims[0] = vec_rho->dimensions[0];
    dims[1] = vec_rho->dimensions[1];
    dims[2] = vec_rho->dimensions[2];

    vec_out = (PyArrayObject *) PyArray_FromDims(3,dims,NPY_FLOAT);
    cout = vec_to_cube_float(vec_out);
    field2[0] = field[0]/cube_size[0];
    field2[1] = field[1]/cube_size[1];
    field2[2] = field[2]/cube_size[2];
    length = 0.5*sqrt(pow(field2[0]*cube_size[0],2)+pow(field2[1]*cube_size[1],2)+pow(field2[2]*cube_size[2],2));
    field_max = max_list(field2,3);
    length /= field_max;
    for (i = 0; i < 3; i++)
    {
        field2[i] /= field_max;
    }
    //Convert density to cube length
    for(i = 0; i < dims[0]; i++)
    {
        for(j= 0; j < dims[1]; j++)
        {
            for(k = 0; k < dims[2]; k++)
            {
                rho_cube[i][j][k] *= (float)length;
            }
        }
    }

    step[0] = (field[0] >= 0)?1:-1;
    step[1] = (field[1] >= 0)?1:-1;
    step[2] = (field[2] >= 0)?1:-1;
    for(i = 0; i < 3; i++)
    {
        weight[i] = pow(field[i],2)/cube_size[i];
        w_sum += weight[i];
    }
    for(i = 0; i < 3; i++)
        weight[i] /= w_sum;
    for(i = 0; i < dims[0]; i++)
    {
        for(j= 0; j < dims[1]; j++)
        {
            for(k = 0; k < dims[2]; k++)
            {
                if(cout[i][j][k] != 0.0)
                    continue;
                point[0] = i;
                point[1] = j;
                point[2] = k;
                cout[i][j][k] = calculate_path_length(cout,rho_cube,dims,point,step,field2,weight);
                //~ printf("%d,%d,%d\n",point[0],point[1],point[2]);
            }
        }
    }
    out = (float *)vec_out->data;
    for(i = 0; i < dims[0]; i++)
    {
        for(j= 0; j < dims[1]; j++)
        {
            for(k = 0; k < dims[2]; k++)
            {
                out[l++] = cout[i][j][k];
            }
            free(cout[i][j]);
            free(rho_cube[i][j]);
        }
        free(cout[i]);
        free(rho_cube[i]);
    }
    free(cout);
    free(rho_cube);

    return PyArray_Return(vec_out);
}

static PyObject * calculate_dist(PyObject *self, PyObject *args)
{
    int i,j,k,l,m;
    PyArrayObject *vec_dist,*vec_cube_size,*vec_center,*vec_basis,*vec_out;
    float * water_cube;
    double *center,*cube_size;
    double ** basis;
    double point[3];
    double dist[3];
    int dims[4];
    float * out;

    if (!PyArg_ParseTuple(args, "OOOO",&vec_dist,&vec_cube_size,&vec_center,&vec_basis))
        return NULL;
    water_cube = (float *)vec_dist->data;
    center = (double *)vec_center->data;
    cube_size = (double *)vec_cube_size->data;
    basis = vec_to_matrix(vec_basis);
    dims[0] = vec_dist->dimensions[0];
    dims[1] = vec_dist->dimensions[1];
    dims[2] = vec_dist->dimensions[2];
    dims[3] = 3;

    vec_out = (PyArrayObject *) PyArray_FromDims(4,dims,NPY_FLOAT);
    out = (float *)vec_out->data;
    l = 0;
    m = 0;
    for(i = 0; i < dims[0]; i++)
    {
        for(j = 0; j < dims[1]; j++)
        {
            for(k = 0; k < dims[2]; k++)
            {
                point[0] = (0.5+k)*cube_size[0];
                point[1] = (0.5+j)*cube_size[1];
                point[2] = (0.5+i)*cube_size[2];

                dist[0] = point[0]-center[0];
                dist[1] = point[1]-center[1];
                dist[2] = point[2]-center[2];

                out[l++] = (float)dot(dist,basis[1],3);
                out[l++] = (float)dot(dist,basis[2],3);
                out[l++] = water_cube[m++];
            }
        }
    }
    return PyArray_Return(vec_out);

}

double **** rastervector_to_array(PyArrayObject * vector)
{
    int i,j,k,l;
    int dims[3];

    double * data;
    double **** out;

    dims[0] = vector->dimensions[0];
    dims[1] = vector->dimensions[1];
    dims[2] = vector->dimensions[2];

    data = (double *)vector->data;
    out = (double ****)malloc(sizeof(double ***)*dims[0]);

    l = 0;
    for (i = 0; i < dims[0]; i++)
    {
        out[i] = (double ***)malloc(sizeof(double **)*dims[1]);
        for(j = 0; j < dims[1]; j++)
        {
            out[i][j] = (double **)malloc(sizeof(double *)*dims[2]);
            for(k = 0; k < dims[2]; k++)
            {
                out[i][j][k] = (double *)malloc(sizeof(double)*3);
                out[i][j][k][0] = data[l++];
                out[i][j][k][1] = data[l++];
                out[i][j][k][2] = data[l++];
            }
        }
    }
    return out;
}

double *** ddd_vector_to_cube(PyArrayObject * vector)
{
    int i,j,k;
    int dims[2];
    double * data;
    double *** out;

    dims[0] = vector->dimensions[0];
    dims[1] = vector->dimensions[1];

    data = (double *)vector->data;
    out = (double ***)malloc(sizeof(double **)*dims[0]);

    k = 0;
    for (i = 0; i < dims[0]; i++)
    {
        out[i] = (double **)malloc(sizeof(double *)*dims[1]);
        for(j = 0; j < dims[1]; j++)
        {
            out[i][j] = (double *)malloc(sizeof(double)*3);
            out[i][j][0] = data[k++];
            out[i][j][1] = data[k++];
            out[i][j][2] = data[k++];
        }
    }
    return out;
}

int lookup_idx_ddd(double ** list,int n,double value)
{
    int bottom = 0;
    int top = n-1;
    int mid = (int)(top+bottom)/2;
    if(list[top][0] < value)
        return -1;
    while(mid != bottom && mid != top)
    {
        if(list[mid][0] > value)
        {
            top = mid;
        }
        else
        {
            bottom = mid;
        }
        mid = (int)(top+bottom)/2;
    }
    return mid;
}

static PyObject * calculate_dose(PyObject *self, PyObject *args)
{
    int i,j,k;
    int dims[1];
    int submachines;
    int ddd_steps;
    int raster_idx[2];
    int ddd_idx;

    double u,t,si1,si2,si3,si4;

    double zero[2];
    double last[2];
    double stepsize[2];

    double tmp_ddd;
    float * point;


    float * dose,*points;
    double ****raster_cube;
    double *** ddd;
    double max_depth;
    PyArrayObject *vec_dist,*vec_dose;
    PyArrayObject *vec_raster,*vec_ddd;
    if (!PyArg_ParseTuple(args, "OOO",&vec_dist,&vec_raster,&vec_ddd))
        return NULL;
    dims[0] = vec_dist->dimensions[0];
    raster_cube = rastervector_to_array(vec_raster);
    ddd = ddd_vector_to_cube(vec_ddd);
    points = (float *)vec_dist->data;
    vec_dose = (PyArrayObject *) PyArray_FromDims(1,dims,NPY_FLOAT);
    dose = (float *)vec_dose->data;
    submachines = (int)vec_raster->dimensions[0];
    ddd_steps = vec_ddd->dimensions[1];
    max_depth = ddd[submachines-1][ddd_steps-1][0];

    zero[0] = raster_cube[0][0][0][0];
    zero[1] = raster_cube[0][0][0][1];

    last[0] = raster_cube[0][vec_raster->dimensions[1]-1][vec_raster->dimensions[2]-1][0];
    last[1] = raster_cube[0][vec_raster->dimensions[1]-1][vec_raster->dimensions[2]-1][1];

    stepsize[0] = raster_cube[0][0][1][0]-raster_cube[0][0][0][0];
    stepsize[1] = raster_cube[0][1][0][1]-raster_cube[0][0][0][1];
    j = 0;
    for(i = 0; i < dims[0]; i++)
    {
        point = &points[3*i];
        dose[i] = 0;
        if(point[0] >= zero[0] && point[0] < last[0] && point[1] >= zero[1] && point[1] < last[1] && point[2] < max_depth)
        {
            raster_idx[0] = (int)((point[0]-zero[0])/stepsize[0]);
            raster_idx[1] = (int)((point[1]-zero[1])/stepsize[1]);
            t = (point[0]-raster_cube[0][raster_idx[1]][raster_idx[0]][0])/(raster_cube[0][raster_idx[1]][raster_idx[0]+1][0]-raster_cube[0][raster_idx[1]][raster_idx[0]][0]);
            u = (point[1]-raster_cube[0][raster_idx[1]][raster_idx[0]][1])/(raster_cube[0][raster_idx[1]+1][raster_idx[0]][1]-raster_cube[0][raster_idx[1]][raster_idx[0]][1]);
            si1 = (1-t)*(1-u);
            si2 = t*(1-u);
            si3 = (1-t)*u;
            si4 = t*u;
            for(j = 0; j < submachines; j++)
            {
                tmp_ddd = 0.0;


                ddd_idx = lookup_idx_ddd(ddd[j],ddd_steps,point[2]);
                if (ddd_idx == -1)
                {
                    tmp_ddd = -1.0;
                    continue;
                }
                tmp_ddd += ((ddd[j][ddd_idx][2]-ddd[j][ddd_idx+1][2])/(ddd[j][ddd_idx][0]-ddd[j][ddd_idx+1][0])*(point[2]-ddd[j][ddd_idx][0])+ddd[j][ddd_idx][2]);

                dose[i] += (float)(si1*raster_cube[j][raster_idx[1]][raster_idx[0]][2]*tmp_ddd);
                dose[i] += (float)(si2*raster_cube[j][raster_idx[1]][raster_idx[0]+1][2]*tmp_ddd);
                dose[i] += (float)(si3*raster_cube[j][raster_idx[1]+1][raster_idx[0]][2]*tmp_ddd);
                dose[i] += (float)(si4*raster_cube[j][raster_idx[1]+1][raster_idx[0]+1][2]*tmp_ddd);
            }
        }
    }
    //Cleanup
    for(i = 0; i < (int)vec_raster->dimensions[0]; i++)
    {
        for(j = 0; j < (int)vec_raster->dimensions[1]; j++)
        {
            for(k = 0; k < (int)vec_raster->dimensions[2]; k++)
            {
                free(raster_cube[i][j][k]);
            }
            free(raster_cube[i][j]);
        }
        free(raster_cube[i]);
    }
    free(raster_cube);
    for(i = 0; i < (int)vec_ddd->dimensions[0]; i++)
    {
        for(j = 0; j < (int)vec_ddd->dimensions[1]; j++)
        {
            free(ddd[i][j]);
        }
        free(ddd[i]);
    }
    free(ddd);
    return PyArray_Return(vec_dose);
}

static PyObject * merge_raster_grid(PyObject *self, PyObject *args)
{
    int i,j;
    PyArrayObject *vec_raster,*vec_out;

    double * raster,*out;
    double dist;
    double factor;
    double a;
    float sigma;
    int dims[2];
    int n;

    if (!PyArg_ParseTuple(args, "Of",&vec_raster,&sigma))
        return NULL;

    n = vec_raster->dimensions[0];
    raster = (double *)vec_raster->data;
    dims[0] = n;
    dims[1] = 3;
    vec_out = (PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
    out = (double *)vec_out->data;
    factor = 1/(2*3.141592*sigma*sigma);
    a = 2*sigma*sigma;

    for (i = 0; i < n; i++)
        out[3*i+2] = 0.0;

    for(i = 0; i < n; i++)
    {
        out[3*i] = raster[3*i];
        out[3*i+1] = raster[3*i+1];
        if(raster[3*i+2] > 0.0)
        {
            for(j = 0; j < n; j++)
            {
                dist = pow(raster[3*i]-raster[3*j],2)+pow(raster[3*i+1]-raster[3*j+1],2);
                out[3*j+2] += raster[3*i+2]*factor*exp(-dist/a);
            }
        }
    }
    return PyArray_Return(vec_out);
}


static PyObject * calculate_wepl(PyObject *self, PyObject *args)
{
    int i,j;
    PyArrayObject *vec_wepl,*vec_start,*vec_basis,*vec_dimensions,*vec_cubesize;
    PyArrayObject *vec_out;
    float *wepl;
    double *start,*step,*a,*b,*cubesize;
    double *out;
    double point[3];
    double * basis;
    int point_id[3];
    int * dimensions;
    int cubedim[3];
    int id;
    int out_dim[2];
    if (!PyArg_ParseTuple(args, "OOOOO",&vec_wepl,&vec_start,&vec_basis,&vec_dimensions,&vec_cubesize))
        return NULL;
    wepl = (float *)vec_wepl->data;
    cubedim[0] = vec_wepl->dimensions[0];
    cubedim[1] = vec_wepl->dimensions[1];
    cubedim[2] = vec_wepl->dimensions[2];
    basis = (double *)vec_basis->data;
    step = &basis[0];
    a = &basis[3];
    b = &basis[6];
    dimensions = (int *)vec_dimensions->data;
    out_dim[0] = dimensions[0];
    out_dim[1] = dimensions[1];
    start = (double *)vec_start->data;
    cubesize = (double *)vec_cubesize->data;
    vec_out = (PyArrayObject *) PyArray_FromDims(2,out_dim,NPY_DOUBLE);
    out = (double *)vec_out->data;

    for(i = 0; i < dimensions[0]; i++)
    {
        for(j = 0; j < dimensions[1]; j++)
        {

            id = i*dimensions[1]+j;
            out[id] = 0.0;
            point[0] = start[0]+a[0]*i+b[0]*j;
            point[1] = start[1]+a[1]*i+b[1]*j;
            point[2] = start[2]+a[2]*i+b[2]*j;

            while(1)
            {
                point_id[0] = (int)(point[0]/cubesize[0]);
                point_id[1] = (int)(point[1]/cubesize[1]);
                point_id[2] = (int)(point[2]/cubesize[2]);
                if(point_id[0] < 0 || point_id[0] >= cubedim[2] || point_id[1] < 0 || point_id[1] >= cubedim[1] || point_id[2] < 0 || point_id[2] >= cubedim[0])
                    break;
                out[id] += wepl[point_id[2]*cubedim[2]*cubedim[1]+point_id[1]*cubedim[2]+point_id[0]];
                point[0] -= step[0];
                point[1] -= step[1];
                point[2] -= step[2];
            }
        }
    }
    return PyArray_Return(vec_out);
}

//static PyObject * raytracing(PyObject *self, PyObject *args) // TODO to implement
//{
//
//}


//First vector outer cube, second inner cube, third is field vector scaled to indices
static PyObject * create_field_shadow(PyObject *self, PyObject *args)
{
    int i,j,k,l;
    PyArrayObject *vec_in1,*vec_in2,*vec_out,*vec_field;
    short * in1,*in2;
    short * out;
    double * field;
    int cubedim[3];
    int a,a_temp,b_temp;
    double point[3];
    int point_idx[3];
    int temp_dose;

    if (!PyArg_ParseTuple(args, "OOO",&vec_in1,&vec_in2,&vec_field))
        return NULL;

    field = (double *)vec_field->data;
    cubedim[0] = vec_in1->dimensions[0];
    cubedim[1] = vec_in1->dimensions[1];
    cubedim[2] = vec_in1->dimensions[2];

    in1 = (short *)vec_in1->data;
    in2 = (short *)vec_in2->data;

    vec_out = (PyArrayObject *) PyArray_FromDims(3,cubedim,NPY_INT16);
    out = (short *)vec_out->data;
    a = cubedim[2]*cubedim[1];

    for (i = 0; i < a*cubedim[0]; i++)
    {
        out[i] = -1;
    }

    for (i = 0; i < cubedim[0]; i++)
    {

        a_temp = i*a;
        for (j = 0; j < cubedim[1]; j++)
        {

            b_temp = a_temp+j*cubedim[1];

            for (k = 0; k < cubedim[2]; k++)
            {
                l = b_temp+k;
                if (in1[l] > 0 || in2[l] > 0)
                {
                    point[0] = k;
                    point[1] = j;
                    point[2] = i;
                    point_idx[0] = (int)point[0];
                    point_idx[1] = (int)point[1];
                    point_idx[2] = (int)point[2];
                    temp_dose = 1000-in2[l];

                    while(1)
                    {
                        l = point_idx[2]*a+point_idx[1]*cubedim[1]+point_idx[0];
                        if (in1[l] > 0 && in1[l] < temp_dose)
                            temp_dose = in1[l];
                        if (in1[l] > 0 && (out[l] == -1 || out[l] >  temp_dose))
                        {
                            out[l] = temp_dose;
                        }
                        point[0] += field[0];
                        point[1] += field[1];
                        point[2] += field[2];
                        point_idx[0] = (int)point[0];
                        point_idx[1] = (int)point[1];
                        point_idx[2] = (int)point[2];

                        if(point_idx[0] < 0 || point_idx[1] < 0 || point_idx[2] < 0 || point_idx[0] >= cubedim[2] || point_idx[1] >= cubedim[1] || point_idx[2] >= cubedim[0])
                        {
                            break;
                        }

                    }
                }
                else if(out[l] == -1)
                {
                    out[l] = in1[l];
                }
            }
        }
    }
    for (i = 0; i < a*cubedim[0]; i++)
    {
        if(out[i] == -1)
            out[i] = 0;
    }
    return PyArray_Return(vec_out);
}

//This code is experimental and does only work for fieldvector [1,0,0]
static PyObject * create_field_ramp(PyObject *self, PyObject *args)
{
    int i,j,k,l,m;
    float extension;
    PyArrayObject *vec_in1,*vec_in2,*vec_out,*vec_field;
    short *in2;
    short * out;
    double * field;
    int tmp;
    int cubedim[3];
    int a,a_temp,b_temp;
    double point[3];
    int point_idx[3];
    int length;
    int length_a,length_b;
    int tmp_length;

    if (!PyArg_ParseTuple(args, "OOO",&vec_in1,&vec_in2,&vec_field))
        return NULL;

    length_a = 0;
    length_b = 0;

    extension = 1.4f;
    field = (double *)vec_field->data;
    cubedim[0] = vec_in1->dimensions[0];
    cubedim[1] = vec_in1->dimensions[1];
    cubedim[2] = vec_in1->dimensions[2];

    in2 = (short *)vec_in2->data;

    vec_out = (PyArrayObject *) PyArray_FromDims(3,cubedim,NPY_INT16);
    out = (short *)vec_out->data;
    a = cubedim[2]*cubedim[1];


    for (i = 0; i < a*cubedim[0]; i++)
    {
        out[i] = -1;
    }

    for (i = 0; i < cubedim[0]; i++)
    {
        a_temp = i*a;
        for (j = 0; j < cubedim[1]; j++)
        {

            b_temp = a_temp+j*cubedim[1];

            for (k = 0; k < cubedim[2]; k++)
            {
                l = b_temp+k;
                if (in2[l] > 0)
                {
                    length = 0;

                    //Calculate forward length of hypoxia
                    tmp_length = 0;
                    point[0] = k;
                    point[1] = j;
                    point[2] = i;
                    point_idx[0] = (int)point[0];
                    point_idx[1] = (int)point[1];
                    point_idx[2] = (int)point[2];

                    while(1)
                    {
                        l = point_idx[2]*a+point_idx[1]*cubedim[1]+point_idx[0];
                        tmp_length++;
                        if (in2[l] > 0)
                        {
                            length_a = tmp_length;
                        }
                        point[0] += field[0];
                        point[1] += field[1];
                        point[2] += field[2];
                        point_idx[0] = (int)point[0];
                        point_idx[1] = (int)point[1];
                        point_idx[2] = (int)point[2];

                        if(point_idx[0] < 0 || point_idx[1] < 0 || point_idx[2] < 0 || point_idx[0] >= cubedim[2] || point_idx[1] >= cubedim[1] || point_idx[2] >= cubedim[0])
                        {
                            break;
                        }
                    }

                    //Calculate backward length of hypoxia
                    tmp_length = 0;
                    point[0] = k;
                    point[1] = j;
                    point[2] = i;
                    point_idx[0] = (int)point[0];
                    point_idx[1] = (int)point[1];
                    point_idx[2] = (int)point[2];

                    while(1)
                    {
                        l = point_idx[2]*a+point_idx[1]*cubedim[1]+point_idx[0];
                        tmp_length++;
                        if (in2[l] > 0)
                        {
                            length_b = tmp_length;
                        }
                        point[0] -= field[0];
                        point[1] -= field[1];
                        point[2] -= field[2];
                        point_idx[0] = (int)point[0];
                        point_idx[1] = (int)point[1];
                        point_idx[2] = (int)point[2];

                        if(point_idx[0] < 0 || point_idx[1] < 0 || point_idx[2] < 0 || point_idx[0] >= cubedim[2] || point_idx[1] >= cubedim[1] || point_idx[2] >= cubedim[0])
                        {
                            break;
                        }
                    }

                    length = length_a+length_b;

                    length = (int)(length*extension);

                    point[0] = k-field[0]*(length_b+length*(extension-1.4)/2);
                    point[1] = j-field[1]*(length_b+length*(extension-1.4)/2);
                    point[2] = i-field[2]*(length_b+length*(extension-1.4)/2);
                    point_idx[0] = (int)point[0];
                    point_idx[1] = (int)point[1];
                    point_idx[2] = (int)point[2];
                    m = 0;

                    while(1)
                    {
                        l = point_idx[2]*a+point_idx[1]*cubedim[1]+point_idx[0];
                        if (m < length)
                        {
                            tmp = (int)((1.0-(double)m/(double)length)*1000.0);
                            if (tmp < out[l] || out[l] == -1)
                            {
                                out[l] = tmp;
                            }
                        }
                        else
                        {
                            out[l] = 0;
                        }
                        m++;
                        point[0] += field[0];
                        point[1] += field[1];
                        point[2] += field[2];
                        point_idx[0] = (int)point[0];
                        point_idx[1] = (int)point[1];
                        point_idx[2] = (int)point[2];

                        if(point_idx[0] < 0 || point_idx[1] < 0 || point_idx[2] < 0 || point_idx[0] >= cubedim[2] || point_idx[1] >= cubedim[1] || point_idx[2] >= cubedim[0])
                        {
                            break;
                        }

                    }

                }

            }

        }
    }
    for (i = 0; i < a*cubedim[0]; i++)
    {
        if(out[i] == -1)
            out[i] = 0;
    }
    return PyArray_Return(vec_out);
}


static PyObject * split_by_plane(PyObject *self, PyObject *args)
{
    PyArrayObject *vec_in,*vec_out,*vec_center,*vec_field;
    short * in;
    double * field;
    double factor;
    double * center;
    int i,j,k,l;
    short * out;
    int cubedim[3];
    double d;
    double f_length;
    double dist;
    int a,a_temp,b_temp;
    if (!PyArg_ParseTuple(args, "OOO",&vec_in,&vec_center,&vec_field))
        return NULL;
    cubedim[0] = vec_in->dimensions[0];
    cubedim[1] = vec_in->dimensions[1];
    cubedim[2] = vec_in->dimensions[2];

    in = (short *)vec_in->data;
    field = (double *)vec_field->data;
    vec_out = (PyArrayObject *) PyArray_FromDims(3,cubedim,NPY_INT16);
    center = (double *)vec_center->data;
    out = (short *)vec_out->data;
    a = cubedim[2]*cubedim[1];
    d = -1*(field[0]*center[0]+field[1]*center[1]+field[2]*center[2]);
    f_length = sqrt(pow(field[0],2)+pow(field[1],2)+pow(field[2],2));
    for (i = 0; i < cubedim[0]; i++)
    {
        a_temp = i*a;
        for (j = 0; j < cubedim[1]; j++)
        {
            b_temp = a_temp+j*cubedim[1];

            for (k = 0; k < cubedim[2]; k++)
            {
                l = b_temp+k;
                if (in[l] > 0)
                {
                    dist = (k*field[0]+j*field[1]+i*field[2]+d)/f_length;
                    factor = dist*0.05+0.50;
                    if(factor > 1.0)
                        factor = 1.0;
                    if (factor > 0)
                    {
                        out[l] = (short)(factor*in[l]);
                    }
                    else
                        out[l] = 0;
                }
                else
                {
                        out[l] = 0;
                }
            }
        }
    }

    return PyArray_Return(vec_out);
}

static PyObject * extend_cube(PyObject *self, PyObject *args)
{
    PyArrayObject *vec_in,*vec_out,*vec_cubesize;
    short * in;
    int i,j,k,l,l2;
    int x,y,z;
    double dist = 1;
    double * cubesize;
    double dist_2;
    double t1,t2;
    short * out;
    int cubedim[3];

    int a,a_temp,b_temp;
    if (!PyArg_ParseTuple(args, "OOd",&vec_in,&vec_cubesize,&dist))
        return NULL;

    cubedim[0] = vec_in->dimensions[0];
    cubedim[1] = vec_in->dimensions[1];
    cubedim[2] = vec_in->dimensions[2];

    cubesize = (double *)vec_cubesize->data;

    in = (short *)vec_in->data;

    vec_out = (PyArrayObject *) PyArray_FromDims(3,cubedim,NPY_INT16);
    dist_2 = pow(dist,2);
    out = (short *)vec_out->data;
    a = cubedim[2]*cubedim[1];

    for (i = 1; i < cubedim[0]; i++)
    {
        a_temp = i*a;
        for (j = 0; j < cubedim[1]; j++)
        {
            b_temp = a_temp+j*cubedim[1];
            for (k = 0; k < cubedim[2]; k++)
            {
                l = b_temp+k;

                if (in[l] > 0)
                {
                    for (x = -(int)dist; x <= dist; x++)
                    {
                        t1 = pow(cubesize[0]*x,2);
                        for (y = -(int)dist; y <= dist; y++)
                        {

                            t2 = t1 + pow(cubesize[1]*y,2);
                            if(t2 > dist_2)
                                continue;
                            for (z = -(int)dist; z <= dist; z++)
                            {
                                l2 = (i+z)*a+(j+y)*cubedim[1]+(k+x);
                                if ((t2+pow(cubesize[2]*z,2)) <= dist_2 && out[l2] < in[l] && l2 > 0)
                                {
                                    out[l2] = in[l];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return PyArray_Return(vec_out);
}

static PyMethodDef unusedMethods[] = {
{"rhocube_to_water",(PyCFunction)rhocube_to_water,METH_VARARGS},
{"calculate_dist",(PyCFunction)calculate_dist,METH_VARARGS},
{"calculate_dose",(PyCFunction)calculate_dose,METH_VARARGS},
{"merge_raster_grid",(PyCFunction)merge_raster_grid,METH_VARARGS},
{"calculate_wepl",(PyCFunction)calculate_wepl,METH_VARARGS},
{"create_field_shadow",(PyCFunction)create_field_shadow,METH_VARARGS},
{"create_field_ramp",(PyCFunction)create_field_ramp,METH_VARARGS},
{"split_by_plane",(PyCFunction)split_by_plane,METH_VARARGS},
{"extend_cube",(PyCFunction)extend_cube,METH_VARARGS},
//{"raytracing",(PyCFunction)raytracing,METH_VARARGS},  // TODO to implement
{NULL,NULL}
};

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "unused",
        "pytrip unused methods - to be tested and refactored",
        -1,
        unusedMethods,
        NULL,
        NULL,
        NULL,
        NULL
};


#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_initunused(void)

#else
#define INITERROR return

void
initunused(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("unused", unusedMethods);
#endif
    import_array();
    if (module == NULL)
        INITERROR;

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
