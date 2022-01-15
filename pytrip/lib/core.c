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

// macro to ease python array manipulation
// PyObject* contour: list of points, where point is a list of doubles
// int pos: index in contour
// int coord: index in point
#define GET_COORDINATE(contour, pos, coord) PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(contour, pos), coord))

// Visual Studio 2010 doesn't support C99, so all the code below should follow C89 standard
// it means first we declare ALL variables, then we assign them values and use them


struct list_el{
    double * point;
    struct list_el * next;
};


typedef struct list_el item;


double _pytriplib_dot(double * a,double *b)
{
    return a[0]*b[0]+a[1]*b[1];
}


double _pytriplib_norm(double * vector)
{
    return sqrt(_pytriplib_dot(vector,vector));
}


static PyObject * points_to_contour(PyObject *self, PyObject *args)
{
    int n_items;
    int j;
    npy_intp i;
    item *head, *element, *prev, *prev2, *element2;
    double * point,*prev_point;
    double a[2],b[2],c[2];
    int n = 0;
    double dot;
    int valid;
    int rm_points;

    // array objects into which input will be unpacked and output packed into
    PyArrayObject *vecin;
    PyArrayObject *vecout;
    npy_intp  dims[2];  // shape of output vector

    if (!PyArg_ParseTuple(args, "O", &vecin))
        return NULL;

    // exit if input vector has less than 3 elements
    if (PyArray_DIM(vecin, 0) < 3)
    {
        return NULL;
    }

    head = (item *)malloc(sizeof(item));
    head->point = (double*)calloc(2, sizeof(double));
    head->point[0] = *((double*)PyArray_GETPTR2(vecin, 0, 0));
    head->point[1] = *((double*)PyArray_GETPTR2(vecin, 0, 1));

    element = (item *)malloc(sizeof(item));
    element->point = (double*)calloc(2, sizeof(double));
    element->point[0] = *((double*)PyArray_GETPTR2(vecin, 1, 0));
    element->point[1] = *((double*)PyArray_GETPTR2(vecin, 1, 1));

    head->next = element;
    n_items = 2;

    prev = NULL;
    prev2 = NULL;
    element2 = NULL;
    point = head->point;
    for(i = 1; i < PyArray_DIM(vecin, 0); i++)
    {
        prev_point = point;

        point = (double*)calloc(2, sizeof(double));
        point[0] = *((double*)PyArray_GETPTR2(vecin, i, 0));
        point[1] = *((double*)PyArray_GETPTR2(vecin, i, 1));

        if(prev_point[0] > point[0])
        {
            if(n != 1)
            {
                element->next = (item *)malloc(sizeof(item));
                element->next->point = prev_point;
                element = element->next;
                n_items++;
            }
            element2 = (item *)malloc(sizeof(item));
            element2->point = (double*)calloc(2, sizeof(double));
            element2->point[0] = *((double*)PyArray_GETPTR2(vecin, i, 0));
            element2->point[1] = *((double*)PyArray_GETPTR2(vecin, i, 1));
            element2->next = head;
            head = element2;
            n_items++;
            n = 0;
        }
        n++;
    }
    element->next = (item *)malloc(sizeof(item));
    element->next->point = point;
    element = element->next;
    n_items++;
    n_items++;
    element->next = head;
    prev = NULL;
    element->next = head;
    rm_points = 0;
    element = head;
    for (i = 0; i < (n_items-rm_points)*2; i++)
    {

        if(prev != NULL)
        {
            a[0] = element->point[0]-prev->point[0];
            a[1] = element->point[1]-prev->point[1];

            b[0] = element->next->point[0]-element->point[0];
            b[1] = element->next->point[1]-element->point[1];

            dot = _pytriplib_dot(a,b)/_pytriplib_norm(a)/_pytriplib_norm(b);
            if (dot > 0.98)
            {
                //~ printf("%f,%f\n")
                prev->next = element->next;
                element = element->next;
                rm_points++;
                continue;
            }
        }
        prev = element;
        element = element->next;
    }
    element2  = element;
    for (i = 0; i < n_items-rm_points-1; i++)
    {
        element = element->next;
        if (element->point[1] > element2->point[1])
            element2 = element;
    }
    element = element2;
    //~ printf("%f,%f\n",element->point[0],element->point[1]);
    for(j = 0; j < 3; j++)
    {

    for (i = 0; i < n_items-rm_points-1; i++)
    //~ for (i = 0; i < 0; i++)
    {

        if(prev2 != NULL)
        {
            a[0] = element->point[0]-prev->point[0];
            a[1] = element->point[1]-prev->point[1];

            b[0] = element->next->point[0]-element->point[0];
            b[1] = element->next->point[1]-element->point[1];

            c[0] = prev->point[0]-prev2->point[0];
            c[1] = prev->point[1]-prev2->point[1];
            dot = _pytriplib_dot(a,b)/_pytriplib_norm(a)/_pytriplib_norm(b);
            if(c[1]*a[1] <= 0 && fabs(c[1]) > 0.30)
                valid = 0;
            else
                valid = 1;
            if (dot > 0.99 || (dot < -0.7 && valid == 1))
            {
                //~ printf("%f,%f\n")
                prev->next = element->next;
                element = element->next;
                rm_points++;
                continue;
            }
        }
        prev2 = prev;
        prev = element;
        element = element->next;
    }
}
    dims[0] = n_items-rm_points;
    dims[1] = 2;
    vecout = (PyArrayObject *) PyArray_ZEROS(2,dims,NPY_DOUBLE, NPY_ANYORDER);

    element = head;
    for (i = 0; i < dims[0]; i++)
    {
        *((double*)PyArray_GETPTR2(vecout, i, 0)) = element->point[0];
        *((double*)PyArray_GETPTR2(vecout, i, 1)) = element->point[1];
        element = element->next;
    }
    return PyArray_Return(vecout);


}


static PyObject * filter_points(PyObject *self, PyObject *args)
{
    int i = 0;
    int k = 0;
    double dist;
    double d, dist_2;
    int to_close;

    double current_point_x = 0.0;
    double current_point_y = 0.0;

    double out_item_x = 0.0;
    double out_item_y = 0.0;

    // array objects into which input will be unpacked and output packed into
    PyArrayObject *vecin;
    PyObject *list_out;
    PyObject *list_item;

    if (!PyArg_ParseTuple(args, "Od", &vecin,&dist))
        return NULL;

    dist_2 = pow(dist,2);

    list_out = PyList_New(0);
    list_item = PyList_New(2);
    PyList_SetItem(list_item, 0, PyFloat_FromDouble(*((double*)PyArray_GETPTR2(vecin, 0, 0))));
    PyList_SetItem(list_item, 1, PyFloat_FromDouble(*((double*)PyArray_GETPTR2(vecin, 0, 1))));
    PyList_Append(list_out, list_item);

    for(i = 0; i <  PyArray_DIM(vecin, 0); i++)
    {
        to_close = 0;
        current_point_x = *((double*)PyArray_GETPTR2(vecin, i, 0));
        current_point_y = *((double*)PyArray_GETPTR2(vecin, i, 1));

        for(k = 0; k < PyList_Size(list_out); k++)
        {
            out_item_x = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(list_out, k), 0));
            out_item_y = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(list_out, k), 1));

            d = pow(current_point_x-out_item_x,2) + pow(current_point_y-out_item_y,2);
            if (d < dist_2)
            {
                to_close = 1;
            }
        }
        if (to_close == 0){
            list_item = PyList_New(2);
            PyList_SetItem(list_item, 0, PyFloat_FromDouble(current_point_x));
            PyList_SetItem(list_item, 1, PyFloat_FromDouble(current_point_y));
            PyList_Append(list_out, list_item);
        }

    }

    return list_out;
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


int point_in_contour(double * point,double * contour,int n_contour)
{
    int a,b;
    int m = 0;
    int count = 0;
    for(m = 0; m < n_contour; m++)
    {
        a = m;
        if(m == n_contour -1)
            b = 0;
        else
            b = m+1;
        if( (contour[3*a+1] <= point[1] && contour[3*b+1] > point[1]) || (contour[3*a+1] > point[1] && contour[3*b+1] <= point[1]))
        {
            if(contour[3*a]-point[0]+(contour[3*b]-contour[3*a])/(contour[3*b+1]-contour[3*a+1])*(point[1]-contour[3*a+1]) >= 0)
                count++;
        }
    }
    return count%2;
}


static PyObject * calculate_dvh_slice(PyObject *self, PyObject *args)
{
    // temporary variables
    npy_intp i,j,m;
    int n;
    double point[2];
    int resolution = 5;
    double tiny_area = 1.0/pow(resolution,2);
    double point_a[2];
    int edge = 0;
    int inside = 0;
    npy_intp out_dim[] = {1500};
    int upper_limit = 1500;
    int p1[2],p2[2];
    double min_x = 0;
    double max_x = 0;
    double min_y = 0;
    double max_y = 0;

    // array objects into which input will be unpacked
    PyArrayObject *vec_dose,*vec_contour,*vec_size;
    PyArrayObject *vec_out;

    // helper variables to read and operate on input data
    npy_int16 slice_element = 0;
    npy_intp n_contour;
    double * contour;
    double contour_element_x = 0.0;
    double contour_element_y = 0.0;
    double voxel_size_x = 0.0;
    double voxel_size_y = 0.0;

    // function expects as input:
    //  vec_dose - 2-D table of int16, shape (X,Y)
    //  vec_contour - 2-D table of float64, shape (Z, 3)
    //  vec_size - 1-D table of float64, shape (3,)
    if (!PyArg_ParseTuple(args, "OOO",&vec_dose,&vec_contour,&vec_size))
        return NULL;

    // allocate memory for output array
    // function returns as output 1-D table of doubles, size out_dim[0] (1500)
    vec_out = (PyArrayObject *) PyArray_ZEROS(1,out_dim,NPY_DOUBLE, NPY_ANYORDER);

    // get number of contour points from length of vec_contour
    n_contour = PyArray_DIM(vec_contour, 0);

    // cast vec_contour structure to a plain C array of doubles, needed by point_in_contour method
    contour = (double*)PyArray_DATA(vec_contour);

    // read voxel size from input data structure vec_size
    voxel_size_x = *((double*)PyArray_GETPTR1(vec_size, 0));
    voxel_size_y = *((double*)PyArray_GETPTR1(vec_size, 1));

    // calculate box envelope of the contour
    min_x = *((double*)PyArray_GETPTR2(vec_contour, 0, 0));
    max_x = *((double*)PyArray_GETPTR2(vec_contour, 0, 0));
    min_y = *((double*)PyArray_GETPTR2(vec_contour, 1, 0));
    max_y = *((double*)PyArray_GETPTR2(vec_contour, 1, 0));
    for(i = 1; i < n_contour; i++)
    {
        contour_element_x = *((double*)PyArray_GETPTR2(vec_contour, i, 0));
        contour_element_y = *((double*)PyArray_GETPTR2(vec_contour, i, 1));

        if(min_x > contour_element_x)
            min_x = contour_element_x;
        else if(max_x < contour_element_x)
            max_x = contour_element_x;
        if(min_y > contour_element_y)
            min_y = contour_element_y;
        else if(max_y < contour_element_y)
            max_y = contour_element_y;
    }
    min_x -= voxel_size_x;
    max_x += voxel_size_x;
    min_y -= voxel_size_y;
    max_y += voxel_size_y;

    // loop over all elements of cube slice
    n = 0;
    for(i = 0; i < PyArray_DIM(vec_dose, 0); i++)
    {
        // move to next iteration if a point is outside contour box envelope (y coordinate)
        if((0.5+i)*voxel_size_y < min_y || (0.5+i)*voxel_size_y > max_y)
            continue;
        for(j = 0; j < PyArray_DIM(vec_dose, 1); j++)
        {
            point[0] = (0.5+j)*voxel_size_x;
            point[1] = (0.5+i)*voxel_size_y;

            // move to next iteration if a point is outside contour box envelope (x coordinate)
            if(point[0] < min_x || point[0] > max_x)
                continue;
            inside = 0;

            // point entirely inside a contour
            if(point_in_contour(point,contour,(int)n_contour) == 1)
            {
                inside = 1;
                slice_element = *((npy_int16*)PyArray_GETPTR2(vec_dose, i, j));
                if(slice_element < upper_limit)
                    *((double*)PyArray_GETPTR1(vec_out, (npy_intp)slice_element)) += 1;
            }
            edge = 0;
            for(m = 0; m < n_contour; m++)
            {

                contour_element_x = *((double*)PyArray_GETPTR2(vec_contour, m%n_contour, 0));
                p1[0] = (int)(contour_element_x/voxel_size_x);

                contour_element_y = *((double*)PyArray_GETPTR2(vec_contour, m%n_contour, 1));
                p1[1] = (int)(contour_element_y/voxel_size_y);

                contour_element_x = *((double*)PyArray_GETPTR2(vec_contour, (m+1)%n_contour, 0));
                p2[0] = (int)(contour_element_x/voxel_size_x);

                contour_element_y = *((double*)PyArray_GETPTR2(vec_contour, (m+1)%n_contour, 1));
                p2[1] = (int)(contour_element_y/voxel_size_y);

                if(p1[0] == j && p1[1] == i)
                {
                    edge = 1;
                    break;
                }
                if( ((p1[0] <= j && p2[0] >= j) || (p1[0] >= j && p2[0] <= j)) && ((p1[1] <= i && p2[1] >= i) || (p1[1] >= i && p2[1] <= i)))
                {
                    edge = 1;
                    break;
                }
            }
            if(edge)
            {
                point[0] = j*voxel_size_x;
                point[1] = i*voxel_size_y;

                for(m = 0; m < resolution; m++)
                {
                    for(n = 0; n < resolution; n++)
                    {
                        point_a[0] = point[0]+(m+0.5)*voxel_size_x/resolution;
                        point_a[1] = point[1]+(n+0.5)*voxel_size_y/resolution;
                        slice_element = *((npy_int16*)PyArray_GETPTR2(vec_dose, i, j));
                        if(point_in_contour(point_a,contour, (int)n_contour))
                        {
                            if(!inside)
                            {
                                *((double*)PyArray_GETPTR1(vec_out, (npy_intp)slice_element)) += tiny_area;
                            }
                        }
                        else
                        {
                            if(inside)
                            {
                                *((double*)PyArray_GETPTR1(vec_out, (npy_intp)slice_element)) -= tiny_area;
                            }
                        }
                    }
                }
            }
        }
    }
    return PyArray_Return(vec_out);
}


static PyObject * calculate_lvh_slice(PyObject *self, PyObject *args)
{
    // temporary variables
    npy_intp i,j,m;
    int n;
    double point[2];
    int resolution = 5;
    double tiny_area = 1.0/pow(resolution,2);
    double point_a[2];
    int edge = 0;
    int inside = 0;
    npy_intp out_dim[] = {3000};
    int upper_limit = 3000;
    int p1[2],p2[2];
    double min_x = 0;
    double max_x = 0;
    double min_y = 0;
    double max_y = 0;

    // array objects into which input will be unpacked
    PyArrayObject *vec_let,*vec_contour,*vec_size;
    PyArrayObject *vec_out;

    // helper variables to read and operate on input data
    double slice_element = 0;
    npy_intp n_contour;
    double * contour;
    double contour_element_x = 0.0;
    double contour_element_y = 0.0;
    double voxel_size_x = 0.0;
    double voxel_size_y = 0.0;

    // function expects as input:
    //  vec_let - 2-D table of double, shape (X,Y)
    //  vec_contour - 2-D table of float64, shape (Z, 3)
    //  vec_size - 1-D table of float64, shape (3,)
    if (!PyArg_ParseTuple(args, "OOO",&vec_let,&vec_contour,&vec_size))
        return NULL;

    // allocate memory for output array
    // function returns as output 1-D table of doubles, size out_dim[0] (1500)
    vec_out = (PyArrayObject *) PyArray_ZEROS(1,out_dim,NPY_DOUBLE, NPY_ANYORDER);

    // get number of contour points from length of vec_contour
    n_contour = PyArray_DIM(vec_contour, 0);

    // cast vec_contour structure to a plain C array of doubles, needed by point_in_contour method
    contour = (double*)PyArray_DATA(vec_contour);

    // read voxel size from input data structure vec_size
    voxel_size_x = *((double*)PyArray_GETPTR1(vec_size, 0));
    voxel_size_y = *((double*)PyArray_GETPTR1(vec_size, 1));

    // calculate box envelope of the contour
    min_x = *((double*)PyArray_GETPTR2(vec_contour, 0, 0));
    max_x = *((double*)PyArray_GETPTR2(vec_contour, 0, 0));
    min_y = *((double*)PyArray_GETPTR2(vec_contour, 1, 0));
    max_y = *((double*)PyArray_GETPTR2(vec_contour, 1, 0));
    for(i = 1; i < n_contour; i++)
    {
        contour_element_x = *((double*)PyArray_GETPTR2(vec_contour, i, 0));
        contour_element_y = *((double*)PyArray_GETPTR2(vec_contour, i, 1));

        if(min_x > contour_element_x)
            min_x = contour_element_x;
        else if(max_x < contour_element_x)
            max_x = contour_element_x;
        if(min_y > contour_element_y)
            min_y = contour_element_y;
        else if(max_y < contour_element_y)
            max_y = contour_element_y;
    }
    min_x -= voxel_size_x;
    max_x += voxel_size_x;
    min_y -= voxel_size_y;
    max_y += voxel_size_y;

    // loop over all elements of cube slice
    n = 0;
    for(i = 0; i < PyArray_DIM(vec_let, 0); i++)
    {
        // move to next iteration if a point is outside contour box envelope (y coordinate)
        if((0.5+i)*voxel_size_y < min_y || (0.5+i)*voxel_size_y > max_y)
            continue;
        for(j = 0; j < PyArray_DIM(vec_let, 1); j++)
        {
            point[0] = (0.5+j)*voxel_size_x;
            point[1] = (0.5+i)*voxel_size_y;

            // move to next iteration if a point is outside contour box envelope (x coordinate)
            if(point[0] < min_x || point[0] > max_x)
                continue;
            inside = 0;

            // point entirely inside a contour
            if(point_in_contour(point,contour,(int)n_contour) == 1)
            {
                inside = 1;
                slice_element = *((double*)PyArray_GETPTR2(vec_let, i, j));
                if(slice_element < upper_limit)
                    *((double*)PyArray_GETPTR1(vec_out, (int)slice_element)) += 1;
            }
            edge = 0;
            for(m = 0; m < n_contour; m++)
            {

                contour_element_x = *((double*)PyArray_GETPTR2(vec_contour, m%n_contour, 0));
                p1[0] = (int)(contour_element_x/voxel_size_x);

                contour_element_y = *((double*)PyArray_GETPTR2(vec_contour, m%n_contour, 1));
                p1[1] = (int)(contour_element_y/voxel_size_y);

                contour_element_x = *((double*)PyArray_GETPTR2(vec_contour, (m+1)%n_contour, 0));
                p2[0] = (int)(contour_element_x/voxel_size_x);

                contour_element_y = *((double*)PyArray_GETPTR2(vec_contour, (m+1)%n_contour, 1));
                p2[1] = (int)(contour_element_y/voxel_size_y);

                if(p1[0] == j && p1[1] == i)
                {
                    edge = 1;
                    break;
                }
                if( ((p1[0] <= j && p2[0] >= j) || (p1[0] >= j && p2[0] <= j)) && ((p1[1] <= i && p2[1] >= i) || (p1[1] >= i && p2[1] <= i)))
                {
                    edge = 1;
                    break;
                }
            }
            if(edge)
            {
                point[0] = j*voxel_size_x;
                point[1] = i*voxel_size_y;

                for(m = 0; m < resolution; m++)
                {
                    for(n = 0; n < resolution; n++)
                    {
                        point_a[0] = point[0]+(m+0.5)*voxel_size_x/resolution;
                        point_a[1] = point[1]+(n+0.5)*voxel_size_y/resolution;
                        slice_element = *((double*)PyArray_GETPTR2(vec_let, i, j));
                        if(point_in_contour(point_a,contour, (int)n_contour))
                        {
                            if(!inside)
                            {
                                *((double*)PyArray_GETPTR1(vec_out, (int)slice_element)) += tiny_area;
                            }

                        }
                        else
                        {
                            if(inside)
                            {
                                *((double*)PyArray_GETPTR1(vec_out, (int)slice_element)) -= tiny_area;
                            }
                        }
                    }
                }
            }
        }
    }
    return PyArray_Return(vec_out);
}

static void append_point_to_list(PyObject *list_out, double x, double y, double z){
    PyObject *list_item; // temporary variable to store point in 3D space (represented as list of 3 floats)

    list_item = PyList_New(3);

    PyList_SetItem(list_item, 0, PyFloat_FromDouble(x));
    PyList_SetItem(list_item, 1, PyFloat_FromDouble(y));
    PyList_SetItem(list_item, 2, PyFloat_FromDouble(z));

    PyList_Append(list_out, list_item);
}


/*******************************************************************************
 * calc_intersect_sagittal checks if segment is intersected 
 *  and if it is, calculates point of intersection and appends it to passed list.
 * Input:
 *  PyObject *vec_slice - chain of points,
 *  PyObject *list_out - output list,
 *  double depth - X coordinate for which intersection is calculated,
 *  int i_0 - index in chain of points of first point of segment,
 *  int i_1 - index in chain of points of second point of segment.
 * 
 * Does not return anything, modifies passed list.
 ******************************************************************************/
static void calc_intersect_sagittal(PyObject *vec_slice, PyObject *list_out, double depth, int i_0, int i_1){
    // sagittal, projection onto YZ
    double x_0, x_1, y_0, y_1, z;
    double slope;
    
    // take X-coordinate of two subsequent points from input segment
    // these two points form a line segment
    x_0 = GET_COORDINATE(vec_slice, i_0, 0);
    x_1 = GET_COORDINATE(vec_slice, i_1, 0);
    // check if current line segment has intersection with given plane
    // for YZ intersection, the plane is defined by equation `x == depth`,
    // hence we check if requested depth is between those two coordinate values
    if((x_0 >= depth && x_1 < depth) || (x_1 >= depth && x_0 < depth))
    {
        y_0 = GET_COORDINATE(vec_slice, i_0, 1);
        y_1 = GET_COORDINATE(vec_slice, i_1, 1);
        z = GET_COORDINATE(vec_slice, i_0, 2);
        slope = (y_1 - y_0) / (x_1 - x_0);

        append_point_to_list(list_out, depth, ((depth-x_0)*slope+y_0), z);
    }   
}

/*******************************************************************************
 * calc_intersect_sagittal checks if segment is intersected 
 *  and if it is, calculates point of intersection and appends it to passed list.
 * Input:
 *  PyObject *vec_slice - chain of points,
 *  PyObject *list_out - output list,
 *  double depth - Y coordinate for which intersection is calculated,
 *  int i_0 - index in chain of points of first point of segment,
 *  int i_1 - index in chain of points of second point of segment.
 * 
 * Does not return anything, modifies passed list.
 ******************************************************************************/
static void calc_intersect_coronal(PyObject *vec_slice, PyObject *list_out, double depth, int i_0, int i_1){
    // coronal, projection onto XZ
    double x_0, x_1, y_0, y_1, z;
    double slope;
    
    y_0 = GET_COORDINATE(vec_slice, i_0, 1);
    y_1 = GET_COORDINATE(vec_slice, i_1, 1);
    if((y_0 >= depth && y_1 < depth) || (y_1 >= depth && y_0 < depth))
    {
        x_0 = GET_COORDINATE(vec_slice, i_0, 0);
        x_1 = GET_COORDINATE(vec_slice, i_1, 0);
        z = GET_COORDINATE(vec_slice, i_0, 2);
        slope = (x_1 - x_0) / (y_1 - y_0);

        append_point_to_list(list_out, ((depth-y_0)*slope+x_0), depth, z);
    }   
}
/*******************************************************************************
 * slice_on_place calculates intersection of given chain of points in 3D space along
 * given plane.
 * Intersection planes are limited to YZ (sagittal, represented as int 2)
 * and XZ (coronal, represented as int 1)
 * Input: open chain of points in 3D space for which intersection is requested.
 *   usually it is contained withing transversal plane (meaning the Z coordinate is fixed for all points)
 *
 * Output: list of points in 3D space contained within intersection plane.
 *   if input chain is contained in transversal plane (as for most applications), then the output
 *   will have two coordinates being fixed: Z (because input lies on transversal plane) and
 *   either X (if sagittal intersection is requested) or Y (if coronal intersection is requested).
 *   That means the list of output points lies on a straight line in 3D space
 *
 * TODO add support for input chains holding only one point !
 * currently empty list if being returned if input chain holds a single point
 ******************************************************************************/
static PyObject * slice_on_plane(PyObject *self, PyObject *args)
{
    int i;
    int plane;
    double depth;
    int length;

    // array objects into which input will be unpacked and output packed into
    PyObject *vec_slice; // input variable - list of points in 3D space
    PyObject *list_out;  // return variable - list of points in 3D space, added as list_item objects

    // digest arguments, we expect:
    //    an object - vec_slice: input chain of points
    //    plane - integer : intersection plane type (2 - sagittal YZ, 1 - coronal XZ)
    //    depth - double : intersection plane location in mm
    if (!PyArg_ParseTuple(args, "Oid",&vec_slice,&plane,&depth))
        return NULL;

    // allocate empty list for output variable
    // it will store list of points in 3D space, represented as 3-elements lists
    list_out = PyList_New(0);

    // loop over all line segments in the input chain of points
    // if input chain has only one element this loop won't be executed
    length = PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(vec_slice)));
    // sagittal, projection onto YZ
    if(plane == 2){
        for(i = 0; i < length; i++){
            if(i != length-1){
                calc_intersect_sagittal(vec_slice, list_out, depth, i, i+1);
            }
            else{
                // if this is the last point, check it with the first one, 
                //  we need to do that virtually, because points form an open chain, not closed one
                calc_intersect_sagittal(vec_slice, list_out, depth, length-1, 0);
            }
        }
    }
    // coronal, projection onto XZ
    else if(plane == 1){
        for(i = 0; i < length; i++){
            if(i != length-1){
                calc_intersect_coronal(vec_slice, list_out, depth, i, i+1);
            }
            else{
                calc_intersect_coronal(vec_slice, list_out, depth, length-1, 0);
            }
        }
    }

    // return empty list if no intersection was found, otherwise
    // return chain of points with X
    return list_out;
}

/*******************************************************************************
 * function_ranges calculates boundaries of ranges, where X (or Y) coordinate changes along indices are monotonic.
 * binary_search_intersection method needs those to work properly.
 * 
 * Input: 
 * open chain of points in 3D space,
 *  usually it is contained within transversal plane (meaning the Z coordinate is fixed for all points).
 * plane, to determine checking by X or Y,
 * 
 * Output: 
 * list of indices' numbers, by which whole chain can be divided into monotonically changing ranges.
 * 
 * i.e. output [0, 100, 200, 300] for sagittal plane means that
 *  points in chain from 0 to 100 have monotically increasing Xs 
 *  points in chain from 100 to 200 have monotically decreasing Xs
 *  and so on. 
 ******************************************************************************/
static PyObject* function_ranges(PyObject *self, PyObject *args){
    int i;
    int plane;
    int last_direction;
    int current_direction;
    int length;
    int coord;

    // array objects into which input will be unpacked and output packed into
    PyObject *vec_slice; // input variable - list of points in 3D space
    PyObject *list_out;  // return variable - list of points in 3D space, added as list_item objects

    // helper variables to read and operate on input data
    double a = 0.0;
    double b = 0.0;

    // digest arguments, we expect:
    //    an object - vec_slice: input chain of points
    //    plane - integer : checking plane type (2 - sagittal YZ, 1 - coronal XZ)
    if (!PyArg_ParseTuple(args, "Oi",&vec_slice,&plane)) return NULL;

    length = PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(vec_slice)));
    // at least two to make initial direction check
    if (length < 2) return NULL;

    // create empty list
    list_out = PyList_New(0);
    // add first index
    PyList_Append(list_out, PyLong_FromLong(0));
    // set coord based on plane
    if(plane == 2){
        coord = 0;
    }
    else if(plane == 1){
        coord = 1;
    }

    // make initial direction check based on plane  
    a = GET_COORDINATE(vec_slice, 0, coord);
    b = GET_COORDINATE(vec_slice, 1, coord);
    last_direction = (a < b);
    // check others
    for(i = 1; i < length - 1; i++){
        a = GET_COORDINATE(vec_slice, i, coord);
        b = GET_COORDINATE(vec_slice, i+1, coord);
        current_direction = (a < b);
        // if direction changes, add current index, replace last_direction
        if(last_direction != current_direction){
            PyList_Append(list_out, PyLong_FromLong(i));
            last_direction = current_direction;
        }
    }

    // add last index 
    PyList_Append(list_out, PyLong_FromLong(length-1));

    return list_out;

}

static void linear_search(PyObject *vec_slice, PyObject *list_out, int start, int end, int plane, double depth){
    int i;
    // sagittal, projection onto YZ
    if(plane == 2){
        for(i = start; i < end; i++){
            calc_intersect_sagittal(vec_slice, list_out, depth, i, i+1);
        }
    }
    // coronal, projection onto XZ
    else if(plane == 1){
        for(i = start; i < end; i++){
            calc_intersect_coronal(vec_slice, list_out, depth, i, i+1);
        }
    }
}

static int binary_search_inner(PyObject *vec_slice, int l, int r, int current_direction, double depth, int coord){
    double a, b;
    int m;

    while (l <= r){
        m = (l+r)/2;
        a = GET_COORDINATE(vec_slice, m, coord);
        b = GET_COORDINATE(vec_slice, m+1, coord);
        if (current_direction){
            if(b >= depth && a < depth){
                break;
            }
            if (depth > a){
                l = m+1;
            }else{
                r = m-1;
            }
        }else{
            if(a >= depth && b < depth){
                break;
            }
            if (depth < a){
                l = m+1;
            }else{
                r = m-1;
            }
        }
    }

    return m;
}

static void binary_search(PyObject *vec_slice, PyObject *list_out, int l, int r, int plane, double depth){
    double x_0, x_1, y_0, y_1, z;
    int m;
    double slope;

    if(plane == 2){ // sagittal
        x_0 = GET_COORDINATE(vec_slice, l, 0);
        x_1 = GET_COORDINATE(vec_slice, r, 0);
        if((x_0 >= depth && x_1 < depth) || (x_1 >= depth && x_0 < depth)){
            m = binary_search_inner(vec_slice, l, r, (x_0 < x_1), depth, 0);

            x_0 = GET_COORDINATE(vec_slice, m, 0);
            x_1 = GET_COORDINATE(vec_slice, m+1, 0);
            y_0 = GET_COORDINATE(vec_slice, m, 1);
            y_1 = GET_COORDINATE(vec_slice, m+1, 1);
            z = GET_COORDINATE(vec_slice, m, 2);
            slope = (y_1 - y_0) / (x_1 - x_0);

            append_point_to_list(list_out, depth, ((depth-x_0)*slope+y_0), z);
        }
    }
    else if(plane == 1){ // coronal
        y_0 = GET_COORDINATE(vec_slice, l, 1);
        y_1 = GET_COORDINATE(vec_slice, r, 1);
        if((y_0 >= depth && y_1 < depth) || (y_1 >= depth && y_0 < depth)){
            m = binary_search_inner(vec_slice, l, r, (y_0 < y_1), depth, 1);

            x_0 = GET_COORDINATE(vec_slice, m, 0);
            x_1 = GET_COORDINATE(vec_slice, m+1, 0);
            y_0 = GET_COORDINATE(vec_slice, m, 1);
            y_1 = GET_COORDINATE(vec_slice, m+1, 1);
            z = GET_COORDINATE(vec_slice, m, 2);
            slope = (x_1 - x_0) / (y_1 - y_0);

            append_point_to_list(list_out, ((depth-y_0)*slope+x_0), depth, z);
        }
    }
}

static PyObject* binary_search_intersection(PyObject *self, PyObject *args){
    int i;
    int plane;
    double depth;
    int l, r;
    int length;
    int binary_search_threshold = 50;

    // array objects into which input will be unpacked and output packed into
    PyObject *vec_slice; // input variable - list of points in 3D space
    PyObject *ranges; // input variable - list of function ranges
    PyObject *list_out;  // return variable - list of points in 3D space, added as list_item objects

    // digest arguments, we expect:
    //    an object - vec_slice: input chain of points
    //    an object - ranges: input chain, each pair tells function ranges
    //    plane - integer : intersection plane type (2 - sagittal YZ, 1 - coronal XZ)
    //    depth - double : intersection plane location in mm
    if (!PyArg_ParseTuple(args, "OOid",&vec_slice,&ranges,&plane,&depth)) return NULL;

    list_out = PyList_New(0);

    length = PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(ranges)));
    for(i = 0; i < length-1; i++){
        l = PyLong_AsLong(PyList_GetItem(ranges, i));
        r = PyLong_AsLong(PyList_GetItem(ranges, i+1));
        if(r-l < binary_search_threshold){
            linear_search(vec_slice, list_out, l, r, plane, depth);
        }
        else{
            binary_search(vec_slice, list_out, l, r, plane, depth);
        }
    }

    // check connection between last and first point
    length = PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(vec_slice)));
    if(plane == 2) // sagittal, projection onto YZ
    {
        calc_intersect_sagittal(vec_slice, list_out, depth, length-1, 0);
    }
    else if(plane == 1) // coronal, projection onto XZ
    {
        calc_intersect_coronal(vec_slice, list_out, depth, length-1, 0);
    }

    return list_out;

}

static PyObject * calculate_dose_center(PyObject *dummy, PyObject *args) {

    // input vector
    PyObject *arg1 = NULL;

    // output vector, 1-dimensional, 3 elements
    // it should hold coordinates of center-of-the-mass of the dose cube
    PyArrayObject *vec_out;
    npy_intp out_dims[1];

    // 3-D array iterators
    npy_intp i, j, k;

    // total dose
    double tot_dose;

    // element of dose cube
    npy_int16 cube_element;

    // parse input argument to a PyObject
    if (!PyArg_ParseTuple(args, "O", &arg1)) return NULL;

    // TODO add check if input is a 3-D table holding int32 integers

    // allocate output vector with zeros
    out_dims[0] = 3;  // set the vector to be 3 elements long
    vec_out = (PyArrayObject *) PyArray_ZEROS(1, out_dims, NPY_DOUBLE, NPY_ANYORDER);

    // loop over all elements of input cube and calculate center-of-mass location
    cube_element = 0;
    tot_dose = 0.0;
    for (i = 0; i < PyArray_DIM(arg1, 0); i++)
    {
        for (j = 0; j < PyArray_DIM(arg1, 1); j++)
        {
            for (k = 0; k < PyArray_DIM(arg1, 2); k++)
            {
                cube_element = *((npy_int16*)PyArray_GETPTR3(arg1, i, j, k));
                if( cube_element > 0){
                    tot_dose += cube_element;
                    *((double*)PyArray_GETPTR1(vec_out, 0)) += (double)cube_element * i;
                    *((double*)PyArray_GETPTR1(vec_out, 1)) += (double)cube_element * j;
                    *((double*)PyArray_GETPTR1(vec_out, 2)) += (double)cube_element * k;
                }
            }
        }
    }

    *((double*)PyArray_GETPTR1(vec_out, 0)) /= tot_dose;
    *((double*)PyArray_GETPTR1(vec_out, 1)) /= tot_dose;
    *((double*)PyArray_GETPTR1(vec_out, 2)) /= tot_dose;

    return PyArray_Return(vec_out);
}


static PyMethodDef pytriplibMethods[] = {
{"filter_points",(PyCFunction)filter_points,METH_VARARGS},
{"points_to_contour",(PyCFunction)points_to_contour,METH_VARARGS},
{"calculate_dvh_slice",(PyCFunction)calculate_dvh_slice,METH_VARARGS},
{"calculate_lvh_slice",(PyCFunction)calculate_lvh_slice,METH_VARARGS},
{"slice_on_plane",(PyCFunction)slice_on_plane,METH_VARARGS},
{"calculate_dose_center",(PyCFunction)calculate_dose_center,METH_VARARGS},
{"function_ranges",(PyCFunction)function_ranges,METH_VARARGS},
{"binary_search_intersection",(PyCFunction)binary_search_intersection,METH_VARARGS},
{NULL,NULL}
};

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "pytriplib",
        "pytriplib docstring (TODO)",
        -1,
        pytriplibMethods,
        NULL,
        NULL,
        NULL,
        NULL
};


#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_pytriplib(void)

#else
#define INITERROR return

void
initpytriplib(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("pytriplib", pytriplibMethods);
#endif
    import_array();
    if (module == NULL)
        INITERROR;

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
