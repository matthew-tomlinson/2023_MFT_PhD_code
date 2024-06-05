import numpy as np
import scipy
import scipy.stats as stats
import pandas as pd
import time
import datetime
#import wmi

import numba as nb
from numba import jit, njit

import misc_functions as misc_fns


def sum_X_bool_cs(X, X_bool, axis=None, f=None, f_args=None):

    
    if f is None:
        f = misc_fns.return_x
    if f_args is None:
        f_args = {}

    if axis is None:
        s = np.sum(f(X[X_bool], **f_args))
    else:

        X = np.moveaxis(X, source=axis, destination=-1)
        X_bool = np.moveaxis(X_bool, source=axis, destination=-1)

        s = np.reshape(misc_fns.diff_zero(np.cumsum(f(X[X_bool], **f_args))[np.cumsum(np.sum(X_bool, axis=-1))-1]), np.shape(X)[:-1])


    return s


def sum_X_bool_filter(X, X_bool, axis=None, f=None, f_args=None):

    if f is None:
        f = misc_fns.return_x
    if f_args is None:
        f_args = {}

    if axis is None:
        s = np.sum(f(X[X_bool], **f_args))
    else:

        X = np.moveaxis(X, source=axis, destination=-1)
        X_bool = np.moveaxis(X_bool, source=axis, destination=-1)

        s = np.sum(X_bool_filter(X=X, X_bool=X_bool, ret_format='array', X_nan_zero=True, f=f, f_args=f_args), axis=-1)

    return s


def X_bool_filter(X, X_bool, ret_format=None, X_nan=None, X_nan_zero=None, axis=None, post_axis=None, f=None, f_args=None):

    if f is None:
        f = misc_fns.return_x
    if f_args is None:
        f_args = {}

    if X_nan_zero is None:
        X_nan_zero = False

    if ret_format is None:
        ret_format = 'array'

    if axis is not None:
        if post_axis is None:
            post_axis = True

        X = np.moveaxis(X, source=axis, destination=-1)
        X_bool = np.moveaxis(X_bool, source=axis, destination=-1)


    if ret_format=='array':

        X = f(X, **f_args)

        if X_nan is None:
            if X.dtype==int:
                if X_nan_zero:
                    X_nan = 0
                else:
                    X_nan = -(2**32 - 1)
            else:
                if X_nan_zero:
                    X_nan = 0.0
                else:
                    X_nan = np.nan


        X_shape = np.array(np.shape(X))
        X_bool_N = np.sum(X_bool, axis=-1)
        max_X_bool_N = np.max(X_bool_N)

        S_shape = np.concatenate([misc_fns.make_iterable_array(X_shape[:-1]), misc_fns.make_iterable_array(max_X_bool_N)])

        S = np.full(S_shape, X_nan)

        S[np.flip(np.sort(X_bool, axis=-1), axis=-1)[...,:max_X_bool_N]] = X[X_bool]

    

    else:
        S =  X[X_bool]

    
    if post_axis:
        if axis is not None:
            S =  np.moveaxis(S, source=-1, destination=axis)


    return S



def sum_X_N(X, N):

    if f is None:
        f = misc_fns.return_x
    if f_args is None:
        f_args = {}

    X_shape = np.array(np.shape(X))

    if len(X_shape)==1:
        return np.sum(X[:N])
    else:
        N_i = np.product(X_shape[:-1])
        X = f(np.reshape(X, (N_i, X_shape[-1])), **f_args)
        N = np.reshape(X, (N_i))
        
        return np.reshape(sum_X_N_nb(X=X, N=N), X_shape[:-1])
    


def sum_X_bool(X, X_bool, axis=None, f=None, f_args=None):

    if f is None:
        f = misc_fns.return_x
    if f_args is None:
        f_args = {}


    if axis is None:
        s = f(X[X_bool], **f_args)
    else:

        X = f(np.moveaxis(X, source=axis, destination=-1), **f_args)
        X_bool = np.moveaxis(X_bool, source=axis, destination=-1)

        X_shape = np.array(np.shape(X))

        if len(X_shape)==1:
            return np.sum(X[X_bool])
        else:
            N_i = np.product(X_shape[:-1])
            X = np.reshape(X, (N_i, X_shape[-1]))
            X_bool = np.reshape(X_bool, (N_i, X_shape[-1]))
            
            return np.reshape(sum_X_bool_nb(X=X, X_bool=X_bool), X_shape[:-1])




def where_X_bool(X_bool, max_X_bool_N=None, X_nan=None, X_nan_zero=None, axis=None, post_axis=None):



    if X_nan_zero is None:
        X_nan_zero = False


    if axis is None:
        S = np.where(X_bool)
    else:

        if post_axis is None:
            post_axis = True

        X_bool = np.moveaxis(X_bool, source=axis, destination=-1)
        X_shape = np.array(np.shape(X_bool))

        if len(X_shape)==1:
            S =  np.where(X_bool)[0]
        else:
            if X_nan is None:
                if X_nan_zero:
                    X_nan = 0
                else:
                    X_nan = -X_shape[-1]

        
            N_i = np.product(X_shape[:-1])

            if N_i==0:
                S = np.full(np.concatenate([misc_fns.make_iterable_array(X_shape[:-1]), misc_fns.make_iterable_array(0)]), X_nan)
            else:

                X_bool = np.reshape(X_bool, (N_i, X_shape[-1]))
                max_X_bool_N = np.max(np.sum(X_bool, axis=-1))
            
                X_where = np.full((N_i, max_X_bool_N), X_nan)
                
                S = np.reshape(where_X_bool_nb(X_bool=X_bool, X_where=X_where), np.concatenate([misc_fns.make_iterable_array(X_shape[:-1]), misc_fns.make_iterable_array(max_X_bool_N)]))
    
            if post_axis:
                S = np.moveaxis(S, source=-1, destination=axis)

    return S



def filter_X_bool(X, X_bool, max_X_bool_N=None, X_nan=None, X_nan_zero=None, axis=None, post_axis=None, f=None, f_args=None):

    if f is None:
        f = misc_fns.return_x
    if f_args is None:
        f_args = {}

    if X_nan_zero is None:
        X_nan_zero = False

    


    if axis is None:
        S = np.sum(f(X[X_bool], **f_args))
    else:

        if post_axis is None:
            post_axis = True

        X = f(np.moveaxis(X, source=axis, destination=-1), **f_args)
        X_bool = np.moveaxis(X_bool, source=axis, destination=-1)

        X_shape = np.array(np.shape(X))

        if len(X_shape)==1:
            S = X[X_bool]
        else:

            if X_nan is None:
                if X.dtype==int:
                    if X_nan_zero:
                        X_nan = 0
                    else:
                        X_nan = -(2**31 - 1)
                else:
                    if X_nan_zero:
                        X_nan = 0.0
                    else:
                        X_nan = np.nan
        
            N_i = np.product(X_shape[:-1])


            if N_i==0:
                S = np.full(np.concatenate([misc_fns.make_iterable_array(X_shape[:-1]), misc_fns.make_iterable_array(0)]), X_nan)

            else:
                X = np.reshape(X, (N_i, X_shape[-1]))
                X_bool = np.reshape(X_bool, (N_i, X_shape[-1]))

                max_X_bool_N = np.max(np.sum(X_bool, axis=-1))
                X_filter = np.full((N_i, max_X_bool_N), X_nan)
                
                S = np.reshape(filter_X_bool_nb(X=X, X_bool=X_bool, X_filter=X_filter), np.concatenate([misc_fns.make_iterable_array(X_shape[:-1]), misc_fns.make_iterable_array(max_X_bool_N)]))
    
            if post_axis:
                S = np.moveaxis(S, source=-1, destination=axis)

    return S





@njit(   [
            
            nb.float64(nb.float64[:], nb.int64),
        ])
def calc_autocorrcoef_h(X, h):

    return np.corrcoef(X[h:], X[:-h])[0,1]


@njit(   [
            
            nb.float64[:](nb.float64[:], nb.int64),
        ])
def calc_autocorrcoef_H(X, H):

    T = np.shape(X)[0]
    if H >= T:
        H = T-1

    rho = np.full(H, np.nan)
    for h in np.arange(H):
        rho[h] = calc_autocorrcoef_h(X=X, h=1+h)

    return rho


@njit(   [
            
            nb.float64[:](nb.float64[:], nb.int32[:]),
        ])
def calc_autocorrcoef_Hlist(X, Hlist):

    T = np.shape(X)[0]
    H_size = np.shape(Hlist)[0]
    rho = np.full(H_size, np.nan)
    for k in range(H_size):
        h = Hlist[k]
        if h < T:
            rho[k] = calc_autocorrcoef_h(X=X, h=h)
    return rho







@njit(   [
            
            nb.bool_[:](nb.float64[:], nb.float64, nb.bool_),
            nb.bool_[:](nb.int32[:], nb.float64, nb.bool_),
            nb.bool_[:](nb.float64[:], nb.int64, nb.bool_),
            nb.bool_[:](nb.int32[:], nb.int64, nb.bool_),
            nb.bool_(nb.float64, nb.float64, nb.bool_),
            nb.bool_(nb.int32, nb.float64, nb.bool_),
            nb.bool_(nb.float64, nb.int64, nb.bool_),
            nb.bool_(nb.int32, nb.int64, nb.bool_),
        ])
def calc_X_GREATER_u_i(X, u, u_inclusive):

    if u_inclusive:
        return X >= u
    else:
        return X > u
    

@njit(   [
            
            nb.bool_[:](nb.float64[:], nb.float64, nb.bool_),
            nb.bool_[:](nb.int32[:], nb.float64, nb.bool_),
            nb.bool_[:](nb.float64[:], nb.int64, nb.bool_),
            nb.bool_[:](nb.int32[:], nb.int64, nb.bool_),
            nb.bool_(nb.float64, nb.float64, nb.bool_),
            nb.bool_(nb.int32, nb.float64, nb.bool_),
            nb.bool_(nb.float64, nb.int64, nb.bool_),
            nb.bool_(nb.int32, nb.int64, nb.bool_),
        ])
def calc_X_LESSER_u_i(X, u, u_inclusive):

    if u_inclusive:
        return X <= u
    else:
        return X < u


@njit(   [
            nb.float64[:](nb.float64[:,:], nb.int32[:]),
            nb.int32[:](nb.int32[:,:], nb.int32[:]),
        ])
def sum_X_N_nb(X, N):

    N_i = len(N)
    X_sum = np.zeros(N_i, dtype=X.dtype)
    for i in range(N_i):
        X_sum[i] = np.sum(X[i,:N[i]])

    return X_sum



@njit(   [
                nb.float64(nb.float64[:], nb.bool_[:]),
                nb.int32(nb.int32[:], nb.bool_[:]),
            ])
def sum_X_bool_nb_1D(X, X_bool):

    return np.sum(X[X_bool])


if False:
    @njit(   [
                nb.float64[:](nb.float64[:,:], nb.bool_[:,:]),
                nb.int32[:](nb.int32[:,:], nb.bool_[:,:]),
            ])
    def sum_X_bool_nb(X, X_bool):

        N_i = np.shape(X)[0]
        X_sum = np.zeros(N_i, dtype=X.dtype)
        for i in range(N_i):
            X_sum[i] = np.sum(X[i,X_bool[i,:]])
            #X_sum[i] = np.sum(X[i,np.where(X_bool[i,:])[0]])

        return X_sum
elif False:
    @njit(   [
                nb.float64[:](nb.float64[:,:], nb.bool_[:,:]),
                nb.int32[:](nb.int32[:,:], nb.bool_[:,:]),
            ])
    def sum_X_bool_nb(X, X_bool):

        N_i = np.shape(X)[0]
        X_sum = np.zeros(N_i, dtype=X.dtype)
        for i in range(N_i):
            X_sum[i] = sum_X_bool_nb_1D(X=X[i,:], X_bool=X_bool[i,:])
            #X_sum[i] = np.sum(X[i,np.where(X_bool[i,:])[0]])

        return X_sum
else:

    @njit(   [
                    nb.int32[:,:](nb.bool_[:,:], nb.int32[:,:]),
                    nb.int32[:,:](nb.bool_[:,:], nb.int32[:,:]),
                ])
    def where_X_bool_nb(X_bool, X_where):

        N_i = np.shape(X_bool)[0]
        for i in range(N_i):
            where_i = np.where(X_bool[i,:])[0]
            X_where[i,:np.shape(where_i)[0]] = where_i

        return X_where
    

    @njit(   [
                    nb.float64[:,:](nb.float64[:,:], nb.bool_[:,:], nb.float64[:,:]),
                    nb.int32[:,:](nb.int32[:,:], nb.bool_[:,:], nb.int32[:,:]),
                    nb.float64[:,:](nb.int32[:,:], nb.bool_[:,:], nb.float64[:,:]),
                ])
    def filter_X_bool_nb(X, X_bool, X_filter):

        N_i = np.shape(X)[0]
        for i in range(N_i):
            where_i = np.where(X_bool[i,:])[0]
            for j in range(where_i.size):
                X_filter[i,j] = X[i,where_i[j]]

        return X_filter

    @njit(   [
                    nb.float64[:](nb.float64[:,:], nb.bool_[:,:]),
                    nb.int32[:](nb.int32[:,:], nb.bool_[:,:]),
                ])
    def sum_X_bool_nb(X, X_bool):

        N_i = np.shape(X)[0]
        X_sum = np.zeros(N_i, dtype=X.dtype)
        for i in range(N_i):
            where_i = np.where(X_bool[i,:])[0]
            for j in where_i:
                X_sum[i] += X[i,j]

        return X_sum