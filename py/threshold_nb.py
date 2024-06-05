import numpy as np

#import wmi

import numba as nb
from numba import jit, njit

import misc_functions_nb as misc_fns_nb



    



@njit(   [
            nb.float64[:](nb.float64[:,:], nb.float64[:]),
            nb.float64[:](nb.int32[:,:], nb.float64[:]),
            nb.int32[:](nb.int32[:,:], nb.float64[:]),
        ])
def calc_row_quantiles_nb(tilde_X, q):

    N_i = np.shape(tilde_X)[0]
    u = np.full(N_i, np.nan)
    for i in range(N_i):
        u[i] = np.quantile(tilde_X[i,:], q[i])

    return u



@njit(   [
            nb.float64(nb.float64[:], nb.float64, nb.int64),
            nb.float64(nb.int32[:], nb.float64, nb.int64),
            nb.int64(nb.int32[:], nb.float64, nb.int64),
        ])
def calc_u_nb_i(tilde_X, a_u, u_dir):

    u = np.quantile(tilde_X, 1-a_u)
    if u_dir==-1:
        u = -u
    elif u_dir==0:
        u = np.abs(u)

    return u


@njit(   [
            nb.float64[:](nb.float64[:,:], nb.float64[:], nb.int32[:]),
            nb.float64[:](nb.int32[:,:], nb.float64[:], nb.int32[:]),
            nb.int32[:](nb.int32[:,:], nb.float64[:], nb.int32[:]),
        ])
def calc_u_nb(tilde_X, a_u, u_dir):

    N_i = np.shape(tilde_X)[0]
    u = np.full(N_i, np.nan)
    for i in range(N_i):
        u[i] = calc_u_nb_i(tilde_X=tilde_X[i,:], a_u=a_u[i], u_dir=u_dir[i])

    return u



@njit(   [
            nb.float64[:](nb.float64[:], nb.int64, nb.float64),
            nb.float64[:](nb.float64[:], nb.int64, nb.int64),
            nb.float64[:](nb.int32[:], nb.int64, nb.float64),
            nb.float64[:](nb.int32[:], nb.int64, nb.int64),
        ])
def calc_tilde_X_nb_i(X, u_dir, u_zero):

    T = np.shape(X)[-1]
    tilde_X = np.full(T, np.nan)

    if u_dir==-1:
        tilde_X[:] = -(X - u_zero)
    elif u_dir==0:
        tilde_X[:] = np.abs(X - u_zero)
    else:
        tilde_X[:] = X - u_zero

    return tilde_X

@njit(   [
            nb.float64[:,:](nb.float64[:], nb.int32[:], nb.float64[:]),
            nb.float64[:,:](nb.float64[:], nb.int32[:], nb.int32[:]),
            nb.float64[:,:](nb.int32[:], nb.int32[:], nb.float64[:]),
            nb.float64[:,:](nb.int32[:], nb.int32[:], nb.int32[:]),
        ])
def calc_tilde_X_nb(X, u_dir, u_zero):

    N_i = np.shape(u_dir)[0]
    T = np.shape(X)[-1]


    dtype=np.float64
    tilde_X = np.zeros((N_i, T), dtype=dtype)

    for i in range(N_i):
        tilde_X[i,:] = calc_tilde_X_nb_i(X=X, u_dir=u_dir[i], u_zero=u_zero[i])

    return tilde_X


@njit(   [
            nb.int32[:](nb.float64[:], nb.int64, nb.float64, nb.bool_),
            nb.int32[:](nb.float64[:], nb.int64, nb.int64, nb.bool_),
            nb.int32[:](nb.int32[:], nb.int64, nb.float64, nb.bool_),
            nb.int32[:](nb.int32[:], nb.int64, nb.int64, nb.bool_),
        ])
def calc_post_sign_nb_i(X, u_dir, u_zero, carry_sign):

    T = np.shape(X)[-1]
    post_sign = np.ones(T, dtype=np.int32)
    if carry_sign:
        if u_dir==-1:
            post_sign[:] = -1
        elif u_dir==0:
            post_sign[:] = np.sign(calc_tilde_X_nb_i(X=X, u_dir=1, u_zero=u_zero))

    return post_sign


@njit(   [
            nb.int32[:,:](nb.float64[:], nb.int32[:], nb.float64[:], nb.bool_),
            nb.int32[:,:](nb.float64[:], nb.int32[:], nb.int32[:], nb.bool_),
            nb.int32[:,:](nb.int32[:], nb.int32[:], nb.float64[:], nb.bool_),
            nb.int32[:,:](nb.int32[:], nb.int32[:], nb.int32[:], nb.bool_),
        ])
def calc_post_sign_nb(X, u_dir, u_zero, carry_sign):

    N_i = np.shape(u_dir)[0]
    T = np.shape(X)[-1]

    post_sign = np.ones((N_i, T), dtype=np.int32)
    if carry_sign:
        for i in range(N_i):
            post_sign[i,:] = calc_post_sign_nb_i(X=X, u_dir=u_dir[i], u_zero=u_zero[i], carry_sign=True)

    return post_sign





@njit(   [
            
            nb.float64[:](nb.float64[:], nb.int64, nb.float64),
            nb.float64[:](nb.float64[:], nb.int64, nb.int64),
            nb.float64[:](nb.int32[:], nb.int64, nb.float64),
            nb.float64[:](nb.int32[:], nb.int64, nb.int64),
        ])
def calc_tilde_X_u_tilde_X_nb_i(tilde_X, u_dir, u):

    T = np.shape(tilde_X)[0]
    tilde_X_u = np.full(T, np.nan)

    if u_dir==-1:
        tilde_X_u[:] = tilde_X - -u
    elif u_dir==0:
        tilde_X_u[:] = tilde_X - np.abs(u)
    else:
        tilde_X_u[:] = tilde_X - u

    return tilde_X_u


@njit(   [
            nb.float64[:](nb.float64[:], nb.int64, nb.float64, nb.float64),
            nb.float64[:](nb.float64[:], nb.int64, nb.float64, nb.int64),
            nb.float64[:](nb.float64[:], nb.int64, nb.int64, nb.float64),
            nb.float64[:](nb.float64[:], nb.int64, nb.int64, nb.int64),
            nb.float64[:](nb.int32[:], nb.int64, nb.float64, nb.float64),
            nb.float64[:](nb.int32[:], nb.int64, nb.float64, nb.int64),
            nb.float64[:](nb.int32[:], nb.int64, nb.int64, nb.float64),
            nb.float64[:](nb.int32[:], nb.int64, nb.int64, nb.int64),
        ])
def calc_tilde_X_u_nb_i(X, u_dir, u_zero, u):

    return calc_tilde_X_u_tilde_X_nb_i(tilde_X=calc_tilde_X_nb_i(X=X, u_dir=u_dir, u_zero=u_zero), u_dir=u_dir, u=u)



@njit(   [
            nb.float64[:,:](nb.float64[:,:], nb.int32[:], nb.float64[:]),
            nb.float64[:,:](nb.float64[:,:], nb.int32[:], nb.int32[:]),
            nb.float64[:,:](nb.int32[:,:], nb.int32[:], nb.float64[:]),
            nb.float64[:,:](nb.int32[:,:], nb.int32[:], nb.int32[:]),
        ])
def calc_tilde_X_u_tilde_X_nb(tilde_X, u_dir, u):

    N_i = np.shape(u_dir)[0]
    T = np.shape(tilde_X)[-1]

    dtype=np.float64
    tilde_X_u = np.zeros((N_i, T), dtype=dtype)

    for i in range(N_i):
        tilde_X_u[i,:] = calc_tilde_X_u_tilde_X_nb_i(tilde_X=tilde_X[i,:], u_dir=u_dir[i], u=u[i])

    return tilde_X_u


@njit(   [
            nb.float64[:,:](nb.float64[:], nb.int32[:], nb.float64[:], nb.float64[:]),
            nb.float64[:,:](nb.float64[:], nb.int32[:], nb.float64[:], nb.int32[:]),
            nb.float64[:,:](nb.float64[:], nb.int32[:], nb.int32[:], nb.float64[:]),
            nb.float64[:,:](nb.float64[:], nb.int32[:], nb.int32[:], nb.int32[:]),
            nb.float64[:,:](nb.int32[:], nb.int32[:], nb.float64[:], nb.float64[:]),
            nb.float64[:,:](nb.int32[:], nb.int32[:], nb.float64[:], nb.int32[:]),
            nb.float64[:,:](nb.int32[:], nb.int32[:], nb.int32[:], nb.float64[:]),
            nb.float64[:,:](nb.int32[:], nb.int32[:], nb.int32[:], nb.int32[:]),
        ])
def calc_tilde_X_u_nb(X, u_dir, u_zero, u):

    N_i = np.shape(u_dir)[0]
    T = np.shape(X)[-1]

    dtype=np.float64
    tilde_X_u = np.zeros((N_i, T), dtype=dtype)

    for i in range(N_i):
        tilde_X_u[i,:] = calc_tilde_X_u_nb_i(X=X, u_dir=u_dir[i], u_zero=u_zero[i], u=u[i])

    return tilde_X_u



@njit(   [
            nb.bool_[:](nb.float64[:], nb.bool_),
            nb.bool_[:](nb.int32[:], nb.bool_),
        ])
def calc_I_u_tilde_X_u_nb_i(tilde_X_u,  u_inclusive):

    return misc_fns_nb.calc_X_GREATER_u_i(X=tilde_X_u, u=0, u_inclusive=u_inclusive)


@njit(   [
            nb.bool_[:](nb.float64[:], nb.int64, nb.float64, nb.bool_),
            nb.bool_[:](nb.float64[:], nb.int64, nb.int64, nb.bool_),
            nb.bool_[:](nb.int32[:], nb.int64, nb.float64, nb.bool_),
            nb.bool_[:](nb.int32[:], nb.int64, nb.int64, nb.bool_),
        ])
def calc_I_u_tilde_X_nb_i(tilde_X, u_dir, u, u_inclusive):

    return calc_I_u_tilde_X_u_nb_i(tilde_X_u=calc_tilde_X_u_tilde_X_nb_i(tilde_X=tilde_X, u_dir=u_dir, u=u),  u_inclusive=u_inclusive)



@njit(   [
            nb.bool_[:](nb.float64[:], nb.int64, nb.float64, nb.float64, nb.bool_),
            nb.bool_[:](nb.float64[:], nb.int64, nb.float64, nb.int64, nb.bool_),
            nb.bool_[:](nb.float64[:], nb.int64, nb.int64, nb.float64, nb.bool_),
            nb.bool_[:](nb.float64[:], nb.int64, nb.int64, nb.int64, nb.bool_),
            nb.bool_[:](nb.int32[:], nb.int64, nb.float64, nb.float64, nb.bool_),
            nb.bool_[:](nb.int32[:], nb.int64, nb.float64, nb.int64, nb.bool_),
            nb.bool_[:](nb.int32[:], nb.int64, nb.int64, nb.float64, nb.bool_),
            nb.bool_[:](nb.int32[:], nb.int64, nb.int64, nb.int64, nb.bool_),
        ])
def calc_I_u_nb_i(X, u_dir, u_zero, u, u_inclusive):

    return calc_I_u_tilde_X_u_nb_i(tilde_X_u=calc_tilde_X_u_nb_i(X=X, u_dir=u_dir, u_zero=u_zero, u=u),  u_inclusive=u_inclusive)




@njit(   [
            nb.bool_[:,:](nb.float64[:,:], nb.bool_[:]),
            nb.bool_[:,:](nb.int32[:,:], nb.bool_[:]),
        ])
def calc_I_u_tilde_X_nb(tilde_X_u, u_inclusive):

    N_i = np.shape(u_inclusive)[0]
    T = np.shape(tilde_X_u)[-1]

    I_u = np.full((N_i, T), False)

    for i in range(N_i):
        I_u[i,:] = calc_I_u_tilde_X_u_nb_i(tilde_X_u=tilde_X_u[i,:], u_inclusive=u_inclusive[i])

    return I_u



@njit(   [
            nb.bool_[:,:](nb.float64[:,:], nb.int32[:], nb.float64[:], nb.bool_[:]),
            nb.bool_[:,:](nb.float64[:,:], nb.int32[:], nb.int32[:], nb.bool_[:]),
            nb.bool_[:,:](nb.int32[:,:], nb.int32[:], nb.float64[:], nb.bool_[:]),
            nb.bool_[:,:](nb.int32[:,:], nb.int32[:], nb.int32[:], nb.bool_[:]),
        ])
def calc_I_u_tilde_X_nb(tilde_X, u_dir, u, u_inclusive):

    N_i = np.shape(u_dir)[0]
    T = np.shape(tilde_X)[-1]

    I_u = np.full((N_i, T), False)

    for i in range(N_i):
        I_u[i,:] = calc_I_u_tilde_X_nb_i(tilde_X=tilde_X[i,:], u_dir=u_dir[i], u=u[i], u_inclusive=u_inclusive[i])

    return I_u


@njit(   [
            nb.bool_[:,:](nb.float64[:], nb.int32[:], nb.float64[:], nb.float64[:], nb.bool_[:]),
            nb.bool_[:,:](nb.float64[:], nb.int32[:], nb.float64[:], nb.int32[:], nb.bool_[:]),
            nb.bool_[:,:](nb.float64[:], nb.int32[:], nb.int32[:], nb.float64[:], nb.bool_[:]),
            nb.bool_[:,:](nb.float64[:], nb.int32[:], nb.int32[:], nb.int32[:], nb.bool_[:]),
            nb.bool_[:,:](nb.int32[:], nb.int32[:], nb.float64[:], nb.float64[:], nb.bool_[:]),
            nb.bool_[:,:](nb.int32[:], nb.int32[:], nb.float64[:], nb.int32[:], nb.bool_[:]),
            nb.bool_[:,:](nb.int32[:], nb.int32[:], nb.int32[:], nb.float64[:], nb.bool_[:]),
            nb.bool_[:,:](nb.int32[:], nb.int32[:], nb.int32[:], nb.int32[:], nb.bool_[:]),
        ])
def calc_I_u_nb(X, u_dir, u_zero, u, u_inclusive):

    N_i = np.shape(u_dir)[0]
    T = np.shape(X)[-1]

    I_u = np.full((N_i, T), False)

    for i in range(N_i):
        I_u[i,:] = calc_I_u_nb_i(X=X, u_dir=u_dir[i], u_zero=u_zero[i], u=u[i], u_inclusive=u_inclusive[i])

    return I_u



@njit([
        nb.types.Tuple((
                        nb.float64[:], nb.float64[:], nb.bool_[:], nb.float64,
                    ))(
                        nb.float64[:], nb.float64, nb.int64, nb.float64, nb.float64, nb.bool_, 
                        nb.bool_, nb.int32[:],   
                    ),
        nb.types.Tuple((
                        nb.float64[:], nb.float64[:], nb.bool_[:], nb.float64,
                    ))(
                        nb.int32[:], nb.float64, nb.int64, nb.float64, nb.float64, nb.bool_, 
                        nb.bool_, nb.int32[:],   
                    ),
                    ])
def calc_events_base_X_nb_i(X, a_u, u_dir, u_zero, u, u_inclusive,
                         set_u, train_t):
    
    tilde_X = calc_tilde_X_nb_i(X, u_dir, u_zero)
    if set_u:
        u = calc_u_nb_i(tilde_X=tilde_X[train_t[0]:train_t[1]], a_u=a_u, u_dir=u_dir)
    tilde_X_u = calc_tilde_X_u_tilde_X_nb_i(tilde_X, u_dir, u)
    I_u = calc_I_u_tilde_X_u_nb_i(tilde_X_u,  u_inclusive)

    return tilde_X, tilde_X_u, I_u, u



@njit([
        nb.types.Tuple((
                        nb.float64[:,:], nb.float64[:,:], nb.bool_[:,:], nb.float64[:],
                    ))(
                        nb.float64[:], nb.float64[:], nb.int32[:], nb.float64[:], nb.float64[:], nb.bool_[:], 
                        nb.bool_, nb.int32[:],   
                    ),
        nb.types.Tuple((
                        nb.float64[:,:], nb.float64[:,:], nb.bool_[:,:], nb.float64[:],
                    ))(
                        nb.int32[:], nb.float64[:], nb.int32[:], nb.float64[:], nb.float64[:], nb.bool_[:], 
                        nb.bool_, nb.int32[:],   
                    ),
                    ])
def calc_events_base_X_nb(X, a_u, u_dir, u_zero, u, u_inclusive,
                         set_u, train_t):
    
    N_i = np.shape(u_dir)[0]
    T = np.shape(X)[-1]

    tilde_X = np.full((N_i, T), np.nan)
    tilde_X_u = np.full((N_i, T), np.nan)
    I_u = np.full((N_i, T), False)

    for i in range(N_i):
        tilde_X[i,:], tilde_X_u[i,:], I_u[i,:], u[i] = calc_events_base_X_nb_i(X=X, a_u=a_u[i], u_dir=u_dir[i], u_zero=u_zero[i], u=u[i], u_inclusive=u_inclusive[i],
                                                                                set_u=set_u, train_t=train_t)


    return tilde_X, tilde_X_u, I_u, u

