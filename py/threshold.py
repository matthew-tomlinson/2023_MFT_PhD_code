import numpy as np
import scipy
import scipy.stats as stats
import pandas as pd
import time
import datetime
#import wmi

import multiprocessing
from multiprocessing import Pool
# Local files
import parameters

import misc_functions as misc_fns
import plot_functions as plot_fns
import dist_functions as dist_fns

import misc_functions_nb as misc_fns_nb 
import threshold_nb as thresh_nb




class threshold(parameters.D_nodes):
    """Class for u_dir
    """
    def __init__(self, 
                 D_nodes=None,
                 a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None,
                 X=None, train_t=None, calc_u=None, calc_a_u=None):

        self._D_nodes = misc_fns.clean_D_nodes(D_nodes=D_nodes)

        self.reset_params_threshold()

        self._set_params_threshold(a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, reset=True,
                                   X=X, train_t=train_t, calc_u=calc_u, calc_a_u=calc_a_u)



    def reset_params_threshold(self):

        #----------------------------
        # Threshold parameters
        #----------------------------

        # Expected intensity
        self.a_u = parameters.p_scalar(D_nodes=self.get_D_nodes(), name="a_u", etype=[int, float], family="threshold", default=0.025, incl_low=0)

        # Threshold type
        self.u_dir = parameters.p_scalar(D_nodes=self.get_D_nodes(), name="u_dir", etype=[str], family="threshold", default="right", permitted=["left", "right", "abs"])

        # Threshold zero
        self.u_zero = parameters.p_scalar(D_nodes=self.get_D_nodes(), name="u_zero", etype=[int, float], family="threshold", default=0)

        # Threshold
        self.u = parameters.p_scalar(D_nodes=self.get_D_nodes(), name="u", etype=[int, float], family="threshold", default=0)

        # Threshold inclusive
        self.u_inclusive = parameters.p_scalar(D_nodes=self.get_D_nodes(), name="u_inclusive", etype=[bool], family="threshold", default=False)


        
        self.reset_threshold_names()

    
    def print_params_threshold(self):

        print("Threshold parameters:")
        print("a_u =", self.a_u.get_value())
        print("u_dir =", self.u_dir.get_value())
        print("u_zero =", self.u_zero.get_value())
        print("u =", self.u.get_value())
        print("u_inclusive =", self.u_inclusive.get_value())
        print("")

    def get_threshold_name(self, name=None, name_rm=None):
        if name is None:
            name = self._threshold_name
        return misc_fns.A_rm(A=name, A_rm=name_rm)
    
    def calc_threshold_names(self, name=None, name_rm=None):

        return self.calc_names(name=self.get_threshold_name(name=name, name_rm=name_rm))

    def get_threshold_names(self, names=None, names_rm=None):
        if names is None:
            names = self._threshold_names
        return misc_fns.A_rm(A=names, A_rm=names_rm)


    def reset_threshold_names(self):

        self._threshold_name = ["a_u", "u_dir", "u_zero", "u", "u_inclusive"]
        self._threshold_names = self.calc_names(name=self._threshold_name)


    def calc_params_threshold_dict(self, name=None, name_rm=None):


        self.get_threshold_name(name=None, name_rm=None)

        name = self.get_threshold_name(name=name, name_rm=name_rm)
        names = self.calc_threshold_names(name=name)
        #_params_threshold_dict = {**self.__dict__[n].calc_dict() for n in name}
        _params_threshold_dict = {}
        for n in name:
            n_dict = self.__dict__[n].calc_dict()
            for ns in n_dict:
                _params_threshold_dict[ns] = n_dict[ns]
                
        return _params_threshold_dict
    


    def _set_params_threshold(self, a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None, 
                    reset=None, reset_a_u=None, reset_u_dir=None, reset_u_zero=None, reset_u=None, reset_u_inclusive=None,
                    X=None, train_t=None,
                    calc_u=None, calc_a_u=None):
        

        resets = calc_resets_params_threshold(reset=reset, a_u=reset_a_u, u_dir=reset_u_dir, u_zero=reset_u_zero, u=reset_u, u_inclusive=reset_u_inclusive)

        self._set_a_u(a_u=a_u, reset=resets['a_u'])
        self._set_u_dir(u_dir=u_dir, reset=resets['u_dir'])
        self._set_u_zero(u_zero=u_zero, reset=resets['u_zero'])
        self._set_u(u=u, reset=resets['u'])
        self._set_u_inclusive(u_inclusive=u_inclusive, reset=resets['u_inclusive'])

        if calc_u is None:
            calc_u = False
        if calc_a_u is None:
            calc_a_u = False

        self._set_u(X=X, train_t=train_t, calc_u=calc_u)
        self._set_a_u(X=X, train_t=train_t, calc_a_u=calc_a_u)


    
    def _set_a_u(self, a_u=None, reset=None, X=None, train_t=None, calc_a_u=None):

        if reset is None:
            reset = False

        if a_u is None:
            if X is not None:
                if calc_a_u is None:
                    calc_a_u = True
            if calc_a_u:
                a_u = self.calc_a_u_func(X=X, train_t=train_t)

        self.a_u.set_value(value=a_u, reset=reset)

    
    def _set_u_dir(self, u_dir=None, reset=None):

        if reset is None:
            reset = False
        
        self.u_dir.set_value(value=u_dir, reset=reset)

        #self._set_u_dir_func()
        #self._set_u_dir_func_inv()



    def _set_u_zero(self, u_zero=None, reset=None):

        if reset is None:
            reset = False
        
        self.u_zero.set_value(value=u_zero, reset=reset)


    def _set_u(self, u=None, reset=None, X=None, train_t=None, calc_u=None):

        if reset is None:
            reset = False

        if u is None:
            if X is not None:
                if calc_u is None:
                    calc_u = True
            if calc_u:
                u = self.calc_u_func(X=X, train_t=train_t)

        self.u.set_value(value=u, reset=reset)


    def _set_u_inclusive(self, u_inclusive=None, reset=None):

        if reset is None:
            reset = False
        
        self.u_inclusive.set_value(value=u_inclusive, reset=reset)

    
    if False:
        def _set_u_dir_func(self):
            self._u_dir_func = self.calc_u_dir_func()

        def _set_u_dir_func_inv(self):
            self._u_dir_func_inv = self.calc_u_dir_func_inv()

        def get_u_dir_func(self):
            return self._u_dir_func
        
        def get_u_dir_func_inv(self):
            return self._u_dir_func_inv
    
    def calc_u_dir_func(self):
        return calc_u_dir_func(u_dir=self.u_dir.get_value())

    def calc_u_dir_func_inv(self):
        return calc_u_dir_func_inv(u_dir=self.u_dir.get_value())
    

    def calc_tilde_X_func_u(self, X, apply_u_dir_func_inc=None):

        return calc_tilde_X_func_u(X=X, u_dir=self.u_dir.get_value(), u_zero=self.u_zero.get_value(), u=self.u.get_value(), apply_u_dir_func_inc=apply_u_dir_func_inc)
    

    def calc_tilde_X_func(self, X, apply_u_dir_func_inc=None):

        return calc_tilde_X_func(X=X, u_dir=self.u_dir.get_value(), u_zero=self.u_zero.get_value(), apply_u_dir_func_inc=apply_u_dir_func_inc)
    

    def calc_I_u_func(self, X):

        return calc_I_u_func(X=X, u_dir=self.u_dir.get_value(), u_zero=self.u_zero.get_value(), u=self.u.get_value(), u_inclusive=self.u_inclusive.get_value())

    if False:
        def calc_u_func(self, X, train_t=None):

            train_t = misc_fns.calc_train_t(train_t=train_t, X=X)
            samp = X[train_t[0]:train_t[1]]

            return np.quantile(self.calc_tilde_X_func_u(X=samp, apply_u_dir_func_inc=False), self.a_u.get_value())


        def calc_a_u_func(self, X, train_t=None):

            train_t = misc_fns.calc_train_t(train_t=train_t, X=X)
            samp = X[train_t[0]:train_t[1]]

            return np.sum(self.calc_I_u(X=samp))/np.size(samp)
    

    def calc_u_func(self, X, train_t=None, train_T=None):

        #self._set_u_dir()

        return calc_u_func(X=X, train_t=train_t, train_T=train_T, a_u=self.a_u.get_value(), u_dir=self.u_dir.get_value(), u_zero=self.u_zero.get_value())


    def calc_a_u_func(self, X, train_t=None, train_T=None):

        return calc_a_u_func(X=X, train_t=train_t, train_T=train_T, u_dir=self.u_dir.get_value(), u_zero=self.u_zero.get_value(), u=self.u.get_value(), u_inclusive=self.u_inclusive.get_value())
        

    def calc_signs_u(self, X, u_dir=None, u_zero=None):
        
        return calc_signs_u(X=X, u_dir=self.u_dir.get_value(), u_zero=self.u_zero.get_value())







class thresholds(threshold):
    """Class for thresholds
    """
    def __init__(self, 
                 D_nodes=None,
                 a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None,
                 X=None, train_t=None, calc_u=None, calc_a_u=None):
        
        if False:

            self._D_nodes = misc_fns.clean_D_nodes(D_nodes=D_nodes)

            self.reset_params_threshold()

            self._set_params_threshold(a_u=a_u, u_dir=u_dir, u_zero=u_zero, u_inclusive=u_inclusive, u=u, reset=True,
                                    X=X, train_t=train_t, calc_u=calc_u, calc_a_u=calc_a_u)
        else:
            super().__init__(D_nodes=D_nodes, 
                                a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, 
                                X=X, train_t=train_t, calc_u=calc_u, calc_a_u=calc_a_u)



    def reset_params_threshold(self):

        #----------------------------
        # Threshold parameters
        #----------------------------

        # Expected intensity
        self.a_u = parameters.p_vector(D_nodes=self.get_D_nodes(), name="a_u", etype=[int, float], family="threshold", default=0.025, incl_low=0)

        # Threshold type
        self.u_dir = parameters.p_vector(D_nodes=self.get_D_nodes(), name="u_dir", etype=[str], family="threshold", default="right", permitted=["left", "right", "abs"])

        # Threshold zero
        self.u_zero = parameters.p_vector(D_nodes=self.get_D_nodes(), name="u_zero", etype=[int, float], family="threshold", default=0)

        # Threshold
        self.u = parameters.p_vector(D_nodes=self.get_D_nodes(), name="u", etype=[int, float], family="threshold", default=0)

        # Threshold inclusive
        self.u_inclusive = parameters.p_vector(D_nodes=self.get_D_nodes(), name="u_inclusive", etype=[bool], family="threshold", default=False)

        
        self.reset_threshold_names()

    
    if False:
        def _set_params_threshold(self, a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None, 
                            reset=None, reset_a_u=None, reset_u_dir=None, reset_u_zero=None, reset_u=None, reset_u_inclusive=None):


            resets = calc_resets_params_threshold(reset=reset, a_u=reset_a_u, u_dir=reset_u_dir, u_zero=reset_u_zero, u=reset_u, u_inclusive=reset_u_inclusive)

            self.a_u.set_value(value=a_u, reset=resets['a_u'])
            self.u_dir.set_value(value=u_dir, reset=resets['u_dir'])
            self.u_zero.set_value(value=u_zero, reset=resets['u_zero'])
            self.u.set_value(value=u, reset=resets['u'])
            self.u_inclusive.set_value(value=u_inclusive, reset=resets['u_inclusive'])


    def calc_u_dir_func(self, nodes=None):
        nodes = self.clean_nodes(nodes)
        return [calc_u_dir_func(u_dir=self.u_dir.get_value()[i]) for i in nodes]

    def calc_u_dir_func_inv(self, nodes=None):
        nodes = self.clean_nodes(nodes)
        return [calc_u_dir_func_inv(u_dir=self.u_dir.get_value()[i]) for i in nodes]

    
    def calc_tilde_X_func_u(self, X, apply_u_dir_func_inc=None, nodes=None):

        nodes = self.clean_nodes(nodes)
        return np.array([
                    calc_tilde_X_func_u(X=X, u_dir=self.u_dir.get_value()[i], u_zero=self.u_zero.get_value()[i], u=self.u.get_value()[i], apply_u_dir_func_inc=apply_u_dir_func_inc)
                for i in nodes])
    
    
    def calc_tilde_X_func(self, X, apply_u_dir_func_inc=None, nodes=None):

        nodes = self.clean_nodes(nodes)
        return np.array([
                    calc_tilde_X_func(X=X, u_dir=self.u_dir.get_value()[i], u_zero=self.u_zero.get_value()[i], apply_u_dir_func_inc=apply_u_dir_func_inc)
                for i in nodes])
    

    def calc_I_u_func(self, X, nodes=None):

        nodes = self.clean_nodes(nodes)

        return np.array([
                    calc_I_u_func(X=X, u_dir=self.u_dir.get_value()[i], u_zero=self.u_zero.get_value()[i], u=self.u.get_value()[i], u_inclusive=self.u_inclusive.get_value()[i])
                for i in nodes])


    def calc_u_func(self, X, train_t=None, train_T=None, nodes=None):

        nodes = self.clean_nodes(nodes)

        self._set_u_dir()

        return np.array([
                    calc_u_func(X=X, train_t=train_t, train_T=train_T, a_u=self.a_u.get_value()[i], u_dir=self.u_dir.get_value()[i], u_zero=self.u_zero.get_value()[i])
                for i in nodes])
    

    def calc_a_u_func(self, X, train_t=None, train_T=None, nodes=None):

        nodes = self.clean_nodes(nodes)

        return np.array([
                    calc_a_u_func(X=X, train_t=train_t, train_T=train_T, u_dir=self.u_dir.get_value()[i], u_zero=self.u_zero.get_value()[i], u=self.u.get_value()[i], u_inclusive=self.u_inclusive.get_value()[i])
                for i in nodes])


    def calc_signs_u(self, X, nodes=None):

        nodes = self.clean_nodes(nodes)

        return np.array([
                    calc_signs_u(X=X, u_dir=self.u_dir.get_value(), u_zero=self.u_zero.get_value())
                for i in nodes])
        
        


#============
#============
#
# Functions
#
#============
#============


def calc_params_threshold_dict(a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None):

    _locals = locals()
    return _locals

def calc_resets_params_threshold(reset=None, a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None):

    return misc_fns.calc_resets(reset=reset, resets=calc_params_threshold_dict(a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive))




def calc_u_dir_int(u_dir=None):

    if u_dir is None:
        u_dir = 'right'

    if np.isin(u_dir, ['abs', 'both', 0]):
        u_dir_int = 0
    elif np.isin(u_dir, ['left', -1]):
        u_dir_int = -1
    else:
        u_dir_int = 1
        
    return u_dir_int


def calc_u_dir_int_array(u_dir=None):

    u_dir = np.array(u_dir, dtype=object)

    u_dir_int = np.full(np.shape(u_dir), calc_u_dir_int(u_dir='right'))

    u_dir_int[np.isin(u_dir, np.array(['abs', 'both', 0], dtype=object))] = calc_u_dir_int(u_dir='abs')
    u_dir_int[np.isin(u_dir, np.array(['left', -1], dtype=object))] = calc_u_dir_int(u_dir='left')
        
    return u_dir_int


def calc_u_dir_func(u_dir=None):

    if u_dir is None:
        u_dir = 'right'

    if np.isin(u_dir, ['abs', 'both', 0]):
        def u_dir_func(X):
            return np.abs(X)
    elif np.isin(u_dir, ['left', -1]):
        def u_dir_func(X):
            return -X
    else:
        def u_dir_func(X):
            return X
        
    return u_dir_func


def calc_u_dir_func_inv(u_dir=None):

    if u_dir is None:
        u_dir = 'right'

    if u_dir=='abs':
        def u_dir_func(X):
            return np.abs(X)
    elif np.isin(u_dir, ['left', -1]):
        def u_dir_func(X):
            return -X
    else:
        def u_dir_func(X):
            return X
        
    return u_dir_func


def calc_params_threshold_defaults(val=None, two_tail=None):

    if two_tail is None:
        two_tail = False

    _params_threshold_defaults = {
                                    'a_u': 0.025,
                                    'u_dir': 'right',
                                    'u_zero': 0,
                                    'u': 0,
                                    'u_inclusive': False,
                                }
    
    if two_tail:
        _params_threshold_defaults['u_dir'] = ['left', 'right']


    if val is None:
        return _params_threshold_defaults
    else:
        return _params_threshold_defaults[val]
    


def match_params_threshold_shape(a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None,
                                 priority=None, two_tail=None, master_shape=None):
    


    _locals = locals()

    _shapes = calc_params_threshold_dict()
    params = np.array([p for p in _shapes])

    P = len(params)
    _dims = calc_params_threshold_dict()
    dims  = np.zeros(P, dtype=int)

    for p in range(P):
        param = params[p]
        _shapes[param] = np.array(np.shape(_locals[param]))
        _dims[param] = len(_shapes[param])
        dims[p] = len(_shapes[param])
    
    

    if master_shape is None:
        if priority is None:
            if u is not None:
                priority = 'u'
            else:
                max_dim = np.max(dims)
                if max_dim==0:
                    priority = 'u'
                else:
                    where_max_dim = params[np.where(dims==max_dim)[0]]
                    shape_dim_max = np.array([_shapes[param] for param in where_max_dim])
                    priority = where_max_dim[np.argmax(shape_dim_max)//max_dim]

        master_shape = _shapes[priority] 
        force_master_shape = False
        
    else:
        master_shape = np.array(master_shape)
        force_master_shape = True


    if np.array_equal(master_shape, np.array([])):
        master_shape = np.array([1])

    if two_tail and _locals['u_dir'] is None and _locals['u'] is None and not force_master_shape:
        if master_shape[-1]==1:
            master_shape[-1] = 2

    output = calc_params_threshold_dict()


    for param in output:

        val = _locals[param]

        if val is None:
            val =  calc_params_threshold_defaults(val=param, two_tail=two_tail)
            if param=='a_u':
                if u is not None:
                    val = np.nan
            if np.size(val)>1:
                #output[param] = np.moveaxis(np.full(np.moveaxis(master_shape, source=-2, destination=-1), misc_fns.list_1D(val, I_max=master_shape[-2])), source=-1, destination=-2)
                output[param] = np.full(master_shape, misc_fns.list_1D(val, I_max=master_shape[-1]))
            else:
                output[param] = np.full(master_shape, val)

        else:
            output[param] = misc_fns.fill_master_shape(val=val, master_shape=master_shape)


            
    return output








def calc_tilde_X_2D(X, u_dir=None, u_zero=None, match_params=None, master_shape=None, two_tail=None, carry_sign=None):

    if match_params is None:
        match_params = True

    if match_params:
        params_threshold = match_params_threshold_shape(u_dir=u_dir, u_zero=u_zero, master_shape=master_shape, two_tail=two_tail)
        u_dir =  params_threshold['u_dir']
        u_zero =  params_threshold['u_zero']

    u_dir = calc_u_dir_int_array(u_dir=u_dir)

    if carry_sign is None:
        carry_sign = False


    tilde_X = np.add.outer(-u_zero, X)
    tilde_X[u_dir==0,:] = np.abs(tilde_X[u_dir==0,:])
    tilde_X[u_dir==-1,:] = -tilde_X[u_dir==-1,:]


    if carry_sign:
        tilde_X[u_dir==0,:] *= np.sign(np.add.outer(-u_zero[u_dir==0], X))
        tilde_X[u_dir==-1,:] *= -1

    return tilde_X


def calc_tilde_X(X, u_dir=None, u_zero=None, match_params=None, master_shape=None, two_tail=None, carry_sign=None, reshape=None):

    if reshape is None:
        reshape = True

    if match_params is None:
        match_params = True

    if match_params:
        params_threshold = match_params_threshold_shape(u_dir=u_dir, u_zero=u_zero, master_shape=master_shape, two_tail=two_tail)
        u_dir =  params_threshold['u_dir']
        u_zero =  params_threshold['u_zero']

    u_dir = calc_u_dir_int_array(u_dir=u_dir)

    
    u_shape = np.shape(u_dir)
    T = np.size(X)

    N_i = np.product(u_shape)

    tilde_X = calc_tilde_X_2D(X=X, 
                                    u_dir=np.reshape(u_dir, (N_i)),
                                    u_zero=np.reshape(u_zero, (N_i)),
                                    carry_sign=carry_sign,
                                    match_params=False)
    
    if reshape:
        return np.reshape(tilde_X, np.concatenate([misc_fns.make_iterable_array(u_shape), misc_fns.make_iterable_array(T)]))
    else:
        return tilde_X



def calc_tilde_u(u=None, u_dir=None, match_params=None, master_shape=None, two_tail=None):

    if match_params is None:
        match_params = True

    if match_params:
        params_threshold = match_params_threshold_shape(u_dir=u_dir, u=u, master_shape=master_shape, two_tail=two_tail)
        u_dir =  params_threshold['u_dir']
        u =  params_threshold['u']

    u_dir = calc_u_dir_int_array(u_dir=u_dir)

    tilde_u = np.copy(u)

    tilde_u[u_dir==0] = np.abs(u[u_dir==0])
    tilde_u[u_dir==-1] = -u[u_dir==-1]

    return tilde_u




def calc_tilde_X_u(X, u_dir=None, u_zero=None, u=None, match_params=None, master_shape=None, two_tail=None, carry_sign=None, tilde_X=None):

    if match_params is None:
        match_params = True

    if match_params:
        params_threshold = match_params_threshold_shape(u_dir=u_dir, u_zero=u_zero, u=u, master_shape=master_shape, two_tail=two_tail)
        u_dir =  params_threshold['u_dir']
        u_zero =  params_threshold['u_zero']
        u =  params_threshold['u']

    u_dir = calc_u_dir_int_array(u_dir=u_dir)

    if carry_sign is None:
        carry_sign = False

    T = np.shape(X)[0]

    if tilde_X is None:
        tilde_X = calc_tilde_X(X, u_dir=u_dir, u_zero=u_zero, carry_sign=False, match_params=False, two_tail=two_tail)

    X_shape = np.array(np.shape(tilde_X))
    N_i = np.product(X_shape[:-1])
    T = X_shape[-1]
    
    tilde_X -= np.reshape(np.outer(np.reshape(calc_tilde_u(u=u, u_dir=u_dir, match_params=False), (N_i)), np.ones(T)), X_shape)


    if carry_sign:
        tilde_X[u_dir==0,:] *= np.sign(np.add.outer(-u_zero[u_dir==0], X))
        tilde_X[u_dir==-1,:] *= -1

    return tilde_X


def calc_I_u(X=None, u_dir=None, u_zero=None, u=None, u_inclusive=None, match_params=None, master_shape=None, two_tail=None, tilde_X_u=None):

    if match_params is None:
        match_params = True

    if match_params:
        params_threshold = match_params_threshold_shape(u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, master_shape=master_shape, two_tail=two_tail)
        u_dir =  params_threshold['u_dir']
        u_zero =  params_threshold['u_zero']
        u =  params_threshold['u']
        u_inclusive =  params_threshold['u_inclusive']

    u_dir = calc_u_dir_int_array(u_dir=u_dir)

    if tilde_X_u is None:
        tilde_X_u = calc_tilde_X_u(X=X, u_dir=u_dir, u_zero=u_zero, u=u, carry_sign=False)

    I_u = tilde_X_u > 0
    if np.any(u_inclusive):
        I_u[u_inclusive,:] = tilde_X_u[u_inclusive,:] >=0

    return I_u


def calc_a_lambda(X=None, train_t=None, train_T=None, u_dir=None, u_zero=None, u=None, u_inclusive=None, match_params=None, master_shape=None, two_tail=None, tilde_X_u=None, I_u=None):


    if I_u is None:
        if tilde_X_u is None:
            train_t = misc_fns.calc_train_t(train_t=train_t, X=X, train_T=train_T)
            I_u = calc_I_u(X=X[train_t[0]:train_t[1]], u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, match_params=match_params, master_shape=master_shape, two_tail=two_tail, tilde_X_u=None)
        else:
            train_t = misc_fns.calc_train_t(train_t=train_t, X=tilde_X_u, train_T=train_T)
            I_u = calc_I_u(X=None, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, match_params=match_params, master_shape=master_shape, two_tail=two_tail, tilde_X_u=tilde_X_u[...,train_t[0]:train_t[1]])

        train_t = misc_fns.calc_train_t(X=I_u)

    else:
        train_t = misc_fns.calc_train_t(train_t=train_t, X=I_u, train_T=train_T)

    return np.sum(I_u[...,train_t[0]:train_t[1]], axis=-1)/np.diff(train_t)[0]


def calc_u(X=None, train_t=None, train_T=None, a_u=None, u_dir=None, u_zero=None, match_params=None, master_shape=None, two_tail=None, tilde_X=None):

    if match_params is None:
        match_params = True

    if match_params:
        params_threshold = match_params_threshold_shape(a_u=a_u, u_dir=u_dir, u_zero=u_zero, master_shape=master_shape, two_tail=two_tail)
        a_u =  params_threshold['a_u']
        u_dir =  params_threshold['u_dir']
        u_zero =  params_threshold['u_zero']


    u_dir = calc_u_dir_int_array(u_dir=u_dir)

    if tilde_X is None:
        train_t = misc_fns.calc_train_t(train_t=train_t, X=X, train_T=train_T)
        tilde_X = calc_tilde_X(X=X[train_t[0]:train_t[1]], u_dir=u_dir, u_zero=u_zero, match_params=False)
        train_t = misc_fns.calc_train_t(X=tilde_X)
    else:
        train_t = misc_fns.calc_train_t(train_t=train_t, X=tilde_X, train_T=train_T)


    u = calc_row_quantiles_nb(tilde_X=tilde_X[...,train_t[0]:train_t[1]], q=1-a_u, match_params=False, reshape=True)
    u[u_dir==0] = np.abs(u[u_dir==0])
    u[u_dir==-1] = -u[u_dir==-1]

    return u


def calc_row_quantiles_nb(tilde_X, q, match_params=None, reshape=None):

    if reshape is None:
        reshape = True

    X_shape = np.array(np.shape(tilde_X))

    q = misc_fns.fill_master_shape(val=q, master_shape=X_shape[:-1])

    T = X_shape[-1]
    N_i = np.product(X_shape[:-1])

    u = thresh_nb.calc_row_quantiles_nb(tilde_X=np.reshape(tilde_X, (N_i,T)), q=np.reshape(q, (N_i)))

    if reshape:
        return np.reshape(u, X_shape[:-1])
    else:
        return u



def calc_tilde_X_u_nb(X, u_dir=None, u_zero=None, u=None, carry_sign=None, match_params=None, two_tail=None, reshape=None):

    if reshape is None:
        reshape = True

    if match_params is None:
        match_params = True

    if match_params:
        params_threshold = match_params_threshold_shape(u_dir=u_dir, u_zero=u_zero, u=u, two_tail=two_tail)
        u_dir =  params_threshold['u_dir']
        u_zero =  params_threshold['u_zero']
        u =  params_threshold['u']

    u_dir = calc_u_dir_int_array(u_dir=u_dir)

    if carry_sign is None:
        carry_sign = False

    u_shape = np.shape(u)
    T = np.size(X)

    N_i = np.product(u_shape)

    tilde_X_u = thresh_nb.calc_tilde_X_u_nb(X=X, 
                                                u_dir=np.reshape(u_dir, (N_i)),
                                                u_zero=np.reshape(u_zero, (N_i)),
                                                u=np.reshape(u, (N_i)), 
                                                )
    
    if carry_sign:
        tilde_X_u *= thresh_nb.calc_post_sign_nb(X=X, 
                                                u_dir=np.reshape(u_dir, (N_i)),
                                                u_zero=np.reshape(u_zero, (N_i)),
                                                carry_sign=carry_sign)

    if reshape:
        return np.reshape(tilde_X_u, np.concatenate([misc_fns.make_iterable_array(u_shape), misc_fns.make_iterable_array(T)]))
    else:
        return tilde_X_u



def calc_I_u_nb(X, u_dir=None, u_zero=None, u=None, u_inclusive=None, match_params=None, two_tail=None, reshape=None):

    if reshape is None:
        reshape = True

    if match_params is None:
        match_params = True

    if match_params:
        params_threshold = match_params_threshold_shape(u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, two_tail=two_tail)
        u =  params_threshold['u']
        u_dir =  params_threshold['u_dir']
        u_zero =  params_threshold['u_zero']
        u_inclusive =  params_threshold['u_inclusive']

    u_dir = calc_u_dir_int_array(u_dir=u_dir)

    u_shape = np.shape(u)
    T = np.size(X)

    N_i = np.product(u_shape)

    I_u = thresh_nb.calc_I_u_nb(X=X, 
                                    u_dir=np.reshape(u_dir, (N_i)),
                                    u_zero=np.reshape(u_zero, (N_i)),
                                    u=np.reshape(u, (N_i)), 
                                    u_inclusive=np.reshape(u_inclusive, (N_i)))
    

    if reshape:    
        return np.reshape(I_u, np.concatenate([misc_fns.make_iterable_array(u_shape), misc_fns.make_iterable_array(T)]))
    else:
        return I_u



def calc_tilde_X_func(X, u_dir=None, u_zero=None, u_dir_func=None, u_dir_func_inv=None, apply_u_dir_func_inc=None):

    return calc_tilde_X_func_u(X=X, u_dir=u_dir, u_zero=u_zero, u=0.0, u_dir_func=u_dir_func, u_dir_func_inv=u_dir_func_inv, apply_u_dir_func_inc=apply_u_dir_func_inc)


def calc_tilde_X_func_u(X, u_dir=None, u_zero=None, u=None, u_dir_func=None, u_dir_func_inv=None, apply_u_dir_func_inc=None):

    if apply_u_dir_func_inc is None:
        apply_u_dir_func_inc = False

    if u is None:
        u = 1.0

    if u_zero is None:
        u = 0.0

    if u_dir is None:
        u_dir = 1

    if u_dir_func is None:
        u_dir_func = calc_u_dir_func(u_dir=u_dir)

    tilde_X = u_dir_func(X-u_zero)          # - u_dir_func(u)

    if apply_u_dir_func_inc:
        if u_dir_func_inv is None:
            u_dir_func_inv = calc_u_dir_func_inv(u_dir=u_dir)
        return u_dir_func_inv(tilde_X)
    else:
        return tilde_X


def calc_I_u_func(X, u_dir=None, u_zero=None, u=None, u_inclusive=None, u_dir_func=None):

    if u_inclusive is None:
        u_inclusive = False
    
    if u_inclusive:
        I = calc_tilde_X_func_u(X=X, u_dir=u_dir, u_zero=u_zero, u=u, u_dir_func=u_dir_func) >= 0
    else:
        I = calc_tilde_X_func_u(X=X, u_dir=u_dir, u_zero=u_zero, u=u, u_dir_func=u_dir_func) > 0

    return I


def calc_signs_u(X, u_dir=None, u_zero=None):


    T = np.size(X)

    signs_u = np.ones(T)
    if np.isin(u_dir, ['left', -1]):
        signs_u *= -1
    elif u_dir=="abs":
        signs_u[X<u_zero] *= -1

    return signs_u



def calc_u_func(X, train_t=None, train_T=None, a_u=None, u_dir=None, u_zero=None, u_dir_func=None, u_dir_func_inv=None):

    samp = misc_fns.calc_samp_t(train_t=train_t, X=X, train_T=train_T)

    if u_dir_func is None:
        u_dir_func = calc_u_dir_func(u_dir=u_dir)
    if u_dir_func_inv is None:
        u_dir_func_inv = calc_u_dir_func_inv(u_dir=u_dir)

    tilde_X = calc_tilde_X_func(X=samp, u_dir=u_dir, u_zero=u_zero, u_dir_func=u_dir_func, apply_u_dir_func_inc=False)

    return u_dir_func_inv(np.quantile(tilde_X, 1-a_u))


def calc_a_u_func(X, train_t=None, train_T=None, u_dir=None, u_zero=None, u=None, u_inclusive=None, u_dir_func=None):
    
    samp = misc_fns.calc_samp_t(train_t=train_t, X=X, train_T=train_T)

    return np.sum(calc_I_u_func(X=samp, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, u_dir_func=u_dir_func))/np.size(samp)





def calc_events_base_old(X, a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None, match_params=None, two_tail=None, set_a_u=None, set_u=None, train_t=None, train_T=None, overwrite_tilde_X=None, include_X=None):


    if overwrite_tilde_X:
        overwrite_tilde_X = False

    if include_X:
        include_X = True

    if set_a_u is None:
        if a_u is None and u is not None:
            set_a_u = True
        else:
            set_a_u = False

    if set_u is None:
        if not set_a_u and u is None and a_u is not None:
            set_u = True
        else:
            set_u = False


    if match_params is None:
        match_params = True

    if match_params:
        params_threshold = match_params_threshold_shape(a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, two_tail=two_tail)
    else:
        params_threshold = calc_params_threshold_dict(a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive)


    events_base =   {
                        **params_threshold,
                    }
    
    events_base['u_dir_int'] = calc_u_dir_int_array(events_base['u_dir'])
    

    if overwrite_tilde_X:
        tilde_X = 'tilde_X_u'
    else:
        tilde_X = 'tilde_X'

    events_base['T'] = np.shape(X)[-1]
    events_base['1:T+1'] = 1 + np.arange(events_base['T'])

    events_base['u_shape'] = np.array(np.shape(events_base['u']))


    events_base['all_t'] = np.array([0, events_base['T']])
    events_base['train_t'] = misc_fns.calc_train_t(train_t=train_t, train_T=train_T, X=X)
    events_base['forecast_t'] = np.array([events_base['train_t'][-1], events_base['all_t'][-1]])

    events_base[tilde_X] = calc_tilde_X(X=X, u_dir=events_base['u_dir'], u_zero=events_base['u_zero'], match_params=False, carry_sign=False, reshape=True)
    if set_u:
        events_base['u'] = calc_u(tilde_X=events_base['tilde_X'], train_t=events_base['train_t'], train_T=None, a_u=events_base['a_u'], u_dir=events_base['u_dir'], u_zero=events_base['u_zero'], match_params=False)
    events_base['tilde_X_u'] = calc_tilde_X_u(X=X, u_dir=events_base['u_dir'], u_zero=events_base['u_zero'], u=events_base['u'], match_params=False, carry_sign=False, tilde_X=events_base[tilde_X])
    events_base['I_u'] = calc_I_u(X=X, u_dir=events_base['u_dir'], u_zero=events_base['u_zero'], u=events_base['u'], u_inclusive=events_base['u_inclusive'], match_params=False, tilde_X_u=events_base['tilde_X_u'])

    events_base['a_lambda_train'] = calc_a_lambda(X=X, train_t=events_base['train_t'], u_dir=events_base['u_dir'], u_zero=events_base['u_zero'], u=events_base['u'], u_inclusive=events_base['u_inclusive'], match_params=False, tilde_X_u=events_base['tilde_X_u'], I_u=events_base['I_u'])
    events_base['a_lambda'] = calc_a_lambda(X=X, u_dir=events_base['u_dir'], u_zero=events_base['u_zero'], u=events_base['u'], u_inclusive=events_base['u_inclusive'], match_params=False, tilde_X_u=events_base['tilde_X_u'], I_u=events_base['I_u'])
    if set_a_u:
        events_base['a_u'] = events_base['a_lambda_train']

    events_base['t'] = misc_fns_nb.where_X_bool(X_bool=events_base['I_u'], axis=-1)

    

    if include_X:
        events_base['X'] = X



    return events_base


def calc_events_base(X, a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None, match_params=None, master_shape=None, two_tail=None, set_a_u=None, set_u=None, train_t=None, train_T=None, include_X=None):


    if include_X:
        include_X = True

    if set_a_u is None:
        if a_u is None and u is not None:
            set_a_u = True
        else:
            set_a_u = False

    if set_u is None:
        if not set_a_u and u is None and a_u is not None:
            set_u = True
        else:
            set_u = False


    if match_params is None:
        match_params = True

    if match_params:
        params_threshold = match_params_threshold_shape(a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, master_shape=master_shape, two_tail=two_tail)
    else:
        params_threshold = calc_params_threshold_dict(a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive)


    events_base =   {
                        **params_threshold,
                    }
    
    events_base['u_dir_int'] = calc_u_dir_int_array(events_base['u_dir'])
    

    events_base['T'] = np.shape(X)[-1]
    events_base['1:T+1'] = 1 + np.arange(events_base['T'])

    events_base['u_shape'] = np.array(np.shape(events_base['u']))


    events_base['all_t'] = np.array([0, events_base['T']])
    events_base['train_t'] = misc_fns.calc_train_t(train_t=train_t, train_T=train_T, X=X)
    events_base['forecast_t'] = np.array([events_base['train_t'][-1], events_base['all_t'][-1]])

    events_base['tilde_X'], events_base['tilde_X_u'], events_base['I_u'], events_base['u'] = calc_events_base_X(X=X,
                                                                                                                        set_u=set_u, train_t=events_base['train_t'], train_T=None, reshape=True,
                                                                                                                        **params_threshold)
    

    events_base['a_lambda_train'] = calc_a_lambda(X=X, train_t=events_base['train_t'], u_dir=events_base['u_dir'], u_zero=events_base['u_zero'], u=events_base['u'], u_inclusive=events_base['u_inclusive'], match_params=False, tilde_X_u=events_base['tilde_X_u'], I_u=events_base['I_u'])
    events_base['a_lambda'] = calc_a_lambda(X=X, u_dir=events_base['u_dir'], u_zero=events_base['u_zero'], u=events_base['u'], u_inclusive=events_base['u_inclusive'], match_params=False, tilde_X_u=events_base['tilde_X_u'], I_u=events_base['I_u'])
    if set_a_u:
        events_base['a_u'] = events_base['a_lambda_train']

    events_base['t'] = misc_fns_nb.where_X_bool(X_bool=events_base['I_u'], axis=-1)

    

    if include_X:
        events_base['X'] = X



    return events_base


def calc_events_base_X(X, a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None, match_params=None, master_shape=None, two_tail=None, 
                                set_u=None, train_t=None, train_T=None, reshape=None):
    

    if match_params is None:
        match_params = True

    if match_params:
        params_threshold = match_params_threshold_shape(a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, master_shape=master_shape, two_tail=two_tail)
    else:
        params_threshold = calc_params_threshold_dict(a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive)


    params_threshold['u_dir'] = calc_u_dir_int_array(u_dir=params_threshold['u_dir'])
    for param in ['a_u', 'u_zero', 'u']:
        params_threshold[param] = np.array(params_threshold[param], dtype=np.float64)

    if reshape is None:
        reshape = True

    u_shape = np.array(np.shape(u_dir))
    T = np.shape(X)[0]
    N_i = np.product(u_shape)

    train_t = misc_fns.calc_train_t(train_t=train_t, train_T=train_T, X=X)
    
    
    tilde_X, tilde_X_u, I_u, u = thresh_nb.calc_events_base_X_nb(X=X, a_u=np.reshape(params_threshold['a_u'], (N_i)), u_dir=np.reshape(params_threshold['u_dir'], (N_i)), u_zero=np.reshape(params_threshold['u_zero'], (N_i)), u=np.reshape(params_threshold['u'], (N_i)), u_inclusive=np.reshape(params_threshold['u_inclusive'], (N_i)),
                                                                   set_u=set_u, train_t=train_t)

    if reshape:
        X_shape = np.concatenate([u_shape, [T]])
        return np.reshape(tilde_X, X_shape), np.reshape(tilde_X_u, X_shape), np.reshape(I_u, X_shape), np.reshape(u, u_shape)
    else:
        return tilde_X, tilde_X_u, I_u, u



def calc_events_field(field, X=None, a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None, match_params=None, two_tail=None, set_a_u=None, set_u=None, train_t=None, train_T=None,
                          events_base=None):
    
    if events_base is None:
        events_base = calc_events_base(X=X, a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, match_params=match_params, two_tail=two_tail, set_a_u=set_a_u, set_u=set_u, train_t=train_t, train_T=train_T, include_X=False)

    a_fields = ['a_lambda', 'a_lambda_train', 'a_u']
    period_fields = ['all', 'train', 'forecast']

    if field=='N_t':
        return np.cumsum(events_base['I_u'], axis=-1)
    elif field=='N':
        return np.sum(events_base['I_u'], axis=-1)
    elif field=='t':
        return events_base['t']
    elif field=='M':
        return misc_fns_nb.filter_X_bool(X=events_base['tilde_X_u'], X_bool=events_base['I_u'], X_nan=np.nan, X_nan_zero=None, axis=-1)
    elif field=='delta_t':
        return misc_fns.diff_zero(events_base['t'], axis=-1)
    elif np.isin(field, [f"{p}_t" for p in period_fields]):
        return events_base[field]
    elif np.isin(field, [f"{p}_N" for p in period_fields]):
        field_eff = f'{field[:-2]}_t'
        N_0 = np.sum(events_base['I_u'][...,:events_base[field_eff][0]], axis=-1)
        N_1 = np.sum(events_base['I_u'][...,:events_base[field_eff][1]], axis=-1)
        return np.moveaxis([N_0, N_1], source=0, destination=-1)

        #return np.moveaxis([np.sum(events_base['I_u'][...,:events_base[f'{field[:-2]}_t'][0]], axis=-1), np.sum(events_base['I_u'][...,:events_base[f'{field[:-2]}_t'][1]], axis=-1)], source=0, destination=-1)
    elif np.isin(field, a_fields):
        return events_base[field]
    elif np.isin(field, [f"lambda_{a}" for a in a_fields]):
        return np.outer(events_base[field[7:]], np.ones(events_base['T']))
    elif np.isin(field, [f"tilde_t_{a}" for a in a_fields]):
        return np.outer(events_base[field[8:]], events_base['1:T+1'])
    else:
        return None
    



def calc_events(X, a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None, match_params=None, master_shape=None, two_tail=None, set_a_u=None, set_u=None, train_t=None, train_T=None,
                    fields=None, fields_rm=None, in_dict=None):

    if in_dict is None:
        in_dict = False

    events_base = calc_events_base(X=X, a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, match_params=match_params, master_shape=master_shape, two_tail=two_tail, set_a_u=set_a_u, set_u=set_u, train_t=train_t, train_T=train_T, include_X=False)

    a_fields = ['a_lambda', 'a_lambda_train', 'a_u']
    period_fields = ['all', 'train', 'forecast']


    

    fields_default = np.concatenate([['N', 't', 'N_t', 'M', 'delta_t'], 
                                     np.concatenate([[f"{p}{post}" for p in period_fields] for post in ['_t', '_N']]),
                                     np.concatenate([[f"{pre}{a}" for a in a_fields] for pre in ['', 'lambda_']]),
                                        ])
    if fields is None:
        fields = fields_default

    fields = misc_fns.A_rm(A=fields, A_in=fields_default, A_rm=fields_rm)

    if not in_dict and np.size(fields)==1:
        return calc_events_field(events_base=events_base, field=fields[0])
    else:
        return {field: calc_events_field(events_base=events_base, field=field) for field in fields}
    




def calc_end_dims_events(events):

    return {'T': np.shape(events['N_t'])[-1], 'N': np.shape(events['M'])[-1], 'p_N': 2}

def calc_shape_events(events):

    return misc_fns.make_iterable_array(np.shape(events['N']), dtype=int)


def reshape_events(events, reshape=None):

    shape = calc_shape_events(events=events)
    end_dims = calc_end_dims_events(events=events)

    period_fields = ['all', 'train', 'forecast']

    no_reshape = np.concatenate([
                                     np.concatenate([[f"{p}{post}" for p in period_fields] for post in ['_t']]),
                                        ])

    return misc_fns.calc_reshape_dict(dic=events, shapes=None, shape=shape, reshape=reshape, end_dims=end_dims, no_reshape=no_reshape)



def calc_events_cat(events, I_s_cat_i=None, I_s_cat_I=None, do_reshape=None, reshape=None):



    a_fields = ['a_lambda', 'a_lambda_train', 'a_u']
    period_fields = ['all', 'train', 'forecast']

    events_shape = calc_shape_events(events)
    N_i = np.product(events_shape)

    if I_s_cat_i is None:
        if I_s_cat_I is not None:
            I_s_cat_i = misc_fns.calc_N_index(index=I_s_cat_I, shape=events_shape)
        else:
            #I_s_cat_i = [[]]
            
            
            if do_reshape is None:
                do_reshape = True
            if reshape is None:
                if np.size(events_shape)==1:
                    reshape = np.array([1])
                else:
                    reshape = events_shape[:-1]

            I_s_cat_i = np.reshape(np.arange(N_i), np.concatenate([misc_fns.make_iterable_array(np.product(events_shape[:-1])), misc_fns.make_iterable_array(events_shape[-1])]))


            

    
    T = np.shape(events['N_t'])[-1]




    N_i_cat = len(I_s_cat_i)

    events_2D = reshape_events(events=events, reshape=np.array([N_i]))

    events_N_max = np.shape(events_2D['t'])[-1] 

    events_cat = calc_events(X=np.full(T, np.nan), train_t=events['train_t'], master_shape=np.array([N_i_cat]), two_tail=False)

    events_cat['N'] = np.array([np.sum(events_2D['N'][I_s_cat_i[i]]) for i in range(N_i_cat)])
    events_N_cat_max = np.max(events_cat['N'])
    events_cat_N_shape = misc_fns.make_iterable_array([N_i_cat, events_N_cat_max])

    N_fields_t = ['t', 'delta_t']
    for field in N_fields_t:
        events_cat[field] = np.full(events_cat_N_shape, -T)
    N_fields_float = ['M']
    for field in N_fields_float:
        events_cat[field] = np.full(events_cat_N_shape, np.nan)


    order = [np.argsort(np.concatenate(events_2D['t'][I_s_cat_i[i],:]))[-events_cat['N'][i]:] for i in range(N_i_cat)]
    order_ij = [[I_s_cat_i[i][order[i]//events_N_max] , order[i]%events_N_max] for i in range(N_i_cat)]

    order_list = ['t', 'M',
                           #'TS', 
                    #'tilde_t', 'tilde_t_eff', 'tilde_t_all', 
                    #'N', 'N_t', 
                    # 'list', 't_unique', 'tilde_t_t_unique'
                    ]
    
    order_source_list = [
                            #'delta_t', 
                           #'tilde_M', 'F_M', 'kappa', 
                           #'sigma', 'lambda', 
                            #'tilde_t', 
                    # 'list', 't_unique', 'tilde_t_t_unique'
                    ]
    
    for field in order_source_list:
       events_cat[f"source_{field}"] = np.full(events_cat_N_shape, np.nan)


    for i in range(N_i_cat):
        I_s = I_s_cat_i[i]

        for field in np.concatenate([a_fields]):
            events_cat[field][i] = np.sum(events_2D[field][I_s])

        for field in np.concatenate([['N_t'], 
                                     np.concatenate([[f"{p}{post}" for p in period_fields] for post in ['_N']]), 
                                     np.concatenate([[f"{pre}{a}" for a in a_fields] for pre in ['lambda_']]),
                                     ]):
            events_cat[field][i,:] = np.sum(events_2D[field][I_s], axis=0)

        for field in order_list:
            for k in I_s:
                events_cat[field][i,np.where(order_ij[i][0]==k)[0]] = events_2D[field][k,:events_2D['N'][k]]

        for field in order_source_list:
            for k in I_s:
                events_cat[f"source_{field}"][i,np.where(order_ij[i][0]==k)[0]] = events_2D[field][k,:events_2D['N'][k]]
        
    events_cat['delta_t'] = misc_fns.diff_zero(events_cat['t'], axis=-1)

    if do_reshape is None:
        do_reshape = False

    if do_reshape and reshape is not None:
        return reshape_events(events=events_cat, reshape=reshape)
    else:
        return events_cat

    
    




######################################################################################################################

def calc_u_a_u(X, a_u=None):

    if a_u is None:
        a_u = 0.025
    a_u = misc_fns.make_iterable_array(a_u)

    return np.quantile(X, 0.5 + np.outer(np.array([-1,1]), (0.5-a_u)))


def calc_events_old(X, a_u=None, u_dir=None, u_zero=None, u=None, override_u2=None, carry_sign=None, ret_val=None, as_array=None, first_u=None):

    if override_u2 is None:
        override_u2 = True

    if as_array is None:
        as_array = False

    if first_u is None:
        first_u = False

    if carry_sign is None:
        carry_sign = False

    if a_u is None and u is None:
        a_u = 0.025
    elif a_u is None and u is not None:
        a_u = np.nan
    a_u = misc_fns.make_iterable_array(a_u)

    if u is None:
        u = calc_u_a_u(X=X, a_u=a_u)

    if override_u2:
        if len(np.shape(u))==1 and np.size(u)==2:
            u = np.array([[u[k]] for k in range(2)])

    if u_dir is None:
        if len(np.shape(u))==2:
            u_dir = np.array([-1,1])
        else:
            u_dir = 1
    u_dir = misc_fns.make_iterable_array(u_dir)

    if u_zero is None:
        if len(np.shape(u))==2:
            u_zero = np.repeat(0.0,2)
        else:
            u_zero = 0.0
    u_zero = misc_fns.make_iterable_array(u_zero)

    events_list = ['t', 'tilde_t_a_u', 'tilde_t_a_lambda', 'delta_t', 'M']

    T_max = np.size(X)


    if len(np.shape(u))==0:
        
        events = {val: None for val in events_list}


        u_func = calc_u_dir_func(dir=u_dir[0])
        if carry_sign:
            u_func_inv = u_func
        else:
            u_func_inv = calc_u_dir_func(dir=1)
        zero = u_zero[0]

        events['N'] = 0
        events['a_lambda'] = np.nan
        events['N_t'] = np.zeros(T_max, dtype=int)


        events['t'] = np.where(sign*X>sign*u)[0]
        events['N'] = np.size(events['t'])
        if T_max>0:
            events['a_lambda'] = events['N']/T_max

        events['tilde_t_a_u'] = events['t'] * a_u[0]
        events['tilde_t_a_lambda'] = events['t'] * events['a_lambda']

        
        events['M'] = post_sign*(sign*(X[events['t']]-zero) - sign*u)
        if events['N']>0:
            events['delta_t'] = misc_fns.diff_zero(events['t'], dtype=int)
            for t in events['t']:
                events['N_t'][t:] += 1
        else:
             events['delta_t'] = []


        events['N'] = np.size(events['t'])

        if as_array:
            ret_events = {val: np.array([events[val]]) for val in events}
        else:
            ret_events = events

    elif len(np.shape(u))==1:

        I_max = np.shape(u)[0]


        events = {val: [None for i in range(I_max)] for val in events_list}

        events['N'] = np.zeros(I_max, dtype=int)
        events['a_lambda'] = np.full(I_max, np.nan)
        events['N_t'] = np.zeros((I_max, T_max), dtype=int)

        for i in range(I_max):
            sign = u_dir[i]
            if carry_sign:
                post_sign = sign
            else:
                post_sign = 1
            zero = u_zero[i]

            events['t'][i] = np.where(sign*X>sign*u[i])[0]
            events['N'][i] = np.size(events['t'][i])
            if T_max>0:
                events['a_lambda'][i] = events['N'][i]/T_max


            events['tilde_t_a_u'][i] = events['t'][i] * a_u[i%a_u.size]
            events['tilde_t_a_lambda'][i] = events['t'][i] * events['a_lambda'][i]

            
            events['M'][i] = post_sign*(sign*(X[events['t'][i]]-zero) - sign*u[i])
            if events['N'][i]>0:
                events['delta_t'][i] = misc_fns.diff_zero(events['t'][i], dtype=int)
                for t in events['t'][i]:
                    events['N_t'][i,t:] += 1
            else:
                events['delta_t'][i] = []
        
        if as_array:
            N_max = np.max(events['N'])
            ret_events = {val: np.array([[np.nan for n in range(N_max)] for i in range(I_max)]) for val in events_list}
            ret_events['N'] = events['N']
            ret_events['a_lambda'] = events['a_lambda']
            ret_events['N_t'] = events['N_t']
            ret_events['t'] = np.full((I_max, N_max), -T_max, dtype=int)
            ret_events['delta_t'] = np.full((I_max, N_max), -T_max, dtype=int)
            for val in events_list:
                for i in range(I_max):
                    ret_events[val][i,:ret_events['N'][i]] = np.array(events[val][i])
        else:
            ret_events = events


    elif len(np.shape(u))==2:

        I_max = np.shape(u)[0]
        J_max = np.shape(u)[1]

        events = {val: [[None for j in range(J_max)] for i in range(I_max)] for val in events_list}


        events['N'] = np.zeros((I_max, J_max), dtype=int)
        events['a_lambda'] = np.full((I_max, J_max), np.nan)
        events['N_t'] = np.zeros((I_max, J_max, T_max), dtype=int)


        for i in range(I_max):
            sign = u_dir[i]
            if carry_sign:
                post_sign = sign
            else:
                post_sign = 1
            zero = u_zero[i]

            for j in range(J_max):
                events['t'][i][j] = np.where(sign*X>sign*u[i,j])[0]
                events['N'][i][j] = np.size(events['t'][i][j])

                if T_max>0:
                    events['a_lambda'][i][j] = events['N'][i][j]/T_max

                events['tilde_t_a_u'][i][j] = events['t'][i][j] * a_u[i%a_u.size]
                events['tilde_t_a_lambda'][i][j] = events['t'][i][j] * events['a_lambda'][i][j]

                events['M'][i][j] = post_sign*(sign*(X[events['t'][i][j]]-zero) - sign*u[i,j])

                if events['N'][i][j] > 0:
                    events['delta_t'][i][j] = misc_fns.diff_zero(events['t'][i][j], dtype=int)
                    for t in events['t'][i][j]:
                        events['N_t'][i,j,t:] += 1
                else:
                    events['delta_t'][i][j] = []

        if as_array:
            N_max = np.max(events['N'])
            ret_events = {val: np.array([[[np.nan for n in range(N_max)] for j in range(J_max)] for i in range(I_max)]) for val in events_list}
            ret_events['N'] = events['N']
            ret_events['a_lambda'] = events['a_lambda']
            ret_events['N_t'] = events['N_t']
            ret_events['t'] = np.full((I_max, J_max, N_max), -T_max, dtype=int)
            ret_events['delta_t'] = np.full((I_max, J_max, N_max), -T_max, dtype=int)
            for val in events_list:
                for i in range(I_max):
                    for j in range(J_max):
                        ret_events[val][i,j,:ret_events['N'][i][j]] = np.array(events[val][i][j])
        else:
            ret_events = events

    if first_u:
        if as_array:
            ret_events_2 = {val: ret_events[val][:,0] for val in ret_events}
        else:
            if len(np.shape(u))==2:
                ret_events_2 = {val: [ret_events[val][k][0] for k in range(2)] for val in ret_events}
            elif len(np.shape(u))==1:
                ret_events_2 = {val: [ret_events[val][0]] for val in ret_events}
            else:
                ret_events_2 = {val: [ret_events[val]] for val in ret_events}
    else:
        ret_events_2 = ret_events
            
    if ret_val=='t':
        return ret_events_2['t']
    if ret_val=='tilde_t_a_u':
        return ret_events_2['tilde_t_a_u']
    if ret_val=='tilde_t_a_lambda':
        return ret_events_2['tilde_t_a_lambda']
    elif ret_val=='delta_t':
        return ret_events_2['delta_t']  
    elif ret_val=='M':
        return ret_events_2['M']  
    elif ret_val=='N':
        return ret_events_2['N']
    elif ret_val=='a_lambda':
        return ret_events_2['a_lambda']
    elif ret_val=='N_t':
        return ret_events_2['N_t'] 
    else: 
        return ret_events_2






def calc_events_cat_old(events, I_s_cat_list=None, tilde_t=None):

    I_max = events['t'].shape[0]

    if I_s_cat_list is None:
        I_s_cat_list = calc_I_s_cat_list_default(I_max=I_max)

    K_max = len(I_s_cat_list) 

    events_cat = dict()
    events_cat['N'] = np.array([np.sum(events['N'][I_s_cat_list[k]]) for k in range(K_max)], dtype=int)


    N_max = np.max(events_cat['N'])
    I_void = np.argmin(events['N'])

    N_shape_events = np.shape(events['t'])[1]
    T = np.shape(events['N_t'])[1]


    order = [np.argsort(np.concatenate(events['t'][I_s_cat_list[k],:]))[-events_cat['N'][k]:] for k in range(K_max)]
    order_ij = [[I_s_cat_list[k][order[k]//N_shape_events] , order[k]%N_shape_events] for k in range(K_max)]

    e_list = [e for e in events]

    for c in ['t', 'delta_t', 'TS', 'M', 'tilde_M', 'F_M', 'kappa', 
                    #'tilde_t', 'tilde_t_eff', 'tilde_t_all', 
                    'sigma', 'lambda', 
                    #'N', 'N_t', 
                    # 'list', 't_unique', 'tilde_t_t_unique'
                    ]:

        if np.isin(c, e_list):
            events_cat[c] = np.full((K_max, N_max), events[c][I_void,-1])
            for k in range(K_max):
                for i in I_s_cat_list[k]:
                    events_cat[c][k,np.where(order_ij[k][0]==i)[0]] = events[c][i,:events['N'][i]]


    for c in [
                'tilde_t_a_u', 'tilde_t_a_lambda', 'N_t',
                ]:

        if np.isin(c, e_list):
            events_cat[c] = np.array([np.sum(events[c][I_s_cat_list[k],:], axis=0) for k in range(K_max)])


    for c in [
                'tilde_t',
                ]:
        
        if np.isin(c, e_list):
            events_cat[c] = np.full((K_max, N_max), events[c][I_void,-1])
            tilde_t_cat = calc_tilde_t_cat(tilde_t=tilde_t, I_max=I_max, T=T, K_max=K_max, I_s_cat_list=I_s_cat_list)
            if tilde_t is not None:
                for k in range(K_max):
                    events_cat[c][k,:events_cat['N'][k]] = tilde_t_cat[k,events_cat['t'][k,:events_cat['N'][k]]]

    return events_cat

def calc_I_s_cat_list_default(I_max):

    return [np.arange(I_max)]

def calc_tilde_t_cat(tilde_t=None, I_max=None, T=None, K_max=None, I_s_cat_list=None):

    if tilde_t is not None:
        set_tilde_t = True
        I_max = np.shape(tilde_t)[0]
        T = np.shape(tilde_t)[1]
    else:
        set_tilde_t = False
        if I_max is None:
            I_max = 0
        if T is None:
            T = 0

    if I_s_cat_list is None:
        I_s_cat_list = calc_I_s_cat_list_default(I_max=I_max)
    
    K_max = len(I_s_cat_list)

    tilde_t_cat = np.full((K_max, T), np.nan)
    if set_tilde_t:
        for k in range(K_max):
            tilde_t_cat[k,:] = np.sum(tilde_t[I_s_cat_list[k], :], axis=0)

    return tilde_t_cat






def calc_events_M(X, a_u=None, u_dir=None, u_zero=None, u=None, override_u2=None, carry_sign=None):

    return calc_events_old(X, a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, override_u2=override_u2, carry_sign=carry_sign, ret_val="M")

def calc_events_t(X=None, a_u=None, u_dir=None, u_zero=None, u=None, override_u2=None):

    return calc_events_old(X=X, a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, override_u2=override_u2, ret_val="t")

def calc_events_delta_t(X=None, a_u=None, u_dir=None, u_zero=None, u=None, override_u2=None):
    
    return calc_events_old(X=X, a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, override_u2=override_u2, ret_val="delta_t")

def calc_events_N(X=None, a_u=None, u_dir=None, u_zero=None, u=None, override_u2=None):

    return calc_events_old(X=X, a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, override_u2=override_u2, ret_val="N")

def calc_events_N_t(X=None, a_u=None, u_dir=None, u_zero=None, u=None, override_u2=None):
    
    return calc_events_old(X=X, a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, override_u2=override_u2, ret_val="N_t")

def calc_tail_fit(X, a_u=None, u_dir=None, u_zero=None, u=None, as_array=None, carry_sign=None, incl_M_stat=None, incl_gp_fit=None, fa_lambda=None, fscale=None, fc=None, fNparams_free=None):


    if incl_M_stat is None:
        incl_M_stat = True
    
    if incl_gp_fit is None:
        incl_gp_fit = True


    if as_array is None:
        as_array = False

    if carry_sign is None:
        carry_sign = False

    


    if a_u is None and u is None:
        a_u = 0.025
    elif a_u is None and u is not None:
        a_u = np.nan

    if u is None:
        u = calc_u_a_u(X=X, a_u=a_u)

    if u_dir is None:
        if len(np.shape(u))==2:
            u_dir = np.array([-1,1])
        else:
            u_dir = 1
    u_dir = misc_fns.make_iterable_array(u_dir)

    if u_zero is None:
        if len(np.shape(u))==2:
            u_zero = np.repeat(0.0,2)
        else:
            u_zero = 0.0
    u_zero = misc_fns.make_iterable_array(u_zero)

    
    fields = {
                'lambda': ['T', 'a_u', 'u', 'u_dir', 'N', 'a_lambda_emp', 'a_lambda', 'log_L', 'N_params', 'AIC', 'N_params_free', 'AIC_free'],
                'M_stat': ['mean', 'std'],
                'gp_fit': ['est', 'se', 'log_L', 'N_params', 'AIC', 'N_params_free', 'AIC_free'],
            }

    incl_fields = ['lambda']


    if incl_M_stat:
        incl_fields.append('M_stat')
    if incl_gp_fit:
        incl_fields.append('gp_fit')

    M_fit = {field: {val: np.full(np.shape(u), np.nan) for val in  fields[field]} for field in incl_fields}
    
    M_fit['lambda']['T'] = np.full(np.shape(u), np.size(X))
    M_fit['lambda']['N'] = np.full(np.shape(u), -1)
    M_fit['lambda']['u'] = u
    M_fit['lambda']['u_dir'] = np.outer(u_dir, np.ones(np.shape(u)[1], dtype=int))
    if incl_gp_fit:
        for val in ['N_params', 'N_params_free']:
            M_fit['gp_fit'][val] = np.full(np.shape(u), np.nan)
        for val in ['est', 'se']:
            M_fit['gp_fit'][val] = np.full(np.concatenate([np.shape(u), [3]]), np.nan)


    



    if len(np.shape(u))==2:

        a_u_ij = misc_fns.list_2D(a_u, I_max=u.shape[0], J_max=u.shape[1], set_J_max=None, mode_1D=None, as_array=True)
        fscale_ij = misc_fns.list_2D(fscale, I_max=u.shape[0], J_max=u.shape[1], set_J_max=None, mode_1D=None, as_array=True)
        fc_ij = misc_fns.list_2D(fc, I_max=u.shape[0], J_max=u.shape[1], set_J_max=None, mode_1D=None, as_array=True)
        fNparams_free_ij = misc_fns.list_2D(fNparams_free, I_max=u.shape[0], J_max=u.shape[1], set_J_max=None, mode_1D=None, as_array=True)
        u_zero_ij = misc_fns.list_2D(u_zero, I_max=u.shape[0], J_max=u.shape[1], set_J_max=None, mode_1D=None, as_array=True)

        M_fit['lambda']['a_u'] = a_u_ij

        for i in range(u.shape[0]):

            sign = u_dir[i]
            if carry_sign:
                post_sign = u_dir[i]
            else:
                post_sign = 1


            if False:

                if len(np.shape(fscale))==0:
                    fscale_i = np.repeat(fscale, u.shape[1])
                elif len(np.shape(fscale))==1:
                    if np.size(fscale)==1:
                        fscale_i = np.repeat(fscale[0], u.shape[1])
                    else:
                        fscale_i = fscale
                else:
                    fscale_i = misc_fns.make_iterable_array(fscale[i])

            for j in range(u.shape[1]):

                #M = sign*X[sign*X>sign*u[i,j]] - sign*
                M = calc_events_M(X=X, u=u[i,j], u_dir=sign, u_zero=u_zero_ij[i,j], carry_sign=False)

                M_fit['lambda']['N'][i,j] = M.size
                if incl_M_stat:
                    M_fit['M_stat']['mean'][i,j] = post_sign*np.mean(M)
                    M_fit['M_stat']['std'][i,j] = np.std(M)
                if incl_gp_fit:
                    gp_fit = dist_fns.calc_scipy_dist_est_se_log_L(data=M, dist='genpareto', floc=0, fscale=fscale_ij[i,j], fc=fc_ij[i,j], fNparams_free=fNparams_free_ij[i,j])
                    for val in ['est', 'se']:
                        M_fit['gp_fit'][val][i,j,:] = gp_fit[val]
                    for val in ['log_L', 'N_params', 'AIC', 'N_params_free', 'AIC_free']:
                        M_fit['gp_fit'][val][i,j] = gp_fit[val]


    elif len(np.shape(u))==1:

        a_u_ij = misc_fns.list_2D(a_u, I_max=u.shape[0], J_max=1, set_J_max=None, mode_1D=None, as_array=True)
        fscale_ij = misc_fns.list_2D(fscale, I_max=u.shape[0], J_max=1, set_J_max=None, mode_1D=None, as_array=True)
        fc_ij = misc_fns.list_2D(fc, I_max=u.shape[0], J_max=1, set_J_max=None, mode_1D=None, as_array=True)
        fNparams_free_ij = misc_fns.list_2D(fNparams_free, I_max=u.shape[0], J_max=1, set_J_max=None, mode_1D=None, as_array=True)
        u_zero_ij = misc_fns.list_2D(u_zero, I_max=u.shape[0], J_max=1, set_J_max=None, mode_1D=None, as_array=True)

        M_fit['lambda']['a_u'] = a_u_ij[:,0]

        for i in range(u.shape[0]):

            sign = u_dir[i]
            if carry_sign:
                post_sign = u_dir[i]
            else:
                post_sign = 1


            #M = sign*X[sign*X>sign*u[i]] - sign*u[i]
            M = calc_events_M(X=X, u=u[i], u_dir=sign, u_zero=u_zero_ij[i,0], carry_sign=False)

            M_fit['lambda']['N'][i] = M.size
            if incl_M_stat:
                M_fit['M_stat']['mean'][i] = post_sign*np.mean(M)
                M_fit['M_stat']['std'][i] = np.std(M)
            if incl_gp_fit:
                gp_fit = dist_fns.calc_scipy_dist_est_se_log_L(data=M, dist='genpareto', floc=0, fscale=fscale_ij[i,0], fc=fc_ij[i,0], fNparams_free=fNparams_free[i,0])
                for val in ['est', 'se']:
                    M_fit['gp_fit'][val][i,:] = gp_fit[val]
                for val in ['log_L', 'N_params', 'AIC', 'N_params_free', 'AIC_free']:
                    M_fit['gp_fit'][val][i] = gp_fit[val]

    else:

        a_u_ij = misc_fns.list_2D(a_u, I_max=1, J_max=1, set_J_max=None, mode_1D=None, as_array=True)
        fscale_ij = misc_fns.list_2D(fscale, I_max=1, J_max=1, set_J_max=None, mode_1D=None, as_array=True)
        fc_ij = misc_fns.list_2D(fc, I_max=1, J_max=1, set_J_max=None, mode_1D=None, as_array=True)
        fNparams_free_ij = misc_fns.list_2D(fNparams_free, I_max=1, J_max=1, set_J_max=None, mode_1D=None, as_array=True)
        u_zero_ij = misc_fns.list_2D(u_zero, I_max=1, J_max=1, set_J_max=None, mode_1D=None, as_array=True)

        sign = u_dir[0]
        if carry_sign:
            post_sign = u_dir[0]
        else:
            post_sign = 1

        M_fit['lambda']['a_u'] = a_u[0,0]

        #M = sign*X[sign*X>sign*u] - sign*u
        M = calc_events_M(X=X, u=u, u_dir=sign, u_zero=u_zero_ij[0,0], carry_sign=False)


        M_fit['lambda']['N'] = M.size
        if incl_M_stat:
            M_fit['M_stat']['mean'] = post_sign*np.mean(M)
            M_fit['M_stat']['std'] = np.std(M)
        if incl_gp_fit:
            gp_fit = dist_fns.calc_scipy_dist_est_se_log_L(data=M, dist='genpareto', floc=0, fscale=fscale_ij[0,0], fc=fc_ij[0,0], fNparams_free=fNparams_free[0,0])
            for val in ['est', 'se']:
                M_fit['gp_fit'][val][:] = gp_fit[val]
            for val in ['log_L', 'N_params', 'AIC', 'N_params_free', 'AIC_free']:
                M_fit['gp_fit'][val] = gp_fit[val]



    M_fit['lambda']['a_lambda_emp'] = M_fit['lambda']['N']/M_fit['lambda']['T']
    if fa_lambda is None:
        M_fit['lambda']['a_lambda'] = M_fit['lambda']['a_lambda_emp']
    else:
        M_fit['lambda']['a_lambda'] = fa_lambda

    M_fit['lambda']['log_L'] = -M_fit['lambda']['T']*M_fit['lambda']['a_lambda'] + M_fit['lambda']['N']*np.log(M_fit['lambda']['a_lambda'])
    M_fit['lambda']['N_params'] = np.full_like(M_fit['lambda']['N'], 1)
    M_fit['lambda']['N_params_free'] = np.full_like(M_fit['lambda']['N'], 1)

    M_fit['lambda']['AIC'] = misc_fns.calc_AIC(N_params=M_fit['lambda']['N_params'], log_L=M_fit['lambda']['log_L'])
    M_fit['lambda']['AIC_free'] = misc_fns.calc_AIC(N_params=M_fit['lambda']['N_params_free'], log_L=M_fit['lambda']['log_L'])




    

    return M_fit


def calc_tail_fit_atf(X, train_t=None, a_u=None, u_dir=None, u_zero=None, u=None, as_array=None, carry_sign=None, incl_M_stat=None, incl_gp_fit=None, fa_lambda=None, fscale=None, fc=None, fNparams_free=None):

    M_fit = {period: None for period in ['all', 'train', 'forecast']}

    if train_t is None:
        train_t = [0, np.size(X)]

    M_fit['train'] = calc_tail_fit(X=X[train_t[0]:train_t[1]], a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, as_array=as_array, carry_sign=carry_sign, incl_M_stat=incl_M_stat, incl_gp_fit=incl_gp_fit, fa_lambda=fa_lambda, fscale=fscale, fc=fc, fNparams_free=fNparams_free)

    M_fit['all'] = calc_tail_fit(X=X, a_u=a_u, u=u, u_dir=u_dir, as_array=as_array, carry_sign=carry_sign, incl_M_stat=incl_M_stat, incl_gp_fit=incl_gp_fit, 
                                 fa_lambda=M_fit['train']['lambda']['a_lambda'], fscale=M_fit['train']['gp_fit']['est'][:,:,-1], fc=M_fit['train']['gp_fit']['est'][:,:,0], fNparams_free=M_fit['train']['gp_fit']['N_params_free'])
    

    M_fit['forecast'] = calc_tail_fit(X=X[train_t[1]:], a_u=a_u, u=u, u_dir=u_dir, as_array=as_array, carry_sign=carry_sign, incl_M_stat=incl_M_stat, incl_gp_fit=incl_gp_fit, 
                                 fa_lambda=M_fit['train']['lambda']['a_lambda'], fscale=M_fit['train']['gp_fit']['est'][:,:,-1], fc=M_fit['train']['gp_fit']['est'][:,:,0], fNparams_free=M_fit['train']['gp_fit']['N_params_free'])
    

    return M_fit