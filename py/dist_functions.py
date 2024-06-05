import numpy as np
import scipy
import scipy.stats as stats
import pandas as pd
import time
import datetime
import numdifftools



# Local files
import misc_functions as misc_fns
#import plot_functions as plot_fns



def get_scipy_dist_symb_letter(dist=None):

    _scipy_dist_symb_letter = {
                                'norm': 'N',
                                't': 'S',
                                'nct': 'S',
                                'expon': 'E',
                                'genpareto': 'P',
                                'dist': 'D',
                                }
    if dist is None:
        return _scipy_dist_symb_letter
    else:
        return _scipy_dist_symb_letter[dist]
    

    

def get_scipy_dist_tex(dist=None, macro=None):

    if macro is None:
        macro = True
    
    if macro:
        macro_str = f"\\rsDist"
    else:
        macro_str = ""

    _scipy_dist_tex = {
                                'norm': 'Norm',
                                't': 'T',
                                'nct': 'TNC',
                                'expon': 'Expon',
                                'genpareto': 'GP',
                                'dist': 'D',
                        }

    _scipy_dist_tex_macro = {d: f"{macro_str}{_scipy_dist_tex[d]}" for d in _scipy_dist_tex}
    
    if dist is None:
        tex =  _scipy_dist_tex_macro
    else:
        return _scipy_dist_tex_macro[dist]


def get_scipy_dist_symb(dist=None, tex=None, macro=None):

    if tex is None:
        tex = False

    if tex:
        return  get_scipy_dist_tex(dist=dist, macro=True)
    else:
        _scipy_dist_symb_letter = get_scipy_dist_symb_letter()

        _scipy_dist_symb = {d: f"\\mathcal{{{_scipy_dist_symb_letter[d]}}}" for d in _scipy_dist_symb_letter}

        if dist is None:
            return _scipy_dist_symb
        else:
            return _scipy_dist_symb[dist]
    
def get_scipy_dist_symb_params(dist=None, specific=None, val_replace=None, rm=None, moment_order=None):

    if moment_order is None:
        moment_order = True
    
    _scipy_dist_symb_letter = get_scipy_dist_symb_letter()

    _scipy_dist_symb_params = {d: misc_fns.str_concatenate(get_scipy_dist_param_symb_list(dist=d, specific=specific, val_replace=val_replace, rm=rm, moment_order=moment_order), fill_str=',', fill_str_void=-1) for d in _scipy_dist_symb_letter}

    if dist is None:
        return _scipy_dist_symb_params
    else:
        return _scipy_dist_symb_params[dist]
    
def get_scipy_dist_symb_with_params(dist=None, specific=None, val_replace=None, rm=None, moment_order=None, brac_lr=None):

    if brac_lr is None:
        brac_lr = True

    brac_str = [f"(", ")"]
    brac_lr_str = [f"\\left", f"\\right"]

    if brac_lr:
        brac = [f"{brac_lr_str[k]}{brac_str[k]}" for k in range(2)]

    _scipy_dist_symb_params = get_scipy_dist_symb_params(dist=None, specific=specific, val_replace=val_replace, rm=rm, moment_order=moment_order)
    _scipy_dist_symb = get_scipy_dist_symb(dist=None)

    _scipy_dist_symb_with_params = {d: f"{_scipy_dist_symb[d]}_{{{brac[0]}{_scipy_dist_symb_params[d]}{brac[1]}}}" for d in _scipy_dist_symb}

    if dist is None:
        return _scipy_dist_symb_with_params
    else:
        return _scipy_dist_symb_with_params[dist]


def get_scipy_dist_param_symb(param=None, dist=None, specific=None):


    _scipy_dist_param_symb = {
                                'c': f'\\xi',
                                'df': f'\\nu',
                                'nc': f'\\eta',
                                'loc': f'\\mu',
                                'scale': f'\\sigma',
                                }
    
    if dist is not None:

        if specific is None:
            specific = True

        if specific:
            if np.isin(dist, ['expon']):
                _scipy_dist_param_symb['scale'] == f'\\lambda'
            if np.isin(dist, ['genpareto']):
                _scipy_dist_param_symb['loc'] == f'u'
    
    if param is None:  
        return _scipy_dist_param_symb
    else:
        return _scipy_dist_param_symb[param]
    

def get_scipy_dist_fparam(param=None):

    _scipy_dist_fparam = {f"f{param}": param for param in get_scipy_dist_param_symb()}

    if param is None:  
        return _scipy_dist_fparam
    else:
        return _scipy_dist_fparam[param]
    

    
def get_scipy_dist_param_symb_list(dist=None, specific=None, rm=None, moment_order=None, as_array=None, val_replace=None):

    if as_array is None:
        as_array = True
    
    _scipy_dist_param_list = get_scipy_dist_param_list(dist=None, rm=rm, moment_order=moment_order)

    _scipy_dist_param_symb_list = {d: None for d in _scipy_dist_param_list}
    for d in _scipy_dist_param_symb_list:
        _scipy_dist_param_symb = get_scipy_dist_param_symb(dist=d, specific=specific)

        if val_replace is not None:
            for val in val_replace:
                _scipy_dist_param_symb[val] = str(val_replace[val])

        if as_array:
            _scipy_dist_param_symb_list[d] = np.array([_scipy_dist_param_symb[p] for p in _scipy_dist_param_list[d]])
        else:
            _scipy_dist_param_symb_list[d] = {p: _scipy_dist_param_symb[p] for p in _scipy_dist_param_list[d]}

    if dist is None:
        return _scipy_dist_param_symb_list
    else:
        return _scipy_dist_param_symb_list[dist]


def get_scipy_dist_param_list(dist=None, rm=None, moment_order=None):

    if moment_order is None:
        moment_order = False

    if rm is None: 
        rm = []

    if moment_order:
        _scipy_dist_param_list = {
                                            'norm': np.array(['loc', 'scale']),
                                            't': np.array(['loc', 'scale', 'df']),
                                            'nct': np.array(['loc', 'scale', 'df', 'nc']),
                                            'expon': np.array(['loc', 'scale']),
                                            'genpareto': np.array(['loc', 'scale', 'c']),
                                        }
    else:
        _scipy_dist_param_list = {
                                'norm': np.array(['loc', 'scale']),
                                't': np.array(['df', 'loc', 'scale']),
                                'nct': np.array(['df', 'nc', 'loc', 'scale']),
                                'expon': np.array(['loc', 'scale']),
                                'genpareto': np.array(['c', 'loc', 'scale']),
                            }
    


    _scipy_dist_param_list_rm = {d: _scipy_dist_param_list[d][~np.isin(_scipy_dist_param_list[d], rm)] for d in _scipy_dist_param_list}

    if dist is None:
        return _scipy_dist_param_list_rm
    else:
        return _scipy_dist_param_list_rm[dist]
    

def get_scipy_dist_param_list_index(dist=None, params=None, rm=None, source_rm=None, source_moment_order=None, as_array=None):

    if as_array is None:
        as_array = True

    _scipy_dist_param_list = get_scipy_dist_param_list(dist=None, rm=source_rm, moment_order=source_moment_order)

    if params is not None:
        _scipy_dist_param_list_key = {d: misc_fns.make_iterable_array(params) for d in _scipy_dist_param_list}
    else:
        _scipy_dist_param_list_key = get_scipy_dist_param_list(dist=None, rm=rm, moment_order=True)


    _scipy_dist_param_list_index = {d: {p: None for p in _scipy_dist_param_list_key[d]} for d in _scipy_dist_param_list_key}
    for d in _scipy_dist_param_list_index:
        for p in range(_scipy_dist_param_list_key[d].size):
            param = _scipy_dist_param_list_key[d][p]
            where = np.where(param==_scipy_dist_param_list[d])[0]
            if np.size(where)>0:
                _scipy_dist_param_list_index[d][param] = where[0]


        if as_array:
            _scipy_dist_param_list_index[d] = np.array([_scipy_dist_param_list_index[d][p] for p in _scipy_dist_param_list_index[d]])

    if dist is None:
        return _scipy_dist_param_list_index
    else:
        return _scipy_dist_param_list_index[dist]
    
def get_scipy_dist_param_fparams_bool(dist, fparams):
    
    names = get_scipy_dist_param_list(dist=dist, rm=None, moment_order=None)
    
    fparams_index = get_scipy_dist_param_list_index(dist=dist, params=[f"{fparam}"[1:] for fparam in fparams], as_array=True)
    fparams_bool = np.full(np.size(names), False)
    for k in fparams_index:
        if k is not None:
            fparams_bool[k] = True

    return fparams_bool



def get_scipy_dist_param_bounds(param=None, dist=None, rm=None, as_array=None):

    if as_array is None:
        as_array = True

    _scipy_dist_param_bounds =   {
                                        'c': np.array([-1, np.infty]),
                                        'df': np.array([2, np.infty]),
                                        'nc': np.array([-np.infty, np.infty]),
                                        'loc': np.array([-np.infty, np.infty]),
                                        'scale': np.array([0, np.infty]),
                                    }
    
    if param is not None:
        return _scipy_dist_param_bounds[param]
    else:
        if dist is None:
            _scipy_dist_param_bounds_dict =  _scipy_dist_param_bounds
        else:
            _scipy_dist_param_bounds_dict = {p: _scipy_dist_param_bounds[p] for p in get_scipy_dist_param_list(dist=dist, rm=rm)}

        if as_array:
            return np.array([_scipy_dist_param_bounds_dict[p] for p in _scipy_dist_param_bounds_dict])


def get_fparams(fparams=None, floc=None, fscale=None, fdf=None, fnc=None, fc=None, dist=None, dist_only=None, add_None=None):

    if dist_only is None:
        dist_only = True

    if add_None is None:
        add_None = False


    if fparams is None:
        fparams = {fparam: None for fparam in get_scipy_dist_fparam()}

    if floc is not None:
        fparams['floc'] = floc
    if fscale is not None:
        fparams['fscale'] = fscale
    if fdf is not None:
        fparams['fdf'] = fdf
    if fnc is not None:
        fparams['fnc'] = fnc
    if fc is not None:
        fparams['fc'] = fc

    if dist_only:
        if dist is None:
            dist = 'norm'

        fparams_list = [fparam for fparam in fparams]
        fparams_dist = dict()
        for param in get_scipy_dist_param_list(dist=dist, rm=None, moment_order=None):
            fparam = f"f{param}"
            if np.isin(fparam, fparams_list):
                if fparams[fparam] is not None or add_None:
                    fparams_dist[fparam] = fparams[fparam]

        return fparams_dist
    else:
        return fparams




def calc_scipy_dist_est(data, dist=None, fparams=None, floc=None, fscale=None, fdf=None, fnc=None, fc=None, 
                        nan_rm=None, train_t=None, train_T=None, train_N=None, train_N_T=None):

    if dist is None:
        dist = 'norm'

    if nan_rm is None:
        nan_rm = True

    train_t = misc_fns.calc_train_t(train_t=train_t, X=data, train_T=train_T)
    if nan_rm:
        data = data[train_t[0]:train_t[1]][~np.isnan(data[train_t[0]:train_t[1]])]
    train_N = misc_fns.calc_train_t(train_t=train_N, X=data, train_T=train_N_T)

    

    fparams = get_fparams(fparams=fparams, floc=floc, fscale=fscale, fdf=fdf, fnc=fnc, fc=fc, dist=dist, dist_only=True, add_None=False)
    args = {
                **{'data': data},
                **fparams
            }
    
    fparams_list_no_f = [param[1:] for param in fparams]
    params_list = get_scipy_dist_param_list(dist=dist, rm=None, moment_order=None)

    if all(ele in fparams_list_no_f for ele in params_list):
        fit = np.array([fparams[f"f{param}"] for param in params_list], dtype=float)
    else:
        if np.size(data)==0:
            args["data"] = [0]
            fit = np.array(scipy.stats.__getattribute__(dist).fit(**args))
            fit[~np.isin(np.arange(fit.size), get_scipy_dist_param_list_index(dist=dist, params=[param for param in fparams_list_no_f], as_array=True))] = np.nan

        else:
            fit = np.array(scipy.stats.__getattribute__(dist).fit(**args))

    return fit


def log_L_scipy_dist(data, dist=None, params=None, fparams=None, floc=None, fscale=None, fdf=None, fnc=None, fc=None,
                     nan_rm=None, train_t=None, train_T=None, train_N=None, train_N_T=None):

    if dist is None:
        dist = 'norm'
    if params is None:
        params = calc_scipy_dist_est(data=data, dist=dist, fparams=fparams, floc=floc, fscale=fscale, fdf=fdf, fnc=fnc, fc=fc, 
                                     nan_rm=nan_rm, train_t=train_t, train_T=train_T, train_N=train_N, train_N_T=train_N_T)

    return np.sum(np.log(scipy.stats.__getattribute__(dist)(*np.array(params)).pdf(data)))




def calc_scipy_dist_est_se_log_L(data, dist=None, est=None, fparams=None, floc=None, fscale=None, fdf=None, fnc=None, fc=None, fNparams_free=None,
                                 nan_rm=None, train_t=None, train_T=None, train_N=None, train_N_T=None,
                                 est_se_dict=None, free_only=None, free_rm=None, push_free=None,
                                pre=None, suff=None, eval_pre=None, eval_suff=None):


    if dist is None:
        dist = 'norm'

    if nan_rm is None:
        nan_rm = True


    train_t = misc_fns.calc_train_t(train_t=train_t, X=data, train_T=train_T)
    if nan_rm:
        data = data[train_t[0]:train_t[1]][~np.isnan(data[train_t[0]:train_t[1]])]
    train_N = misc_fns.calc_train_t(train_t=train_N, X=data, train_T=train_N_T)


    return_dict = {val: None for val in ['est', 'se', 'eval']}


    fparams = get_fparams(fparams=fparams, floc=floc, fscale=fscale, fdf=fdf, fnc=fnc, fc=fc, dist=dist, dist_only=True, add_None=False)

    if est is None:
        est = calc_scipy_dist_est(data=data[train_N[0]:train_N[1]], dist=dist, fparams=fparams, nan_rm=False)
    else:
        est = np.array(est)


    save_est = np.copy(est)

    def calc_log_L(params, unconstrained_bool=None):
        
        temp_est = np.copy(save_est)

        if unconstrained_bool is None:
            unconstrained_bool = np.full(np.size(temp_est), True)

        temp_est[unconstrained_bool] = params
        
        return log_L_scipy_dist(data=data[train_N[0]:train_N[1]], dist=dist, params=temp_est)
    

    


    bounds = get_scipy_dist_param_bounds(dist=dist, rm=None, as_array=True)


    log_L = calc_log_L(params=save_est)

    step_min = 1E-8
    unconstrained_nz = np.logical_and((bounds[:,0]+step_min) <= save_est, (bounds[:,1]-step_min) >= save_est)
    
    se_steps = np.max([
                        np.full(np.size(save_est), step_min),
                        save_est/1E5,
                        ], axis=0)
    

    fparams_bool = get_scipy_dist_param_fparams_bool(dist=dist, fparams=fparams)
    unconstrained_nz[fparams_bool] = False


    def calc_log_L_unconstrained(params):
        return calc_log_L(params=params, unconstrained_bool=unconstrained_nz)
    


    # Standard error
    se = np.full(np.size(save_est), np.nan)
    
    # Ad-hoc check for fitting across u range
    #if self._parameters.res.fun < 500

    if not np.isnan(log_L) and np.sum(unconstrained_nz)>0:
        fitted_success = True
        hess = numdifftools.Hessian(calc_log_L_unconstrained, full_output=True, method="central", step=se_steps[unconstrained_nz])(save_est[unconstrained_nz])[0]
        try:
            np.linalg.inv(hess)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                se_unconstrained = np.full(np.sum(unconstrained_nz), np.nan)
                fitted_success = False
        
        if fitted_success:
            se_unconstrained = np.sqrt(np.abs(np.diag(np.linalg.inv(hess))))     
        for i in range(se_unconstrained.size):
            se[np.where(unconstrained_nz)[0][i]] = se_unconstrained[i]
    


    if fNparams_free is None:
        N_params_free = np.sum(unconstrained_nz)
    else:
        N_params_free = fNparams_free


    if est_se_dict is None:
        est_se_dict = False


    if free_only is None:
        free_only is False

    
    if free_only:
        include = ~fparams_bool
        push_free = True
    else:
        include = np.full(np.size(est), True)

    if est_se_dict:
        names = get_scipy_dist_param_list(dist=dist, rm=None, moment_order=None)
        where_include = np.where(include)[0]
        return_dict['est'] = misc_fns.dict_pre_suff(dic={names[p]: est[p] for p in where_include}, pre=pre, suff=suff)
        return_dict['se'] = misc_fns.dict_pre_suff(dic={names[p]: se[p] for p in where_include}, pre=pre, suff=suff)
    else:
        return_dict['est'] = est[include]
        return_dict['se'] = se[include]
        
    if eval_pre is None:
        eval_pre = pre
    if eval_suff is None:
        eval_suff = suff




    return_dict['eval'] = misc_fns.calc_log_L_eval_dict_pre_suff(N_data=np.size(data[train_N[0]:train_N[1]]), N_params=np.size(save_est), N_params_free=N_params_free, log_L=log_L, AIC=None, AIC_free=None, BIC=None, BIC_free=None,
                                  log_L_eval_dict=None, complete=True, push_free=push_free, free_rm=free_rm, None_rm=None, pre=pre, suff=suff, keys_rm=None)

    
    

    return return_dict




    
def calc_X_dist_fit(X, dist=None, train_t=None, train_T=None, train_N=None, train_N_T=None, est_se_dict=None,
                fparams=None, floc=None, fscale=None, fdf=None, fnc=None, fc=None, fNparams_free=None):

    if est_se_dict is None:
        est_se_dict = False

    X_shape = np.array(np.shape(X))

    N_i = np.product(X_shape[:-1])
    T = X_shape[-1]

    int_void = np.full(N_i, 0)
    float_void = np.full(N_i, np.nan)

    X_dist_fit = {d: None for d in ['est', 'se', 'eval']}

    names = get_scipy_dist_param_list(dist=dist, rm=None, moment_order=None)
    N_params = names.size

    if est_se_dict:
        params_shape = X_shape[:-1]
        for d in ['est', 'se']:
            X_dist_fit[d] = {name: np.full((N_i), np.nan) for name in names}
    else:
        params_shape = np.concatenate([X_shape[:-1], [N_params]])
        for d in ['est', 'se']:
            X_dist_fit[d] = np.full((N_i,N_params), np.nan)



    X_dist_fit['eval'] = misc_fns.calc_log_L_eval_dict_pre_suff(N_data=np.copy(int_void), N_params=np.copy(int_void), N_params_free=None, log_L=np.copy(float_void), AIC=np.copy(float_void), AIC_free=None, BIC=np.copy(float_void), BIC_free=None,
                                  log_L_eval_dict=None, complete=False, push_free=None, free_rm=None, None_rm=None, pre=None, suff=None, keys_rm=None)
    

    X = np.reshape(X, (N_i, X_shape[-1]))

    if train_t is None:
        if train_T is None:
            train_T = np.repeat(T, N_i)
        elif np.size(train_T)==1:
            train_T = np.repeat(train_T, N_i)
        train_t = np.array([[0,train_T[i]] for i in range(N_i)])
    elif len(np.shape(train_t))==1:
        train_t = np.array([train_t for i in range(N_i)])

    if train_N is None:
        if train_N_T is None:
            train_N_T = np.repeat(T, N_i)
        elif np.size(train_N_T)==1:
            train_N_T = np.repeat(train_N_T, N_i)
        train_N = np.array([[0,train_N_T[i]] for i in range(N_i)])
    elif len(np.shape(train_N))==1:
        train_N = np.array([train_N for i in range(N_i)])

    train_t = np.reshape(train_t, (N_i, 2))
    train_N = np.reshape(train_N, (N_i, 2))


    for i in range(N_i):
        dist_est_se_eval = calc_scipy_dist_est_se_log_L(data=X[i,:], train_t=train_t[i,:], train_N=train_N[i,:],  dist=dist, est=None, fparams=fparams, floc=floc, fscale=fscale, fdf=fdf, fnc=fnc, fc=fc, fNparams_free=fNparams_free, est_se_dict=est_se_dict, 
                                                #suff=f"_{1+i}", 
                                                free_only=False, push_free=True)
        
        
        if est_se_dict:
            for d in ['est', 'se']:
                for name in names:
                    X_dist_fit[d][name][i] = dist_est_se_eval[d][name]
        else:
            for d in ['est', 'se']:
                X_dist_fit[d][i,:] = dist_est_se_eval[d]

        for field in X_dist_fit['eval']:
            X_dist_fit['eval'][field][i] = dist_est_se_eval['eval'][field]
        

    if est_se_dict:
        for d in ['est', 'se']:
            for name in names:
                X_dist_fit[d][name] = np.reshape(X_dist_fit[d][name], params_shape)
    else:
        for d in ['est', 'se']:
            X_dist_fit[d] = np.reshape(X_dist_fit[d], params_shape)

    for field in X_dist_fit['eval']:
        X_dist_fit['eval'][field] = np.reshape(X_dist_fit['eval'][field], X_shape[:-1])
    
    return X_dist_fit








    



def calc_u_a_u(x, a_u=None):

    if a_u is None:
        a_u = 0.025
    a_u = misc_fns.make_iterable_array(a_u)

    return np.quantile(x, 0.5 + np.outer(np.array([-1,1]), (0.5-a_u)))


def calc_events(x, a_u=None, u=None, u_dir=None, u_zero=None, override_u2=None, carry_sign=None, ret_val=None, as_array=None, first_u=None):

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
        u = calc_u_a_u(x=x, a_u=a_u)

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

    T_max = np.size(x)


    if len(np.shape(u))==0:
        
        events = {val: None for val in events_list}

        sign = u_dir[0]
        if carry_sign:
            post_sign = sign
        else:
            post_sign = 1
        zero = u_zero[0]

        events['N'] = 0
        events['a_lambda'] = np.nan
        events['N_t'] = np.zeros(T_max, dtype=int)


        events['t'] = np.where(sign*x>sign*u)[0]
        events['N'] = np.size(events['t'])
        if T_max>0:
            events['a_lambda'] = events['N']/T_max

        events['tilde_t_a_u'] = events['t'] * a_u[0]
        events['tilde_t_a_lambda'] = events['t'] * events['a_lambda']

        
        events['M'] = post_sign*(sign*(x[events['t']]-zero) - sign*u)
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

            events['t'][i] = np.where(sign*x>sign*u[i])[0]
            events['N'][i] = np.size(events['t'][i])
            if T_max>0:
                events['a_lambda'][i] = events['N'][i]/T_max


            events['tilde_t_a_u'][i] = events['t'][i] * a_u[i%a_u.size]
            events['tilde_t_a_lambda'][i] = events['t'][i] * events['a_lambda'][i]

            
            events['M'][i] = post_sign*(sign*(x[events['t'][i]]-zero) - sign*u[i])
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
                events['t'][i][j] = np.where(sign*x>sign*u[i,j])[0]
                events['N'][i][j] = np.size(events['t'][i][j])

                if T_max>0:
                    events['a_lambda'][i][j] = events['N'][i][j]/T_max

                events['tilde_t_a_u'][i][j] = events['t'][i][j] * a_u[i%a_u.size]
                events['tilde_t_a_lambda'][i][j] = events['t'][i][j] * events['a_lambda'][i][j]

                events['M'][i][j] = post_sign*(sign*(x[events['t'][i][j]]-zero) - sign*u[i,j])

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






def calc_events_cat(events, I_s_cat_list=None, tilde_t=None):

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






def calc_events_M(x, a_u=None, u=None, u_dir=None, u_zero=None, override_u2=None, carry_sign=None):

    return calc_events(x, a_u=a_u, u=u, u_dir=u_dir, u_zero=u_zero, override_u2=override_u2, carry_sign=carry_sign, ret_val="M")

def calc_events_t(x=None, a_u=None, u=None, u_dir=None, u_zero=None, override_u2=None):

    return calc_events(x=x, a_u=a_u, u=u, u_dir=u_dir, u_zero=u_zero, override_u2=override_u2, ret_val="t")

def calc_events_delta_t(x=None, a_u=None, u=None, u_dir=None, u_zero=None, override_u2=None):
    
    return calc_events(x=x, a_u=a_u, u=u, u_dir=u_dir, u_zero=u_zero, override_u2=override_u2, ret_val="delta_t")

def calc_events_N(x=None, a_u=None, u=None, u_dir=None, u_zero=None, override_u2=None):

    return calc_events(x=x, a_u=a_u, u=u, u_dir=u_dir, u_zero=u_zero, override_u2=override_u2, ret_val="N")

def calc_events_N_t(x=None, a_u=None, u=None, u_dir=None, u_zero=None, override_u2=None):
    
    return calc_events(x=x, a_u=a_u, u=u, u_dir=u_dir, u_zero=u_zero, override_u2=override_u2, ret_val="N_t")

def calc_tail_fit(x, a_u=None, u=None, u_dir=None, u_zero=None, as_array=None, carry_sign=None, incl_M_stat=None, incl_gp_fit=None, fa_lambda=None, fscale=None, fc=None, fNparams_free=None, train_t=None, train_T=None, train_N=None, train_N_T=None):


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
        u = calc_u_a_u(x=x, a_u=a_u)


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

    int_void = np.full(np.shape(u), 0)
    float_void = np.full(np.shape(u), np.nan)
        

    M_fit = {field: {val: np.copy(float_void) for val in fields[field]} for field in incl_fields}
    
    
    M_fit['lambda']['T'] = np.full(np.shape(u), np.size(x))
    M_fit['lambda']['N'] = np.full(np.shape(u), -1)
    M_fit['lambda']['u'] = u
    M_fit['lambda']['u_dir'] = np.outer(u_dir, np.ones(np.shape(u)[1], dtype=int))
    if incl_gp_fit:
        incl_fields.append('gp_fit')
        M_fit['gp_fit'] = {field: None for field in ['est', 'se', 'eval']}
        for val in ['est', 'se']:
            M_fit['gp_fit'][val] = np.full(np.concatenate([np.shape(u), [3]]), np.nan)

            M_fit['gp_fit']['eval'] = misc_fns.calc_log_L_eval_dict_pre_suff(N_data=np.copy(int_void), N_params=np.copy(int_void), N_params_free=np.copy(int_void), log_L=np.copy(float_void), AIC=np.copy(float_void), AIC_free=np.copy(float_void), BIC=np.copy(float_void), BIC_free=np.copy(float_void),
                                                                                    log_L_eval_dict=None, complete=False, push_free=None, free_rm=None, None_rm=None, pre=None, suff=None, keys_rm=None)
        



    



    if len(np.shape(u))==2:

        a_u_ij = misc_fns.list_2D(a_u, I_max=u.shape[0], J_max=u.shape[1], set_J_max=None, mode_1D='j', as_array=True)
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


            for j in range(u.shape[1]):

                #M = sign*x[sign*x>sign*u[i,j]] - sign*
                M = calc_events_M(x=x, u=u[i,j], u_dir=sign, u_zero=u_zero_ij[i,j], carry_sign=False)

                M_fit['lambda']['N'][i,j] = M.size
                if incl_M_stat:
                    M_fit['M_stat']['mean'][i,j] = post_sign*np.mean(M)
                    M_fit['M_stat']['std'][i,j] = np.std(M)
                if incl_gp_fit:

                    gp_fit = calc_scipy_dist_est_se_log_L(data=M, dist='genpareto', floc=0, fscale=fscale_ij[i,j], fc=fc_ij[i,j], fNparams_free=fNparams_free_ij[i,j], train_t=train_t, train_T=train_T, train_N=train_N, train_N_T=train_N_T)
                    for val in ['est', 'se']:
                        M_fit['gp_fit'][val][i,j,:] = gp_fit[val]
                    for field in M_fit['gp_fit']['eval']:
                        M_fit['gp_fit']['eval'][field][i,j] = gp_fit['eval'][field]


    elif len(np.shape(u))==1:

        a_u_ij = misc_fns.list_2D(a_u, I_max=u.shape[0], J_max=1, set_J_max=None, mode_1D='j', as_array=True)
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


            #M = sign*x[sign*x>sign*u[i]] - sign*u[i]
            M = calc_events_M(x=x, u=u[i], u_dir=sign, u_zero=u_zero_ij[i,0], carry_sign=False)

            M_fit['lambda']['N'][i] = M.size
            if incl_M_stat:
                M_fit['M_stat']['mean'][i] = post_sign*np.mean(M)
                M_fit['M_stat']['std'][i] = np.std(M)
            if incl_gp_fit:

                gp_fit = calc_scipy_dist_est_se_log_L(data=M, dist='genpareto', floc=0, fscale=fscale_ij[i,j], fc=fc_ij[i,j], fNparams_free=fNparams_free_ij[i,j])
                for val in ['est', 'se']:
                    M_fit['gp_fit'][val][i,:] = gp_fit[val]
                for field in M_fit['gp_fit']['eval']:
                    M_fit['gp_fit']['eval'][field][i] = gp_fit['eval'][field]

    else:

        a_u_ij = misc_fns.list_2D(a_u, I_max=1, J_max=1, set_J_max=None, mode_1D='j', as_array=True)
        fscale_ij = misc_fns.list_2D(fscale, I_max=1, J_max=1, set_J_max=None, mode_1D=None, as_array=True)
        fc_ij = misc_fns.list_2D(fc, I_max=1, J_max=1, set_J_max=None, mode_1D=None, as_array=True)
        fNparams_free_ij = misc_fns.list_2D(fNparams_free, I_max=1, J_max=1, set_J_max=None, mode_1D=None, as_array=True)
        u_zero_ij = misc_fns.list_2D(u_zero, I_max=1, J_max=1, set_J_max=None, mode_1D=None, as_array=True)

        sign = u_dir[0]
        if carry_sign:
            post_sign = u_dir[0]
        else:
            post_sign = 1

        M_fit['lambda']['a_u'] = a_u_ij[0,0]

        #M = sign*x[sign*x>sign*u] - sign*u
        M = calc_events_M(x=x, u=u, u_dir=sign, u_zero=u_zero_ij[0,0], carry_sign=False)


        M_fit['lambda']['N'] = M.size
        if incl_M_stat:
            M_fit['M_stat']['mean'] = post_sign*np.mean(M)
            M_fit['M_stat']['std'] = np.std(M)
        if incl_gp_fit:

            M_fit['gp_fit'] = calc_scipy_dist_est_se_log_L(data=M, dist='genpareto', floc=0, fscale=fscale_ij[0,0], fc=fc_ij[0,0], fNparams_free=fNparams_free[0,0])
            




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


def calc_tail_fit_atf(x, train_t=None, a_u=None, u=None, u_dir=None, u_zero=None, as_array=None, carry_sign=None, incl_M_stat=None, incl_gp_fit=None, fa_lambda=None, fscale=None, fc=None, fNparams_free=None):

    M_fit = {period: None for period in ['all', 'train', 'forecast']}

    if train_t is None:
        train_t = [0, np.size(x)]

    M_fit['train'] = calc_tail_fit(x=x[train_t[0]:train_t[1]], a_u=a_u, u=u, u_dir=u_dir, u_zero=u_zero, as_array=as_array, carry_sign=carry_sign, incl_M_stat=incl_M_stat, incl_gp_fit=incl_gp_fit, fa_lambda=fa_lambda, fscale=fscale, fc=fc, fNparams_free=fNparams_free)

    M_fit['all'] = calc_tail_fit(x=x, a_u=a_u, u=u, u_dir=u_dir, as_array=as_array, carry_sign=carry_sign, incl_M_stat=incl_M_stat, incl_gp_fit=incl_gp_fit, 
                                 fa_lambda=M_fit['train']['lambda']['a_lambda'], fscale=M_fit['train']['gp_fit']['est'][:,:,-1], fc=M_fit['train']['gp_fit']['est'][:,:,0], fNparams_free=M_fit['train']['gp_fit']['eval']['N_params_free'])
    

    M_fit['forecast'] = calc_tail_fit(x=x[train_t[1]:], a_u=a_u, u=u, u_dir=u_dir, as_array=as_array, carry_sign=carry_sign, incl_M_stat=incl_M_stat, incl_gp_fit=incl_gp_fit, 
                                 fa_lambda=M_fit['train']['lambda']['a_lambda'], fscale=M_fit['train']['gp_fit']['est'][:,:,-1], fc=M_fit['train']['gp_fit']['est'][:,:,0], fNparams_free=M_fit['train']['gp_fit']['eval']['N_params_free'])
    

    return M_fit

    


def F_D1_inv_F_D0(X, F_D1_inv=None, F_D0=None, varsigma=None, excl_bounds_Y=None, excl_inf_Z=None, F_D1_inv_min_max=None):

    min_res = 1E-16

    if F_D0 is None:
        if varsigma is None:
            varsigma = 1
        F_D0 = scipy.stats.expon(scale=varsigma).cdf

    if F_D1_inv is None:
        F_D1_inv = scipy.stats.norm().ppf

    if F_D1_inv_min_max is None:   
        F_D1_inv_min_max = np.array([F_D1_inv(min_res), F_D1_inv(1-min_res)])


    Y = F_D0(X)

    if excl_bounds_Y is None:
        excl_bounds_Y = False
    if excl_bounds_Y:
        Y[Y==0] = min_res
        Y[Y==1] = 1 - min_res


    Z = F_D1_inv(Y)


    if excl_inf_Z is None:
        excl_inf_Z = False
    if excl_inf_Z:
        sign = np.sign(Z)
        isinf = np.isinf(Z)

        for k in range(2):
            Z[np.logical_and(sign==(-1)**(1+k), isinf)] = F_D1_inv_min_max[k]

    return Z


def calc_dir_func(dir=None):

    if dir is None:
        dir = 'right'

    if dir=='abs':
        def dir_dn(x):
            return np.abs(x)
    elif dir=='left':
        def dir_dn(x):
            return -x
    else:
        def dir_dn(x):
            return x
        
    return dir_dn

          