import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

# Local files
import misc_functions as misc_fns
import plot_functions as plot_fns
import dist_functions as dist_fns
import hawkes_functions as hwks_fns

import misc_functions_nb as misc_fns_nb




#============
#============
#
# Functions
#
#============
#============

def calc_index_t(index, bound_vals, pt_correct):

    if pt_correct is None:
        pt_correct = np.array([0,1])
    
    t = np.where(np.logical_and(index >= bound_vals[0], index < bound_vals[1]))[0][[0,-1]]+pt_correct 

    return t




def calc_ps_tT(self, tick, r=None, overwrite_t_ps=None):

    if r is None:
        r = self.tk[tick].pd.lr['p']['all']

    if overwrite_t_ps is None:
        overwrite_t_ps = {}
    t_dict = self.tk[tick].pd.get_periods_t_offset(forecast_as_list=False)
    if not np.isin('forecast', t_dict):
        t_dict['forecast'] = np.array([t_dict['train'][-1]-1, t_dict['all'][-1]-1], dtype=int)
    for p in overwrite_t_ps:
        if type(overwrite_t_ps[p][0])==str:
            overwrite_t_ps[p] = [pd.Timestamp(overwrite_t_ps[p][k]) for k in range(len(overwrite_t_ps[p]))]
        if type(overwrite_t_ps[p][0])==type(pd.Timestamp('2000')):
            overwrite_t_ps[p] = calc_index_t(index=r.index, bound_vals=overwrite_t_ps[p], pt_correct=np.array([0,1]))
        t_dict[p] = overwrite_t_ps[p]
       
    ret_dict = {p: {
                        't': t_dict[p],
                        'T': np.diff(t_dict[p])[0]
                    } for p in t_dict}

    return ret_dict




def calc_ps_N(N_t, ps_tT, I_max=None):
    
    if I_max is None:
        I_max = np.shape(N_t)[0]

    ret_dict = {p: {calc_p_N(N_t=N_t, **ps_tT[p], I_max=I_max)} for p in ps_tT}

    return ret_dict



def calc_p_N(N_t, T, t, I_max=None):
    
    if I_max is None:
        I_max = np.shape(N_t)[0]

    ret_dict = {}

    ret_dict['N_t'] = np.zeros((I_max, T), dtype=int)
    ret_dict['N'] = np.zeros(I_max, dtype=int)
    ret_dict['N_t_0'] = np.zeros(I_max, dtype=int)

    ret_dict['N_t'][:I_max,:] = N_t[:I_max,t[0]:t[1]]
    if np.shape(ret_dict['N_t'])[1] > 0:
        ret_dict['N'] = np.diff(ret_dict['N_t'][:I_max,[0,-1]], axis=1)[:,0]
    if t[0] > 0:
        ret_dict['N_t_0'] = N_t[:I_max,t[0]-1]


    return ret_dict


def calc_ps_tT_N(self, tick, r, N_t, overwrite_t_ps=None):

    ps_tT = calc_ps_tT(self=self, tick=tick, r=r, overwrite_t_ps=overwrite_t_ps)

    ps_N = calc_ps_N(N_t=N_t, ps_tT=ps_tT)

    return {p: {**ps_tT[p], **ps_N} for p in ps_tT}



def get_tf_a_u_u(self, tick=None, 
            tf_mode=None, tf_u_key=None, a_u=None, u=None, a_lambda=None,
            mode=None, mark=None,
            as_dict=None):

    if as_dict is None:
        as_dict = False

    if tf_mode is None:
        if a_u is None and a_u is not None:
            tf_mode = 'u'
        else:
            tf_mode = 'a_u'

    if tf_u_key is not None:
        if a_u is None:
            a_u = self.tk[tick]._tail_fit[tf_mode]['lambda']['a_u'][:,tf_u_key]
        if u is None:
            u = self.tk[tick]._tail_fit[tf_mode]['lambda']['u'][:,tf_u_key]
        if a_lambda is None:
            a_lambda = self.tk[tick]._tail_fit[tf_mode]['lambda']['a_lambda'][:,tf_u_key]

    else:
        if mode is None:
            mode = 'ci'
        if mark is None:
            mark = 'mark'
        

        if u is None:
            u = self.tk[tick].tt.tt[mode][mark]._parameters.u.get_value()
        if a_u is None:
            a_u = self.tk[tick].tt.tt[mode][mark]._parameters.a_u.get_value()
        if a_lambda is None:
            a_lambda = self.tk[tick].tt.tt[mode][mark]._parameters.a_lambda.get_value()

    u = misc_fns.make_iterable_array(u)
    a_u = misc_fns.make_iterable_array(a_u)

    if tf_u_key is None:
        where_tf_u_key = np.where(u[0]==self.tk[tick]._tail_fit[tf_mode]['lambda']['u'][0,:])[0]
        if len(where_tf_u_key)>0:
            tf_u_key = where_tf_u_key[0]
        else:
            tf_u_key = None

    if as_dict:
        return {
                    'tf_mode': tf_mode, 
                    'tf_u_key': tf_u_key, 
                    'u': u, 
                    'a_u': a_u,
                    'a_lambda': a_lambda,
                }
    else:
        return tf_mode, tf_u_key, u, a_u,a_lambda


def calc_val_key_func(val_key=None):

    if np.isin(val_key, [f'N_t', f'N_tilde_t']):
        val_key_func = f'N_t'
    elif np.isin(val_key, [f'delta_t', f'delta_tilde_t']):
        val_key_func = f'delta_t'
    elif np.isin(val_key, [f'M', f'M']):
        val_key_func = f'M'
    else:
        val_key_func = None
    return val_key_func





def KS_dict(self, tick=None, r=None, period=None, val_key=None, I_s_cat_list=None,
            tf_mode=None, tf_u_key=None, a_u=None, u=None, a_lambda=None,
            overwrite_t_ps=None,
            events=None, tilde_t=None, mode=None, mark=None,
            a_mode=None, tilde_t_0_mode=None,
            ):
    


    if tick is None:
        tick = [t for t in self.tk][0]


    if mode is None:
        mode='ci'
    if mark is None:
        mark='mark'
    

    if r is None:
        r = self.tk[tick].pd.lr['p']['all']
    

    ret_dict = {}

    ret_dict['u'] = get_tf_a_u_u(self=self, tick=tick, 
                                                tf_mode=tf_mode, tf_u_key=tf_u_key, a_u=a_u, u=u, a_lambda=a_lambda,
                                                mode=mode, mark=mark, 
                                                as_dict=True)
    

    ret_dict['ps'] = calc_ps_tT(self=self, tick=tick, r=r, overwrite_t_ps=overwrite_t_ps)
    ret_dict['p'] = ret_dict['ps'][period]



    if events is None:
        if val_key.find('tilde')>=0:
            events = self.tk[tick].tt.tt[mode][mark]._events
        else:
            events = dist_fns.calc_events(x=r.values, u=ret_dict['u']['u'], a_u=ret_dict['u']['a_u'], as_array=True, first_u=True)

    


    I_max = np.shape(events['N'])[0]



    if I_s_cat_list is None:
        I_s_cat_list = dist_fns.calc_I_s_cat_list_default(I_max=I_max)
    K_max = len(I_s_cat_list)


    if a_mode is None:
        if ret_dict['u']['tf_mode']=='u':
            a_mode = f"lambda"
        else:
            a_mode = f"u"
        
    a_plot = ret_dict['u'][f'a_{a_mode}']


    I_list = ['I', 'I_cat']
    tilde_t_all_dict = {I: None for I in I_list}
    if tilde_t is not None:
        tilde_t_all_dict['I'] = tilde_t
    else:
        if np.isin(val_key, ["N_t", "delta_t"]):
            tilde_t_all_dict['I'] = np.array([(1+np.arange(ret_dict['ps']['all']['T']))*a_plot[i%a_plot.size] for i in range(I_max)])
        elif np.isin(val_key, ["N_tilde_t", "delta_tilde_t"]):
            tilde_t_all_dict['I'] = self.tk[tick].tt.tt[mode][mark]._t_series['tilde_t']


    tilde_t_all_dict['I_cat'] = dist_fns.calc_tilde_t_cat(tilde_t=tilde_t_all_dict['I'], I_max=None, T=None, K_max=None, I_s_cat_list=I_s_cat_list)

    
    events_dict = {
                        'I': events,
                        'I_cat': dist_fns.calc_events_cat(events=events, I_s_cat_list=I_s_cat_list, tilde_t=tilde_t_all_dict['I']),
                    }

    
    ret_dict['I'] = {I: calc_p_N(N_t=events_dict[I]['N_t'], **ret_dict['p']) for I in events_dict}
    ret_dict['I']['I']['I_max'] = I_max
    ret_dict['I']['I_cat']['I_max'] = K_max


    for sub in ['u', 'lambda']:
        ret_dict['I']['I'][f'a_{sub}'] = ret_dict['u'][f'a_{sub}']
        ret_dict['I']['I_cat'][f'a_{sub}'] = np.array([np.sum(ret_dict['u'][f'a_{sub}'][I_s_cat_list[k]])] for k in range(K_max))



    for I in ret_dict['I']:
        ret_dict['I'][I]['a_plot'] = ret_dict['I'][I][f'a_{a_mode}']
        if val_key.find('tilde')<0:
            events_dict[I]['tilde_t'] = events_dict[I][f'tilde_t_a_{a_mode}']

    
    ret_dict['I']['I']['K_factor'] = np.ones(I_max, dtype=int)
    ret_dict['I']['I_cat']['K_factor'] = np.array([len(I_s_cat_list[k]) for k in range(K_max)], dtype=int)


    if tilde_t_0_mode is None:
        tilde_t_0_mode = 'N'


    val_key_func = calc_val_key_func(val_key=val_key)
    ret_dict['meta'] = {
                            'val_key': val_key,
                            'val_key_func': val_key_func,
                            'tick': tick,
                            'period': period,
                            'overwrite_t_ps': overwrite_t_ps,
                            'tilde_t_0_mode': tilde_t_0_mode,
                            'a_mode': a_mode,
                            'I_max': I_max,
                            'K_max': K_max,
                            'I_s_cat_list': I_s_cat_list,
                            'train_t': ret_dict['ps']['train']['t'],
                        }
    
    for I in ret_dict['I']:
        #ret_dict['I'][I]['train_N'] = ret_dict['I'][I]['N_t'][:,[ret_dict['meta']['train_t'][0],ret_dict['meta']['train_t'][1]-1]]
        ret_dict['I'][I]['train_N'] = events_dict[I]['N_t'][:,[ret_dict['meta']['train_t'][0],ret_dict['meta']['train_t'][1]-1]]

    



    

    ret_dict['all'] = {}
    ret_dict['all']['r'] = r

    ret_dict['Hawkes'] = {
                            'mode': mode,
                            'mark': mark,
                            'a_u': self.tk[tick].tt.tt[mode][mark]._parameters.a_u.get_value(),
                            'u': self.tk[tick].tt.tt[mode][mark]._parameters.u.get_value(),
                        }


    
    if False:

        if np.isin(val_key_func, ["N_t", "delta_t"]):

        
            if tilde_t is not None:
                ret_dict['I']['I']['tilde_t_all'] = tilde_t
            else:
                if np.isin(val_key, ["N_t", "delta_t"]):
                    ret_dict['I']['I']['tilde_t_all'] = np.array([(1+np.arange(ret_dict['ps']['all']['T']))*a_plot[i%a_plot.size] for i in range(ret_dict['I']['I']['I_max'])])
                elif np.isin(val_key, ["N_tilde_t", "delta_tilde_t"]):
                    ret_dict['I']['I']['tilde_t_all'] = self.tk[tick].tt.tt[mode][mark]._t_series['tilde_t']

            ret_dict['I']['I_cat']['tilde_t_all'] = dist_fns.calc_tilde_t_cat(ret_dict['I']['I']['tilde_t_all'], I_max=None, T=None, K_max=None, I_s_cat_list=I_s_cat_list)


    

    for I in ret_dict['I']:
        ret_dict['I'][I]['kstest'] = np.full((ret_dict['I'][I]['I_max'],2), np.nan)
        ret_dict['I'][I]['kstest_exact'] = np.full((ret_dict['I'][I]['I_max'],2), np.nan)
        ret_dict['I'][I]['kstest_asymp'] = np.full((ret_dict['I'][I]['I_max'],2), np.nan)

        ret_dict['I'][I]['kstest_alt'] = np.full((ret_dict['I'][I]['I_max'],2), np.nan)
        ret_dict['I'][I]['kstest_N_t'] = np.full((ret_dict['I'][I]['I_max'],2), np.nan)


        


    
        if np.isin(val_key_func, ["N_t", "delta_t"]):

            ret_dict['I'][I]['tilde_t_all'] = tilde_t_all_dict[I]
            ret_dict['I'][I]['tilde_t'] = ret_dict['I'][I]['tilde_t_all'][:,ret_dict['p']['t'][0]:ret_dict['p']['t'][1]]


            # mod_cat
            ret_dict['I'][I]['t'] = [np.array([]) for i in range(ret_dict['I'][I]['I_max'])]
            ret_dict['I'][I]['t_all'] = [np.array([]) for i in range(ret_dict['I'][I]['I_max'])]
            for i in range(ret_dict['I'][I]['I_max']):
                if ret_dict['I'][I]['N'][i]>0:
                    ret_dict['I'][I]['t'][i] = events_dict[I][f't'][i,ret_dict['I'][I]['N_t'][i,0]:ret_dict['I'][I]['N_t'][i,-1]] 
                if events_dict[I]['N'][i]>0:
                    ret_dict['I'][I]['t_all'][i] = events_dict[I][f't'][i,:events_dict[I]['N'][i]] 



            ret_dict['I'][I]['tilde_t_0'] = np.zeros(ret_dict['I'][I]['I_max'], dtype=float)
            if ret_dict['p']['t'][0] > 0:
                ret_dict['I'][I]['tilde_t_0'] = ret_dict['I'][I]['tilde_t_all'][:,ret_dict['p']['t'][0]-1]
            if tilde_t_0_mode=='N':
                for i in range(ret_dict['I'][I]['I_max']):
                    if ret_dict['I'][I]['N'][i]>0 and ret_dict['I'][I]['N_t'][i,0]>0:
                        ret_dict['I'][I]['tilde_t_0'][i] = ret_dict['I'][I]['tilde_t_all'][i,ret_dict['I'][I]['t_all'][i][ret_dict['I'][I]['N_t'][i,0]-1]]
                


            ret_dict['I'][I]['tilde_N'] = [np.array([])  for i in range(ret_dict['I'][I]['I_max'])]
            #ret_dict['I'][I]['tilde_N_0'] = np.zeros(ret_dict['I'][I]['I_max'])
            for i in range(ret_dict['I'][I]['I_max']):
                if ret_dict['I'][I]['N'][i]>0:
                    ret_dict['I'][I]['tilde_N'][i] = ret_dict['I'][I]['tilde_t_all'][i,ret_dict['I'][I]['t'][i]]
                    if False:
                        if ret_dict['I'][I]['N_t'][i,0]>0:
                            #ret_dict['I'][I]['tilde_N_0'][i] = events_dict[I][f'tilde_t'][i,ret_dict['I'][I]['N_t'][i,0]-1]
                            ret_dict['I'][I]['tilde_N_0'][i] = ret_dict['I'][I]['tilde_t_all'][i,ret_dict['I'][I]['t_all'][i][ret_dict['I'][I]['N_t'][i,0]-1]]
                            ret_dict['I'][I]['tilde_N'][i] -= ret_dict['I'][I]['tilde_N_0'][i]
                    else:
                        ret_dict['I'][I]['tilde_N'][i] -= ret_dict['I'][I]['tilde_t_0'][i]

                    
            if False:
                if ret_dict['p']['t'][0] > 0:
                    ret_dict['I'][I]['N_t_0'] = events_dict[I]['N_t'][:,ret_dict['p']['t'][0]-1]
                else:
                    ret_dict['I'][I]['N_t_0'] = np.zeros(ret_dict['I'][I]['I_max'], dtype=int)

        
        if val_key_func=='N_t':
                

            
            for i in range(ret_dict['I'][I]['I_max']):
                if ret_dict['I'][I]['N'][i]>0:
                    #ret_dict['I'][I]['kstest'][i,:] = misc_fns.KS_Poisson_N_t_p(N_t=ret_dict['I'][I]['N_t'][i,:], tilde_t=ret_dict['I'][I]['tilde_t'][i,:], N_t_0=ret_dict['I'][I]['N_t_0'][i], tilde_t_0=ret_dict['I'][I]['tilde_t_0'])
                    ret_dict['I'][I]['kstest'][i,:] = misc_fns.KS_test_Poisson_scipy(tilde_t=ret_dict['I'][I]['tilde_N'][i])

                    ret_dict['I'][I]['kstest_exact'][i,:] = misc_fns.KS_test_Poisson_scipy(tilde_t=ret_dict['I'][I]['tilde_N'][i], method='exact')
                    ret_dict['I'][I]['kstest_asymp'][i,:] = misc_fns.KS_test_Poisson_scipy(tilde_t=ret_dict['I'][I]['tilde_N'][i], method='asymp')

                    ret_dict['I'][I]['kstest_alt'][i,:] = misc_fns.KS_test_Poisson(tilde_t=ret_dict['I'][I]['tilde_N'][i])
                    ret_dict['I'][I]['kstest_N_t'][i,:] = misc_fns.KS_Poisson_N_t_p(N_t=ret_dict['I'][I]['N_t'][i,:], tilde_t=ret_dict['I'][I]['tilde_t'][i,:], N_t_0=ret_dict['I'][I]['N_t_0'][i], tilde_t_0=ret_dict['I'][I]['tilde_t_0'][i])

        else:
            ret_dict['I'][I]['D0'] = [None for i in range(ret_dict['I'][I]['I_max'])]
            ret_dict['I'][I]['D1'] = [None for i in range(ret_dict['I'][I]['I_max'])]
            ret_dict['I'][I]['x'] = [np.array([]) for i in range(ret_dict['I'][I]['I_max'])]
            ret_dict['I'][I]['a_p'] = np.zeros(ret_dict['I'][I]['I_max'])
            
            
            if val_key=='delta_t':

                for i in range(ret_dict['I'][I]['I_max']):
                    ret_dict['I'][I]['x'][i] = (1/a_plot[i])*misc_fns.diff_zero(ret_dict['I'][I]['tilde_N'][i])
                    ret_dict['I'][I]['D0'][i] = scipy.stats.expon(scale=1/a_plot[i])


            elif val_key=='delta_tilde_t':
                for i in range(ret_dict['I'][I]['I_max']):
                    if ret_dict['I'][I]['N'][i]>0:
                        ret_dict['I'][I]['x'][i] = misc_fns.diff_zero(ret_dict['I'][I]['tilde_N'][i])
                    ret_dict['I'][I]['D0'][i] = scipy.stats.expon()

            if val_key=='M':
                for i in range(ret_dict['I'][I]['I_max']):
                    if ret_dict['I'][I]['N'][i]>0:
                        ret_dict['I'][I]['x'][i] = events_dict[I][val_key_func][i,ret_dict['I'][I]['N_t'][i,0]:ret_dict['I'][I]['N_t'][i,-1]]
                        #ret_dict['I'][I]['D0'][i] = scipy.stats.genpareto(*scipy.stats.genpareto.fit(ret_dict['I'][I]['x'][i], floc=0))
                        x_train = events_dict[I][val_key_func][i,ret_dict['I'][I]['train_N'][i,0]:ret_dict['I'][I]['train_N'][i,-1]]
                        ret_dict['I'][I]['D0'][i] = scipy.stats.genpareto(*scipy.stats.genpareto.fit(x_train, floc=0))
                    else:
                        ret_dict['I'][I]['D0'][i] = scipy.stats.genpareto(c=0, loc=0, scale=1)

            elif val_key=='tilde_M':

                for i in range(ret_dict['I'][I]['I_max']):
                    if ret_dict['I'][I]['N'][i]>0:
                        ret_dict['I'][I]['x'][i] = events_dict[I]["F_M"][i,ret_dict['I'][I]['N_t'][i,0]:ret_dict['I'][I]['N_t'][i,-1]]

                    ret_dict['I'][I]['D0'][i] = scipy.stats.uniform()


            for i in range(ret_dict['I'][I]['I_max']):
                # if app_norm:
                if True:
                    ret_dict['I'][I]['D1'][i] = scipy.stats.norm()
                    ret_dict['I'][I]['x'][i] = dist_fns.F_D1_inv_F_D0(X=ret_dict['I'][I]['x'][i], F_D1_inv=ret_dict['I'][I]['D1'][i].ppf, F_D0=ret_dict['I'][I]['D0'][i].cdf, excl_bounds_Y=None, excl_inf_Z=True, F_D1_inv_min_max=None)
                else:
                    ret_dict['I'][I]['D1'][i] = ret_dict['I'][I]['D0'][i]

                if ret_dict['I'][I]['N'][i]>0:
                    ret_dict['I'][I]['kstest'][i,:] = scipy.stats.kstest(ret_dict['I'][I]['x'][i], ret_dict['I'][I]['D1'][i].cdf)

            
        
    return ret_dict



def KS_plot_dict(self, tick=None, r=None, period=None, val_key=None, I_s_cat_list=None,
            tf_mode=None, tf_u_key=None, a_u=None, u=None, a_lambda=None,
            overwrite_t_ps=None,
            events=None, tilde_t=None, mode=None, mark=None,
            tilde_t_0_mode=None, a_mode=None,
            I_s=None, I_cat_s=None,
            set_tilde_t_0=None, set_N_t_0=None, 
            CI=None, 
            xs_minmax=None, xs_step=None, 
            plot_date=None,
            xlim=None, ylim=None, 
            xlabel=None, ylabel=None, linelabels=None,
            Nlim_rf=None, Nlim_rf_lower=None, Nlim_rf_upper=None,
            Nlim_rb=None, Nlim_rb_lower=None, Nlim_rb_upper=None, 
            t_symb_base=None, t_bracket=None, k_sub=None, a_subscript=None, val_symb=None,  
            super_I=None, super_I_cat=None, 
            str_tick=None, str_mod=None, 
            normal_K_factor=None,
            display_I_s=None,
            plot_I_list=None,
            ):
    
    
    
    
    if plot_I_list is None:
        plot_I_list = ['I', 'I_s']
    

    if I_s_cat_list is None:
        if I_s is not None:
            I_s_cat_list = [misc_fns.make_iterable_array(I_s)]
        
    

    ret_dict = KS_dict(self, tick=tick, r=r, period=period, val_key=val_key, I_s_cat_list=I_s_cat_list,
                            tf_mode=tf_mode, tf_u_key=tf_u_key, a_u=a_u, u=u, a_lambda=a_lambda,
                            overwrite_t_ps=overwrite_t_ps,
                            events=events, tilde_t=tilde_t, mode=mode, mark=mark,
                            tilde_t_0_mode=tilde_t_0_mode, a_mode=a_mode,
                            )
    
    ret_dict['plot_I'] = {I: {} for I in ret_dict['I']}
    ret_dict['plot_meta'] = {}
    ret_dict['plot_labels'] = {}

    if normal_K_factor is None:
        normal_K_factor = True

    ret_dict['plot_meta']['plot_I_list'] = plot_I_list
    ret_dict['plot_meta']['normal_K_factor'] = normal_K_factor


    
    if I_s is None:
        I_s = np.arange(ret_dict['I']['I']['I_max'])
    I_s = misc_fns.make_iterable_array(I_s)
    ret_dict['plot_I']['I']['I_s'] = I_s
    ret_dict['plot_I']['I']['I_s_size'] = np.size(I_s)



    I_s_cat_list = ret_dict['meta']['I_s_cat_list']

    if I_cat_s is None:
        I_cat_s = np.arange(ret_dict['I']['I_cat']['I_max'])
    I_cat_s = misc_fns.make_iterable_array(I_cat_s)
    ret_dict['plot_I']['I_cat']['I_s'] = I_cat_s
    ret_dict['plot_I']['I_cat']['I_s_size'] = np.size(I_cat_s)

    ret_dict['plot_I']['I']['meta_I_s'] = ret_dict['plot_I']['I']['I_s']
    ret_dict['plot_I']['I_cat']['meta_I_s'] = np.sort(np.unique(np.concatenate([ret_dict['meta']['I_s_cat_list'][i] for i in ret_dict['plot_I']['I_cat']['I_s']])))

    
    ret_dict['plot_meta']['meta_I_s'] = np.sort(np.unique(np.concatenate([ret_dict['plot_I'][I]['meta_I_s'] for I in ret_dict['plot_I']])))

    if display_I_s is None:
        display_I_s = ret_dict['plot_meta']['meta_I_s']
    elif display_I_s=='all':
        display_I_s = np.arange(ret_dict['I']['I']['I_max'])
    ret_dict['plot_meta']['display_I_s'] = display_I_s




    if str_tick is None:
        str_tick = f"$r_{{t}}$ [{ret_dict['meta']['tick']}]"
    ret_dict['plot_labels']['str_tick'] = str_tick

    
    if t_symb_base is None:
        t_symb_base = f"t"
    ret_dict['plot_labels']['t_symb_base'] = t_symb_base

    if t_bracket is None:
        t_bracket = False
    ret_dict['plot_labels']['t_bracket'] = t_bracket

    if a_subscript is None:
        a_subscript = f"\\lambda"
    ret_dict['plot_labels']['a_subscript'] = a_subscript

    
    if super_I is None:
        super_I = np.concatenate([["\\leftarrow", "\\rightarrow"], [f"I{k}" for k in range(2,ret_dict['I']['I']['I_max'])]])[:ret_dict['I']['I']['I_max']]
        
    if super_I_cat is None:
        super_I_cat = np.concatenate([["\\leftrightarrow"], [f"CAT{k}" for k in range(1,ret_dict['I']['I_cat']['I_max'])]])[:ret_dict['I']['I_cat']['I_max']]

    ret_dict['plot_I']['I']['super'] = super_I
    ret_dict['plot_I']['I_cat']['super'] = super_I_cat



    if len(plot_I_list)==0:
        I_base = plot_I_list[0]
    elif np.size(I_s)==0:
        I_base = 'I_cat'
    else:
        I_base = 'I'
    ret_dict['plot_meta']['I_base'] = I_base



    

    if np.size(ret_dict['plot_I'][I_base]['I_s'])==1:
        xylabel_super = ret_dict['plot_I'][I_base]['super'][ret_dict['plot_I'][I_base]['I_s'][0]]
    else:
        xylabel_super = f"\\leftrightharpoons"
    ret_dict['plot_labels']['xylabel_super'] = xylabel_super



    if k_sub is None:
        k_sub = "k"
    if k_sub=="":
        k_sub_str = ""
    else:
        k_sub_str = f"_{{{k_sub}}}"
    ret_dict['plot_labels']['k_sub'] = k_sub
    ret_dict['plot_labels']['k_sub_str'] = k_sub_str

    if val_symb is None:
        if val_key==f'N_t':
            if t_bracket:
                val_symb = f"N^{{i}}{{\\left(t\\right)}}"
            else:
                val_symb = f"N_{{t}}"
        elif val_key==f'N_tilde_t':
            if t_bracket:
                # AM3-9
                #val_symb = f"N^{{i}}{{\\left(\\tilde{{t}}^{{i}}\\right)}}"
                val_symb = f"N^{{i}}{{\\left(\\Pi^{{i}}\\right)}}"
            else:
                # AM3-9
                #val_symb = f"N_{{\\tilde{{t}}}}"
                val_symb = f"N_{{\\Pi}}"
        elif val_key==f'delta_t':
            val_symb = f"\\Delta{{t}}"
        elif val_key==f'delta_tilde_t':
            # AM3-9
            #val_symb = f"\\Delta{{\\tilde{{t}}}}"
            val_symb = f"\\Delta{{\\Pi}}"
        elif val_key==f'M':
            val_symb = f"M"
        elif val_key==f"tilde_M":
            val_symb = f"\\tilde{{M}}"

    if np.isin(val_key, ['N_t', 'N_tilde_t']):
        if t_bracket:
            val_symb_arrow = f"{val_symb}".replace(f"^{{i}}", f"^{{{xylabel_super}}}")
            #val_symb_arrow = f"{val_symb[0]}^{{{xylabel_super}}}{val_symb[1:]}"
        else:
            val_symb_arrow = f"{val_symb}^{{{xylabel_super}}}"
    else:
        val_symb_arrow = f"{val_symb}{k_sub_str}^{{{xylabel_super}}}"
        
    ret_dict['plot_labels']['val_symb'] = val_symb
    ret_dict['plot_labels']['val_symb_arrow'] = val_symb_arrow


    if np.isin(val_key, ['N_t', 'N_tilde_t']):
        if np.isin(val_key, ['N_t']):
            resid_symb = f"t"
        else:
            # AM3-9
            #resid_symb = f"\\tilde{{t}}"
            resid_symb = f"\\Pi"
        resid_symb_arrow = f"{resid_symb}{k_sub_str}^{{{xylabel_super}}}"
    else:
        resid_symb = val_symb
        resid_symb_arrow = val_symb_arrow

    ret_dict['plot_labels']['resid_symb'] = resid_symb
    ret_dict['plot_labels']['resid_symb_arrow'] = resid_symb_arrow
    

    if CI is None:
        CI = [0.95, 0.99]
    CI = misc_fns.make_iterable_array(CI)
    ret_dict['plot_meta']['CI'] = CI




    if np.isin(ret_dict['meta']['val_key_func'], ["N_t", "delta_t",]):
        if np.isin(val_key, ["N_t", "delta_t"]):
            
            t_symb = f"{t_symb_base}"
            t_symb_super_I = [f"" for a in super_I]
            t_symb_xylabel_super = f""
            t_symb_xy = t_symb

        elif np.isin(val_key, ["N_tilde_t", "delta_tilde_t"]):

            # AM3-9
            #t_symb = f"\\tilde{{{t_symb_base}}}"
            if t_symb_base[-3:]=='tau':
                t_symb = "\\Lambda"
            else: 
                t_symb = f"\\Pi"
            t_symb_super_I = [f"^{{{a}}}" for a in super_I]
            t_symb_xylabel_super = f"^{{{xylabel_super}}}"
            t_symb_xy = f"{t_symb}{t_symb_xylabel_super}"






    if np.isin(ret_dict['meta']['val_key_func'], ['N_t']):

        if plot_date is None:
            plot_date = True
        ret_dict['plot_meta']['plot_date'] = plot_date

        if set_N_t_0 is None:
            set_N_t_0 = 'all'
        if set_tilde_t_0 is None:
            set_tilde_t_0 = 'all'
        ret_dict['plot_meta']['set_N_t_0'] = set_N_t_0
        ret_dict['plot_meta']['set_tilde_t_0'] = set_tilde_t_0

        ret_dict['plot_meta']['index_lim'] = ret_dict['all']['r'].index[ret_dict['p']['t'] - np.array([0,1])]


        if Nlim_rf is None:
            Nlim_rf = 1

            dir = ['floor', 'ceil']
            Nlim_rfs = np.repeat(Nlim_rf,2)
            if Nlim_rf_lower is not None:
                Nlim_rfs[0] = Nlim_rf_lower
            if Nlim_rf_upper is not None:
                Nlim_rfs[1] = Nlim_rf_upper

            Nlim_rbs = np.repeat(Nlim_rb,2)
            if Nlim_rb_lower is not None:
                Nlim_rbs[0] = Nlim_rb_lower
            if Nlim_rf_upper is not None:
                Nlim_rbs[1] = Nlim_rb_upper



        for I in ret_dict['plot_I']:

            if normal_K_factor:
                K_factor_T1 = np.outer(ret_dict['I'][I]['K_factor'], np.ones(1+ret_dict['p']['T']))
            #else:
            #    K_factor_T1 = np.ones((ret_dict['I'][I]['I_max'], 1+ret_dict['p']['T']), dtype=float)
        
            if set_tilde_t_0=='all':
                ret_dict['plot_I'][I]['tilde_t_0'] = np.zeros(ret_dict['I'][I]['I_max'], dtype=float)
            elif set_tilde_t_0=='min':
                ret_dict['plot_I'][I]['tilde_t_0'] = ret_dict['I'][I]['tilde_t_0'] - np.min(ret_dict['I'][I]['tilde_t_0'][I_s])
            else:
                ret_dict['plot_I'][I]['tilde_t_0'] = ret_dict['I'][I]['tilde_t_0']
            ret_dict['plot_I'][I]['tilde_t'] = ret_dict['I'][I]['tilde_t'] - np.outer(ret_dict['I'][I]['tilde_t_0']-ret_dict['plot_I'][I]['tilde_t_0'], np.ones(ret_dict['p']['T'], dtype=float))

            if normal_K_factor:
                ret_dict['plot_I'][I]['tilde_t_CI'] = (ret_dict['plot_I'][I]['tilde_t'] - np.outer(ret_dict['plot_I'][I]['tilde_t_0'], np.ones(ret_dict['p']['T'])))/K_factor_T1[:,1:]
            else:
                ret_dict['plot_I'][I]['tilde_t_CI'] = (ret_dict['plot_I'][I]['tilde_t'] - np.outer(ret_dict['plot_I'][I]['tilde_t_0'], np.ones(ret_dict['p']['T'])))


            if set_N_t_0=='all':
                ret_dict['plot_I'][I]['N_t_0'] = np.zeros(ret_dict['I'][I]['I_max'], dtype=int)
            elif set_N_t_0=='min':
                ret_dict['plot_I'][I]['N_t_0'] = ret_dict['I'][I]['N_t_0'] - ret_dict['I'][I]['K_factor']*np.min(ret_dict['I'][I]['N_t_0'][I_s])
            else:
                ret_dict['plot_I'][I]['N_t_0'] = ret_dict['I'][I]['N_t_0']
            ret_dict['plot_I'][I]['N_t'] = ret_dict['I'][I]['N_t'] - np.outer(ret_dict['I'][I]['N_t_0']-ret_dict['plot_I'][I]['N_t_0'], np.ones(ret_dict['p']['T'], dtype=int))


            ret_dict['plot_I'][I]['TS_t'] = [pd.Timestamp(ret_dict['all']['r'].index[k]) for k in range(ret_dict['p']['t'][0], ret_dict['p']['t'][1])]
            if ret_dict['p']['t'][0]==0:
                ret_dict['plot_I'][I]['TS_0'] = ret_dict['all']['r'].index[ret_dict['p']['t'][0]] - pd.Timedelta(days=1)
            else:
                ret_dict['plot_I'][I]['TS_0'] = ret_dict['all']['r'].index[ret_dict['p']['t'][0]-1]

            if ret_dict['p']['t'][1]==ret_dict['ps']['all']['t'][1]:
                ret_dict['plot_I'][I]['TS_T1'] = ret_dict['all']['r'].index[ret_dict['p']['t'][1]-1] + pd.Timedelta(days=1)
            else:
                ret_dict['plot_I'][I]['TS_T1'] = ret_dict['all']['r'].index[ret_dict['p']['t'][1]]
                


            if ret_dict['plot_meta']['plot_date'] and ret_dict['meta']['val_key']=="N_t":
                ret_dict['plot_I'][I]['x'] = np.array([np.concatenate([[ret_dict['plot_I'][I]['TS_0']], ret_dict['plot_I'][I]['TS_t']]) for i in range(ret_dict['I'][I]['I_max'])])
                ret_dict['plot_I'][I]['CI_x'] = ret_dict['plot_I'][I]['TS_t']

            else:
                ret_dict['plot_I'][I]['x'] = np.full((ret_dict['I'][I]['I_max'], 1+ret_dict['p']['T']), np.nan)
                if normal_K_factor:
                    ret_dict['plot_I'][I]['x'][:,0] = ret_dict['plot_I'][I]['tilde_t_0']/K_factor_T1[:,0]
                    ret_dict['plot_I'][I]['x'][:,1:] = ret_dict['plot_I'][I]['tilde_t']/K_factor_T1[:,1:]
                else:
                    ret_dict['plot_I'][I]['x'][:,0] = ret_dict['plot_I'][I]['tilde_t_0']
                    ret_dict['plot_I'][I]['x'][:,1:] = ret_dict['plot_I'][I]['tilde_t']

                if False:
                    if normal_K_factor:
                        ret_dict['plot_I'][I]['x'] = ret_dict['plot_I'][I]['x']/K_factor_T1


                #ret_dict['plot_I'][I]['CI_x'] = np.mean(ret_dict['plot_I'][I]['tilde_t'][ret_dict['plot_I'][I]['I_s'],:], axis=0)
                ret_dict['plot_I'][I]['CI_x'] = np.mean(ret_dict['plot_I'][I]['x'][ret_dict['plot_I'][I]['I_s'],1:], axis=0)

            if normal_K_factor:
                ret_dict['plot_I'][I]['y'] = np.full((ret_dict['I'][I]['I_max'], 1+ret_dict['p']['T']), np.nan)
                ret_dict['plot_I'][I]['y'][:,0] = ret_dict['plot_I'][I]['N_t_0']/K_factor_T1[:,0]
                ret_dict['plot_I'][I]['y'][:,1:] = ret_dict['plot_I'][I]['N_t']/K_factor_T1[:,1:]
            else:
                ret_dict['plot_I'][I]['y'] = np.zeros((ret_dict['I'][I]['I_max'], 1+ret_dict['p']['T']), dtype=int)
                ret_dict['plot_I'][I]['y'][:,0] = ret_dict['plot_I'][I]['N_t_0']
                ret_dict['plot_I'][I]['y'][:,1:] = ret_dict['plot_I'][I]['N_t']

            

                

            #ret_dict['plot_I'][I]['CI_y'] = np.array([[np.mean(ret_dict['plot_I'][I]['N_t_0'][ret_dict['plot_I'][I]['I_s']])+scipy.stats.poisson.ppf(mu=np.mean((ret_dict['I'][I]['tilde_t'] - np.outer(ret_dict['I'][I]['tilde_t_0'], np.ones(ret_dict['p']['T'])))[ret_dict['plot_I'][I]['I_s'],:],axis=0), q=(1+((-1)**(1+k))*CI[j])/2) for j in range(len(CI))] for k in range(2)])

            ret_dict['plot_I'][I]['CI_y'] = np.array([[np.mean(ret_dict['plot_I'][I]['y'][ret_dict['plot_I'][I]['I_s'],0]) + 
                                                    scipy.stats.poisson.ppf(mu=np.mean(ret_dict['plot_I'][I]['tilde_t_CI'][ret_dict['plot_I'][I]['I_s'],:],axis=0), q=(1+((-1)**(1+k))*CI[j])/2) for j in range(len(CI))] for k in range(2)])



            # (ret_dict['I'][I]['tilde_t'] - np.outer(ret_dict['I'][I]['tilde_t_0'], np.ones(ret_dict['p']['T'])))[ret_dict['plot_I'][I]['I_s'],:]

            if ret_dict['plot_meta']['plot_date'] and ret_dict['meta']['val_key']=="N_t":
                ret_dict['plot_I'][I]['xlim'] = ret_dict['plot_meta']['index_lim']
            else:
                #ret_dict['plot_I'][I]['xlim'] = np.array([np.min(ret_dict['plot_I'][I]['tilde_t'][ret_dict['plot_I'][I]['I_s'],0]), np.max(ret_dict['plot_I'][I]['tilde_t'][ret_dict['plot_I'][I]['I_s'],-1])])
                ret_dict['plot_I'][I]['xlim'] = np.array([np.min(ret_dict['plot_I'][I]['x'][ret_dict['plot_I'][I]['I_s'],0]), np.max(ret_dict['plot_I'][I]['x'][ret_dict['plot_I'][I]['I_s'],-1])])

            if True:
                ret_dict['plot_I'][I]['Nlim_eff'] = np.array([
                                        np.min(ret_dict['plot_I'][I]['y'][ret_dict['plot_I'][I]['I_s'],0]),
                                        np.max(ret_dict['plot_I'][I]['y'][ret_dict['plot_I'][I]['I_s'],-1]),
                                    ])
            else:
                ret_dict['plot_I'][I]['Nlim_eff'] = np.array([
                                        np.min((ret_dict['plot_I'][I]['N_t_0'][ret_dict['plot_I'][I]['I_s']])/ret_dict['I'][I]['K_factor'][ret_dict['plot_I'][I]['I_s']]),
                                        np.max((ret_dict['plot_I'][I]['N_t_0'][ret_dict['plot_I'][I]['I_s']] + ret_dict['I'][I]['N'][ret_dict['plot_I'][I]['I_s']])/ret_dict['I'][I]['K_factor'][ret_dict['plot_I'][I]['I_s']]),
                                    ])

            ret_dict['plot_I'][I]['ylim'] = [misc_fns.multiple_round(val=ret_dict['plot_I'][I]['Nlim_eff'][k], factor=Nlim_rfs[k], buffer=Nlim_rbs[k], round_type=dir[k]) for k in range(2)]

                

        


        if xlim is None:
            xlim = ret_dict['plot_I'][I_base]['xlim']
        if ylim is None:
            ylim = ret_dict['plot_I'][I_base]['ylim']

        if ret_dict['plot_meta']['plot_date'] and ret_dict['meta']['val_key']=="N_t":
            if xlabel is None:
                xlabel = f"Date"
        else:
            if xlabel is None:
                xlabel = f"${t_symb_xy}$"




        linelabels_post = {I: [f"" for i in range(ret_dict['I'][I]['I_max'])] for I in ret_dict['plot_I']}
        linelabels_pre = {I: [f"" for i in range(ret_dict['I'][I]['I_max'])] for I in ret_dict['plot_I']}
        linelabels_post_t = {I: [f"" for i in range(ret_dict['I'][I]['I_max'])] for I in ret_dict['plot_I']}
        linelabels_pre_t = {I: [f"" for i in range(ret_dict['I'][I]['I_max'])] for I in ret_dict['plot_I']}

        for I in ret_dict['plot_I']:
            for i in range(ret_dict['I'][I]['I_max']):
                if ret_dict['I'][I]['K_factor'][i]!=1:
                    linelabels_post[I][i] = f"/{ret_dict['I'][I]['K_factor'][i]}"
                    linelabels_pre[I][i] = f"{ret_dict['I'][I]['K_factor'][i]}"
                if ret_dict['meta']['val_key'].find('tilde')>=0:
                    if t_bracket:
                        linelabels_post_t[I][i] = f"^{{{ret_dict['plot_I'][I]['super'][i]}}}"
                        linelabels_pre_t[I][i] = linelabels_pre[I][i]
                    else:
                        linelabels_post_t[I][i] = linelabels_post[I][i]


        if t_bracket:
            if linelabels is None:
                linelabels = np.concatenate([[f"$N^{{{ret_dict['plot_I'][I]['super'][i]}}}{{\\left({linelabels_pre_t[I][i]}{t_symb}{linelabels_post_t[I][i]}\\right)}}{linelabels_post[I][i]}$" for i in ret_dict['plot_I'][I]['I_s']] for I in ret_dict['plot_I']])
            if ylabel is None:
                ylabel = f"$N^{{{xylabel_super}}}{{\\left({t_symb}{t_symb_xylabel_super}\\right)}}$"
        else:
            if linelabels is None:
                linelabels = np.concatenate([[f"$N_{{{linelabels_pre_t[I][i]}{t_symb}{linelabels_post_t[I][i]}}}^{{{ret_dict['plot_I'][I]['super'][i]}}}{linelabels_post[I][i]}$" for i in ret_dict['plot_I'][I]['I_s']]  for I in ret_dict['plot_I']])
            if ylabel is None:
                ylabel = f"$N_{{{t_symb}}}^{{{xylabel_super}}}$"

            



    else:

        if xs_minmax is None:
            xs_minmax = np.array([-4,4])
        if xs_step is None:
            xs_step = 0.025
        ret_dict['plot_meta']['xs'] = np.linspace(xs_minmax[0], xs_minmax[1], 1+int(np.diff(xs_minmax)[0]/xs_step))
        ret_dict['plot_meta']['xs_minmax'] = xs_minmax
        ret_dict['plot_meta']['xs_step'] = xs_step
        

        if val_key=='delta_t':
            D0_symb = f"\\mathcal{{E}}_{{(1/a_{{{ret_dict['plot_labels']['a_subscript']}}})}}"
        elif val_key=='delta_tilde_t':
            D0_symb = f"\\mathcal{{E}}_{{(1)}}"
        elif val_key=='M':
            D0_symb = f"\\hat{{\\mathcal{{P}}}}_{{M}}^{{{xylabel_super}}}"
        elif val_key=='tilde_M':
            D0_symb = f"\\mathcal{{E}}_{{(1)}}"
        ret_dict['plot_meta']['D0_symb'] = D0_symb

        #if ret_dict['meta']['app_norm']:
        if True:
            D1_symb = f"\\mathcal{{N}}_{{(0,1)}}"
        else:
            D1_symb = D0_symb
        ret_dict['plot_meta']['D1_symb'] = D1_symb


        for I in ret_dict['plot_I']:
            ret_dict['plot_I'][I]['x'] = [np.concatenate([[np.nan], np.sort(ret_dict['I'][I]['x'][i])]) for i in range(ret_dict['I'][I]['I_max'])]
            ret_dict['plot_I'][I]['y'] = [(np.arange(ret_dict['I'][I]['N'][i]+1))/ret_dict['I'][I]['N'][i] for i in range(ret_dict['I'][I]['I_max'])]

            ret_dict['plot_I'][I]['CI_x'] = ret_dict['plot_meta']['xs']
            ret_dict['plot_I'][I]['CI_y'] = np.array([[scipy.stats.norm().cdf(ret_dict['plot_meta']['xs']) + ((-1)**(1+k))*np.sqrt(np.log(2/((1-CI[j])/2)))/np.sqrt(2*np.product(ret_dict['I'][I]['N'][ret_dict['plot_I'][I]['I_s']])**(1/ret_dict['I'][I]['N'][ret_dict['plot_I'][I]['I_s']].size)) for j in range(len(CI))] for k in range(2)])

            ret_dict['plot_I'][I]['xlim'] = xs_minmax
            ret_dict['plot_I'][I]['ylim'] = plot_fns.buffer(b=1.04, x=np.array([0,1]))



        linelabels = np.concatenate([[f"$F_{{{val_symb}}}^{{{ret_dict['plot_I'][I]['super'][i]}}}$" for i in ret_dict['plot_I'][I]['I_s']] for I in ret_dict['plot_I']])

        if ylabel is None:
            ylabel = f"Empirical $F_{{{val_symb}}}^{{{xylabel_super}}}$"
        if xlabel is None:
            xlabel = f"Normal quantiles $\\left[{D0_symb}\\right]$"

        if xlim is None:
            xlim = ret_dict['plot_I'][I_base]['xlim']
        if ylim is None:
            ylim = ret_dict['plot_I'][I_base]['ylim']


    

    if np.isin(ret_dict['meta']['val_key_func'], ['N_t']):

        if plot_date is None:
            plot_date = True



    ret_dict['plot_labels']['linelabels'] = linelabels
    ret_dict['plot_labels']['xlabel'] = xlabel
    ret_dict['plot_labels']['ylabel'] = ylabel

    ret_dict['plot_meta']['xlim'] = xlim
    ret_dict['plot_meta']['ylim'] = ylim



    return ret_dict
    


def KS_plot(self, tick=None, r=None, period=None, val_key=None, I_s_cat_list=None,
            tf_mode=None, tf_u_key=None, a_u=None, u=None, a_lambda=None,
            overwrite_t_ps=None,
            events=None, tilde_t=None, mode=None, mark=None,
            tilde_t_0_mode=None, a_mode=None,
            I_s=None,
            set_tilde_t_0=None, set_N_t_0=None, 
            CI=None, 
            xs_minmax=None, xs_step=None, 
            plot_date=None,
            xlim=None, ylim=None, 
            xlabel=None, ylabel=None, linelabels=None,
            Nlim_rf=None, Nlim_rf_lower=None, Nlim_rf_upper=None,
            Nlim_rb=None, Nlim_rb_lower=None, Nlim_rb_upper=None, 
            t_symb_base=None, t_bracket=None, k_sub=None, a_subscript=None, val_symb=None,  super_I=None, str_tick=None, str_mod=None, 
            normal_K_factor=None,
            plot_I_list=None, plot_I_list_plot=None, plot_I_list_p=None,
            ax=None, ax2=None, 
            thesis=None, 
            lw=None,
            an_mode=None, an_xycoords=None, an_fs=None, an_x0=None, an_y0=None, an_yi=None,
            an_h_p=None, an_v_p=None, an_fs_p=None, an_x0_p=None, an_y0_p=None, an_yi_p=None, 
            an_h_t=None, an_v_t=None, an_fs_t=None, an_x0_t=None, an_y0_t=None, an_yi_t=None, 
            an_yM_tick=None, an_yM_mod=None, an_yM_u=None, an_yM_a_lambda=None, 
            plot_with_step=None,
            plot_main_legend=None, plot_ci_legend=None, 
            plot_pvals=None, 
            plot_order=None,
            plot_tick=None, plot_mod=None, plot_u=None, plot_a_lambda=None,
            uncond=None, hwk_uncond=None,
            main_legend_bbox_to_anchor=None, main_legend_loc=None, main_legend_ncols=None, main_legend_markerfirst=None,
            ci_legend_bbox_to_anchor=None, ci_legend_loc=None, ci_legend_ncols=None, ci_legend_markerfirst=None,
            u_scif_pow10=None, u_dp=None, u_dp_buff=None, u_val_plus=None,
            a_u_scif_pow10=None, a_u_dp=None, a_u_dp_buff=None, a_u_val_plus=None,
            a_lambda_scif_pow10=None, a_lambda_dp=None, a_lambda_dp_buff=None, a_lambda_val_plus=None,
            cols_I=None, cols_I_cat=None, cols_CI=None,
            ):
    

    if plot_I_list_plot is None:
        if plot_I_list is not None:
            plot_I_list_plot = plot_I_list
        else:
            plot_I_list_plot = ['I']
    if plot_I_list_p is None:
        if plot_I_list is not None:
            plot_I_list_p = plot_I_list
        else:
            plot_I_list_p = ['I']
    if plot_I_list is None:
        if len(plot_I_list_plot)==1 and len(plot_I_list_plot)==1 and plot_I_list_plot[0]==plot_I_list_p[0]:
            plot_I_list = [plot_I_list_plot[0]]
        else:
            plot_I_list = ['I', 'I_cat']
    

    ret_dict = KS_plot_dict(self, tick=tick, r=r, period=period, val_key=val_key, I_s_cat_list=I_s_cat_list,
                            tf_mode=tf_mode, tf_u_key=tf_u_key, a_u=a_u, u=u, a_lambda=a_lambda,
                            overwrite_t_ps=overwrite_t_ps,
                            events=events, tilde_t=tilde_t, mode=mode, mark=mark,
                            tilde_t_0_mode=tilde_t_0_mode, a_mode=a_mode,
                            I_s=I_s,
                            set_tilde_t_0=set_tilde_t_0, set_N_t_0=set_N_t_0, 
                            CI=CI, 
                            xs_minmax=xs_minmax, xs_step=xs_step, 
                            plot_date=plot_date,
                            xlim=xlim, ylim=ylim, 
                            xlabel=xlabel, ylabel=ylabel, linelabels=linelabels,
                            Nlim_rf=Nlim_rf, Nlim_rf_lower=Nlim_rf_lower, Nlim_rf_upper=Nlim_rf_upper,
                            Nlim_rb=Nlim_rb, Nlim_rb_lower=Nlim_rb_lower, Nlim_rb_upper=Nlim_rb_upper, 
                            t_symb_base=t_symb_base, t_bracket=t_bracket, k_sub=k_sub, a_subscript=a_subscript, val_symb=val_symb,  super_I=super_I, str_tick=str_tick, str_mod=str_mod, 
                            normal_K_factor=normal_K_factor,
                            plot_I_list=plot_I_list,
                            )
    
    
    
    if thesis is None:
        thesis = True

    an_mode = None
    if thesis:
        an_mode = 'thesis'

    if lw is None:
        lw = 1.5

    if an_h_p is None:
        an_h_p = 'left'
    if an_v_p is None:
        an_v_p = 'top'
    if an_h_t is None:
        an_h_t = 'right'
    if an_v_t is None:
        an_v_t = 'bottom'


    an_params = {'lambda': plot_fns.calc_an_params(an_fs=an_fs, an_x0=an_x0, an_y0=an_y0, an_yi=an_yi, an_mode=an_mode, an_xycoords=an_xycoords)}
    an_params['spec'] = {
                            'p': plot_fns.calc_an_params_dir(**an_params['lambda'], flip_x0ry0t=True,
                                                                an_h=an_h_p, an_v=an_v_p, an_fs_ow=an_fs_p, an_x0_ow=an_x0_p, an_y0_ow=an_y0_p, an_yi_ow=an_yi_p, 
                                                            ),
                            't': plot_fns.calc_an_params_dir(**an_params['lambda'], flip_x0ry0t=True,
                                                                an_h=an_h_t, an_v=an_v_t, an_fs_ow=an_fs_t, an_x0_ow=an_x0_t, an_y0_ow=an_y0_t, an_yi_ow=an_yi_t, 
                                                            ),
                        }









    if plot_main_legend is None:
        plot_main_legend = True
    if plot_ci_legend is None:
        plot_ci_legend = True
    if plot_pvals is None:
        plot_pvals = True

    if plot_tick is None:
        plot_tick = True

    
    if plot_mod is None:
        plot_mod = True
        if False:
            if ret_dict['meta']['val_key'].find('tilde')>=0 or str_mod is not None:
                plot_mod = True
            else:
                plot_mod = False

        
    if plot_u is None:
        plot_u = True
    if plot_a_lambda is None:
        plot_a_lambda = True

    
    if uncond is None:
        if ret_dict['meta']['val_key'].find('tilde')>=0 or str_mod is not None:
            uncond = False
        else:
            uncond = True



    if ax is None:
        ax_arg = False
        fig, ax = plt.subplots()
    else:
        ax_arg = True
    
    ax.xaxis.set_label_position('bottom')
    ax.yaxis.set_label_position('left')
    ax.tick_params(left=True, right=True, top=True, bottom=True,
                                        labelleft=True, labelright=False, labeltop=False, labelbottom=True)

    if ax2 is None:
        ax2 = ax.twiny()


    
    ax.tick_params(left=True, right=True, top=True, bottom=True,
                                        labelleft=True, labelright=False, labeltop=False, labelbottom=True)
    
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(left=False, right=False, top=False, bottom=False,
                                        labelleft=False, labelright=False, labeltop=False, labelbottom=False)
    


    I_base = ret_dict['plot_meta']['I_base']
    

    a_plot = ret_dict['I']['I']['a_plot']
    super_I_i = ret_dict['plot_I']['I']['super']
    super_I_cat = ret_dict['plot_I']['I_cat']['super']
    
    u = ret_dict['u']['u']
    a_u = ret_dict['u']['a_u']
    a_lambda = ret_dict['u']['a_lambda']


    a_mode = ret_dict['meta']['a_mode']
    a_subscript = ret_dict['plot_labels']['a_subscript']

    resid = ret_dict['plot_labels']['resid_symb_arrow']


    if plot_with_step is None:
        plot_with_step = True




    linelabels = ret_dict['plot_labels']['linelabels']

    ylabel = ret_dict['plot_labels']['ylabel']
    xlabel = ret_dict['plot_labels']['xlabel']

    ylim = ret_dict['plot_meta']['ylim']
    xlim = ret_dict['plot_meta']['xlim']


    CI = ret_dict['plot_meta']['CI']
    CI_x = ret_dict['plot_I'][I_base]['CI_x']
    CI_y = ret_dict['plot_I'][I_base]['CI_y']





    plot_I_s_cat_plot_I_s_size =  np.array([ret_dict['plot_I'][I]['I_s_size'] for I in plot_I_list_plot])
    plot_I_s_cat_p_I_s_size =  np.array([ret_dict['plot_I'][I]['I_s_size'] for I in plot_I_list_p])





    


    if cols_I is None:
        cols_I = plot_fns.calc_cols_tail(rows=False)
    if cols_I_cat is None:
        cols_I_cat = cols_I[ret_dict['I']['I']['I_max']:]

    cols = {'I': cols_I, 'I_cat': cols_I_cat}


    N_CI = np.size(CI)
    if cols_CI is None:
        cols_CI = plot_fns.calc_cols_CI(N_CI=N_CI)


    for i in range(N_CI):
        j = N_CI-(1+i)
        ax.fill_between(CI_x, y1=CI_y[0][j], y2=CI_y[1][j], color=cols_CI[j])
            
            
    for I_i in range(len(plot_I_list_plot)):
        I = plot_I_list_plot[I_i]
        for i in ret_dict['plot_I'][I]['I_s']:
            if plot_with_step:
                ax.plot(np.repeat(ret_dict['plot_I'][I]['x'][i],2)[2:], np.repeat(ret_dict['plot_I'][I]['y'][i],2)[1:-1], color=cols[I][i], lw=lw)
            else:
                ax.plot(ret_dict['plot_I'][I]['x'][i], ret_dict['plot_I'][I]['y'][i], color=cols[I][i], lw=lw)

    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.set_ylim(ylim)
    ax2.set_ylim(ylim)

    ax.set_xlim(xlim)
    ax2.set_xlim(xlim)

    ax.tick_params(labeltop=False)


    if plot_pvals:

        p_vals = np.concatenate([ret_dict['I'][I]['kstest'][ret_dict['plot_I'][I]['I_s'],1] for I in plot_I_list_p])
        p_cols = np.concatenate([cols[I][ret_dict['plot_I'][I]['I_s']] for I in plot_I_list_p])
        p_symb = np.concatenate([[f"p_{{\mathrm{{KS}}}}^{{{ret_dict['plot_I'][I]['super'][i]}}}" for i in ret_dict['plot_I'][I]['I_s']] for I in plot_I_list_p])


        xy_box_p = plot_fns.annotate_pvals(
                   p_vals=p_vals, p_cols=p_cols, p_symb=p_symb,
                   p_thresh=1-CI,
                   **an_params['spec']['p'], flip_x0ry0t=False,
                   ax=ax,
                   )
    else:
        xy_box_p = plot_fns.calc_xy_box(default_full=False)




    xy_box_t = hwks_fns.annotate_model_Hawkes(
                                                            ax=ax, 
                                                            str_tick=None, str_mod=None, str_u=None, str_a_lambda=None, 
                                                            plot_tick=plot_tick, plot_mod=plot_mod, plot_u=plot_u, plot_a_lambda=plot_a_lambda, 
                                                            an_yM_mod=an_yM_mod, an_yM_tick=an_yM_tick, an_yM_u=an_yM_u, an_yM_a_lambda=an_yM_a_lambda,
                                                            u_scif_pow10=u_scif_pow10, u_dp=u_dp, u_dp_buff=u_dp_buff, u_val_plus=u_val_plus, 
                                                            a_lambda_scif_pow10=a_lambda_scif_pow10, a_lambda_dp=a_lambda_dp, a_lambda_dp_buff=a_lambda_dp_buff, a_lambda_val_plus=a_lambda_val_plus, 
                                                            a_u_scif_pow10=a_u_scif_pow10, a_u_dp=a_u_dp, a_u_dp_buff=a_u_dp_buff, a_u_val_plus=a_u_val_plus, 
                                                            flip_x0ry0t=False,
                                                            #an_mode=None, an_xycoords=None, an_h=None, an_v=None, an_fs=None, an_x0=None, an_y0=None, an_xi=None, an_yi=None, an_xs=None, an_ys=None,
                                                            **an_params['spec']['t'],
                                                            u=u, a_u=a_u, a_lambda=a_plot, 
                                                            display_I_s=ret_dict['plot_meta']['meta_I_s'], super_I_i=None, super_I_cat=None,
                                                            tf_mode=tf_mode, a_mode=a_mode, a_subscript=a_subscript, plot_order=plot_order,
                                                            tick=tick,
                                                            resid=resid, mode=mode, mark=mark, uncond=uncond, hwk_uncond=hwk_uncond,
                                                            )
        

        



        

    
    if main_legend_loc is None:
        main_legend_loc = 'upper left'
    if main_legend_ncols is None:
        main_legend_ncols = np.min([np.size(I_s), 1])
    if main_legend_markerfirst is None:
        main_legend_markerfirst = True
    if ci_legend_loc is None:
        ci_legend_loc = 'lower right'
    if ci_legend_ncols is None:
        ci_legend_ncols = np.min([N_CI, 1])
    if ci_legend_markerfirst is None:
        ci_legend_markerfirst = True


    if False:
        an_y_bound = dict()
        
        an_y_bound['an'] = {
                            'p': {
                                    'plot': bool(plot_pvals),
                                    'y': np.repeat(0.0, 2),
                                    'an_v': an_params['spec']['p']['an_v']
                                },
                            't': {
                                    'plot': bool((plot_tick+plot_mod+plot_u+plot_a_lambda)>0),
                                    'y': np.repeat(0.0, 2),
                                    'an_v': an_params['spec']['t']['an_v']
                                },
                        }
        
        if plot_pvals:
            an_y_bound['an']['p']['y'] = np.cumsum([an_params['spec']['p']['an_y0'], an_params['spec']['p']['an_ys']*np.sum(plot_I_s_cat_p_I_s_size)*an_params['spec']['p']['an_yi']])
        if np.size(plot_index)>0:
            an_y_bound['an']['t']['y'] = np.cumsum([an_params['spec']['t']['an_y0'], an_params['spec']['t']['an_ys']*(1+np.max(an_yM_all[plot_index]))*an_params['spec']['t']['an_yi']])



        an_y_bound['y_set'] = {
                                'top': [[1]],
                                'bottom': [[0]],
                            }
        for an in an_y_bound['an']:
            if an_y_bound['an'][an]:
                an_y_bound['y_set'][an_y_bound['an'][an]['an_v']].append(an_y_bound['an'][an]['y'])
        
        an_y_bound['y']  = {
                                'top': np.min(np.concatenate(an_y_bound['y_set']['top'])),
                                'bottom': np.max(np.concatenate(an_y_bound['y_set']['bottom'])),
                            }

            
        default_legend_bbox_to_anchor = (
                                            0,
                                            an_y_bound['y']['bottom'],
                                            1,
                                            an_y_bound['y']['top']-an_y_bound['y']['bottom'],
                                        )
    else:
        default_legend_bbox_to_anchor = plot_fns.calc_bbox(xy_box=plot_fns.calc_between_xy_boxes(xy_boxes=[xy_box_p, xy_box_t], along_x=False, along_y=True))
         


    if main_legend_bbox_to_anchor is None:
        main_legend_bbox_to_anchor = default_legend_bbox_to_anchor
    if ci_legend_bbox_to_anchor is None:
        ci_legend_bbox_to_anchor = default_legend_bbox_to_anchor


    ret_dict['plot_meta']['xy_box'] = {
                                        'p': xy_box_p,
                                        't': xy_box_t,
                                    }
    
    ret_dict['plot_meta']['bbox'] = {
                                        'main': main_legend_bbox_to_anchor,
                                        'ci': ci_legend_bbox_to_anchor,
                                    }
    

    if plot_main_legend:
        ax.legend([ax.get_children()[N_CI+i] for i in range(np.sum(plot_I_s_cat_plot_I_s_size))], linelabels, 
                  loc=main_legend_loc, bbox_to_anchor=main_legend_bbox_to_anchor, ncol=main_legend_ncols, markerfirst=main_legend_markerfirst,
                  fontsize=an_params['lambda']['an_fs'])
    if plot_ci_legend:
        ax2.legend([ax.get_children()[N_CI-(1+k)] for k in range(N_CI)], 
                   [f"${100*CI[k]:.0f}\\%$ CI" for k in range(N_CI)],
                   loc=ci_legend_loc, bbox_to_anchor=ci_legend_bbox_to_anchor, ncol=ci_legend_ncols, markerfirst=ci_legend_markerfirst,
                   fontsize=an_params['lambda']['an_fs'])
        




    return ret_dict



def LB_dict(self, tick=None, r=None, period=None, val_key=None, I_s_cat_list=None,
            tf_mode=None, tf_u_key=None, a_u=None, u=None, a_lambda=None,
            overwrite_t_ps=None,
            events=None, tilde_t=None, mode=None, mark=None,
            a_mode=None, tilde_t_0_mode=None,
            H=None, app_norm=None,
            ):
    




    if tick is None:
        tick = [t for t in self.tk][0]


    if mode is None:
        mode='ci'
    if mark is None:
        mark='mark'
    

    if r is None:
        r = self.tk[tick].pd.lr['p']['all']
    

    ret_dict = {}

    ret_dict['u'] = get_tf_a_u_u(self=self, tick=tick, 
                                                tf_mode=tf_mode, tf_u_key=tf_u_key, a_u=a_u, u=u, a_lambda=a_lambda,
                                                mode=mode, mark=mark, 
                                                as_dict=True)
    

    ret_dict['ps'] = calc_ps_tT(self=self, tick=tick, r=r, overwrite_t_ps=overwrite_t_ps)
    ret_dict['p'] = ret_dict['ps'][period]

    





    if events is None:
        if val_key.find('tilde')>=0:
            events = self.tk[tick].tt.tt[mode][mark]._events
        else:
            events = dist_fns.calc_events(x=r.values, u=ret_dict['u']['u'], a_u=ret_dict['u']['a_u'], as_array=True, first_u=True)

    I_max = np.shape(events['N'])[0]


    if I_s_cat_list is None:
        I_s_cat_list = dist_fns.calc_I_s_cat_list_default(I_max=I_max)
    K_max = len(I_s_cat_list)


    if a_mode is None:
        if ret_dict['u']['tf_mode']=='u':
            a_mode = f"lambda"
        else:
            a_mode = f"u"
        
    a_plot = ret_dict['u'][f'a_{a_mode}']


    I_list = ['I', 'I_cat']
    tilde_t_all_dict = {I: None for I in I_list}
    if tilde_t is not None:
        tilde_t_all_dict['I'] = tilde_t
    else:
        if np.isin(val_key, ["N_t", "delta_t"]):
            tilde_t_all_dict['I'] = np.array([(1+np.arange(ret_dict['ps']['all']['T']))*a_plot[i%a_plot.size] for i in range(I_max)])
        elif np.isin(val_key, ["N_tilde_t", "delta_tilde_t"]):
            tilde_t_all_dict['I'] = self.tk[tick].tt.tt[mode][mark]._t_series['tilde_t']


    tilde_t_all_dict['I_cat'] = dist_fns.calc_tilde_t_cat(tilde_t=tilde_t_all_dict['I'], I_max=None, T=None, K_max=None, I_s_cat_list=I_s_cat_list)

    
    events_dict = {
                        'I': events,
                        'I_cat': dist_fns.calc_events_cat(events=events, I_s_cat_list=I_s_cat_list, tilde_t=tilde_t_all_dict['I']),
                    }

    
    ret_dict['I'] = {I: calc_p_N(N_t=events_dict[I]['N_t'], **ret_dict['p']) for I in events_dict}
    ret_dict['I']['I']['I_max'] = I_max
    ret_dict['I']['I_cat']['I_max'] = K_max


    for sub in ['u', 'lambda']:
        ret_dict['I']['I'][f'a_{sub}'] = ret_dict['u'][f'a_{sub}']
        ret_dict['I']['I_cat'][f'a_{sub}'] = np.array([np.sum(ret_dict['u'][f'a_{sub}'][I_s_cat_list[k]])] for k in range(K_max))



    for I in ret_dict['I']:
        ret_dict['I'][I]['a_plot'] = ret_dict['I'][I][f'a_{a_mode}']
        if val_key.find('tilde')<0:
            events_dict[I]['tilde_t'] = events_dict[I][f'tilde_t_a_{a_mode}']

    
    ret_dict['I']['I']['K_factor'] = np.ones(I_max, dtype=int)
    ret_dict['I']['I_cat']['K_factor'] = np.array([len(I_s_cat_list[k]) for k in range(K_max)], dtype=int)


    if tilde_t_0_mode is None:
        tilde_t_0_mode = 'N'


    if H is None:
        H = 50
    if app_norm is None:
        app_norm = True

    val_key_func = calc_val_key_func(val_key=val_key)
    ret_dict['meta'] = {
                            'val_key': val_key,
                            'val_key_func': val_key_func,
                            'tick': tick,
                            'period': period,
                            'overwrite_t_ps': overwrite_t_ps,
                            'tilde_t_0_mode': tilde_t_0_mode,
                            'a_mode': a_mode,
                            'I_max': I_max,
                            'K_max': K_max,
                            'I_s_cat_list': I_s_cat_list,
                            'H': H,
                            'app_norm': app_norm,
                            'train_t': ret_dict['ps']['train']['t'],
                        }

    for I in ret_dict['I']:
        #ret_dict['I'][I]['train_N'] = ret_dict['I'][I]['N_t'][:,[ret_dict['meta']['train_t'][0],ret_dict['meta']['train_t'][1]-1]]
        ret_dict['I'][I]['train_N'] = events_dict[I]['N_t'][:,[ret_dict['meta']['train_t'][0],ret_dict['meta']['train_t'][1]-1]]

    
    

    ret_dict['all'] = {}
    ret_dict['all']['r'] = r

    ret_dict['Hawkes'] = {
                            'mode': mode,
                            'mark': mark,
                            'a_u': self.tk[tick].tt.tt[mode][mark]._parameters.a_u.get_value(),
                            'u': self.tk[tick].tt.tt[mode][mark]._parameters.u.get_value(),
                        }


    




    for I in ret_dict['I']:
        ret_dict['I'][I]['lbtest'] = np.full((ret_dict['I'][I]['I_max'],H,2), np.nan)
        ret_dict['I'][I]['rho'] = np.full((ret_dict['I'][I]['I_max'],H), np.nan)


        if np.isin(val_key_func, ["delta_t"]):

            ret_dict['I'][I]['tilde_t_all'] = tilde_t_all_dict[I]
            ret_dict['I'][I]['tilde_t'] = ret_dict['I'][I]['tilde_t_all'][:,ret_dict['p']['t'][0]:ret_dict['p']['t'][1]]


            # mod_cat
            ret_dict['I'][I]['t'] = [np.array([]) for i in range(ret_dict['I'][I]['I_max'])]
            ret_dict['I'][I]['t_all'] = [np.array([]) for i in range(ret_dict['I'][I]['I_max'])]
            for i in range(ret_dict['I'][I]['I_max']):
                if ret_dict['I'][I]['N'][i]>0:
                    ret_dict['I'][I]['t'][i] = events_dict[I][f't'][i,ret_dict['I'][I]['N_t'][i,0]:ret_dict['I'][I]['N_t'][i,-1]] 
                if events_dict[I]['N'][i]>0:
                    ret_dict['I'][I]['t_all'][i] = events_dict[I][f't'][i,:events_dict[I]['N'][i]] 



            ret_dict['I'][I]['tilde_t_0'] = np.zeros(ret_dict['I'][I]['I_max'], dtype=float)
            if ret_dict['p']['t'][0] > 0:
                ret_dict['I'][I]['tilde_t_0'] = ret_dict['I'][I]['tilde_t_all'][:,ret_dict['p']['t'][0]-1]
            if tilde_t_0_mode=='N':
                for i in range(ret_dict['I'][I]['I_max']):
                    if ret_dict['I'][I]['N'][i]>0 and ret_dict['I'][I]['N_t'][i,0]>0:
                        ret_dict['I'][I]['tilde_t_0'][i] = ret_dict['I'][I]['tilde_t_all'][i,ret_dict['I'][I]['t_all'][i][ret_dict['I'][I]['N_t'][i,0]-1]]
                


            ret_dict['I'][I]['tilde_N'] = [np.array([])  for i in range(ret_dict['I'][I]['I_max'])]
            #ret_dict['I'][I]['tilde_N_0'] = np.zeros(ret_dict['I'][I]['I_max'])
            for i in range(ret_dict['I'][I]['I_max']):
                if ret_dict['I'][I]['N'][i]>0:
                    ret_dict['I'][I]['tilde_N'][i] = ret_dict['I'][I]['tilde_t_all'][i,ret_dict['I'][I]['t'][i]]
                    if False:
                        if ret_dict['I'][I]['N_t'][i,0]>0:
                            #ret_dict['I'][I]['tilde_N_0'][i] = events_dict[I][f'tilde_t'][i,ret_dict['I'][I]['N_t'][i,0]-1]
                            ret_dict['I'][I]['tilde_N_0'][i] = ret_dict['I'][I]['tilde_t_all'][i,ret_dict['I'][I]['t_all'][i][ret_dict['I'][I]['N_t'][i,0]-1]]
                            ret_dict['I'][I]['tilde_N'][i] -= ret_dict['I'][I]['tilde_N_0'][i]
                    else:
                        ret_dict['I'][I]['tilde_N'][i] -= ret_dict['I'][I]['tilde_t_0'][i]

                    



        ret_dict['I'][I]['D0'] = [None for i in range(ret_dict['I'][I]['I_max'])]
        ret_dict['I'][I]['D1'] = [None for i in range(ret_dict['I'][I]['I_max'])]
        ret_dict['I'][I]['x'] = [np.array([]) for i in range(ret_dict['I'][I]['I_max'])]
        ret_dict['I'][I]['a_p'] = np.zeros(ret_dict['I'][I]['I_max'])
        
        
        if val_key=='delta_t':

            for i in range(ret_dict['I'][I]['I_max']):
                ret_dict['I'][I]['x'][i] = (1/a_plot[i])*misc_fns.diff_zero(ret_dict['I'][I]['tilde_N'][i])
                ret_dict['I'][I]['D0'][i] = scipy.stats.expon(scale=1/a_plot[i])


        elif val_key=='delta_tilde_t':
            for i in range(ret_dict['I'][I]['I_max']):
                if ret_dict['I'][I]['N'][i]>0:
                    ret_dict['I'][I]['x'][i] = misc_fns.diff_zero(ret_dict['I'][I]['tilde_N'][i])
                ret_dict['I'][I]['D0'][i] = scipy.stats.expon()

        if val_key=='M':
            for i in range(ret_dict['I'][I]['I_max']):
                if ret_dict['I'][I]['N'][i]>0:
                    ret_dict['I'][I]['x'][i] = events_dict[I][val_key_func][i,ret_dict['I'][I]['N_t'][i,0]:ret_dict['I'][I]['N_t'][i,-1]]
                    #ret_dict['I'][I]['D0'][i] = scipy.stats.genpareto(*scipy.stats.genpareto.fit(ret_dict['I'][I]['x'][i], floc=0))
                    x_train = events_dict[I][val_key_func][i,ret_dict['I'][I]['train_N'][i,0]:ret_dict['I'][I]['train_N'][i,-1]]
                    ret_dict['I'][I]['D0'][i] = scipy.stats.genpareto(*scipy.stats.genpareto.fit(x_train, floc=0))
                else:
                    ret_dict['I'][I]['D0'][i] = scipy.stats.genpareto(c=0, loc=0, scale=1)

        elif val_key=='tilde_M':

            for i in range(ret_dict['I'][I]['I_max']):
                if ret_dict['I'][I]['N'][i]>0:
                    ret_dict['I'][I]['x'][i] = events_dict[I]["F_M"][i,ret_dict['I'][I]['N_t'][i,0]:ret_dict['I'][I]['N_t'][i,-1]]
                ret_dict['I'][I]['D0'][i] = scipy.stats.uniform()


        for i in range(ret_dict['I'][I]['I_max']):
            if app_norm:
                ret_dict['I'][I]['D1'][i] = scipy.stats.norm()
                ret_dict['I'][I]['x'][i] = dist_fns.F_D1_inv_F_D0(X=ret_dict['I'][I]['x'][i], F_D1_inv=ret_dict['I'][I]['D1'][i].ppf, F_D0=ret_dict['I'][I]['D0'][i].cdf, excl_bounds_Y=None, excl_inf_Z=True, F_D1_inv_min_max=None)
            else:
                ret_dict['I'][I]['D1'][i] = ret_dict['I'][I]['D0'][i]


            if ret_dict['I'][I]['N'][i]>0:
                H_eff = np.min([H, ret_dict['I'][I]['N'][i]-1])

                rho, ret_dict['I'][I]['lbtest'][i,:H_eff,0], ret_dict['I'][I]['lbtest'][i,:H_eff,1] = acf(ret_dict['I'][I]['x'][i], nlags=H_eff, qstat=True, adjusted=False)
                ret_dict['I'][I]['rho'][i,:H_eff] = rho[1:1+H_eff]
                if False:
                    ret_dict['I'][I]['lbtest'][i,:H_eff,:] = np.transpose(acorr_ljungbox(x=ret_dict['I'][I]['x'][i], lags=H_eff, return_df=False))
                    ret_dict['I'][I]['rho'][i,:H_eff] = misc_fns_nb.calc_autocorrcoef_H(X=ret_dict['I'][I]['x'][i], H=H_eff)


        
    
    return ret_dict




def LB_plot_dict(self, tick=None, r=None, period=None, val_key=None, I_s_cat_list=None,
            tf_mode=None, tf_u_key=None, a_u=None, u=None, a_lambda=None,
            overwrite_t_ps=None,
            events=None, tilde_t=None, mode=None, mark=None,
            tilde_t_0_mode=None, a_mode=None,
            H=None, app_norm=None,
            I_s=None, I_cat_s=None,
            H_plot=None, H_LB=None,
            CI=None, 
            ylim_abs=None,
            xlim=None, ylim=None,
            xlabel=None, ylabel=None, linelabels=None,
            k_sub=None, a_subscript=None, val_symb=None,  
            super_I=None, super_I_cat=None, 
            str_tick=None, str_mod=None, 
            display_I_s=None,
            plot_I_list=None,
            ):
    
    if plot_I_list is None:
        plot_I_list = ['I', 'I_s']
    

    if I_s_cat_list is None:
        if I_s is not None and len(I_s)>0:
            I_s_cat_list = [misc_fns.make_iterable_array(I_s)]
        else:
            I_s_cat_list

    
    if H_LB is None:
        if H is not None:
            H_LB = np.min([H, 10])
        else: 
            H_LB = 10

    if H_plot is None:
        if H is not None:
            H_plot = H
        else: 
            H_plot = 50

    if H is None:
        H = int(np.max([H_LB, H_plot]))
        
    

    ret_dict = LB_dict(self, tick=tick, r=r, period=period, val_key=val_key, I_s_cat_list=I_s_cat_list,
                            tf_mode=tf_mode, tf_u_key=tf_u_key, a_u=a_u, u=u, a_lambda=a_lambda,
                            overwrite_t_ps=overwrite_t_ps,
                            events=events, tilde_t=tilde_t, mode=mode, mark=mark,
                            tilde_t_0_mode=tilde_t_0_mode, a_mode=a_mode,
                            H=H, app_norm=app_norm,
                            )
    
    ret_dict['plot_I'] = {I: {} for I in ret_dict['I']}
    ret_dict['plot_meta'] = {}
    ret_dict['plot_labels'] = {}

    ret_dict['plot_meta']['plot_I_list'] = plot_I_list
    ret_dict['plot_meta']['H_LB'] = H_LB
    ret_dict['plot_meta']['H_plot'] = H_plot
    ret_dict['plot_meta']['h_plot'] = 1 + np.arange(H_plot)



    # AM3-9
    if ret_dict['meta']['app_norm']:
        
        #norm_str = f"{dist_fns.get_scipy_dist_symb(dist='norm')}, "
        norm_str = [f"\\mathrm{{PIT}}^{{{dist_fns.get_scipy_dist_symb(dist='norm')}}}{{\\left(", f"\\right)}}"]
    else:
        #norm_str = f""
        norm_str = [f"", f""]
        
    ret_dict['plot_labels']['norm_str'] = norm_str
    
    if I_s is None:
        I_s = np.arange(ret_dict['I']['I']['I_max'])
    I_s = misc_fns.make_iterable_array(I_s)
    ret_dict['plot_I']['I']['I_s'] = I_s
    ret_dict['plot_I']['I']['I_s_size'] = np.size(I_s)

    



    I_s_cat_list = ret_dict['meta']['I_s_cat_list']

    if I_cat_s is None:
        I_cat_s = np.arange(ret_dict['I']['I_cat']['I_max'])
    I_cat_s = misc_fns.make_iterable_array(I_cat_s)
    ret_dict['plot_I']['I_cat']['I_s'] = I_cat_s
    ret_dict['plot_I']['I_cat']['I_s_size'] = np.size(I_cat_s)

    ret_dict['plot_I']['I']['meta_I_s'] = ret_dict['plot_I']['I']['I_s']
    ret_dict['plot_I']['I_cat']['meta_I_s'] = np.sort(np.unique(np.concatenate([ret_dict['meta']['I_s_cat_list'][i] for i in ret_dict['plot_I']['I_cat']['I_s']])))

    
    ret_dict['plot_meta']['meta_I_s'] = np.sort(np.unique(np.concatenate([ret_dict['plot_I'][I]['meta_I_s'] for I in ret_dict['plot_I']])))

    


    if display_I_s is None:
        display_I_s = ret_dict['plot_meta']['meta_I_s']
    elif display_I_s=='all':
        display_I_s = np.arange(ret_dict['I']['I']['I_max'])
    ret_dict['plot_meta']['display_I_s'] = display_I_s


    if str_tick is None:
        str_tick = f"$r_{{t}}$ [{ret_dict['meta']['tick']}]"
    ret_dict['plot_labels']['str_tick'] = str_tick

    

    if a_subscript is None:
        a_subscript = f"\\lambda"
    ret_dict['plot_labels']['a_subscript'] = a_subscript

    
    if super_I is None:
        super_I = np.concatenate([["\\leftarrow", "\\rightarrow"], [f"I{k}" for k in range(2,ret_dict['I']['I']['I_max'])]])[:ret_dict['I']['I']['I_max']]
        
    if super_I_cat is None:
        super_I_cat = np.concatenate([["\\leftrightarrow"], [f"CAT{k}" for k in range(1,ret_dict['I']['I_cat']['I_max'])]])[:ret_dict['I']['I_cat']['I_max']]

    ret_dict['plot_I']['I']['super'] = super_I
    ret_dict['plot_I']['I_cat']['super'] = super_I_cat






    if len(plot_I_list)==0:
        I_base = plot_I_list[0]
    elif np.size(I_s)==0:
        I_base = 'I_cat'
    else:
        I_base = 'I'
    ret_dict['plot_meta']['I_base'] = I_base



    

    if np.size(ret_dict['plot_I'][I_base]['I_s'])==1:
        xylabel_super = ret_dict['plot_I'][I_base]['super'][ret_dict['plot_I'][I_base]['I_s'][0]]
    else:
        xylabel_super = f"\\leftrightharpoons"
    ret_dict['plot_labels']['xylabel_super'] = xylabel_super


    if k_sub is None:
        k_sub = "k"
    if k_sub=="":
        k_sub_str = ""
    else:
        k_sub_str = f"_{{{k_sub}}}"
    ret_dict['plot_labels']['k_sub'] = k_sub
    ret_dict['plot_labels']['k_sub_str'] = k_sub_str


    if val_symb is None:
        if val_key==f'delta_t':
            val_symb = f"\\Delta{{t}}"
        elif val_key==f'delta_tilde_t':
            # AM3-9
            #val_symb = f"\\Delta{{\\tilde{{t}}}}"
            val_symb = f"\\Delta{{\\Pi}}"
        elif val_key==f'M':
            val_symb = f"M"
        elif val_key==f"tilde_M":
            val_symb = f"\\tilde{{M}}"

    val_symb_arrow = f"{val_symb}{k_sub_str}^{{{xylabel_super}}}"
    ret_dict['plot_labels']['val_symb'] = val_symb
    ret_dict['plot_labels']['val_symb_arrow'] = val_symb_arrow

    ret_dict['plot_labels']['resid_symb'] = ret_dict['plot_labels']['val_symb']
    ret_dict['plot_labels']['resid_symb_arrow'] = ret_dict['plot_labels']['val_symb_arrow']
    

    if CI is None:
        CI = [0.95, 0.99]
    CI = misc_fns.make_iterable_array(CI)
    ret_dict['plot_meta']['CI'] = CI

    if val_key=='delta_t':
        D0_symb = f"\\mathcal{{E}}_{{(1/a_{{{ret_dict['plot_labels']['a_subscript']}}})}}"
    elif val_key=='delta_tilde_t':
        D0_symb = f"\\mathcal{{E}}_{{(1)}}"
    elif val_key=='M':
        D0_symb = f"\\hat{{\\mathcal{{P}}}}_{{M}}^{{{xylabel_super}}}"
    elif val_key=='tilde_M':
        D0_symb = f"\\mathcal{{E}}_{{(1)}}"
    ret_dict['plot_meta']['D0_symb'] = D0_symb

    if ret_dict['meta']['app_norm']:
        D1_symb = f"\\mathcal{{N}}_{{(0,1)}}"
    else:
        D1_symb = D0_symb
    ret_dict['plot_meta']['D1_symb'] = D1_symb

    for I in ret_dict['plot_I']:

        

        ret_dict['plot_I'][I]['x'] = ret_dict['plot_meta']['h_plot']
        ret_dict['plot_I'][I]['CI_x'] = np.arange(2+H_plot)

        ret_dict['plot_I'][I]['xlim'] = np.array([0, H_plot])





        if len(ret_dict['plot_I'][I]['I_s'])>0:
            ret_dict['plot_I'][I]['H_max'] = np.min(ret_dict['I'][I]['N'][ret_dict['plot_I'][I]['I_s']]) - 1
            ret_dict['plot_I'][I]['N_I_s'] = np.product(ret_dict['I'][I]['N'][ret_dict['plot_I'][I]['I_s']])**(1/np.size(ret_dict['I'][I]['N'][ret_dict['plot_I'][I]['I_s']]))
            ret_dict['plot_I'][I]['y'] = ret_dict['I'][I]['rho'][ret_dict['plot_I'][I]['I_s'],:H_plot]
            ret_dict['plot_I'][I]['y_abs_max_i'] = np.array([np.max(np.abs(ret_dict['plot_I'][I]['y'][i])) for i in range(ret_dict['plot_I'][I]['I_s_size'])])
            ret_dict['plot_I'][I]['y_abs_max'] = np.max(ret_dict['plot_I'][I]['y_abs_max_i'])
        else:
            ret_dict['plot_I'][I]['H_max'] = -1
            ret_dict['plot_I'][I]['N_I_s'] = 0
            ret_dict['plot_I'][I]['y'] = np.full((0,H_plot), np.nan)
            ret_dict['plot_I'][I]['y_abs_max_i'] = np.ones(len(ret_dict['plot_I'][I]['I_s']), dtype=float)
            ret_dict['plot_I'][I]['y_abs_max'] = 1.0


        ret_dict['plot_I'][I]['ylim_abs'] = np.min([1, 1.04*ret_dict['plot_I'][I]['y_abs_max']])
        ret_dict['plot_I'][I]['ylim'] = ret_dict['plot_I'][I]['ylim_abs']*np.array([-1,1])


        N_I_s = ret_dict['plot_I'][I]['N_I_s']
        ret_dict['plot_I'][I]['CI_y'] = np.array([[((-1)**(1+k))*scipy.stats.norm().ppf((1+CI[j])/2)*np.sqrt(N_I_s - ret_dict['plot_I'][I]['CI_x'])/np.sqrt((N_I_s*(N_I_s+2))) for j in range(len(CI))] for k in range(2)])

        if False:
            ret_dict['plot_I'][I]['CI_y'] = np.array([[((-1)**(1+k))*(scipy.stats.norm().ppf((1 + CI[j])/2)/np.sqrt(ret_dict['plot_I'][I]['N_I_s'] - ret_dict['plot_I'][I]['CI_x'])) for j in range(len(CI))] for k in range(2)])
        
        

        

    linelabels = np.concatenate([[f"$\\hat{{\\rho}}_{{h}}{{\\left[{norm_str[0]}{val_symb}{k_sub_str}^{{{ret_dict['plot_I'][I]['super'][i]}}}{norm_str[1]}\\right]}}$" for i in ret_dict['plot_I'][I]['I_s']] for I in ret_dict['plot_I']])

    
    if ylabel is None:
        ylabel = f"$\\hat{{\\rho}}_{{h}}{{\\left[{norm_str[0]}{val_symb}{k_sub_str}^{{{xylabel_super}}}{norm_str[1]}\\right]}}$"
    if xlabel is None:
        xlabel = f"Serial lag $h$"

    if xlim is None:
        xlim = ret_dict['plot_I'][I_base]['xlim']


    if ylim_abs is None:
        #ylim_abs = ret_dict['plot_I'][I_base]['ylim_abs']
        ylim_abs = 1

    if ylim is None:
        ylim = ylim_abs * np.array([-1,1])



    ret_dict['plot_labels']['linelabels'] = linelabels
    ret_dict['plot_labels']['xlabel'] = xlabel
    ret_dict['plot_labels']['ylabel'] = ylabel

    ret_dict['plot_meta']['xlim'] = xlim
    ret_dict['plot_meta']['ylim'] = ylim
    ret_dict['plot_meta']['ylim_abs'] = ylim_abs



    return ret_dict









def LB_plot(self, tick=None, r=None, period=None, val_key=None, I_s_cat_list=None,
            tf_mode=None, tf_u_key=None, a_u=None, u=None, a_lambda=None,
            overwrite_t_ps=None,
            events=None, tilde_t=None, mode=None, mark=None,
            tilde_t_0_mode=None, a_mode=None,
            H=None, app_norm=None,
            I_s=None,
            H_plot=None, H_LB=None,
            CI=None, 
            ylim_abs=None,
            set_ylim_abs_max=None, set_H_plot_max=None, set_H_LB_max=None, 
            xlim=None, ylim=None, 
            xlabel=None, ylabel=None, linelabels=None,
            k_sub=None, a_subscript=None, val_symb=None,  
            super_I=None, super_I_cat=None, 
            str_tick=None, str_mod=None, 
            plot_I_list=None, plot_I_list_plot=None, plot_I_list_p=None,
            ret_dict=None,
            ax=None, ax2=None, 
            #plot_I_s_cat_plot=None, plot_I_s_cat_p=None,
            thesis=None, 
            lw=None, bw=None, balign=None,
            an_mode=None, an_xycoords=None, an_fs=None, an_x0=None, an_y0=None, an_yi=None,
            an_h_p=None, an_v_p=None, an_fs_p=None, an_x0_p=None, an_y0_p=None, an_yi_p=None, 
            an_h_t=None, an_v_t=None, an_fs_t=None, an_x0_t=None, an_y0_t=None, an_yi_t=None, 
            an_yM_tick=None, an_yM_mod=None, an_yM_u=None, an_yM_a_lambda=None, 
            plot_main_legend=None, plot_ci_legend=None, 
            plot_pvals=None, 
            plot_order=None,
            plot_tick=None, plot_mod=None, plot_u=None, plot_a_lambda=None,
            uncond=None, hwk_uncond=None,
            main_legend_bbox_to_anchor=None, main_legend_loc=None, main_legend_ncols=None, main_legend_markerfirst=None,
            ci_legend_bbox_to_anchor=None, ci_legend_loc=None, ci_legend_ncols=None, ci_legend_markerfirst=None,
            u_scif_pow10=None, u_dp=None, u_dp_buff=None, u_val_plus=None,
            a_u_scif_pow10=None, a_u_dp=None, a_u_dp_buff=None, a_u_val_plus=None,
            a_lambda_scif_pow10=None, a_lambda_dp=None, a_lambda_dp_buff=None, a_lambda_val_plus=None,
            cols_I=None, cols_I_cat=None, cols_CI=None,
            ):
    

    if plot_I_list_plot is None:
        if plot_I_list is not None:
            plot_I_list_plot = plot_I_list
        else:
            plot_I_list_plot = ['I']
    if plot_I_list_p is None:
        if plot_I_list is not None:
            plot_I_list_p = plot_I_list
        else:
            plot_I_list_p = ['I']
    if plot_I_list is None:
        if len(plot_I_list_plot)==1 and len(plot_I_list_plot)==1 and plot_I_list_plot[0]==plot_I_list_p[0]:
            plot_I_list = [plot_I_list_plot[0]]
        else:
            plot_I_list = ['I', 'I_cat']
    
    if ret_dict is None:
        ret_dict = LB_plot_dict(self, tick=tick, r=r, period=period, val_key=val_key, I_s_cat_list=I_s_cat_list,
                                tf_mode=tf_mode, tf_u_key=tf_u_key, a_u=a_u, u=u, a_lambda=a_lambda,
                                overwrite_t_ps=overwrite_t_ps,
                                events=events, tilde_t=tilde_t, mode=mode, mark=mark,
                                tilde_t_0_mode=tilde_t_0_mode, a_mode=a_mode,
                                H=None, app_norm=app_norm,
                                I_s=I_s,
                                H_plot=H_plot, H_LB=H_LB,
                                CI=CI, 
                                ylim_abs=ylim_abs, 
                                xlim=xlim, ylim=ylim,
                                xlabel=xlabel, ylabel=ylabel, linelabels=linelabels,
                                k_sub=k_sub, a_subscript=a_subscript, val_symb=val_symb,  
                                super_I=super_I, #super_I_cat=super_I_cat,
                                str_tick=str_tick, str_mod=str_mod, 
                                plot_I_list=plot_I_list,
                                #display_I_s=display_I_s,
                                )

    if plot_a_lambda is None:
        plot_a_lambda = False
    
    
    if thesis is None:
        thesis = True

    an_mode = None
    if thesis:
        an_mode = 'thesis'

    if lw is None:
        lw = 1.5

    if bw is None:
        bw = 0.8

    if balign is None:
        balign = 'center'

    if balign=='center':
        xlim_buff = np.array([0, 1-(bw/2)])
    else:
        xlim_buff = np.array([0, 1])
        




    if an_h_p is None:
        an_h_p = 'left'
    if an_v_p is None:
        an_v_p = 'bottom'
    if an_h_t is None:
        an_h_t = 'right'
    if an_v_t is None:
        an_v_t = 'top'


    an_params = {'lambda': plot_fns.calc_an_params(an_fs=an_fs, an_x0=an_x0, an_y0=an_y0, an_yi=an_yi, an_mode=an_mode, an_xycoords=an_xycoords)}
    an_params['spec'] = {
                            'p': plot_fns.calc_an_params_dir(**an_params['lambda'], flip_x0ry0t=True,
                                                                an_h=an_h_p, an_v=an_v_p, an_fs_ow=an_fs_p, an_x0_ow=an_x0_p, an_y0_ow=an_y0_p, an_yi_ow=an_yi_p, 
                                                            ),
                            't': plot_fns.calc_an_params_dir(**an_params['lambda'], flip_x0ry0t=True,
                                                                an_h=an_h_t, an_v=an_v_t, an_fs_ow=an_fs_t, an_x0_ow=an_x0_t, an_y0_ow=an_y0_t, an_yi_ow=an_yi_t, 
                                                            ),
                        }





    if plot_main_legend is None:
        plot_main_legend = True
    if plot_ci_legend is None:
        plot_ci_legend = True
    if plot_pvals is None:
        plot_pvals = True

    if plot_tick is None:
        plot_tick = True

    
    if plot_mod is None:
        plot_mod = True

    if uncond is None:
        if ret_dict['meta']['val_key'].find('tilde')>=0 or str_mod is not None:
            uncond = False
        else:
            uncond = True

        
    if plot_u is None:
        plot_u = True
    if plot_a_lambda is None:
        plot_a_lambda = True



    if ax is None:
        ax_arg = False
        fig, ax = plt.subplots()
    else:
        ax_arg = True
    
    ax.xaxis.set_label_position('bottom')
    ax.yaxis.set_label_position('left')
    ax.tick_params(left=True, right=True, top=True, bottom=True,
                                        labelleft=True, labelright=False, labeltop=False, labelbottom=True)

    if ax2 is None:
        ax2 = ax.twiny()


    
    ax.tick_params(left=True, right=True, top=True, bottom=True,
                                        labelleft=True, labelright=False, labeltop=False, labelbottom=True)
    
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(left=False, right=False, top=False, bottom=False,
                                        labelleft=False, labelright=False, labeltop=False, labelbottom=False)
    


    I_base = ret_dict['plot_meta']['I_base']
    

    a_plot = ret_dict['I']['I']['a_plot']
    super_I_i = ret_dict['plot_I']['I']['super']
    super_I_cat = ret_dict['plot_I']['I_cat']['super']
    
    u = ret_dict['u']['u']
    a_u = ret_dict['u']['a_u']
    a_lambda = ret_dict['u']['a_lambda']


    a_mode = ret_dict['meta']['a_mode']
    a_subscript = ret_dict['plot_labels']['a_subscript']

    resid = ret_dict['plot_labels']['resid_symb_arrow']





    


    CI = ret_dict['plot_meta']['CI']
    CI_x = ret_dict['plot_I'][I_base]['CI_x']
    CI_y = ret_dict['plot_I'][I_base]['CI_y']



    H_plot_max = np.min([ret_dict['plot_I'][I]['H_max'] for I in plot_I_list_plot])
    H_LB_max = np.min([ret_dict['plot_I'][I]['H_max'] for I in plot_I_list_p])

    plot_I_s_cat_plot_I_s_size =  np.array([ret_dict['plot_I'][I]['I_s_size'] for I in plot_I_list_plot])
    plot_I_s_cat_p_I_s_size =  np.array([ret_dict['plot_I'][I]['I_s_size'] for I in plot_I_list_p])

    linelabels = ret_dict['plot_labels']['linelabels']

    ylabel = ret_dict['plot_labels']['ylabel']
    xlabel = ret_dict['plot_labels']['xlabel']

    if set_ylim_abs_max is None:
        if ylim_abs is None:
            set_ylim_abs_max = True
        else:
            set_ylim_abs_max = False

    if set_ylim_abs_max:
        ylim_abs = np.max([ret_dict['plot_I'][I]['ylim_abs'] for I in plot_I_list_plot])
    else:
        ylim_abs = ret_dict['plot_meta']['ylim_abs']
    ylim = ylim_abs*np.array([-1,1])

    if set_H_plot_max is None:
        set_H_plot_max = True

    if set_H_plot_max:
        H_plot = np.min([H_plot_max, ret_dict['plot_meta']['H_plot']])
    else:
        H_plot = ret_dict['plot_meta']['H_plot']
    xlim = np.array([0, H_plot]) + xlim_buff


    if cols_I is None:
        cols_I = plot_fns.calc_cols_tail(rows=False)
    if cols_I_cat is None:
        cols_I_cat = cols_I[ret_dict['I']['I']['I_max']:]

    cols = {'I': cols_I, 'I_cat': cols_I_cat}


    N_CI = np.size(CI)
    if cols_CI is None:
        cols_CI = plot_fns.calc_cols_CI(N_CI=N_CI)




    for i in range(N_CI):
        j = N_CI-(1+i)
        ax.fill_between(CI_x, y1=CI_y[0][j], y2=CI_y[1][j], color=cols_CI[j])
            
            
    for I_i in range(len(plot_I_list_plot)):
        I = plot_I_list_plot[I_i]

    plot_corr = np.concatenate([ret_dict['plot_I'][I]['y'] for I in plot_I_list_plot])
    plot_I = np.concatenate([np.repeat(I, ret_dict['plot_I'][I]['I_s_size']) for I in plot_I_list_plot])
    plot_i = np.concatenate([np.arange(ret_dict['plot_I'][I]['I_s_size']) for I in plot_I_list_plot])
    

    _argsort = np.flip(np.argsort(np.abs(plot_corr), axis=0), axis=0)
    _i_loc_0 = np.argsort(_argsort[:,0])
    _wheresort = np.arange(np.shape(_argsort)[0])[_argsort]



    for l in range(H_plot):
        for k in range(np.shape(_argsort)[0]):
            I = plot_I[_wheresort[k,l]]
            i = plot_i[_wheresort[k,l]]
            i_prime = ret_dict['plot_I'][I]['I_s'][i]
            ax.bar(ret_dict['plot_I'][I]['x'][l], ret_dict['plot_I'][I]['y'][i,l], width=bw, color=cols[I][i_prime])

    
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.set_ylim(ylim)
    ax2.set_ylim(ylim)

    ax.set_xlim(xlim)
    ax2.set_xlim(xlim)

    ax.tick_params(labeltop=False)


    if plot_pvals:
        if set_H_LB_max is None:
            set_H_LB_max = True

        if set_H_LB_max:
            H_LB = np.min([H_LB_max, ret_dict['plot_meta']['H_LB']])
        else:
            H_LB = ret_dict['plot_meta']['H_LB']

        p_vals = np.concatenate([ret_dict['I'][I]['lbtest'][ret_dict['plot_I'][I]['I_s'],H_LB-1,1] for I in plot_I_list_p])
        p_cols = np.concatenate([cols[I][ret_dict['plot_I'][I]['I_s']] for I in plot_I_list_p])
        p_symb = np.concatenate([[f"p_{{\mathrm{{LB}}_{{{H_LB}}}}}^{{{ret_dict['plot_I'][I]['super'][i]}}}" for i in ret_dict['plot_I'][I]['I_s']] for I in plot_I_list_p])


        xy_box_p = plot_fns.annotate_pvals(
                   p_vals=p_vals, p_cols=p_cols, p_symb=p_symb,
                   p_thresh=1-CI,
                   **an_params['spec']['p'], flip_x0ry0t=False,
                   ax=ax,
                   )
    else:
        xy_box_p = plot_fns.calc_xy_box(default_full=False)




    xy_box_t = hwks_fns.annotate_model_Hawkes(
                                                            ax=ax, 
                                                            str_tick=None, str_mod=None, str_u=None, str_a_lambda=None, 
                                                            plot_tick=plot_tick, plot_mod=plot_mod, plot_u=plot_u, plot_a_lambda=plot_a_lambda, 
                                                            an_yM_mod=an_yM_mod, an_yM_tick=an_yM_tick, an_yM_u=an_yM_u, an_yM_a_lambda=an_yM_a_lambda,
                                                            u_scif_pow10=u_scif_pow10, u_dp=u_dp, u_dp_buff=u_dp_buff, u_val_plus=u_val_plus, 
                                                            a_lambda_scif_pow10=a_lambda_scif_pow10, a_lambda_dp=a_lambda_dp, a_lambda_dp_buff=a_lambda_dp_buff, a_lambda_val_plus=a_lambda_val_plus, 
                                                            a_u_scif_pow10=a_u_scif_pow10, a_u_dp=a_u_dp, a_u_dp_buff=a_u_dp_buff, a_u_val_plus=a_u_val_plus, 
                                                            flip_x0ry0t=False,
                                                            #an_mode=None, an_xycoords=None, an_h=None, an_v=None, an_fs=None, an_x0=None, an_y0=None, an_xi=None, an_yi=None, an_xs=None, an_ys=None,
                                                            **an_params['spec']['t'],
                                                            u=u, a_u=a_u, a_lambda=a_plot, 
                                                            display_I_s=ret_dict['plot_meta']['meta_I_s'], super_I_i=None, super_I_cat=None,
                                                            tf_mode=tf_mode, a_mode=a_mode, a_subscript=a_subscript, plot_order=plot_order,
                                                            tick=tick,
                                                            resid=resid, mode=mode, mark=mark, uncond=uncond, hwk_uncond=hwk_uncond,
                                                            )
        

    
    if main_legend_loc is None:
        main_legend_loc = 'lower center'
    if main_legend_ncols is None:
        main_legend_ncols = np.min([np.size(I_s), 1])
    if main_legend_markerfirst is None:
        main_legend_markerfirst = True
    if ci_legend_loc is None:
        ci_legend_loc = 'lower right'
    if ci_legend_ncols is None:
        ci_legend_ncols = np.min([N_CI, 1])
    if ci_legend_markerfirst is None:
        ci_legend_markerfirst = True


    if False:
        default_legend_bbox_to_anchor = plot_fns.calc_bbox(xy_box=plot_fns.calc_between_xy_boxes(xy_boxes=[xy_box_p, xy_box_t], along_x=False, along_y=True))
    else:
        #default_legend_bbox_to_anchor = xy_box_t
        default_legend_bbox_to_anchor = plot_fns.calc_bbox()
         


    if main_legend_bbox_to_anchor is None:
        main_legend_bbox_to_anchor = default_legend_bbox_to_anchor
    if ci_legend_bbox_to_anchor is None:
        ci_legend_bbox_to_anchor = default_legend_bbox_to_anchor


    ret_dict['plot_meta']['xy_box'] = {
                                        'p': xy_box_p,
                                        't': xy_box_t,
                                    }
    
    ret_dict['plot_meta']['bbox'] = {
                                        'main': main_legend_bbox_to_anchor,
                                        'ci': ci_legend_bbox_to_anchor,
                                    }
    

    if plot_main_legend:
        ax.legend([ax.get_children()[N_CI+i] for i in _i_loc_0], linelabels, 
                  loc=main_legend_loc, 
                  bbox_to_anchor=main_legend_bbox_to_anchor, 
                  ncol=main_legend_ncols, 
                  markerfirst=main_legend_markerfirst,
                  fontsize=an_params['lambda']['an_fs'])
    if plot_ci_legend:
        ax2.legend([ax.get_children()[N_CI-(1+k)] for k in range(N_CI)], 
                   [f"${100*CI[k]:.0f}\\%$ CI" for k in range(N_CI)],
                   loc=ci_legend_loc, 
                   bbox_to_anchor=ci_legend_bbox_to_anchor, 
                   ncol=ci_legend_ncols, 
                   markerfirst=ci_legend_markerfirst,
                   fontsize=an_params['lambda']['an_fs'])
        




    return ret_dict



    


if False:
    def annotate_model_Hawkes(self,
                    str_tick=None, str_mod=None, str_u=None, str_a_lambda=None, 
                    plot_tick=None, plot_mod=None, plot_u=None, plot_a_lambda=None, 
                    an_yM_mod=None, an_yMick=None, an_yM_u=None, an_yM_a_lambda=None,
                    u_scif_pow10=None, u_dp=None, u_dp_buff=None, 
                    a_lambda_scif_pow10=None, a_lambda_dp=None, a_lambda_dp_buff=None, 
                    an_mode=None, an_xycoords=None, an_h=None, an_v=None, an_fs=None, an_x0=None, an_y0=None, an_xi=None, an_yi=None, an_xs=None, an_ys=None,
                    I_base=None, I_s=None, super_I=None, super_I_i=None,
                    u=None, a_lambda=None, a_subscript=None, tf_mode=None, plot_order=None,
                    ax=None, ):
        


        _locals = locals()
        return hwks_fns.annotate_model(**{c: _locals[c] for c in plot_fns.calc_annotate_model_args(cat=True)})











        

    




def N_plot(t_events, t, lambda_t=None, vline=None, xlabel=None, ylabel=None, xlim=None, ylim=None, colours=None, linewidths=None, ax=None):

    if ax is None:
        ax_arg = False
        fig, ax = plt.subplots()
    else:
        ax_arg = True
        

    if lambda_t is None:
        lambda_t = t

    if colours is None:
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if linewidths is None:
        linewidths = np.ones(len(t_events))

    CI = [0.99, 0.95]
    for i in range(2):
    #    ax.plot(t, scipy.stats.poisson.ppf(mu=lambda_t, q=(0.5-((-1)**i)*0.5*0.95)), color="grey", linewidth=0.5)
    #    ax.plot(t, scipy.stats.poisson.ppf(mu=lambda_t, q=(0.5-((-1)**i)*0.5*0.99)), color="black", linewidth=0.5)
        ax.fill_between(t, y1=scipy.stats.poisson.ppf(mu=lambda_t, q=(0.5-0.5*CI[i])),
                            y2=scipy.stats.poisson.ppf(mu=lambda_t, q=(0.5+0.5*CI[i])),
                            color=plot_fns.calc_cols_CI(N_CI=len(CI), bound_cols=None, high_to_low=True)[i])

    for i in range(len(t_events)):
        ax.plot(t_events[i], 1+np.array(range(t_events[i].size)), color=colours[i], linewidth=linewidths[i])

    if vline is not None:
        ax.axvline(vline, color="black", linewidth=0.2)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(bottom=0)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(f"$N$")

    ax.legend((ax.get_children()[1-i] for i in range(2)), ("95% CI", "99% CI"))

    ax.tick_params(
                        which='both',      
                        bottom=True, labelbottom=True,     
                        top=True, labeltop=False,
                        left=True, labelleft=True,     
                        right=True, labelright=False,         
                        direction="out",
                    )
    
    #fig.set_size_inches(5.5, 2.5)
    if ax_arg == False:
        fig.set_size_inches(5.5, 4)
        fig.tight_layout()

        return fig, ax
    else:
        return ax



def N_delta_plot_setup_a(name="fig", nrows=3):

    fig = {
                "name": name,
                "fig": None,
                "gs": None,
                "ax":   [
                            [None],
                            [None for i in range(nrows)]
                        ],
            }
    fig["fig"] = plt.figure(constrained_layout=True)
    fig["gs"] = fig["fig"].add_gridspec(nrows, 2)
    fig["ax"][0][0] = fig["fig"].add_subplot(fig["gs"][:, 0])
    for i in range(nrows):
        fig["ax"][1][i] = fig["fig"].add_subplot(fig["gs"][i, 1])


    fig["fig"].set_size_inches(7, 3)
    fig["fig"].tight_layout()

    return fig


def interval_fig(t, bins, lambda_scale=1, rwidth=0.8, alpha=0.25, align="mid", kde_density=None, hist_colour=None, theory_colour=None, xlabel=None, ylabel=None, ylim=None, lambda_unit=None, fontsize=None, ax=None):

    if ax is None:
        ax_arg = False
        fig, ax = plt.subplots()
    else:
        ax_arg = True

    t_interval = misc_fns.diff_zero(t)

    xlim = np.array([np.min(bins), np.max(bins)])

    if hist_colour is None:
        hist_colour = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    if theory_colour is None:
        theory_colour = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
    hist = ax.hist(t_interval, bins=bins, color=plot_fns.colour_transform_alpha(hist_colour, alpha=alpha), rwidth=rwidth, align=align)

    x = np.linspace(bins[0], bins[-1], 1001)
    norm = t_interval.size*np.diff(hist[1][0:2])

    theory = scipy.stats.expon(scale=lambda_scale)
    ax.plot(x, norm*theory.pdf(x), color=theory_colour)

    if kde_density is not None:
        density = scipy.stats.gaussian_kde(t_interval)
        density.covariance_factor = lambda : kde_density 
        density._compute_covariance()
        ax.plot(x, norm*density(x), color=hist_colour)

    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(bottom=0)

    ax.tick_params(
                        which='both',      
                        bottom=True, labelbottom=True,     
                        top=True, labeltop=False,
                        left=True, labelleft=True,     
                        right=True, labelright=False,         
                        direction="out",
                    )

    if False:
        if lambda_scale <= 1:
            theory_label = f"$f_{{\mathcal{{E}}}}{{\\left(\\lambda={1/lambda_scale}\\right)}}$"
        else:
            theory_label = f"$f_{{\\mathcal{{E}}}}{{\\left(\\lambda=1/{lambda_scale}\\right)}}$"
        ax.legend((ax.get_children()[hist[0].size],), (theory_label,))

    if fontsize is None:
        fontsize = plt.rcParams['font.size']

    if lambda_unit is None:
        theory_label = f"$f_{{\\mathcal{{E}}}}(\\lambda={1/lambda_scale})$"
    else:
        theory_label = f"$f_{{\\mathcal{{E}}}}(\\lambda={1/lambda_scale} \: {lambda_unit})$"
    ax.legend((ax.get_children()[hist[0].size],), (theory_label,), loc='upper right', fontsize=fontsize)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if ax_arg == False:
    #fig.set_size_inches(5,2.5)
        fig.set_size_inches(5,2.7)
        fig.tight_layout()

        output = fig, ax, t_interval, hist, theory
    else: 
        output = ax, t_interval, hist, theory  

    if kde_density is not None:
        output.append(density)

    return output


def get_dir_dates_format(period=None):

    _dir_dates_format =  {
                            'all': 't0_t1_a1',
                            'train': 't0_t1',
                            'forecast': 't0_t1/_a1',
                        }
    
    if period is None:
        return _dir_dates_format
    else:
        return _dir_dates_format[period]
    

        
def dir_fn_tick(dir_res=None, tick=None, fn=None, periods=None, mkdir=None, mkdir_incl=None, mkdir_excl=None, val=None):

    if dir_res is None:
        dir_res = f"./results/"

    if tick is None:
        tick = 'TICK'

    if fn is None:
        fn = "fit"

    if periods is None:
        periods = ""

    dir_list = ['res', 'fn', 'tick', 'periods']
    dir_dict = {suff: None for suff in dir_list }

    dir_dict['res'] = dir_res
    dir_dict['fn'] = f"{dir_dict['res']}{fn}/"
    dir_dict['tick'] = f"{dir_dict['fn']}{tick}/"
    dir_dict['periods'] = f"{dir_dict['tick']}{periods}/"

    if mkdir:
        if mkdir is None:
            mkdir = False
        if mkdir_incl is None:
            mkdir_incl = dir_list

        mkdir_list = misc_fns.A_rm(A=mkdir_incl, A_rm=mkdir_excl)
        for d in mkdir_list:
            plot_fns.mkdir_export(dir_dict[d])

    if val is None:
        return dir_dict
    else:
        return dir_dict[val]
    

    


def calc_dir_suff(dir_suff=None, suff=None, dir_base=None, pref_base=None, use_defaults=None):
        
        if dir_suff is None:
            
            if dir_base is None:
                dir_base = ""

            if pref_base is None:
                pref_base = True

            if pref_base:
                _dir_base = dir_base
            else:
                _dir_base = ""

            if suff is None:
                suff = ""

            if use_defaults is None:
                use_defaults = False

            _suff = suff
            if use_defaults:
                _defaults = get_dir_defaults(suff=None)
                if np.isin(suff, misc_fns.d_keys(d=_defaults, array=None)):
                    _suff = _defaults[suff]

            if len(_suff)==0:
                _post = ""
            else:
                _post = "/"

            dir_suff = f"{_dir_base}{_suff}{_post}"

        return dir_suff


def get_dir_defaults(suff=None):

    dir_defaults = {
                        'data': "data",
                        'res': "results",
                    }

    if suff is None:
        return dir_defaults
    else:
        return dir_defaults[suff]


def dir_fn_tick_period_w(dir_res=None, tick=None, fn=None, periods_dict=None, mkdir=None, mkdir_p=None, mkdir_w=None, mkdir_incl=None, mkdir_excl=None, write_period=None, datetime_w_str=None):

        _get_dir_dates_format = get_dir_dates_format()

        if mkdir is None:
            mkdir = True
        if mkdir_p is None:
            mkdir_p = mkdir
        if mkdir_w is None:
            mkdir_w = mkdir_p

        if periods_dict is None:
            periods_dict = get_dir_dates_format()

        if datetime_w_str is None:
            datetime_w_str = ''

        if mkdir_excl is None:
            mkdir_excl = ['periods']

        if write_period is None:
            write_period = 'train'

        dir_periods = {period: dir_fn_tick(dir_res=dir_res, tick=tick, fn=fn, periods=periods_dict[period], mkdir=mkdir, mkdir_incl=mkdir_incl, mkdir_excl=mkdir_excl, val='periods') for period in  _get_dir_dates_format}

        if write_period=='forecast':
            plot_fns.mkdir_export(dir_periods['train'], mkdir=mkdir_p)
        dir_period = plot_fns.mkdir_export(dir_periods[write_period], mkdir=mkdir_p)

        if datetime_w_str=='':
            return dir_period
        else:
            dir_period_w = plot_fns.mkdir_export(f"{dir_period}{datetime_w_str}/", mkdir=mkdir_w)
            return dir_period_w
        

def calc_table_periods_head_old(self, tick=None, sample_dates=None, periods=None, pre=None, post=None, sample_labels=None, split_labels=None, same_row=None, P_row=None, lead=None, c0=None, cw=None, calign=None, incl_cmidrules=None):

    if incl_cmidrules is None:
        incl_cmidrules = True

    if tick is None:
        tick = self._ticks[0]
    
    if periods is None:
        periods = ['train', 'forecast']
    periods = misc_fns.make_iterable_array(periods)
    N_p = np.size(periods)

    if pre is None:
        pre = ''

    if post is None:
        post = ''
    

    if sample_labels is None:
        sample_labels = ['name', 'dates']
    sample_labels = misc_fns.make_iterable_array(sample_labels)
    N_sl = np.size(sample_labels)

    if lead is None:
        lead = ''
    lead = misc_fns.make_iterable_array(lead)
    if c0 is None:
        c0 = [lead[k].count("&") for k in range(lead.size)]
    c0 = misc_fns.make_iterable_array(c0)

    N_lead = np.size(lead)
    N_c0 = np.size(c0)

    if cw is None:
        cw = 1

    if calign is None:
        calign = 'c'

    if split_labels is None:
        split_labels = False

    if same_row is None:
        same_row = False

    if P_row is None:
        if same_row:
            P_row = N_p
        else:
            P_row = 1

    N_rows = 1+((N_p-1)//P_row)

    P_rows = np.repeat(P_row, N_rows)
    P_rows[-1] = 1 + ((N_p-1) % P_row)

    if sample_dates is None:
        sample_dates = {
                            'train': [self.tk[tick].pd._dates['train'][0].strftime('%Y-%m-%d'), self.tk[tick].pd._dates['train'][-1].strftime('%Y-%m-%d')],
                            'forecast': [self.tk[tick].pd._dates['train'][-1].strftime('%Y-%m-%d'), self.tk[tick].pd._dates['all'][-1].strftime('%Y-%m-%d')],
                            'all': [self.tk[tick].pd._dates['train'][0].strftime('%Y-%m-%d'), self.tk[tick].pd._dates['all'][-1].strftime('%Y-%m-%d')],
                            }
        
    sample_str = {
                    'name': {"train": f"In-sample",
                            "forecast": f"Out-of-sample",
                            "all": f"In- \\& Out-of-sample",},
                    'dates': {period: f"[{sample_dates[period][0]}..{sample_dates[period][-1]})" for period in sample_dates}
                }


    space = ' '
    row_strs = ['' for i in range(N_rows)]
    for i in range(N_rows):
        if split_labels:
            for sl in sample_labels:
                if sl!=sample_labels[0]:
                    row_strs[i] += '\n'
                row_strs[i] += lead[i%N_lead]
                row_strs[i] += '\n'
                row_strs[i] += misc_fns.str_concatenate([f'\\multicolumn{{{cw}}}{{{calign}}}{{{pre}{misc_fns.str_concatenate([sample_str[sl][periods[i*P_row+p]]], fill_str=space, fill_str_void=-1)}{post}}}' for p in range(P_rows[i])], fill_str=' & & ', fill_str_void=-1)
                row_strs[i] += '\n'
                row_strs[i] += '\\\\'
            if incl_cmidrules:
                row_strs[i] += '\n'
                row_strs[i] += misc_fns.str_concatenate([f"\\cmidrule{{{1+c0[i%N_c0]+p*(1+cw)}-{1+c0[i%N_c0]+(1+p)*(1+cw)-2}}}" for p in range(P_rows[i])])
        else:
            row_strs[i] += lead[i%N_lead]
            row_strs[i] += '\n'
            row_strs[i] += misc_fns.str_concatenate([f'\\multicolumn{{{cw}}}{{{calign}}}{{{pre}{misc_fns.str_concatenate([sample_str[sl][periods[i*P_row+p]] for sl in sample_labels], fill_str=space, fill_str_void=-1)}{post}}}' for p in range(P_rows[i])], fill_str=' & & ', fill_str_void=-1)
            row_strs[i] += '\n'
            row_strs[i] += '\\\\'
            if incl_cmidrules:
                row_strs[i] += '\n'
                row_strs[i] += misc_fns.str_concatenate([f"\\cmidrule{{{1+c0[i%N_c0]+p*(1+cw)}-{1+c0[i%N_c0]+(1+p)*(1+cw)-2}}}" for p in range(P_rows[i])])

    return row_strs



def calc_table_periods_head(self, tick=None, sample_dates=None, periods=None, sample_labels=None, lead=None, pre=None, post=None, c0=None, cw=None, calign=None, q_same_row=None, q_space=None, q_above_empty_lead=None, last_head=None, N_p_max=None, p_same_rows=None, p_space=None, repeat=None, r_space=None, incl_hline=None, incl_cmidrules=None, return_heads=None):



    if incl_cmidrules is None:
        incl_cmidrules = True

    if tick is None:
        tick = self._ticks[0]
    
    if periods is None:
        periods = ['train', 'forecast']
    periods = misc_fns.make_iterable_array(periods)
    N_p = np.size(periods)

    if pre is None:
        pre = ''

    if post is None:
        post = ''
    

    if sample_labels is None:
        sample_labels = ['name', 'dates']
    sample_labels = misc_fns.make_iterable_array(sample_labels)
    N_sl = np.size(sample_labels)


    if sample_dates is None:
        sample_dates = {
                            'train': [self.tk[tick].pd._dates['train'][0].strftime('%Y-%m-%d'), self.tk[tick].pd._dates['train'][-1].strftime('%Y-%m-%d')],
                            'forecast': [self.tk[tick].pd._dates['train'][-1].strftime('%Y-%m-%d'), self.tk[tick].pd._dates['all'][-1].strftime('%Y-%m-%d')],
                            'all': [self.tk[tick].pd._dates['train'][0].strftime('%Y-%m-%d'), self.tk[tick].pd._dates['all'][-1].strftime('%Y-%m-%d')],
                            }
        
    sample_str = {
                    'name': {"train": f"In-sample",
                            "forecast": f"Out-of-sample",
                            "all": f"In- \\& Out-of-sample",},
                    'dates': {period: f"[{sample_dates[period][0]}..{sample_dates[period][-1]})" for period in sample_dates}
                }
    

    heads = [[sample_str[sl][period] for sl in sample_labels] for period in periods]

    if return_heads is None:
        return_heads = False
    
    if return_heads:
        return heads
    else:
        return plot_fns.calc_table_head(heads=heads, lead=lead, pre=pre, post=post, c0=c0, cw=cw, calign=calign, q_same_row=q_same_row, q_space=q_space, q_above_empty_lead=q_above_empty_lead, last_head=last_head, N_p_max=N_p_max, p_same_rows=p_same_rows, p_space=p_space, repeat=repeat, r_space=r_space, incl_hline=incl_hline, incl_cmidrules=incl_cmidrules)
    

