import numpy as np
import scipy
import scipy.stats as stats
from scipy.optimize import minimize
import pandas as pd
import copy
import datetime
import os
import matplotlib.pyplot as plt

# Local files
import hawkes as hwks
import cvol as cvl
import ssv_abm
import financial_data_handling as fin_data_handl

import misc_functions as misc_fns
import plot_functions as plot_fns
import dist_functions as dist_fns
import log_ret_functions as lr_fns



class lr_dir():

    def __init__(self, dir_base=None, dir_data=None, dir_res=None, 
                 datetime_r_str=None, datetime_r_TS=None, datetime_w_str=None, datetime_w_TS=None, reset_datetime_w=None):
        """Constructor log_ret_tick class
        """
        self.set_dir(dir_base=dir_base,  dir_data=dir_data, dir_res=dir_res, pref_base=True, reset=True)
        self.set_datetime(datetime_r_str=datetime_r_str, datetime_r_TS=datetime_r_TS, datetime_w_str=datetime_w_str, datetime_w_TS=datetime_w_TS, reset_datetime_w=reset_datetime_w, reset=True)



    def set_dir(self, dir_base=None, dir_data=None, dir_res=None, reset=None, pref_base=None):

        self.set_dir_base(dir_base=dir_base, reset=reset)
        self.set_dir_data(dir_data=dir_data, reset=reset)
        self.set_dir_res(dir_res=dir_res, reset=reset)
        #self.set_datetime(datetime_r_str=datetime_r_str, datetime_r_TS=datetime_r_TS, reset_datetime_w=True, reset=True)


    def set_dir_base(self, dir_base=None, reset=False):
        if dir_base is not None:
            self._dir_base = dir_base
        elif reset:
            self._dir_base = os.getcwd() + '/'


    def calc_dir_suff(self, dir_suff=None, suff=None, dir_base=None, pref_base=None, use_defaults=None):

        if pref_base is None:
            pref_base = True

        if use_defaults is None:
            use_defaults = True

        if suff is None:
            suff = ""

        if dir_base is None:
            dir_base = self.get_dir_base()

        return lr_fns.calc_dir_suff(dir_suff=dir_suff, suff=suff, pref_base=pref_base, dir_base=dir_base, use_defaults=use_defaults)



    def set_dir_data(self, suff=None, dir_data=None, pref_base=None, reset=None):

        if reset is None:
            reset = False

        if pref_base is None:
            pref_base = False

        if suff is None:
            suff = 'data'

        if reset or dir_data is not None:
            self._dir_data = self.calc_dir_suff(suff=suff, dir_suff=dir_data, pref_base=pref_base, use_defaults=True)
            
            
    def set_dir_res(self, suff=None, dir_res=None, pref_base=None, reset=None):


        if reset is None:
            reset = False

        if pref_base is None:
            pref_base = False

        if suff is None:
            suff = 'res'

        if reset or dir_res is not None:
            self._dir_res = self.calc_dir_suff(suff=suff, dir_suff=dir_res, pref_base=pref_base, use_defaults=True)


    def get_dir_base(self):
        return self._dir_base

    def get_dir_data(self, pref_base=None):

        if pref_base is None:
            pref_base = True
        if pref_base:
            return f"{self.get_dir_base()}{self._dir_data}"
        else:
            return self._dir_data
    
    def get_dir_res(self, pref_base=None):
        
        if pref_base is None:
            pref_base = True
        if pref_base:
            return f"{self.get_dir_base()}{self._dir_res}"
        else:
            return self._dir_res
    

    def dir_fn_tick(self, dir_res=None, tick=None, fn=None, periods=None, pref_base=None, mkdir=None, mkdir_incl=None, mkdir_excl=None, val=None):

        if dir_res is None:
            dir_resres = self.get_dir_res(pref_base=pref_base)
        

        return lr_fns.dir_fn_tick(dir_res=dir_res, tick=tick, fn=fn, periods=periods, mkdir=mkdir, mkdir_incl=mkdir_incl, mkdir_excl=mkdir_excl, val=val)
    

    def dir_fn_tick_period_w(self, dir_res=None, tick=None, fn=None, periods_dict=None, pref_base=None, mkdir=None, mkdir_p=None, mkdir_w=None, mkdir_incl=None, mkdir_excl=None, write_period=None, datetime_w_str=None):

        if dir_res is None:
            dir_res = self.get_dir_res(pref_base=pref_base)

        if datetime_w_str is None:
            datetime_w_str = self._datetime['w']['str']

        return lr_fns.dir_fn_tick_period_w(dir_res=dir_res, tick=tick, fn=fn, periods_dict=periods_dict, mkdir=mkdir, mkdir_p=mkdir_p, mkdir_w=mkdir_w, mkdir_incl=mkdir_incl, mkdir_excl=mkdir_excl, write_period=write_period, datetime_w_str=datetime_w_str)
    
    
    def reset_datetime_w(self):

        self._datetime['w']['TS'] = pd.Timestamp(datetime.datetime.now(datetime.timezone.utc))
        self._datetime['w']['str'] = misc_fns.dt_str(self._datetime['w']['TS'])

    def set_datetime(self, datetime_r_str=None, datetime_r_TS=None, datetime_w_str=None, datetime_w_TS=None, reset_datetime_w=None, reset=None):

        if reset is None:
            reset = False
        

        if reset:
            self._datetime = {rw: {t: None for t in ['TS', 'str']} for rw in ['r', 'w']}
        #if datetime_w_str is not None:
        #    self._datetime['w']['str'] = datetime_w_str
        #    self._datetime['w']['TS'] = None
        #elif reset_datetime_w:

        
        if datetime_r_str is not None:
            self._datetime['r']['str'] = datetime_r_str
            self._datetime['r']['TS'] = None
        elif datetime_r_TS is not None:
            self._datetime['r']['TS'] = datetime_r_TS
            self._datetime['r']['str'] = misc_fns.dt_str(self._datetime['r']['TS'])

        if datetime_w_str is None and datetime_w_TS is None:
            if reset_datetime_w is None:
                reset_datetime_w = True
        else:
            if reset_datetime_w is None:
                reset_datetime_w = False

            if datetime_w_str is not None:
                self._datetime['w']['str'] = datetime_w_str
                self._datetime['w']['TS'] = None
            elif datetime_w_TS is not None:
                self._datetime['w']['TS'] = datetime_w_TS
                self._datetime['w']['str'] = misc_fns.dt_str(self._datetime['w']['TS'])
            
                
        if reset_datetime_w:
            self.reset_datetime_w()


class log_ret(lr_dir):
    """Class for log returns analysis of tick
    """
    def __init__(self, dir_base=None, dir_data=None, dir_res=None, datetime_r_str=None, datetime_r_TS=None, ticks=None):
        """Constructor log_ret_tick class
        """
        self.set_setup(dir_base=dir_base,  dir_data=dir_data, dir_res=dir_res, datetime_r_str=datetime_r_str, datetime_r_TS=datetime_r_TS, ticks=ticks)
        


    def set_setup(self, dir_base=None, dir_data=None, dir_res=None, datetime_r_str=None, datetime_r_TS=None, datetime_w_str=None, datetime_w_TS=None, reset_datetime_w=None, ticks=None):

        self.set_dir(dir_base=dir_base,  dir_data=dir_data, dir_res=dir_res, pref_base=True, reset=True)
        self.set_datetime(datetime_r_str=datetime_r_str, datetime_r_TS=datetime_r_TS, datetime_w_str=datetime_w_str, datetime_w_TS=datetime_w_TS, reset_datetime_w=reset_datetime_w, reset=True)
        self._ticks = misc_fns.make_iterable_array(ticks)


    def dir_tick_csv(self, tick):
        return f"{self.get_dir_data()}^{tick.lower()}_d.csv"



    #Checkpoint
    def dir_fn_tick_old(self, tick, fn=None, periods=None, mkdir=None):

        if fn is None:
            fn = "fit"

        if periods is None:
            periods = ""

        if mkdir is None:
            mkdir = False

        dir = {suff: None for suff in ['res', 'fn', 'tick']}

        dir_res = f"{self.get_dir_res()}"
        dir_fn = f"{dir_res}{fn}/"
        dir_tick = f"{dir_fn}{tick}/"
        dir = f"{dir_tick}{periods}/"
        if mkdir:
            plot_fns.mkdir_export(dir_res)
            plot_fns.mkdir_export(dir_fn)
            plot_fns.mkdir_export(dir_tick)
            plot_fns.mkdir_export(dir)

        return dir
    

    

    def dir_fn_tick_period_w(self, dir_res=None, tick=None, fn=None, periods_dict=None, pref_base=None, mkdir=None, mkdir_p=None, mkdir_w=None, mkdir_incl=None, mkdir_excl=None, write_period=None, datetime_w_str=None):
        
        _get_dir_dates_format = lr_fns.get_dir_dates_format()

        if dir_res is None:
            dir_res = self.get_dir_res(pref_base=pref_base)

        if datetime_w_str is None:
            datetime_w_str = self._datetime['w']['str']

        if periods_dict is None:
            periods_dict = {period: self.tk[tick].pd.get_dates_str(dates_format=_get_dir_dates_format[period]) for period in _get_dir_dates_format}

        return lr_fns.dir_fn_tick_period_w(dir_res=dir_res, tick=tick, fn=fn, periods_dict=periods_dict, mkdir=mkdir, mkdir_p=mkdir_p, mkdir_w=mkdir_w, mkdir_incl=mkdir_incl, mkdir_excl=mkdir_excl, write_period=write_period, datetime_w_str=datetime_w_str)
    


    

            




    def set_tk(self, ticks_rng=None,
                            all_start=None, all_end=None, train_start=None, train_end=None, forecast=None,
                            T_all=None, T_train=None,
                            set_cv=None,
                            cv_vol=None, cv_p=None, cv_o=None, cv_q=None, cv_dist=None,
                            cv_a_u=None, 
                            set_tt=None, 
                            tt_a_u=None, tt_dt_dt_fine=None,
                            tt_opt_method=None, tt_fix_a_lambda=None,
                            hb_dist=None, hb_bound=None, hb_mode_bound=None, hb_set_fit=None,
                            a_q=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None,
                            cv_write_tick=None, cv_set_am_stats=None, cv_set_eval=None, append_fit_cv_w=None, append_vrs_cv_w=None,
                            tt_i_0=None, tt_read_params=None, tt_hb_read_params=None, tt_fit_if_need=None,
                            set_vabm=None,
                            vabm_opt_method=None, vabm_rmode_s0s=None, vabm_rmode_s1s=None, vabm_rmode_s2s=None, vabm_bmodes=None, vabm_emodes=None, vabm_use_fp0=None, vabm_check_opt_last=None, vabm_epsilon=None, vabm_step_norm_cutoff=None, vabm_N_k=None, vabm_N_k_trunc=None, vabm_gauss_sigma=None, vabm_datetime_w_str=None,
                            vabm_export_fit_df=None, vabm_export_figs=None, vabm_fig_suppress=None, vabm_reset_fit_df=None, vabm_reset_datetime_w_str=None, vabm_print_fit_time=None,
                            factor=None, print_mp_time=None):

        if set_cv is None:
            set_cv = True
        if set_tt is None:
            set_tt = True
        if set_vabm is None:
            set_vabm = False


        if vabm_datetime_w_str is None:
            vabm_datetime_w_str = self._datetime["w"]["str"]
        
        
        if ticks_rng is None:
            ticks_rng = range(len(self._ticks))
       

        if type(all_start)!=dict:
            all_start_dict = {tick: all_start for tick in self._ticks}
        else:
            all_start_dict = all_start
        if type(all_end)!=dict:
            all_end_dict = {tick: all_end for tick in self._ticks}
        else:
            all_end_dict = all_end
        if type(train_start)!=dict:
            train_start_dict = {tick: train_start for tick in self._ticks}
        else:
            train_start_dict = train_start
        if type(train_end)!=dict:
            train_end_dict = {tick: train_end for tick in self._ticks}
        else:
            train_end_dict = all_end
        if type(forecast)!=dict:
            forecast_dict = {tick: forecast for tick in self._ticks}
        else:
            forecast_dict = forecast
        

        self.tk = dict()
        for i in ticks_rng:
            tick = self._ticks[i]
            self.tk[tick] = log_ret_tick(tick=tick, csv_path=self.dir_tick_csv(tick),
                                            all_start=all_start_dict[tick], all_end=all_end_dict[tick], train_start=train_start_dict[tick], train_end=train_end_dict[tick], forecast=forecast_dict[tick],
                                            T_all=T_all, T_train=T_train,
                                            dir_base=self.get_dir_base(), dir_data=self.get_dir_data(pref_base=False), dir_res=self.get_dir_res(pref_base=False),
                                            datetime_r_str=self._datetime['r']['str'], datetime_r_TS=self._datetime['r']['TS'], datetime_w_str=self._datetime['w']['str'], datetime_w_TS=self._datetime['w']['TS'], reset_datetime_w=None,
                                            set_cv=False,
                                            cv_vol=cv_vol, cv_p=cv_p, cv_o=cv_o, cv_q=cv_q, cv_dist=cv_dist,
                                            cv_a_u=cv_a_u,
                                            set_tt=False, 
                                            tt_a_u=tt_a_u, tt_dt_dt_fine=tt_dt_dt_fine, 
                                            tt_opt_method=tt_opt_method, tt_fix_a_lambda=tt_fix_a_lambda,
                                            hb_dist=hb_dist, hb_bound=hb_bound, hb_mode_bound=hb_mode_bound, hb_set_fit=hb_set_fit,
                                            a_q=a_q, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t,
                                            cv_write_tick=cv_write_tick, cv_set_am_stats=cv_set_am_stats, cv_set_eval=cv_set_eval, append_fit_cv_w=append_fit_cv_w, append_vrs_cv_w=append_vrs_cv_w,
                                            tt_i_0=tt_i_0, tt_read_params=tt_read_params, tt_hb_read_params=tt_hb_read_params, tt_fit_if_need=tt_fit_if_need,
                                            vabm_opt_method=vabm_opt_method, vabm_rmode_s0s=vabm_rmode_s0s, vabm_rmode_s1s=vabm_rmode_s1s, vabm_rmode_s2s=vabm_rmode_s2s, vabm_bmodes=vabm_bmodes, vabm_emodes=vabm_emodes, vabm_use_fp0=vabm_use_fp0, vabm_check_opt_last=vabm_check_opt_last, vabm_epsilon=vabm_epsilon, vabm_step_norm_cutoff=vabm_step_norm_cutoff, vabm_N_k=vabm_N_k, vabm_N_k_trunc=vabm_N_k_trunc, vabm_gauss_sigma=vabm_gauss_sigma, vabm_datetime_w_str=vabm_datetime_w_str,
                                            vabm_export_fit_df=vabm_export_fit_df, vabm_export_figs=vabm_export_figs, vabm_fig_suppress=vabm_fig_suppress, vabm_reset_fit_df=vabm_reset_fit_df, vabm_reset_datetime_w_str=vabm_reset_datetime_w_str, vabm_print_fit_time=vabm_print_fit_time)
        if set_cv:
            self.set_cv_mp_tick(write_tick=cv_write_tick,
                                a_u=cv_a_u,
                                set_am_stats=cv_set_am_stats, set_eval=cv_set_eval, append_fit_cv_w=append_fit_cv_w, append_vrs_cv_w=append_vrs_cv_w,
                                factor=factor, print_mp_time=print_mp_time)
        if set_tt:
            self.set_tt_mp_tick(i_0=tt_i_0, tt_read_params=tt_read_params, tt_hb_read_params=tt_hb_read_params, tt_fit_if_need=tt_fit_if_need, tt_opt_method=tt_opt_method, tt_fix_a_lambda=tt_fix_a_lambda,
                                factor=factor, print_mp_time=print_mp_time)
            
        if set_vabm:
            self.set_vabm_mp_tick(vabm_opt_method=vabm_opt_method, vabm_rmode_s0s=vabm_rmode_s0s, vabm_rmode_s1s=vabm_rmode_s1s, vabm_rmode_s2s=vabm_rmode_s2s, vabm_bmodes=vabm_bmodes, vabm_emodes=vabm_emodes, vabm_use_fp0=vabm_use_fp0, vabm_check_opt_last=vabm_check_opt_last, vabm_epsilon=vabm_epsilon, vabm_step_norm_cutoff=vabm_step_norm_cutoff, vabm_N_k=vabm_N_k, vabm_N_k_trunc=vabm_N_k_trunc, vabm_gauss_sigma=vabm_gauss_sigma, vabm_datetime_w_str=vabm_datetime_w_str,
                                factor=factor, print_mp_time=print_mp_time)


    def calc_ticks(self, tick_rng=None):
        if tick_rng is None:
            tick_rng = range(len(self._ticks))
        
        return [self._ticks[i] for i in tick_rng]


    def set_cv_mp_tick(self, tick_rng=None, write_tick=None,
                                a_u=None,
                                set_am_stats=None, set_eval=None, append_fit_cv_w=None, append_vrs_cv_w=None,
                                factor=None, print_mp_time=None):

        ticks = self.calc_ticks(tick_rng=tick_rng)
        if write_tick is None:
            write_tick = False

        for tick in ticks:
            self.tk[tick].set_cv_sp_params(a_u=a_u, set_am_stats=set_am_stats, set_eval=set_eval, append_fit_cv_w=append_fit_cv_w, append_vrs_cv_w=append_vrs_cv_w, return_self=True)
        
        res_list, res_time = misc_fns.mp_funs_factor(funs=[self.tk[tick].calc_cv_sp for tick in ticks],
                                    factor=factor, print_mp_time=print_mp_time)

        for i in range(len(ticks)):
            self.tk[ticks[i]].cv = res_list[i]
            if write_tick:
                if append_fit_cv_w:
                    dir_fit = self.dir_fn_tick_period_w(tick=ticks[i], fn='fit', mkdir=True, write_period='forecast')
                    for fn in self.tk[ticks[i]].cv._rw['fit']['w']['cv']:
                        self.tk[ticks[i]].cv._rw['fit']['w']['cv'][fn].to_csv(path_or_buf=f"{dir_fit}fit_cv_{fn}.csv", sep=',', index=False)
                if append_vrs_cv_w:
                    dir_vrs = self.dir_fn_tick_period_w(tick=ticks[i], fn='vares', mkdir=True, write_period='forecast')
                    for fn in self.tk[ticks[i]].cv._rw['vares']['w']['cv']:
                        self.tk[ticks[i]].cv._rw['vares']['w']['cv'][fn].to_csv(path_or_buf=f"{dir_vrs}vares_cv_{fn}.csv", sep=',', index=False)

        return res_time

    def set_cv_evt_mp_tick(self, tick_rng=None, write_tick=None,
                                evt_a_u=None,
                                set_eval=None, append_fit_cv_evt_w=None, append_vrs_cv_evt_w=None,
                                factor=None, print_mp_time=None):

        ticks = self.calc_ticks(tick_rng=tick_rng)
        if write_tick is None:
            write_tick = False

        for tick in ticks:
            self.tk[tick].set_cv_evt_sp_params(evt_a_u=evt_a_u, set_eval=set_eval, append_fit_cv_evt_w=append_fit_cv_evt_w, append_vrs_cv_evt_w=append_vrs_cv_evt_w, return_self=True)
        
        res_list, res_time = misc_fns.mp_funs_factor(funs=[self.tk[tick].calc_cv_evt_sp for tick in ticks],
                                    factor=factor, print_mp_time=print_mp_time)

        for i in range(len(ticks)):
            self.tk[ticks[i]].cv = res_list[i]
            if write_tick:
                if append_fit_cv_evt_w:
                    dir_fit = self.dir_fn_tick_period_w(tick=ticks[i], fn='fit', mkdir=True, write_period='forecast')
                    for fn in self.tk[ticks[i]].cv._rw['fit']['w']['cv_evt']:
                        self.tk[ticks[i]].cv._rw['fit']['w']['cv_evt'][fn].to_csv(path_or_buf=f"{dir_fit}fit_cv_evt_{fn}.csv", sep=',', index=False)
                if append_vrs_cv_evt_w:
                    dir_vrs = self.dir_fn_tick_period_w(tick=ticks[i], fn='vares', mkdir=True, write_period='forecast')
                    for fn in self.tk[ticks[i]].cv._rw['vares']['w']['cv_evt']:
                        self.tk[ticks[i]].cv._rw['vares']['w']['cv_evt'][fn].to_csv(path_or_buf=f"{dir_vrs}vares_cv_evt_{fn}.csv", sep=',', index=False)

        return res_time


    def set_cv_am_stats_mp_tick(self, tick_rng=None, write_tick=None,
                                        set_eval=None, append_fit_cv_w=None, append_vrs_cv_w=None, 
                                        factor=None, print_mp_time=None):

        ticks = self.calc_ticks(tick_rng=tick_rng)
        if write_tick is None:
            write_tick = False

        for tick in ticks:
            self.tk[tick].set_cv_sp_params(set_eval=set_eval, append_fit_cv_w=append_fit_cv_w, append_vrs_cv_w=append_vrs_cv_w, return_self=True)

        res_list, res_time = misc_fns.mp_funs_factor(funs=[self.tk[tick].calc_cv_am_stats_sp for tick in ticks],
                                    factor=factor, print_mp_time=print_mp_time)

        for i in range(len(ticks)):
            self.tk[ticks[i]].cv = res_list[i]
            if write_tick:
                if append_fit_cv_w:
                    dir_fit = self.dir_fn_tick_period_w(tick=ticks[i], fn='fit', mkdir=True, write_period='forecast')
                    for fn in self.tk[ticks[i]].cv._rw['fit']['w']['cv']:
                        self.tk[ticks[i]].cv._rw['fit']['w']['cv'][fn].to_csv(path_or_buf=f"{dir_fit}fit_cv_{fn}.csv", sep=',', index=False)
                if append_vrs_cv_w:
                    dir_vrs = self.dir_fn_tick_period_w(tick=ticks[i], fn='vares', mkdir=True, write_period='forecast')
                    for fn in self.tk[ticks[i]].cv._rw['vares']['w']['cv']:
                        self.tk[ticks[i]].cv._rw['vares']['w']['cv'][fn].to_csv(path_or_buf=f"{dir_vrs}vares_cv_{fn}.csv", sep=',', index=False)

        return res_time
    


    def set_vabm_mp_tick(self, tick_rng=None, 
                                vabm_opt_method=None, vabm_rmode_s0s=None, vabm_rmode_s1s=None, vabm_rmode_s2s=None, vabm_bmodes=None, vabm_emodes=None, vabm_use_fp0=None, vabm_check_opt_last=None, vabm_epsilon=None, vabm_step_norm_cutoff=None, vabm_N_k=None, vabm_N_k_trunc=None, vabm_gauss_sigma=None, vabm_datetime_w_str=None,
                                factor=None, print_mp_time=None):
        

        if vabm_datetime_w_str is None:
            vabm_datetime_w_str = self._datetime["w"]["str"]

        ticks = self.calc_ticks(tick_rng=tick_rng)

        for tick in ticks:
            self.tk[tick].set_vabm_sp_params(vabm_opt_method=vabm_opt_method, vabm_rmode_s0s=vabm_rmode_s0s, vabm_rmode_s1s=vabm_rmode_s1s, vabm_rmode_s2s=vabm_rmode_s2s, vabm_bmodes=vabm_bmodes, vabm_emodes=vabm_emodes, vabm_use_fp0=vabm_use_fp0, vabm_check_opt_last=vabm_check_opt_last, vabm_epsilon=vabm_epsilon, vabm_step_norm_cutoff=vabm_step_norm_cutoff, vabm_N_k=vabm_N_k, vabm_N_k_trunc=vabm_N_k_trunc, vabm_gauss_sigma=vabm_gauss_sigma, vabm_datetime_w_str=vabm_datetime_w_str)
        
        res_list, res_time = misc_fns.mp_funs_factor(funs=[self.tk[tick].calc_vabm_sp for tick in ticks],
                                    factor=factor, print_mp_time=print_mp_time)

        for i in range(len(ticks)):
            self.tk[ticks[i]].vabm = res_list[i]

        return res_time
    

    def set_fit_vabm_mp_tick(self, tick_rng=None, 
                                vabm_opt_method=None, vabm_rmode_s0s=None, vabm_rmode_s1s=None, vabm_rmode_s2s=None, vabm_bmodes=None, vabm_emodes=None, vabm_use_fp0=None, vabm_check_opt_last=None, vabm_epsilon=None, vabm_step_norm_cutoff=None, vabm_N_k=None, vabm_N_k_trunc=None, vabm_gauss_sigma=None, vabm_datetime_w_str=None,
                                vabm_export_fit_df=None, vabm_export_figs=None, vabm_fig_suppress=None, vabm_reset_fit_df=None, vabm_reset_datetime_w_str=None, vabm_print_fit_time=None,
                                factor=None, print_mp_time=None):
        
        if vabm_datetime_w_str is None:
            vabm_datetime_w_str = self._datetime["w"]["str"]

        ticks = self.calc_ticks(tick_rng=tick_rng)

        for tick in ticks:
            self.tk[tick].set_vabm_sp_params(vabm_opt_method=vabm_opt_method, vabm_rmode_s0s=vabm_rmode_s0s, vabm_rmode_s1s=vabm_rmode_s1s, vabm_rmode_s2s=vabm_rmode_s2s, vabm_bmodes=vabm_bmodes, vabm_emodes=vabm_emodes, vabm_use_fp0=vabm_use_fp0, vabm_check_opt_last=vabm_check_opt_last, vabm_epsilon=vabm_epsilon, vabm_step_norm_cutoff=vabm_step_norm_cutoff, vabm_N_k=vabm_N_k, vabm_N_k_trunc=vabm_N_k_trunc, vabm_gauss_sigma=vabm_gauss_sigma, vabm_datetime_w_str=vabm_datetime_w_str)
            self.tk[tick].set_fit_vabm_sp_params(vabm_export_fit_df=vabm_export_fit_df, vabm_export_figs=vabm_export_figs, vabm_fig_suppress=vabm_fig_suppress, vabm_reset_fit_df=vabm_reset_fit_df, vabm_reset_datetime_w_str=vabm_reset_datetime_w_str, vabm_print_fit_time=vabm_print_fit_time, vabm_return_self=True)
        
        res_list, res_time = misc_fns.mp_funs_factor(funs=[self.tk[tick].calc_fit_vabm_sp for tick in ticks],
                                    factor=factor, print_mp_time=print_mp_time)

        for i in range(len(ticks)):
            self.tk[ticks[i]].vabm = res_list[i]

        return res_time




    def set_tt_mp_tick(self, tick_rng=None, 
                                i_0=None, tt_read_params=None, tt_hb_read_params=None, tt_fit_if_need=None, tt_opt_method=None, tt_fix_a_lambda=None,
                                factor=None, print_mp_time=None):

        ticks = self.calc_ticks(tick_rng=tick_rng)

        for tick in ticks:
            self.tk[tick].set_tt_sp_params(i_0=i_0, tt_read_params=tt_read_params, tt_hb_read_params=tt_hb_read_params, tt_fit_if_need=tt_fit_if_need, 
                                            tt_opt_method=tt_opt_method, tt_fix_a_lambda=tt_fix_a_lambda,
                                            return_self=True)
        
        res_list, res_time = misc_fns.mp_funs_factor(funs=[self.tk[tick].calc_tt_sp for tick in ticks],
                                    factor=factor, print_mp_time=print_mp_time)

        for i in range(len(ticks)):
            self.tk[ticks[i]].tt = res_list[i]

        return res_time

    #Checkpoint
    def set_fit_tt_mp_tick(self, tick_rng=None, write_tick=None, 
                                a_u_rng=None, tt_read_params=None, tt_fit_if_need=None, print_tt_fit_time=None, print_res_time=None, append_fit_tt_w=None, 
                                factor=None, print_mp_time=None,
                                tt_opt_method=None, tt_fix_a_lambda=None):

        ticks = self.calc_ticks(tick_rng=tick_rng)
        if write_tick is None:
            write_tick = False


        for tick in ticks:
            self.tk[tick].set_tt_hb_sp_params(a_u_rng=a_u_rng, tt_read_params=tt_read_params, tt_fit_if_need=tt_fit_if_need, print_tt_fit_time=print_tt_fit_time, print_res_time=print_res_time, append_fit_tt_w=append_fit_tt_w, return_self=True, tt_opt_method=tt_opt_method, tt_fix_a_lambda=tt_fix_a_lambda)
        
        res_list, res_time = misc_fns.mp_funs_factor(funs=[self.tk[tick].calc_fit_tt_sp for tick in ticks],
                                    factor=factor, print_mp_time=print_mp_time)

        for i in range(len(ticks)):
            self.tk[ticks[i]].tt = res_list[i]
            if write_tick:
                if append_fit_tt_w:
                    dir_fit = self.dir_fn_tick_period_w(tick=ticks[i], fn='fit', mkdir=True, write_period='forecast')
                    for fn in self.tk[ticks[i]].tt._rw['fit']['w']['tt']:
                        #self.tk[ticks[i]].tt._rw['fit']['w']['tt'][fn].to_csv(path_or_buf=f"{dir_fit}fit_tt_c_{fn}.csv", sep=',', index=False)
                        self.tk[ticks[i]].tt._rw['fit']['w']['tt'][fn].to_csv(path_or_buf=f"{dir_fit}fit_tt_{fn}.csv", sep=',', index=False) # Paper 1
        
        return res_time
    

    def set_fit_tt_mp_tick_a_lambda(self, tick_rng=None, write_tick=None, 
                                a_u_rng=None, tt_read_params=None, tt_fit_if_need=None, print_tt_fit_time=None, print_res_time=None, append_fit_tt_w=None, 
                                factor=None, print_mp_time=None,
                                tt_opt_method=None, tt_fix_a_lambda=None):

        ticks = self.calc_ticks(tick_rng=tick_rng)
        if write_tick is None:
            write_tick = False

        tt_fix_a_lambda = misc_fns.make_iterable_array(tt_fix_a_lambda)
        res_times = [None for f in range(len(tt_fix_a_lambda))]

        for f in range(len(tt_fix_a_lambda)):

            for tick in ticks:
                self.tk[tick].set_tt_hb_sp_params(a_u_rng=a_u_rng, tt_read_params=tt_read_params, tt_fit_if_need=tt_fit_if_need, print_tt_fit_time=print_tt_fit_time, print_res_time=print_res_time, append_fit_tt_w=append_fit_tt_w, return_self=True, tt_opt_method=tt_opt_method, tt_fix_a_lambda=tt_fix_a_lambda[f])
            
            res_list, res_time = misc_fns.mp_funs_factor(funs=[self.tk[tick].calc_fit_tt_sp for tick in ticks],
                                        factor=factor, print_mp_time=print_mp_time)

            for i in range(len(ticks)):
                self.tk[ticks[i]].tt = res_list[i]
                if write_tick:
                    if append_fit_tt_w:
                        dir_fit = self.dir_fn_tick_period_w(tick=ticks[i], fn='fit', mkdir=True, write_period='forecast')
                        for fn in self.tk[ticks[i]].tt._rw['fit']['w']['tt']:
                            #self.tk[ticks[i]].tt._rw['fit']['w']['tt'][fn].to_csv(path_or_buf=f"{dir_fit}fit_tt_c_{fn}.csv", sep=',', index=False)
                            self.tk[ticks[i]].tt._rw['fit']['w']['tt'][fn].to_csv(path_or_buf=f"{dir_fit}fit_tt_{fn}.csv", sep=',', index=False) # Paper 1
            
            res_times[f] = res_time

        return res_times


    def set_tt_hb_mp_tick(self, tick_rng=None, write_tick=None, 
                                a_u_rng=None, tt_read_params=None, tt_hb_read_params=None, tt_fit_if_need=None, print_tt_fit_time=None, print_res_time=None, append_fit_tt_w=None, set_eval=None, append_fit_tt_hb_w=None, append_vrs_tt_hb_w=None, 
                                factor=None, print_mp_time=None,
                                tt_opt_method=None, tt_fix_a_lambda=None):

        ticks = self.calc_ticks(tick_rng=tick_rng)
        if write_tick is None:
            write_tick = False

        for tick in ticks:
            self.tk[tick].set_tt_hb_sp_params(a_u_rng=a_u_rng, tt_read_params=tt_read_params, tt_hb_read_params=tt_hb_read_params, tt_fit_if_need=tt_fit_if_need, print_tt_fit_time=print_tt_fit_time, print_res_time=print_res_time, append_fit_tt_w=append_fit_tt_w, set_eval=set_eval, append_fit_tt_hb_w=append_fit_tt_hb_w, append_vrs_tt_hb_w=append_vrs_tt_hb_w, return_self=True, tt_opt_method=tt_opt_method, tt_fix_a_lambda=tt_fix_a_lambda)
        
        res_list, res_time = misc_fns.mp_funs_factor(funs=[self.tk[tick].calc_tt_hb_sp for tick in ticks],
                                    factor=factor, print_mp_time=print_mp_time)

        for i in range(len(ticks)):
            self.tk[ticks[i]].tt = res_list[i]
            if write_tick:
                if append_fit_tt_hb_w:
                    dir_fit = self.dir_fn_tick_period_w(tick=ticks[i], fn='fit', mkdir=True, write_period='forecast')
                    for fn in self.tk[ticks[i]].tt._rw['fit']['w']['tt_hb']:
                        self.tk[ticks[i]].tt._rw['fit']['w']['tt_hb'][fn].to_csv(path_or_buf=f"{dir_fit}fit_tt_hb_{fn}.csv", sep=',', index=False)
                if append_vrs_tt_hb_w:
                    dir_vrs = self.dir_fn_tick_period_w(tick=ticks[i], fn='vares', mkdir=True, write_period='forecast')
                    for fn in self.tk[ticks[i]].tt._rw['vares']['w']['tt_hb']:
                        self.tk[ticks[i]].tt._rw['vares']['w']['tt_hb'][fn].to_csv(path_or_buf=f"{dir_vrs}vares_tt_hb_{fn}.csv", sep=',', index=False)
        return res_time
    

    def calc_table_periods_head(self, tick=None, sample_dates=None, periods=None, sample_labels=None, lead=None, pre=None, post=None, c0=None, cw=None, calign=None, q_same_row=None, q_space=None, q_above_empty_lead=None, last_head=None, N_p_max=None, p_same_rows=None, p_space=None, repeat=None, r_space=None, incl_hline=None, incl_cmidrules=None, return_heads=None):

        return lr_fns.calc_table_periods_head(self=self, tick=tick, sample_dates=sample_dates, periods=periods, sample_labels=sample_labels, lead=lead, pre=pre, post=post, c0=c0, cw=cw, calign=calign, q_same_row=q_same_row, q_space=q_space, q_above_empty_lead=q_above_empty_lead, last_head=last_head, N_p_max=N_p_max, p_same_rows=p_same_rows, p_space=p_space, repeat=repeat, r_space=r_space, incl_hline=incl_hline, incl_cmidrules=incl_cmidrules, return_heads=return_heads)



class log_ret_tick(lr_dir):
    """Class for log returns analysis of tick
    """
    def __init__(self, tick, set_cal_sch=None, csv=None, csv_path=None, sch_index=None, 
                        all_start=None, all_end=None, train_start=None, train_end=None, forecast=None,
                        T_all=None, T_train=None,
                        dir_base=None, dir_data=None, dir_res=None,
                        datetime_r_str=None, datetime_r_TS=None, datetime_w_str=None, datetime_w_TS=None, reset_datetime_w=None,
                        set_cv=None,
                        cv_vol=None, cv_p=None, cv_o=None, cv_q=None, cv_dist=None,
                        cv_a_u=None,
                        set_tt=None, 
                        tt_a_u=None, tt_dt_dt_fine=None, 
                        tt_opt_method=None, tt_fix_a_lambda=None,
                        hb_dist=None, hb_bound=None, hb_mode_bound=None, hb_set_fit=None,
                        a_q=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None,
                        cv_write_tick=None, cv_set_am_stats=None, cv_set_eval=None, append_fit_cv_w=None, append_vrs_cv_w=None,
                        tt_i_0=None, tt_read_params=None, tt_hb_read_params=None, tt_fit_if_need=None,
                        vabm_opt_method=None, vabm_rmode_s0s=None, vabm_rmode_s1s=None, vabm_rmode_s2s=None, vabm_bmodes=None, vabm_emodes=None, vabm_use_fp0=None, vabm_check_opt_last=None, vabm_epsilon=None, vabm_step_norm_cutoff=None, vabm_N_k=None, vabm_N_k_trunc=None, vabm_gauss_sigma=None, vabm_datetime_w_str=None,
                        vabm_export_fit_df=None, vabm_export_figs=None, vabm_fig_suppress=None, vabm_reset_fit_df=None, vabm_reset_datetime_w_str=None, vabm_print_fit_time=None
                        ):
        """Constructor log_ret_tick class
        """
        if set_cv is None:
            set_cv = False
        if set_tt is None:
            set_tt = False

        self.set_dir(dir_base=dir_base,  dir_data=dir_data, dir_res=dir_res, pref_base=True, reset=True)
        self.set_datetime(datetime_r_str=datetime_r_str, datetime_r_TS=datetime_r_TS, datetime_w_str=datetime_w_str, datetime_w_TS=datetime_w_TS, reset_datetime_w=reset_datetime_w, reset=True)


        self.pd = price_data(tick=tick, set_cal_sch=set_cal_sch, csv=csv, csv_path=csv_path, sch_index=sch_index,
                                all_start=all_start, all_end=all_end, train_start=train_start, train_end=train_end, forecast=forecast,
                                T_all=T_all, T_train=T_train)

        
        self.set_vrs_params(a_q=a_q, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)
        self.set_dir_params()
        self.set_cv_params(vol=cv_vol, p=cv_p, o=cv_o, q=cv_q, dist=cv_dist)
        self.set_tt_params(a_u=tt_a_u, dt_dt_fine=tt_dt_dt_fine, opt_method=tt_opt_method, fix_a_lambda=tt_fix_a_lambda)
        self.set_tt_hb_params(dist=hb_dist, bound=hb_bound, mode_bound=hb_mode_bound, set_fit=hb_set_fit)

        self.set_cv_sp_params(a_u=cv_a_u, set_am_stats=cv_set_am_stats, set_eval=cv_set_eval, append_fit_cv_w=append_fit_cv_w, append_vrs_cv_w=append_vrs_cv_w)
        self.set_tt_sp_params(i_0=tt_i_0, tt_read_params=tt_read_params, tt_hb_read_params=tt_hb_read_params, tt_fit_if_need=tt_fit_if_need, tt_opt_method=tt_opt_method, tt_fix_a_lambda=tt_fix_a_lambda)    
        self.set_tt_hb_sp_params()

        self.set_vabm_sp_params(vabm_opt_method=vabm_opt_method, vabm_rmode_s0s=vabm_rmode_s0s, vabm_rmode_s1s=vabm_rmode_s1s, vabm_rmode_s2s=vabm_rmode_s2s, vabm_bmodes=vabm_bmodes, vabm_emodes=vabm_emodes, vabm_use_fp0=vabm_use_fp0, vabm_check_opt_last=vabm_check_opt_last, vabm_epsilon=vabm_epsilon, vabm_step_norm_cutoff=vabm_step_norm_cutoff, vabm_N_k=vabm_N_k, vabm_N_k_trunc=vabm_N_k_trunc, vabm_gauss_sigma=vabm_gauss_sigma, vabm_datetime_w_str=vabm_datetime_w_str)
        self.set_fit_vabm_sp_params(vabm_export_fit_df=vabm_export_fit_df, vabm_export_figs=vabm_export_figs, vabm_fig_suppress=vabm_fig_suppress, vabm_reset_fit_df=vabm_reset_fit_df, vabm_reset_datetime_w_str=vabm_reset_datetime_w_str, vabm_print_fit_time=vabm_print_fit_time, vabm_return_self=True)



        if set_cv:
            self.cv = self.calc_cv_sp()
        if set_tt:
            self.tt = self.calc_tt_sp()



            


    def calc_cv_sp(self):

        set_am_stats = self._cv_sp_params['set_am_stats']
        set_eval = self._cv_sp_params['set_eval']
        append_fit_cv_w = self._cv_sp_params['append_fit_cv_w']
        append_vrs_cv_w = self._cv_sp_params['append_vrs_cv_w']

        if set_am_stats is None:
            set_am_stats = True
        if set_eval is None:
            set_eval = False
        if append_fit_cv_w is None:
            append_fit_cv_w = True
        if append_vrs_cv_w is None:
            append_vrs_cv_w = True
        
        cv =  cvl.cvol_mod(y=self.pd.lr['p']['all'], 
                                vol=self._cv_params['vol'], p=self._cv_params['p'], o=self._cv_params['o'], q=self._cv_params['q'], dist=self._cv_params['dist'], 
                                a_u=self._cv_sp_params['a_u'], 
                                last_obs=self.pd._periods['train'][1], meta_series=self.pd._meta_series,
                                set_am_stats=set_am_stats,
                                a_q=self._vrs_params['a_q'], set_eval=set_eval, 
                                CT_Q=self._vrs_params['CT_Q'], CT_E=self._vrs_params['CT_E'], S_Q=self._vrs_params['S_Q'], S_E=self._vrs_params['S_E'], save_t=self._vrs_params['save_t'], 
                                append_fit_cv_w=append_fit_cv_w, append_vrs_cv_w=append_vrs_cv_w)

        #if self._dir_params['vrs_dir']['train'] is not None:
        #    cv.read_vares_cv(vrs_dir=self._dir_params['vrs_dir']['train']) # Paper 1
        
        return cv

    
    def calc_cv_evt_sp(self):


        evt_a_u = self._cv_evt_sp_params['evt_a_u']
        set_eval = self._cv_evt_sp_params['set_eval']
        append_fit_cv_evt_w = self._cv_evt_sp_params['append_fit_cv_evt_w']
        append_vrs_cv_evt_w = self._cv_evt_sp_params['append_vrs_cv_evt_w']

        if evt_a_u is None:
            evt_a_u = None
        if set_eval is None:
            set_eval = False
        if append_fit_cv_evt_w is None:
            append_fit_cv_evt_w = True
        if append_vrs_cv_evt_w is None:
            append_vrs_cv_evt_w = True

        self.cv.set_bdist_evt(evt_a_u=evt_a_u, 
                                a_q=self._vrs_params['a_q'], set_eval=set_eval, 
                                CT_Q=self._vrs_params['CT_Q'], CT_E=self._vrs_params['CT_E'], S_Q=self._vrs_params['S_Q'], S_E=self._vrs_params['S_E'], save_t=self._vrs_params['save_t'], 
                                append_fit_cv_evt_w=append_fit_cv_evt_w, append_vrs_cv_evt_w=append_vrs_cv_evt_w)
        
        return self.cv

  
    def set_cv_params(self, vol=None, p=None, o=None, q=None, dist=None):

        self._cv_params =   {
                                'vol': vol,
                                'p': p,
                                'o': o,
                                'q': q,
                                'dist': dist,
                            }
    
    def set_cv_sp_params(self, a_u=None, set_am_stats=None, set_eval=None, append_fit_cv_w=None, append_vrs_cv_w=None, return_self=None):

        self._cv_sp_params =    {
                                    'a_u': a_u,
                                    'set_am_stats': set_am_stats,
                                    'set_eval': set_eval,
                                    'append_fit_cv_w': append_fit_cv_w,
                                    'append_vrs_cv_w': append_vrs_cv_w,
                                    'return_self': return_self,
                                }

    def set_cv_evt_sp_params(self, evt_a_u=None, set_eval=None, append_fit_cv_evt_w=None, append_vrs_cv_evt_w=None, return_self=None):

        self._cv_evt_sp_params =    {
                                        'evt_a_u': evt_a_u,
                                        'set_eval': set_eval,
                                        'append_fit_cv_evt_w': append_fit_cv_evt_w,
                                        'append_vrs_cv_evt_w': append_vrs_cv_evt_w,
                                        'return_self': return_self,
                                    }


    def calc_cv_am_stats_sp(self):

        self.cv.set_am_stats(last_obs=self.pd._periods['train'][1], a_q=self._vrs_params['a_q'], set_eval=self._cv_sp_params['set_eval'], 
                                CT_Q=self._vrs_params['CT_Q'], CT_E=self._vrs_params['CT_E'], S_Q=self._vrs_params['S_Q'], S_E=self._vrs_params['S_E'], 
                                save_t=self._vrs_params['save_t'], 
                                append_fit_cv_w=self._cv_sp_params['append_fit_cv_w'], append_vrs_cv_w=self._cv_sp_params['append_vrs_cv_w'])

        if self._cv_sp_params['return_self']:
            return self.cv
        else:
            return self.cv._rw['fit']['w']['cv']   


    def calc_cv_vrs_sp(self):

        self.cv.set_vrs(a_q=self._vrs_params['a_q'], set_eval=self._cv_sp_params['set_eval'], 
                                CT_Q=self._vrs_params['CT_Q'], CT_E=self._vrs_params['CT_E'], S_Q=self._vrs_params['S_Q'], S_E=self._vrs_params['S_E'], 
                        save_t=self._vrs_params['save_t'], append_w=self._cv_sp_params['append_vrs_cv_w'])

        if self._cv_sp_params['return_self']:
            return self.cv
        else:
            return self.cv._rw['vrs']['w']['cv']   




    def set_vabm(self, vabm_opt_method=None, vabm_rmode_s0s=None, vabm_rmode_s1s=None, vabm_rmode_s2s=None, vabm_bmodes=None, vabm_emodes=None, vabm_use_fp0=None, vabm_check_opt_last=None, vabm_epsilon=None, vabm_step_norm_cutoff=None, vabm_N_k=None, vabm_N_k_trunc=None, vabm_gauss_sigma=None, vabm_datetime_w_str=None):

        self.set_dir_params()
        self.set_vabm_sp_params(vabm_opt_method=vabm_opt_method, vabm_rmode_s0s=vabm_rmode_s0s, vabm_rmode_s1s=vabm_rmode_s1s, vabm_rmode_s2s=vabm_rmode_s2s, vabm_bmodes=vabm_bmodes, vabm_emodes=vabm_emodes, vabm_use_fp0=vabm_use_fp0, vabm_check_opt_last=vabm_check_opt_last, vabm_epsilon=vabm_epsilon, vabm_step_norm_cutoff=vabm_step_norm_cutoff, vabm_N_k=vabm_N_k, vabm_N_k_trunc=vabm_N_k_trunc, vabm_gauss_sigma=vabm_gauss_sigma, vabm_datetime_w_str=vabm_datetime_w_str)
        self.vabm = self.calc_vabm_sp()


    def calc_vabm_sp(self):

        vabm_opt_method = self._vabm_sp_params["vabm_opt_method"]

        vabm_rmode_s0s = self._vabm_sp_params["vabm_rmode_s0s"]
        vabm_rmode_s1s = self._vabm_sp_params["vabm_rmode_s1s"]
        vabm_rmode_s2s = self._vabm_sp_params["vabm_rmode_s2s"]
        vabm_bmodes = self._vabm_sp_params["vabm_bmodes"]
        vabm_emodes = self._vabm_sp_params["vabm_emodes"]


        vabm_use_fp0 = self._vabm_sp_params["vabm_use_fp0"]
        vabm_check_opt_last = self._vabm_sp_params["vabm_check_opt_last"]

        vabm_epsilon = self._vabm_sp_params["vabm_epsilon"]
        vabm_step_norm_cutoff = self._vabm_sp_params["vabm_step_norm_cutoff"]
        vabm_N_k = self._vabm_sp_params["vabm_N_k"]
        vabm_N_k_trunc = self._vabm_sp_params["vabm_N_k_trunc"]
        vabm_gauss_sigma = self._vabm_sp_params["vabm_gauss_sigma"]

        vabm_datetime_w_str = self._vabm_sp_params["vabm_datetime_w_str"]
        


        vabm = ssv_abm.pruna_process(N_s=None,
                                    p_t=self.pd.calc_lp_p(period="all", m1=True), r_t=None, p_0=None, 
                                    meta_series=self.pd._meta_series, fit_dir_r=self._dir_params["fit_dir"]['train'], datetime_w_str=vabm_datetime_w_str,
                                    opt_method=vabm_opt_method,
                                    rmode_s0s=vabm_rmode_s0s, rmode_s1s=vabm_rmode_s1s, rmode_s2s=vabm_rmode_s2s, bmodes=vabm_bmodes, emodes=vabm_emodes, 
                                    #T=None, T_burn=None,
                                    #train_t_start=None, train_t_end=None,
                                    use_fp0=vabm_use_fp0, check_opt_last=vabm_check_opt_last,
                                    epsilon=vabm_epsilon, step_norm_cutoff=vabm_step_norm_cutoff, 
                                    N_k=vabm_N_k, N_k_trunc=vabm_N_k_trunc, gauss_sigma=vabm_gauss_sigma,
                                )
        

        return vabm
    

    def set_fit_vabm(self, vabm_opt_method=None, vabm_rmode_s0s=None, vabm_rmode_s1s=None, vabm_rmode_s2s=None, vabm_bmodes=None, vabm_emodes=None, vabm_use_fp0=None, vabm_check_opt_last=None, vabm_epsilon=None, vabm_step_norm_cutoff=None, vabm_N_k=None, vabm_N_k_trunc=None, vabm_gauss_sigma=None, vabm_datetime_w_str=None,
                        vabm_export_fit_df=None, vabm_export_figs=None, vabm_fig_suppress=None, vabm_reset_fit_df=None, vabm_reset_datetime_w_str=None, vabm_print_fit_time=None):

        self.set_dir_params()
        self.set_vabm_sp_params(vabm_opt_method=vabm_opt_method, vabm_rmode_s0s=vabm_rmode_s0s, vabm_rmode_s1s=vabm_rmode_s1s, vabm_rmode_s2s=vabm_rmode_s2s, vabm_bmodes=vabm_bmodes, vabm_emodes=vabm_emodes, vabm_use_fp0=vabm_use_fp0, vabm_check_opt_last=vabm_check_opt_last, vabm_epsilon=vabm_epsilon, vabm_step_norm_cutoff=vabm_step_norm_cutoff, vabm_N_k=vabm_N_k, vabm_N_k_trunc=vabm_N_k_trunc, vabm_gauss_sigma=vabm_gauss_sigma, vabm_datetime_w_str=vabm_datetime_w_str)
        self.set_fit_vabm_sp_params(vabm_export_fit_df=vabm_export_fit_df, vabm_export_figs=vabm_export_figs, vabm_fig_suppress=vabm_fig_suppress, vabm_reset_fit_df=vabm_reset_fit_df, vabm_reset_datetime_w_str=vabm_reset_datetime_w_str, vabm_print_fit_time=vabm_print_fit_time, vabm_return_self=True)
        self.vabm = self.calc_fit_vabm_sp()
    

    def calc_fit_vabm_sp(self):

        vabm_opt_method = self._vabm_sp_params["vabm_opt_method"]

        vabm_rmode_s0s = self._vabm_sp_params["vabm_rmode_s0s"]
        vabm_rmode_s1s = self._vabm_sp_params["vabm_rmode_s1s"]
        vabm_rmode_s2s = self._vabm_sp_params["vabm_rmode_s2s"]
        vabm_bmodes = self._vabm_sp_params["vabm_bmodes"]
        vabm_emodes = self._vabm_sp_params["vabm_emodes"]

        vabm_use_fp0 = self._vabm_sp_params["vabm_use_fp0"]
        vabm_check_opt_last = self._vabm_sp_params["vabm_check_opt_last"]

        vabm_N_k = self._vabm_sp_params["vabm_N_k"]
        vabm_N_k_trunc = self._vabm_sp_params["vabm_N_k_trunc"]
        vabm_gauss_sigma = self._vabm_sp_params["vabm_gauss_sigma"]


        vabm_export_fit_df = self._fit_vabm_sp_params["vabm_export_fit_df"]
        vabm_export_figs = self._fit_vabm_sp_params["vabm_export_figs"]
        vabm_fig_suppress = self._fit_vabm_sp_params["vabm_fig_suppress"]
        vabm_reset_fit_df = self._fit_vabm_sp_params["vabm_reset_fit_df"]
        vabm_reset_datetime_w_str = self._fit_vabm_sp_params["vabm_reset_datetime_w_str"]
        vabm_print_fit_time = self._fit_vabm_sp_params["vabm_print_fit_time"]

        vabm_return_self = self._fit_vabm_sp_params["vabm_return_self"]


            
        self.vabm.all_loop_fit(rmode_s0s=vabm_rmode_s0s, rmode_s1s=vabm_rmode_s1s, rmode_s2s=vabm_rmode_s2s, bmodes=vabm_bmodes, emodes=vabm_emodes, 
                            opt_method=vabm_opt_method, use_fp0=vabm_use_fp0, check_opt_last=vabm_check_opt_last, 
                            N_k=vabm_N_k, N_k_trunc=vabm_N_k_trunc, gauss_sigma=vabm_gauss_sigma, 
                            export_fit_df=vabm_export_fit_df, export_figs=vabm_export_figs, fig_suppress=vabm_fig_suppress, reset_fit_df=vabm_reset_fit_df, reset_datetime_w_str=vabm_reset_datetime_w_str, print_fit_time=vabm_print_fit_time)
        
        if vabm_return_self is None:
            vabm_return_self = True
        
        if vabm_return_self:
            return self.vabm
        else:
            return self.vabm._rw['fit']['w']
    

    def set_vabm_sp_params(self, vabm_opt_method=None, vabm_rmode_s0s=None, vabm_rmode_s1s=None, vabm_rmode_s2s=None, vabm_bmodes=None, vabm_emodes=None, vabm_use_fp0=None, vabm_check_opt_last=None, vabm_epsilon=None, vabm_step_norm_cutoff=None, vabm_N_k=None, vabm_N_k_trunc=None, vabm_gauss_sigma=None, vabm_datetime_w_str=None):
        self._vabm_sp_params = {
                                "vabm_opt_method": vabm_opt_method,

                                "vabm_rmode_s0s": vabm_rmode_s0s,
                                "vabm_rmode_s1s": vabm_rmode_s1s,
                                "vabm_rmode_s2s": vabm_rmode_s2s,
                                "vabm_bmodes": vabm_bmodes,
                                "vabm_emodes": vabm_emodes,

                                "vabm_use_fp0": vabm_use_fp0,
                                "vabm_check_opt_last": vabm_check_opt_last,

                                "vabm_epsilon": vabm_epsilon,
                                "vabm_step_norm_cutoff": vabm_step_norm_cutoff,
                                "vabm_N_k": vabm_N_k,
                                "vabm_N_k_trunc": vabm_N_k_trunc,
                                "vabm_gauss_sigma": vabm_gauss_sigma,

                                "vabm_datetime_w_str": vabm_datetime_w_str,

                            }
        

    def set_fit_vabm_sp_params(self, vabm_export_fit_df=None, vabm_export_figs=None, vabm_fig_suppress=None, vabm_reset_fit_df=None, vabm_reset_datetime_w_str=None, vabm_print_fit_time=None, vabm_return_self=None):
        self._fit_vabm_sp_params = {

                                    "vabm_export_fit_df": vabm_export_fit_df, 
                                    "vabm_export_figs": vabm_export_figs, 
                                    "vabm_fig_suppress": vabm_fig_suppress, 
                                    "vabm_reset_fit_df": vabm_reset_fit_df, 
                                    "vabm_reset_datetime_w_str": vabm_reset_datetime_w_str, 
                                    "vabm_print_fit_time": vabm_print_fit_time,
                                    "vabm_return_self": vabm_return_self,

                                }


    

    def set_tt(self, a_u=None, dt_dt_fine=None,
                    i_0=None, tt_read_params=None, tt_hb_read_params=None, tt_fit_if_need=None,
                    tt_opt_method=None, tt_fix_a_lambda=None):

        self.set_dir_params()
        self.set_tt_params(a_u=a_u, dt_dt_fine=dt_dt_fine, opt_method=tt_opt_method, fix_a_lambda=tt_fix_a_lambda)
        self.set_tt_sp_params(i_0=i_0, tt_read_params=tt_read_params, tt_hb_read_params=tt_hb_read_params, tt_fit_if_need=tt_fit_if_need, tt_opt_method=tt_opt_method, tt_fix_a_lambda=tt_fix_a_lambda)
        self.tt = self.calc_tt_sp()

    def calc_tt_sp(self):

        i_0 = self._tt_sp_params['i_0']
        tt_read_params = self._tt_sp_params['tt_read_params']
        tt_hb_read_params = self._tt_sp_params['tt_hb_read_params']
        tt_fit_if_need = self._tt_sp_params['tt_fit_if_need']
        tt_opt_method = self._tt_sp_params['tt_opt_method']
        tt_fix_a_lambda = self._tt_sp_params['tt_fix_a_lambda']
        
        if i_0 is None:
            i_0 = 0 
        if tt_read_params is None:
            tt_read_params = True
        if tt_hb_read_params is None:
            tt_hb_read_params = True
        if tt_fit_if_need is None:
            tt_fit_if_need = False

        tt = hwks.tt_pot(data=self.pd.lr['p']['all'], exchanges=self.pd._exchange,
                                    a_u=self._tt_params['a_u'][i_0], dt_dt_fine=self._tt_params['dt_dt_fine'], 
                                train_t_end=self.pd._periods['train'][1], meta_series=self.pd._meta_series,
                                opt_method=tt_opt_method, fix_a_lambda=tt_fix_a_lambda)

        if self._dir_params['fit_dir']['train'] is not None:
            print('read_fit_tt...')
            tt.read_fit_tt(fit_dir=self._dir_params['fit_dir']['train'])
            print('read_fit_tt_hb...')
            tt.read_fit_tt_hb(fit_dir=self._dir_params['fit_dir']['train'])
            print('done!')
            #tt.load_a_u(a_u=self._tt_params['a_u'][i_0], tt_read_params=tt_read_params, tt_hb_read_params=tt_hb_read_params, opt_method=tt_opt_method, fix_a_lambda=tt_fix_a_lambda)
            tt.load_a_u(a_u=self._tt_params['a_u'][i_0], tt_read_params=tt_read_params, tt_hb_read_params=tt_hb_read_params)

        #if self._dir_params['vrs_dir'] is not None:
        #    tt.read_vares_tt_hb(vrs_dir=self._dir_params['vrs_dir']) # Paper 1


        return tt


    
    
    def dir_fn_tick(self, dir_res=None, tick=None, fn=None, periods=None, pref_base=None, mkdir=None, mkdir_incl=None, mkdir_excl=None, val=None):

        if dir_res is None:
            dir_res = self.get_dir_res(pref_base=pref_base)

        if tick is None:
            tick = self.pd._tick

        return lr_fns.dir_fn_tick(dir_res=dir_res, tick=tick, fn=fn, periods=periods, mkdir=mkdir, mkdir_incl=mkdir_incl, mkdir_excl=mkdir_excl, val=val)
    

    def dir_fn_tick_period_w(self, dir_res=None, tick=None, fn=None, periods_dict=None, pref_base=None, mkdir=None, mkdir_p=None, mkdir_w=None, mkdir_incl=None, mkdir_excl=None, write_period=None, datetime_w_str=None):
        
        _get_dir_dates_format = lr_fns.get_dir_dates_format()

        if dir_res is None:
            dir_res = self.get_dir_res(pref_base=pref_base)

        if tick is None:
            tick = self.pd._tick

        if datetime_w_str is None:
            datetime_w_str = self._datetime['w']['str']

        if periods_dict is None:
            periods_dict = {period: self.pd.get_dates_str(dates_format=_get_dir_dates_format[period]) for period in _get_dir_dates_format}

        return lr_fns.dir_fn_tick_period_w(dir_res=dir_res, tick=tick, fn=fn, periods_dict=periods_dict, mkdir=mkdir, mkdir_p=mkdir_p, mkdir_w=mkdir_w, mkdir_incl=mkdir_incl, mkdir_excl=mkdir_excl, write_period=write_period, datetime_w_str=datetime_w_str)
    

    

    def set_dir_params(self):

        _get_dir_dates_format = lr_fns.get_dir_dates_format()


        self._dir_params =  {
                                'fit_dir': {period: self.dir_fn_tick_period_w(fn='fit', mkdir=False, mkdir_p=False, mkdir_w=False,  write_period=period, datetime_w_str='') for period in  _get_dir_dates_format},
                                'vrs_dir': {period: self.dir_fn_tick_period_w(fn='vares', mkdir=False, mkdir_p=False, mkdir_w=False,  write_period=period, datetime_w_str='') for period in  _get_dir_dates_format},
                            }
        

    def set_tt_params(self, a_u=None, dt_dt_fine=None, opt_method=None, fix_a_lambda=None):
        self._tt_params =   {
                                'a_u': misc_fns.make_iterable_array(a_u),
                                'dt_dt_fine': dt_dt_fine,
                                'opt_method': opt_method,
                                'fix_a_lambda': fix_a_lambda,
                            }


    def set_tt_sp_params(self, i_0=None, tt_read_params=None, tt_hb_read_params=None, tt_fit_if_need=None, tt_opt_method=None, tt_fix_a_lambda=None, return_self=None):
        self._tt_sp_params = {
                                'i_0': i_0,
                                'tt_read_params': tt_read_params,
                                'tt_hb_read_params': tt_hb_read_params,
                                'tt_fit_if_need': tt_fit_if_need,
                                'tt_opt_method': tt_opt_method,
                                'tt_fix_a_lambda': tt_fix_a_lambda,
                                'return_self': return_self,
                            }
    

    def load_a_u(self, i=0, a_u=None, tt_read_params=None, tt_hb_read_params=None, tt_fit_if_need=None, print_tt_fit_time=None, print_res_time=None, append_fit_tt_w=None, tt_opt_method=None, tt_fix_a_lambda=None):

        if a_u is None:
            a_u = self._tt_params['a_u'][i]

        if tt_fit_if_need is None:
            tt_fit_if_need = False

        read = self.tt.load_a_u(a_u=a_u, tt_read_params=tt_read_params, tt_hb_read_params=tt_hb_read_params, tt_opt_method=tt_opt_method, tt_fix_a_lambda=tt_fix_a_lambda)
        if tt_fit_if_need and read==False:
            self.tt.set_tt_fit(print_tt_fit_time=print_tt_fit_time, print_res_time=print_res_time, append_w=append_fit_tt_w, opt_method=tt_opt_method, fix_a_lambda=tt_fix_a_lambda)




    def set_tt_tt_fit(self, print_tt_fit_time=None, print_res_time=None, append_fit_tt_w=None, a_u=None, a_u_rng=None, tt_opt_method=None, tt_fix_a_lambda=None):

        if a_u is not None:
            self._tt_params['a_u'] = misc_fns.make_iterable_array(a_u)
        if a_u_rng is None:
            a_u_rng = range(len(self._tt_params['a_u']))

        if append_fit_tt_w is None:
            append_fit_tt_w = True

        for i in a_u_rng:
            self.load_a_u(i=i, tt_read_params=False, tt_hb_read_params=False, tt_fit_if_need=True, print_tt_fit_time=print_tt_fit_time, print_res_time=print_res_time, append_fit_tt_w=append_fit_tt_w, tt_opt_method=tt_opt_method, tt_fix_a_lambda=tt_fix_a_lambda)


    def set_tt_hb_mp(self, set_eval=None, append_fit_tt_hb_w=None, append_vrs_tt_hb_w=None, a_u_rng=None, tt_read_params=None, tt_hb_read_params=None, tt_fit_if_need=None, print_tt_fit_time=None, print_res_time=None, append_fit_tt_w=None, tt_opt_method=None, tt_fix_a_lambda=None):

        if a_u_rng is None:
            a_u_rng = range(len(self._tt_params['a_u']))
        for i in a_u_rng:
            self.load_a_u(i=i, tt_read_params=tt_read_params, tt_hb_read_params=tt_hb_read_params, 
                        tt_fit_if_need=tt_fit_if_need, print_tt_fit_time=print_tt_fit_time, print_res_time=print_res_time, append_fit_tt_w=append_fit_tt_w, tt_opt_method=tt_opt_method, tt_fix_a_lambda=tt_fix_a_lambda)
    
            if np.any([self.tt.tt[mode][mark]._hb is None for mode in self.tt.tt for mark in self.tt.tt[mode]]):
                self.tt.set_hb_mp(dist=self._tt_hb_params['dist'], 
                                        bound=self._tt_hb_params['bound'], mode_bound=self._tt_hb_params['mode_bound'], set_fit=self._tt_hb_params['set_fit'], 
                                        a_q=self._vrs_params['a_q'], 
                                        set_eval=set_eval, 
                                        CT_Q=self._vrs_params['CT_Q'], CT_E=self._vrs_params['CT_E'], S_Q=self._vrs_params['S_Q'], S_E=self._vrs_params['S_E'], 
                                        save_t=self._vrs_params['save_t'], 
                                        append_fit_tt_hb_w=append_fit_tt_hb_w,
                                        append_vrs_tt_hb_w=append_vrs_tt_hb_w)
            else:
                self.tt.set_vrs_mp(a_q=self._vrs_params['a_q'], 
                                        set_eval=set_eval, 
                                        CT_Q=self._vrs_params['CT_Q'], CT_E=self._vrs_params['CT_E'], S_Q=self._vrs_params['S_Q'], S_E=self._vrs_params['S_E'], 
                                        save_t=self._vrs_params['save_t'], 
                                        append_vrs_tt_hb_w=append_vrs_tt_hb_w)

    def calc_tt_hb_sp_inp(self, a_u_rng=None, tt_read_params=None, tt_hb_read_params=None, tt_fit_if_need=None, print_tt_fit_time=None, print_res_time=None, append_fit_tt_w=None, set_eval=None, append_fit_tt_hb_w=None, append_vrs_tt_hb_w=None, return_self=None):

        self.set_tt_hb_params(a_u_rng=a_u_rng, tt_read_params=tt_read_params, tt_hb_read_params=tt_hb_read_params, tt_fit_if_need=tt_fit_if_need, print_tt_fit_time=print_tt_fit_time, print_res_time=print_res_time, append_fit_tt_w=append_fit_tt_w, set_eval=set_eval, append_fit_tt_hb_w=append_fit_tt_hb_w, append_vrs_tt_hb_w=append_vrs_tt_hb_w, return_self=return_self)
        self.calc_tt_hb_sp()


    def set_tt_hb_sp_params(self, a_u_rng=None, tt_read_params=None, tt_hb_read_params=None, tt_fit_if_need=None, print_tt_fit_time=None, print_res_time=None, append_fit_tt_w=None, set_eval=None, append_fit_tt_hb_w=None, append_vrs_tt_hb_w=None, return_self=None, tt_opt_method=None, tt_fix_a_lambda=None):

        self._tt_hb_sp_params = {
                                    'a_u_rng': a_u_rng,
                                    'tt_read_params': tt_read_params,
                                    'tt_hb_read_params': tt_hb_read_params,
                                    'tt_fit_if_need': tt_fit_if_need,
                                    'print_tt_fit_time': print_tt_fit_time,
                                    'print_res_time': print_res_time,
                                    'append_fit_tt_w': append_fit_tt_w,
                                    'set_eval': set_eval,
                                    'append_fit_tt_hb_w': append_fit_tt_hb_w,
                                    'append_vrs_tt_hb_w': append_vrs_tt_hb_w,
                                    'return_self': return_self,
                                    'tt_opt_method': tt_opt_method,
                                    'tt_fix_a_lambda': tt_fix_a_lambda,
                                }


    def calc_fit_tt_sp(self):

        if self._tt_hb_sp_params['a_u_rng'] is None:
            self._tt_hb_sp_params['a_u_rng'] = range(len(self._tt_params['a_u']))
        for i in self._tt_hb_sp_params['a_u_rng']:
            self.load_a_u(i=i, tt_read_params=self._tt_hb_sp_params['tt_read_params'], tt_hb_read_params=self._tt_hb_sp_params['tt_hb_read_params'], 
                        tt_fit_if_need=self._tt_hb_sp_params['tt_fit_if_need'], 
                        print_tt_fit_time=self._tt_hb_sp_params['print_tt_fit_time'], print_res_time=self._tt_hb_sp_params['print_res_time'], append_fit_tt_w=self._tt_hb_sp_params['append_fit_tt_w'],
                        tt_opt_method=self._tt_hb_sp_params['tt_opt_method'], tt_fix_a_lambda=self._tt_hb_sp_params['tt_fix_a_lambda'])
        if self._tt_hb_sp_params['return_self']:
            return self.tt
        else:
            return self.tt._rw['fit']['w']



    def calc_tt_hb_sp(self):

        if self._tt_hb_sp_params['a_u_rng'] is None:
            self._tt_hb_sp_params['a_u_rng'] = range(len(self._tt_params['a_u']))
        for i in self._tt_hb_sp_params['a_u_rng']:
            self.load_a_u(i=i, tt_read_params=self._tt_hb_sp_params['tt_read_params'], tt_hb_read_params=self._tt_hb_sp_params['tt_hb_read_params'], 
                        tt_fit_if_need=self._tt_hb_sp_params['tt_fit_if_need'], 
                        print_tt_fit_time=self._tt_hb_sp_params['print_tt_fit_time'], print_res_time=self._tt_hb_sp_params['print_res_time'], append_fit_tt_w=self._tt_hb_sp_params['append_fit_tt_w'],
                        tt_opt_method=self._tt_hb_sp_params['tt_opt_method'], tt_fix_a_lambda=self._tt_hb_sp_params['tt_fix_a_lambda'])

            #if np.any([self.tt.tt[mode][mark]._hb is None for mode in self.tt.tt for mark in self.tt.tt[mode]]):
            if True:
                self.tt.set_hb(dist=self._tt_hb_params['dist'], 
                                        bound=self._tt_hb_params['bound'], mode_bound=self._tt_hb_params['mode_bound'], set_fit=self._tt_hb_params['set_fit'], 
                                        a_q=self._vrs_params['a_q'], 
                                        set_eval=self._tt_hb_sp_params['set_eval'], 
                                        CT_Q=self._vrs_params['CT_Q'], CT_E=self._vrs_params['CT_E'], S_Q=self._vrs_params['S_Q'], S_E=self._vrs_params['S_E'], 
                                        save_t=self._vrs_params['save_t'], 
                                        append_fit_tt_hb_w=self._tt_hb_sp_params['append_fit_tt_hb_w'], 
                                        append_vrs_tt_hb_w=self._tt_hb_sp_params['append_vrs_tt_hb_w'])

            else:
                self.tt.set_vrs(a_q=self._vrs_params['a_q'], 
                                        set_eval=self._tt_hb_sp_params['set_eval'], 
                                        CT_Q=self._vrs_params['CT_Q'], CT_E=self._vrs_params['CT_E'], S_Q=self._vrs_params['S_Q'], S_E=self._vrs_params['S_E'], 
                                        save_t=self._vrs_params['save_t'], 
                                        append_w=self._tt_hb_sp_params['append_vrs_tt_hb_w'])


        if self._tt_hb_sp_params['return_self']:
            return self.tt
        else:
            return self.tt._rw['vares']['w']


    

    def set_tt_hb_params(self, dist=None, bound=None, mode_bound=None, set_fit=None):

        self._tt_hb_params = {
                                'dist': dist,
                                'bound': bound,
                                'mode_bound': mode_bound,
                                'set_fit': set_fit,
                            }

    def set_vrs_params(self, a_q=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):

        self._vrs_params =  {
                                'a_q': a_q,
                                'CT_Q': CT_Q,
                                'CT_E': CT_E,
                                'S_Q': S_Q,
                                'S_E': S_E,
                                'save_t': save_t,
                            }
            
         

    









class price_data():
    """Class for price data
    """
    def __init__(self, tick, set_cal_sch=None, csv=None, csv_path=None, sch_index=None, all_start=None, all_end=None, train_start=None, train_end=None, forecast=None, T_all=None, T_train=None):
        """Constructor price_data class
        """
        self.set_tick(tick=tick, set_cal_sch=set_cal_sch)
        self.set_csv(csv=csv, csv_path=csv_path, sch_index=sch_index)

        self.set_dates(all_start=all_start, all_end=all_end, train_start=train_start, train_end=train_end, forecast=forecast, T_all=T_all, T_train=T_train)
        self.set_lr()
        
    
    def set_tick(self, tick, set_cal_sch=None):
        self._tick = tick
        self.set_exchange(set_cal_sch=set_cal_sch)

    def set_exchange(self, set_cal_sch=None):

        invert = misc_fns.invert_dict_lists(_exchange_tick)

        if np.isin(self._tick, np.array([key for key in invert])):
            self._exchange = invert[self._tick]
        else:
            self._exchange = None

        if set_cal_sch is None:
            set_cal_sch = True
        if set_cal_sch:
            self._cal, self._sch = fin_data_handl.get_calender_schedule(self._exchange, start_date="1927-12-30", end_date="2029-12-31")

    def set_csv(self, csv=None, csv_path=None, sch_index=None):
        if csv_path is not None:
            self._csv = pd.read_csv(csv_path, sep=',', index_col='Date')
        else:
            self._csv = csv
        self._csv.index = np.array([pd.Timestamp(self._csv.index[k]) for k in range(self._csv.index.size)])

        self.set_clean_index(sch_index=sch_index)



    def set_dates(self, all_start=None, all_end=None, train_start=None, train_end=None, forecast=None, T_all=None, T_train=None):

        self._dates = {period: None for period in ['all', 'train']}



        if all_start is None:
            all_start = self._clean_index[0]
        if all_end is None:
            if T_all is not None:
                T_0_all =  np.where(self._clean_index>=all_start)[0][0]
                all_end = self._clean_index[T_0_all+T_all]
            else:
                all_end=self._clean_index[-1] + pd.Timedelta(days=1)
                
            
        self._dates['all'] = np.array([all_start, all_end])

        if train_start is None:
            train_start = all_start
        if train_end is None:
            if T_train is not None:
                T_0_train = np.where(self._clean_index>=train_start)[0][0]
                train_end = self._clean_index[T_0_train+T_train]
            else:
                train_end=all_end
        self._dates['train'] = np.array([train_start, train_end])

        self.set_meta_series()
        self.set_period(forecast=forecast)

    def get_dates_str(self, dates_format=None):

        if dates_format is None:
            dates_format = lr_fns.get_dir_dates_format('all')
        
        if dates_format=='t0_t1':
            dates_str = f"{self._dates['train'][0].strftime('%Y-%m-%d')}_{self._dates['train'][-1].strftime('%Y-%m-%d')}"
        elif dates_format=='_a1':
            dates_str = f"_{self._dates['all'][-1].strftime('%Y-%m-%d')}"
        elif dates_format=='t0_t1/_a1':
            dates_str = f"{self._dates['train'][0].strftime('%Y-%m-%d')}_{self._dates['train'][-1].strftime('%Y-%m-%d')}/_{self._dates['all'][-1].strftime('%Y-%m-%d')}"
        else:
            dates_str = f"{self._dates['train'][0].strftime('%Y-%m-%d')}_{self._dates['train'][-1].strftime('%Y-%m-%d')}_{self._dates['all'][-1].strftime('%Y-%m-%d')}"

        return dates_str

                        
    def set_period(self, forecast=None):     

        self._periods = copy.deepcopy(self._dates)
        f1 = np.array([self._dates['train'][1], self._dates['all'][1]])
        if f1[0]==f1[1]:
            self._periods['forecast'] = []
        else:
            self._periods['forecast'] = [f1]
        if forecast is not None:
            for i in len(forecast):
                self._periods['forecast'].append(forecast[i])


    def set_meta_series(self):
        self._meta_series = {
                                'tick': self._tick,
                                'exchange': self._exchange,
                                'train_start': self._dates['train'][0],
                                'train_end': self._dates['train'][-1],
                                'all_end': self._dates['all'][-1]

                            }

    def set_clean_index(self, sch_index=None):

        log_p = np.log(self._csv['Close'])


        log_r = log_p.diff()[1:]


        if sch_index is None:
            if self._sch is not None:
                sch_index = self._sch.index
        if sch_index is not None:
            self._clean_index = log_r.iloc[np.isin(log_r.index, sch_index)].dropna().index
        else:
            self._clean_index = log_r.dropna()._clean_index



    def set_lr(self):

        self.lr = dict()

        log_p = np.log(self._csv['Close'])


        self.lr['csv'] = log_p.diff()[1:]

        if False:
            if sch_index is None:
                if self._sch is not None:
                    sch_index = self._sch.index
            if sch_index is not None:
                self.lr['c'] = self.lr['csv'].iloc[np.isin(self.lr['csv'].index, sch_index)].dropna()
            else:
                self.lr['c'] = self.lr['csv'].dropna()
        else:

            self.lr['c'] = self.lr['csv'].loc[self._clean_index]



        pt_correct = np.array([0,1])
        if self._periods is not None:
            
            #self._periods_t = {p: np.where(np.logical_and(self._clean_index >= self._periods[p][0], self._clean_index < self._periods[p][1]))[0][[0,-1]]+pt_correct for p in ['all', 'train']}
            
            self._periods_t = {p: lr_fns.calc_index_t(index=self._clean_index, bound_vals=self._periods[p], pt_correct=pt_correct) for p in ['all', 'train']}
            self._periods_start_date = {p: self.lr['c'].index[self._periods_t[p][0]] for p in ['all', 'train']}
            self._periods_start_lp = {p: log_p[self._periods_start_date[p]] for p in ['all', 'train']}


            self.lr['p'] = {p: self.lr['c'].iloc[np.logical_and(self._clean_index >= self._periods[p][0], self._clean_index < self._periods[p][1])] for p in ['all', 'train']}



            #self.lr['p'] = {p: self.lr['c'].iloc[self._periods_t[p][0]:self._periods_t[p][1]] for p in ['all', 'train']}


            if len(self._periods['forecast']) > 0:
                
                #self._periods_t['forecast'] = [np.where(np.logical_and(self._clean_index >= self._periods['forecast'][k][0], 
                #                                    self._clean_index < self._periods['forecast'][k][1]))[0][[0,-1]]+pt_correct for k in range(len(self._periods['forecast']))]

                self._periods_t['forecast'] = [lr_fns.calc_index_t(index=self._clean_index, bound_vals=self._periods['forecast'][k], pt_correct=pt_correct) for k in range(len(self._periods['forecast']))]
                self._periods_start_date["forecast"] = [self._clean_index[self._periods_t["forecast"][k][0]] for k in range(len(self._periods['forecast']))]
                self._periods_start_lp["forecast"] = [log_p[self._periods_start_date["forecast"][k]] for k in range(len(self._periods['forecast']))]


                self.lr['p']['forecast'] = [self.lr['c'].iloc[np.logical_and(self._clean_index >= self._periods['forecast'][k][0],
                                                            self._clean_index < self._periods['forecast'][k][1])] for k in range(len(self._periods['forecast']))],
                
                #self.lr['p']['forecast'] = [self.lr['c'].iloc[self._periods_t['forecast'][k][0]:self._periods_t['forecast'][k][1]] for k in range(len(self._periods['forecast']))]



    def calc_lp_p(self, period=None, m1=None, k=None):

        if m1 is None:
            m1 = False
        if period is None:
            period = False
        
        if period=="forecast":
            if k is None:
                k=0
            lr = np.cumsum(self.lr['p']['forecast'][k])
            p_0 = self._periods_start_lp['forecast'][k]
        else:
            lr = self.lr['p'][period]
            p_0 = self._periods_start_lp[period]

        lp = np.repeat(p_0-lr[0], lr.size+1)

        lp[1:] += np.cumsum(lr)

        if m1:
            return lp
        else:
            return lp[1:]




    def get_periods_t_offset(self, T=None, forecast_as_list=None):

        if T is None:
            T = self._periods_t['all'][0]
        if forecast_as_list is None:
            forecast_as_list = True

        periods_t_offset = {p: self._periods_t[p]-T for p in ['all', 'train']}
        if len(self._periods['forecast']) > 0:
            if forecast_as_list:
                periods_t_offset['forecast'] = [self._periods_t['forecast'][k]-T for k in range(len(self._periods['forecast']))]
            else:
                periods_t_offset['forecast'] = self._periods_t['forecast'][0]-T
            
        return periods_t_offset
    
    
    def calc_table_periods_head(self, tick=None, sample_dates=None, periods=None, sample_labels=None, lead=None, pre=None, post=None, c0=None, cw=None, calign=None, q_same_row=None, q_space=None, q_above_empty_lead=None, last_head=None, N_p_max=None, p_same_rows=None, p_space=None, repeat=None, r_space=None, incl_hline=None, incl_cmidrules=None, return_heads=None):
        
        if tick is None:
            tick = self._tick

        if sample_dates is None:
            sample_dates = {
                    'train': [self._dates['train'][0].strftime('%Y-%m-%d'), self._dates['train'][-1].strftime('%Y-%m-%d')],
                    'forecast': [self._dates['train'][-1].strftime('%Y-%m-%d'), self._dates['all'][-1].strftime('%Y-%m-%d')],
                    'all': [self._dates['train'][0].strftime('%Y-%m-%d'), self._dates['all'][-1].strftime('%Y-%m-%d')],
                    }

        return lr_fns.calc_table_periods_head(self=self, tick=tick, sample_dates=sample_dates, periods=periods, sample_labels=sample_labels, lead=lead, pre=pre, post=post, c0=c0, cw=cw, calign=calign, q_same_row=q_same_row, q_space=q_space, q_above_empty_lead=q_above_empty_lead, last_head=last_head, N_p_max=N_p_max, p_same_rows=p_same_rows, p_space=p_space, repeat=repeat, r_space=r_space, incl_hline=incl_hline, incl_cmidrules=incl_cmidrules, return_heads=return_heads)



    
    







#============
#============
#
# Variables
#
#============
#============

_exchange_tick = {
                    'NASDAQ': ["SP", "SPX", "DJI"],
                    'LSE': ["FTSE", "FTM", "UKX"],
                    'EUREX': ["DAX", "CAC"],
                    'TSX': ["NKX"],
                    'HKEX': ["HSI"],
                    'SSE': ["SHC"],
                    'BSE': ["SNX"],
                }