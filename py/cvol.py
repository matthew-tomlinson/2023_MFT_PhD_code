import numpy as np
import scipy
from scipy.optimize import minimize
import arch
import pandas as pd
import numdifftools
import os
import copy
# Local files
import hawkes as hwks
import bulk_gp

import misc_functions as misc_fns

class cvol_mod():
    """Class for two-tailed condtional volatility model group
    """
    def __init__(self, y, vol=None, p=None, o=None, q=None, dist=None, a_u=None, last_obs=None, set_am_stats=None,
                        a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, meta_series=None, 
                        append_fit_cv_w=None, append_vrs_cv_w=None):
        """Constructor cvol_mod class
        """
        self.reset_rw(meta_series=meta_series)
        self.set_cv(y=y, vol=vol, p=p, o=o, q=q, dist=dist, a_u=a_u, last_obs=last_obs, set_am_stats=set_am_stats,
                        a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

        if append_fit_cv_w is None:
            append_fit_cv_w = True
        if set_am_stats and append_fit_cv_w:
            self.append_fit_cv_w()
        if append_vrs_cv_w is None:
            append_vrs_cv_w = True
        if set_am_stats and set_eval and append_vrs_cv_w:
            self.append_vrs_cv_w()




    def reset_rw(self, meta_series=None):
        self._rw = dict()
        self._rw['meta'] = dict()
        self.set_meta_series(meta_series=meta_series)

        for cat in ['fit', 'vares']:
            self._rw[cat] = {rw: dict() for rw in ['r', 'w']}

        fit_mod_fn ={
                        'cv': ['est', 'se'],
                        'cv_evt': ['est'],
                    }

        for mod in ['cv', 'cv_evt']:
            self._rw['fit']['w'][mod] = {
                                                fn: pd.DataFrame() for fn in fit_mod_fn[mod]
                                        }
            self._rw['vares']['w'][mod] = {
                                                fn: pd.DataFrame() for fn in ['vrs']
                                            }

    def set_meta_series(self, meta_series=None):
        if meta_series is not None:
            self._rw['meta']['series'] = meta_series

    def read_fit_cv(self, fit_dir):

        suf_list = ['est', 'se']
        self._rw['fit']['r']['cv'] = {suf: pd.DataFrame() for suf in suf_list}
    
        for suf in suf_list:
            f_dir = f"{fit_dir}fit_cv_{suf}.csv"
            if os.path.isfile(f_dir):
                self._rw['fit']['r']['cv'][suf] = pd.read_csv(f_dir, sep=',')
    
    def read_vares_cv(self, vrs_dir):

        suf_list = ['vrs']
        self._rw['vares']['r']['cv'] = {suf: pd.DataFrame() for suf in suf_list}
        
        for suf in suf_list:
            f_dir = f"{vrs_dir}vares_cv_{suf}.csv"
            if os.path.isfile(f_dir):
                self._rw['vares']['r']['cv'][suf] = pd.read_csv(f_dir, sep=',')



    def set_cv(self, y, vol=None, p=None, o=None, q=None, dist=None, a_u=None, last_obs=None, set_am_stats=None,
                        a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):

        self.cv = cvol_vol(y=y, vol=vol, p=p, o=o, q=q, dist=dist, a_u=a_u, last_obs=last_obs, set_am_stats=set_am_stats,
                        a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)



    def set_am_stats(self, last_obs=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, append_fit_cv_w=None, append_vrs_cv_w=None):

        self.cv.set_am_stats(last_obs=last_obs, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)
        
        if append_fit_cv_w is None:
            append_fit_cv_w = True
        if append_vrs_cv_w is None:
            append_vrs_cv_w = False

        if append_fit_cv_w:
            self.append_fit_cv_w()
        if set_eval and append_vrs_cv_w:
            self.append_vrs_cv_w()


    def set_bdist_evt(self, evt_a_u=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, append_fit_cv_evt_w=None, append_vrs_cv_evt_w=None):

        self.cv.set_bdist_evt(evt_a_u=evt_a_u, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

        if append_fit_cv_evt_w is None:
            append_fit_cv_evt_w = True
        if append_vrs_cv_evt_w is None:
            append_vrs_cv_evt_w = False

        if append_fit_cv_evt_w:
            self.append_fit_cv_evt_w()
        if set_eval and append_vrs_cv_evt_w:
            self.append_vrs_cv_evt_w()




    def set_vrs(self, a_q, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, append_vrs_cv_w=None):
   
        self.cv.set_vrs(a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

        if append_vrs_cv_w is None:
            append_vrs_cv_w = False
        if set_eval and append_vrs_cv_w:
            self.append_vrs_cv_w()


    def set_vrs_eval(self, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, append_vrs_cv_w=None):
 
        self.cv.set_vrs_eval(CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

        if append_vrs_cv_w is None:
            append_vrs_cv_w = True
        if append_vrs_cv_w:
            self.append_vrs_cv_w()





    def append_fit_cv_w(self):
        for fn in ['est', 'se']:
            self._rw['fit']['w']['cv'][fn] = self._rw['fit']['w']['cv'][fn].append(self.calc_fit_cv_w(lead=self._rw['meta']['series'], page=fn))

    def calc_fit_cv_w(self, lead=None, page=None):
        return self.cv.calc_fit_cv_w(lead=lead, page=page)

    def append_vrs_cv_w(self):
        for fn in ['vrs']:
            self._rw['vares']['w']['cv'][fn] = self._rw['vares']['w']['cv'][fn].append(self.calc_vrs_eval(lead=self._rw['meta']['series']))

    def calc_vrs_eval(self, lead=None, atf=None):
        return self.cv.calc_vrs_eval(lead=lead, atf=atf)

    def append_fit_cv_evt_w(self):
        for fn in ['est']:
            self._rw['fit']['w']['cv_evt'][fn] = self._rw['fit']['w']['cv_evt'][fn].append(self.calc_fit_cv_evt_w(lead=self._rw['meta']['series'], page=fn))

    def calc_fit_cv_evt_w(self, lead=None, page=None):
        return self.cv.calc_fit_cv_evt_w(lead=lead, page=page)

    def append_vrs_cv_evt_w(self):
        for fn in ['vrs']:
            self._rw['vares']['w']['cv_evt'][fn] = self._rw['vares']['w']['cv_evt'][fn].append(self.calc_evt_vrs_eval(lead=self._rw['meta']['series']))

    def calc_evt_vrs_eval(self, lead=None, atf=None):
        return self.cv.calc_evt_vrs_eval(lead=lead, atf=atf)






    def set_tt_mp(self, a_u=None, dt_dt_fine=None,  opt_bounds=None, set_fit=None, fix_a_lambda=None, opt_method=None, factor=None, print_mp_time=None):

        fn = []
        for v in self.cv.cv_v:
            for p in self.cv.cv_v[v].cv_p:
                for o in self.cv.cv_v[v].cv_p[p]:
                    for q in self.cv.cv_v[v].cv_p[p][o]:
                        for d in self.cv.cv_v[v].cv_p[p][o][q].cv_d:
                            meta_series =   {
                                                **self._rw['meta']['series'],
                                                **self.cv.cv_v[v].cv_p[p][o][q].cv_d[d]._stats_res['cv'],
                                            }
                            self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].set_tt_sp_params(a_u=a_u, dt_dt_fine=dt_dt_fine, meta_series=meta_series, opt_bounds=opt_bounds, set_fit=set_fit, fix_a_lambda=fix_a_lambda, opt_method=opt_method)
                            fn.append(self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].set_tt_sp)


        res_list, res_time = misc_fns.mp_funs_factor(funs=fn,
                                    factor=factor, print_mp_time=print_mp_time)

        i=0
        for v in self.cv.cv_v:
            for p in self.cv.cv_v[v].cv_p:
                for o in self.cv.cv_v[v].cv_p[p]:
                    for q in self.cv.cv_v[v].cv_p[p][o]:
                        for d in self.cv.cv_v[v].cv_p[p][o][q].cv_d:
                            self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].tt = res_list[i]
                            i += 1
                    
        return res_time

    
    def set_tt_a_u(self, a_u=None, u=None, fix_a_lambda=None):

        for v in self.cv.cv_v:
            for p in self.cv.cv_v[v].cv_p:
                for o in self.cv.cv_v[v].cv_p[p]:
                    for q in self.cv.cv_v[v].cv_p[p][o]:
                        for d in self.cv.cv_v[v].cv_p[p][o][q].cv_d:
                            #if u is None:
                            #    self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].tt.load_a_u(a_u=a_u, tt_read_params=False, tt_hb_read_params=False)
                            #else:
                            self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].tt._a_u = a_u
                            for mode in self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].tt.tt:
                                for mark in self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].tt.tt[mode]:
                                    if fix_a_lambda is not None:
                                        self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].tt.tt[mode][mark]._parameters.set_fix_a_lambda(fix_a_lambda)

                                    self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].tt.tt[mode][mark]._parameters.a_lambda.set_value(a_u)
                                    if u is not None:
                                        self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].tt.tt[mode][mark]._parameters.u.set_value(u)
                                    else:
                                        self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].tt.tt[mode][mark].set_u()
                                    self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].tt.tt[mode][mark].detect_events()

                            

    def fit_tt_mp(self, factor=None, print_mp_time=None):

        fn = []
        for v in self.cv.cv_v:
            for p in self.cv.cv_v[v].cv_p:
                for o in self.cv.cv_v[v].cv_p[p]:
                    for q in self.cv.cv_v[v].cv_p[p][o]:
                        for d in self.cv.cv_v[v].cv_p[p][o][q].cv_d:
                            fn.append(self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].fit_tt_sp)

        res_list, res_time = misc_fns.mp_funs_factor(funs=fn,
                                    factor=factor, print_mp_time=print_mp_time)

        i=0
        for v in self.cv.cv_v:
            for p in self.cv.cv_v[v].cv_p:
                for o in self.cv.cv_v[v].cv_p[p]:
                    for q in self.cv.cv_v[v].cv_p[p][o]:
                        for d in self.cv.cv_v[v].cv_p[p][o][q].cv_d:
                            self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].tt = res_list[i]
                            i += 1
                    
        return res_time

    def set_tt_bi_lr_mp(self, fix_a_lambda=None, opt_method=None, fit_bi_lr=None, fp0_bi=None, factor=None, print_mp_time=None):

        fn = []
        for v in self.cv.cv_v:
            for p in self.cv.cv_v[v].cv_p:
                for o in self.cv.cv_v[v].cv_p[p]:
                    for q in self.cv.cv_v[v].cv_p[p][o]:
                        for d in self.cv.cv_v[v].cv_p[p][o][q].cv_d:
                            self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].set_tt_sp_params(fix_a_lambda=fix_a_lambda, opt_method=opt_method, fit_bi_lr=fit_bi_lr, fp0_bi=fp0_bi)
                            fn.append(self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].set_tt_bi_lr_sp)

        res_list, res_time = misc_fns.mp_funs_factor(funs=fn,
                                    factor=factor, print_mp_time=print_mp_time)

        i=0
        for v in self.cv.cv_v:
            for p in self.cv.cv_v[v].cv_p:
                for o in self.cv.cv_v[v].cv_p[p]:
                    for q in self.cv.cv_v[v].cv_p[p][o]:
                        for d in self.cv.cv_v[v].cv_p[p][o][q].cv_d:
                            self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].tt = res_list[i]
                            i += 1
                    
        return res_time


    def set_sim_mp(self, metaseed=None, set_sim_data=None, set_sim_tt=None, N_sim=None, i=None, a_u=None, dt_dt_fine=None, opt_bounds=None, set_fit=None, fix_a_lambda=None, opt_method=None, sim_number=None, factor=None, print_mp_time=None):

        fn = []
        for v in self.cv.cv_v:
            for p in self.cv.cv_v[v].cv_p:
                for o in self.cv.cv_v[v].cv_p[p]:
                    for q in self.cv.cv_v[v].cv_p[p][o]:
                        for d in self.cv.cv_v[v].cv_p[p][o][q].cv_d:
                            meta_series =   {
                                                **self._rw['meta']['series'],
                                                **self.cv.cv_v[v].cv_p[p][o][q].cv_d[d]._stats_res['cv'],
                                            }
                            self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].set_sim_tt_sp_params(metaseed=metaseed, set_sim_data=set_sim_data, set_sim_tt=set_sim_tt, N_sim=N_sim, i=i, a_u=a_u, dt_dt_fine=dt_dt_fine, meta_series=meta_series, opt_bounds=opt_bounds, set_fit=set_fit, fix_a_lambda=fix_a_lambda, opt_method=opt_method, sim_number=sim_number)
                            fn.append(self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].set_sim_sp)

        res_list, res_time = misc_fns.mp_funs_factor(funs=fn,
                                    factor=factor, print_mp_time=print_mp_time)

        i=0
        for v in self.cv.cv_v:
            for p in self.cv.cv_v[v].cv_p:
                for o in self.cv.cv_v[v].cv_p[p]:
                    for q in self.cv.cv_v[v].cv_p[p][o]:
                        for d in self.cv.cv_v[v].cv_p[p][o][q].cv_d:
                            self.cv.cv_v[v].cv_p[p][o][q].cv_d[d]._sim = res_list[i]
                            i += 1
                    
        return res_time


    def set_sim_data_tt_fit_mp(self, sim_number=None, outdir=None, factor=None, print_mp_time=None):

        if outdir is not None:
            if os.path.exists(outdir)==False:
                os.mkdir(outdir)

        fn = []
        for v in self.cv.cv_v:
            for p in self.cv.cv_v[v].cv_p:
                for o in self.cv.cv_v[v].cv_p[p]:
                    for q in self.cv.cv_v[v].cv_p[p][o]:
                        for d in self.cv.cv_v[v].cv_p[p][o][q].cv_d:
                            if sim_number is not None:
                                self.cv.cv_v[v].cv_p[p][o][q].cv_d[d]._sim_tt_sp_params['sim_number'] = sim_number
                            fn.append(self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].set_sim_data_tt_fit_sp)

        res_list, res_time = misc_fns.mp_funs_factor(funs=fn,
                                    factor=factor, print_mp_time=print_mp_time)

        i=0
        for v in self.cv.cv_v:
            for p in self.cv.cv_v[v].cv_p:
                for o in self.cv.cv_v[v].cv_p[p]:
                    for q in self.cv.cv_v[v].cv_p[p][o]:
                        for d in self.cv.cv_v[v].cv_p[p][o][q].cv_d:
                            self.cv.cv_v[v].cv_p[p][o][q].cv_d[d]._sim = res_list[i]
                            if outdir is not None:
                                cv_tt = f"{v}-{p}{o}{q}-{d}"
                                for fn in self.cv.cv_v[v].cv_p[p][o][q].cv_d[d]._sim.tt._rw['fit']['w']['tt']:
                                    self.cv.cv_v[v].cv_p[p][o][q].cv_d[d]._sim.tt._rw['fit']['w']['tt'][fn].to_csv(path_or_buf=f"{outdir}fit_cv_sim_r_tt_{cv_tt}_{fn}.csv", sep=',', index=False)
                            i += 1
                    
        return res_time


    def set_sim_data_tt_fit_cmu_mp_old(self, sim_number=None, u=None, outdir=None, factor=None, print_mp_time=None):

        tt_list = ['tt_c', 'tt_u']
        if outdir is not None:
            if os.path.exists(outdir)==False:
                os.mkdir(outdir)
            for tt in tt_list:
                ttdir = f'{outdir}{tt}/'
                if os.path.exists(ttdir)==False:
                    os.mkdir(ttdir)

        fn = []
        for v in self.cv.cv_v:
            for p in self.cv.cv_v[v].cv_p:
                for o in self.cv.cv_v[v].cv_p[p]:
                    for q in self.cv.cv_v[v].cv_p[p][o]:
                        for d in self.cv.cv_v[v].cv_p[p][o][q].cv_d:
                            self.cv.cv_v[v].cv_p[p][o][q].cv_d[d]._sim_tt_sp_params['sim_number'] = sim_number
                            fn.append(self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].set_sim_data_tt_fit_cmu_sp)

        res_list, res_time = misc_fns.mp_funs_factor(funs=fn,
                                    factor=factor, print_mp_time=print_mp_time)

        i=0
        for v in self.cv.cv_v:
            for p in self.cv.cv_v[v].cv_p:
                for o in self.cv.cv_v[v].cv_p[p]:
                    for q in self.cv.cv_v[v].cv_p[p][o]:
                        for d in self.cv.cv_v[v].cv_p[p][o][q].cv_d:
                            self.cv.cv_v[v].cv_p[p][o][q].cv_d[d]._sim = res_list[i]
                            if outdir is not None:
                                cv_tt = f"{v}-{p}{o}{q}-{d}"
                                for tt in ['tt_c', 'tt_u']:
                                    ttdir = f'{outdir}{tt}/'
                                    for fn in self.cv.cv_v[v].cv_p[p][o][q].cv_d[d]._sim.tt._rw['fit']['w'][tt]:
                                        self.cv.cv_v[v].cv_p[p][o][q].cv_d[d]._sim.tt._rw['fit']['w'][tt][fn].to_csv(path_or_buf=f"{ttdir}fit_cv_sim_r_{tt}_{cv_tt}_{fn}.csv", sep=',', index=False)
                            i += 1
                    
        return res_time
    


    def set_sim_data_tt_fit_cmu_mp(self, sim_number=None, u=None, outdir=None, factor=None, print_mp_time=None):

        if outdir is not None:
            if os.path.exists(outdir)==False:
                os.mkdir(outdir)

        fn = []
        for v in self.cv.cv_v:
            for p in self.cv.cv_v[v].cv_p:
                for o in self.cv.cv_v[v].cv_p[p]:
                    for q in self.cv.cv_v[v].cv_p[p][o]:
                        for d in self.cv.cv_v[v].cv_p[p][o][q].cv_d:
                            self.cv.cv_v[v].cv_p[p][o][q].cv_d[d]._sim_tt_sp_params['sim_number'] = sim_number
                            fn.append(self.cv.cv_v[v].cv_p[p][o][q].cv_d[d].set_sim_data_tt_fit_cmu_sp)

        res_list, res_time = misc_fns.mp_funs_factor(funs=fn,
                                    factor=factor, print_mp_time=print_mp_time)

        i=0
        for v in self.cv.cv_v:
            for p in self.cv.cv_v[v].cv_p:
                for o in self.cv.cv_v[v].cv_p[p]:
                    for q in self.cv.cv_v[v].cv_p[p][o]:
                        for d in self.cv.cv_v[v].cv_p[p][o][q].cv_d:
                            self.cv.cv_v[v].cv_p[p][o][q].cv_d[d]._sim = res_list[i]
                            if outdir is not None:
                                cv_tt = f"{v}-{p}{o}{q}-{d}"
                                for fn in self.cv.cv_v[v].cv_p[p][o][q].cv_d[d]._sim.tt._rw['fit']['w']['tt']:
                                    self.cv.cv_v[v].cv_p[p][o][q].cv_d[d]._sim.tt._rw['fit']['w']['tt'][fn].to_csv(path_or_buf=f"{outdir}fit_cv_sim_r_tt_{cv_tt}_{fn}.csv", sep=',', index=False)
                            i += 1
                    
        return res_time

    

    

    





class cvol_vol():
    def __init__(self, y, vol=None, p=None, o=None, q=None, dist=None, a_u=None, last_obs=None, set_am_stats=None,
                        a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        """Constructor for cvol_vol class
        """
        self.set_cv_v(y=y, vol=vol, p=p, o=o, q=q, dist=dist, a_u=a_u, last_obs=last_obs, set_am_stats=set_am_stats,
                    a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t, reset=True)

    def set_cv_v(self, y, vol=None, p=None, o=None, q=None, dist=None, a_u=None, last_obs=None, set_am_stats=None,
                    a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, reset=False):
        if vol is not None:
            self._vol = misc_fns.make_iterable_array(vol)
        elif reset:
            self._vol = misc_fns.make_iterable_array("GARCH")
   
        self.cv_v = calc_cv_v(y=y, vol=self._vol, p=p, o=o, q=q, dist=dist, a_u=a_u, last_obs=last_obs, set_am_stats=set_am_stats,
                                a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

    def set_am_stats(self, last_obs=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        for vol in self.cv_v:
            self.cv_v[vol].set_am_stats(last_obs=last_obs, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

    def set_bdist_evt(self, evt_a_u=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        for vol in self.cv_v:
            self.cv_v[vol].set_bdist_evt(evt_a_u=evt_a_u, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

    def set_vrs(self, a_q, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        for vol in self.cv_v:
            self.cv_v[vol].set_vrs(a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

    def set_vrs_eval(self, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, t_start=None, t_end=None, period=None):
        for vol in self.cv_v:
            self.cv_v[vol].set_vrs_eval(CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)


    def calc_fit_cv_w(self, lead=None, page=None):
        df = pd.DataFrame()
        for vol in self.cv_v:
            df = df.append(self.cv_v[vol].calc_fit_cv_w(lead=lead, page=page))
        return df

    def calc_vrs_eval(self, lead=None, atf=None):
        df = pd.DataFrame()
        for vol in self.cv_v:
            df = df.append(self.cv_v[vol].calc_vrs_eval(lead=lead, atf=atf))
        return df

    def calc_fit_cv_evt_w(self, lead=None, page=None):
        df = pd.DataFrame()
        for vol in self.cv_v:
            df = df.append(self.cv_v[vol].calc_fit_cv_evt_w(lead=lead, page=page))
        return df

    def calc_evt_vrs_eval(self, lead=None, atf=None):
        df = pd.DataFrame()
        for vol in self.cv_v:
            df = df.append(self.cv_v[vol].calc_evt_vrs_eval(lead=lead, atf=atf))
        return df



class cvol_poq():
    def __init__(self, y, vol=None, p=None, o=None, q=None, dist=None, a_u=None, last_obs=None, set_am_stats=None,
                        a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        """Constructor for cvol_poq class
        """
        self.set_cv_p(y=y, vol=vol, p=p, o=o, q=q, dist=dist, a_u=a_u, last_obs=last_obs, set_am_stats=set_am_stats,
                    a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t, reset=True)


    def set_cv_p(self, y, vol=None, p=None, o=None, q=None, dist=None, a_u=None, last_obs=None, set_am_stats=None,
                    a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, reset=False):
        if p is not None:
            self._p = misc_fns.make_iterable_array(p, dtype=int)
        elif reset:
            self._p = misc_fns.make_iterable_array(1, dtype=int)
        if o is not None:
            self._o = misc_fns.make_iterable_array(o, dtype=int)
        elif reset:
            self._o = misc_fns.make_iterable_array(0, dtype=int)
        if q is not None:
            self._q = misc_fns.make_iterable_array(q, dtype=int)
        elif reset:
            self._q = misc_fns.make_iterable_array(1, dtype=int)

        self.cv_p = calc_cv_p(y=y, p=self._p, o=self._o, q=self._q, vol=vol, dist=dist, a_u=a_u, last_obs=last_obs, set_am_stats=set_am_stats,
                                a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

    def set_vrs(self, a_q, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        for p in self.cv_p:
            for o in self.cv_p[p]:
                for q in self.cv_p[p][o]:
                    self.cv_p[p][o][q].set_vrs(a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

    def set_am_stats(self, last_obs=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        for p in self.cv_p:
            for o in self.cv_p[p]:
                for q in self.cv_p[p][o]:
                   self.cv_p[p][o][q].set_am_stats(last_obs=last_obs, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)


    def set_bdist_evt(self, evt_a_u=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        for p in self.cv_p:
            for o in self.cv_p[p]:
                for q in self.cv_p[p][o]:
                    self.cv_p[p][o][q].set_bdist_evt(evt_a_u=evt_a_u, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)


    def set_vrs_eval(self, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        for p in self.cv_p:
            for o in self.cv_p[p]:
                for q in self.cv_p[p][o]:
                    self.cv_p[p][o][q].set_vrs_eval(CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)


    def calc_fit_cv_w(self, lead=None, page=None):
        df = pd.DataFrame()
        for p in self.cv_p:
            for o in self.cv_p[p]:
                for q in self.cv_p[p][o]:
                    df = df.append(self.cv_p[p][o][q].calc_fit_cv_w(lead=lead, page=page))
        return df

    def calc_vrs_eval(self, lead=None, atf=None):
        df = pd.DataFrame()
        for p in self.cv_p:
            for o in self.cv_p[p]:
                for q in self.cv_p[p][o]:
                    df = df.append(self.cv_p[p][o][q].calc_vrs_eval(lead=lead, atf=atf))
        return df

    def calc_fit_cv_evt_w(self, lead=None, page=None):
        df = pd.DataFrame()
        for p in self.cv_p:
            for o in self.cv_p[p]:
                for q in self.cv_p[p][o]:
                    df = df.append(self.cv_p[p][o][q].calc_fit_cv_evt_w(lead=lead, page=page))
        return df
        
    def calc_evt_vrs_eval(self, lead=None, atf=None):
        df = pd.DataFrame()
        for p in self.cv_p:
            for o in self.cv_p[p]:
                for q in self.cv_p[p][o]:
                    df = df.append(self.cv_p[p][o][q].calc_evt_vrs_eval(lead=lead, atf=atf))
        return df

    

class cvol_dist():
    def __init__(self, y, vol=None, p=None, o=None, q=None, dist=None, a_u=None, last_obs=None, set_am_stats=None,
                        a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        """Constructor for cvol_dist class
        """
        self.set_cv_d(y=y, vol=vol, p=p, o=o, q=q, dist=dist, a_u=a_u, last_obs=last_obs, set_am_stats=set_am_stats,
                    a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t, reset=True)


    def set_cv_d(self, y, vol=None, p=None, o=None, q=None, dist=None, a_u=None, last_obs=None, set_am_stats=None,
                    a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, reset=False):
        if dist is not None:
            self._dist = misc_fns.make_iterable_array(dist)
        elif reset:
           self._dist = misc_fns.make_iterable_array("norm")

        self.cv_d = calc_cv_d(y=y, dist=self._dist, vol=vol, p=p, o=o, q=q, a_u=a_u, last_obs=last_obs, set_am_stats=set_am_stats,
                                a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

    def set_am_stats(self, last_obs=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        for dist in self.cv_d:
            self.cv_d[dist].set_am_stats(last_obs=last_obs, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

    def set_bdist_evt(self, evt_a_u=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        for dist in self.cv_d:
            self.cv_d[dist].set_bdist_evt(evt_a_u=evt_a_u, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)


    def set_vrs(self, a_q, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        for dist in self.cv_d:
            self.cv_d[dist].set_vrs(a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

    def set_vrs_eval(self, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        for dist in self.cv_d:
            self.cv_d[dist].set_vrs_eval(CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

    def calc_fit_cv_w(self, lead=None, page=None):
        df = pd.DataFrame()
        for dist in self.cv_d:
            df = df.append(self.cv_d[dist].calc_fit_cv_w(lead=lead, page=page), ignore_index=True)
        return df

    def calc_vrs_eval(self, lead=None, atf=None):
        df = pd.DataFrame()
        for dist in self.cv_d:
            df = df.append(self.cv_d[dist].calc_vrs_eval(lead=lead, atf=atf))
        return df

    def calc_fit_cv_evt_w(self, lead=None, page=None):
        df = pd.DataFrame()
        for dist in self.cv_d:
            df = df.append(self.cv_d[dist].calc_fit_cv_evt_w(lead=lead, page=page), ignore_index=True)
        return df

    def calc_evt_vrs_eval(self, lead=None, atf=None):
        df = pd.DataFrame()
        for dist in self.cv_d:
            df = df.append(self.cv_d[dist].calc_evt_vrs_eval(lead=lead, atf=atf))
        return df


        

class cvol():
    """Class for conditional volatility models
    """
    def __init__(self, y, vol=None, p=None, o=None, q=None, dist=None, a_u=None, last_obs=None, set_am_stats=None,
                        a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        """Constructor for cvol class
        """

        self.set_cvol_params(vol=vol, p=p, o=o, q=q, dist=dist, a_u=a_u, reset=True)
        self.set_evt_a_u(reset=True)
        self.set_am(y=y)
        self.set_last_obs(last_obs=last_obs)

        if set_am_stats is None:
            set_am_stats = True
        if set_am_stats:
            self.set_am_stats(a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)
    
        self._sim = None

        
    def set_cvol_params(self, vol=None, p=None, o=None, q=None, dist=None, a_u=None, reset=False):

        self.set_vol(vol=vol, reset=reset)
        self.set_p(p=p, reset=reset)
        self.set_o(o=o, reset=reset)
        self.set_q(q=q, reset=reset)
        self.set_dist(dist=dist, reset=reset)
        self.set_a_u(a_u=a_u, reset=reset)

    def calc_cvol_dict(self):
        return {p: self.__dict__[f"_{p}"] for p in ["vol", "p", "o", "q", "dist", "a_u"]}


    def set_vol(self, vol=None, reset=True):
        if vol is not None:
            self._vol = vol
        elif reset:
            self._vol = "GARCH"
    def set_p(self, p=None, reset=True):
        if p is not None:
            self._p = p
        elif reset:
            self._p = 1
    def set_o(self, o=None, reset=True):
        if o is not None:
            self._o = o
        elif reset:
            self._o = 0
    def set_q(self, q=None, reset=True):
        if q is not None:
            self._q = q
        elif reset:
            self._q = 1
    def set_dist(self, dist=None, reset=True):
        if dist is not None:
            self._dist = dist
        elif reset:
            self._dist = "norm"
        self._arch_dist = _arch_str[self._dist]
    def set_a_u(self, a_u=None, reset=True):
        if a_u is not None:
            self._a_u = a_u
        elif reset:
            self._a_u = 0
    def set_evt_a_u(self, evt_a_u=None, reset=True):
        if evt_a_u is not None:
            self._evt_a_u = evt_a_u
        elif reset:
            self._evt_a_u = 0


    def calc_cv_dict(self):

        return {name: self.__dict__[f'_{name}'] for name in ['vol', 'p', 'o', 'q', 'dist', 'a_u']}

    
    def set_am(self, y):
        self._am = arch.arch_model(y, vol=self._vol, p=self._p, o=self._o, q=self._q, dist=self._arch_dist, rescale=True)
        

    def set_last_obs(self, last_obs=None):
        self._last_obs = last_obs

    def set_am_stats(self, last_obs=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        self.set_am_res(last_obs=last_obs)
        self.set_stats_res()
        self.set_bdist()
        self.set_F_X()
        if a_q is not None:
            self.set_vrs(a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)


    def set_am_res(self, last_obs=None):

        if last_obs is not None:
            self.set_last_obs(last_obs=last_obs)
        if self._last_obs is not None:
            self._am_res = self._am.fit(disp='on', last_obs=self._last_obs)
        else:
            self._am_res = self._am.fit(disp='on')
        self._scale = self._am.scale
        self._train_t_end = np.sum(~np.isnan(self._am_res._volatility))

    def set_stats_res(self):
        self._stats_res = self.calc_stats_res()


    def calc_stats_res(self):

        p_l = [param for param in self._am_res.params.keys()]
        b_a = np.array([_bound_stats_res[param] for param in p_l])

        max_poq = 1
        params = np.concatenate([
                                    ['mu', 'omega'],
                                    [f'alpha_{1+i}' for i in range(max_poq)],
                                    [f'gamma_{1+i}' for i in range(max_poq)],
                                    [f'beta_{1+i}' for i in range(max_poq)],
                                    ['df', 'nc'],
                                ])

        stats_res = dict()
        stats_res['cv'] = self.calc_cv_dict()         
        stats_res['param'] = {es: {param: np.nan for param in params} for es in ['est', 'se']}
        stats_res['res'] = {"x0": self._am_res.params.values}
        stats_res['opt_time'] = {t: None for t in ['start', 'end', 'delta']}

        def mll(x):
            return self.cvol_mll(params=x, y=self._am._y[~np.isnan(self._am_res._volatility)], s2_0=self._am_res._volatility[0]**2, scale=1)

        opt_timer = misc_fns.timer(print_timer=False)
        stats_res['res']['res'] = scipy.optimize.minimize(mll, x0=stats_res['res']['x0'], 
                                            method='SLSQP', options={'disp': True},
                                            bounds=scipy.optimize.Bounds(lb=b_a[:,0], ub=b_a[:,1]))
        opt_timer.stop()
        stats_res['opt_time'] = opt_timer.get_timer_dict()


        stats_res['res']['log_L'] = -stats_res['res']['res'].fun


        stats_res['x'] = {'est': stats_res['res']['res'].x}
        stats_res['res']['success'] = True
        hess = numdifftools.Hessian(mll, full_output=True, method="central", step=stats_res['x']['est']/1E6)(stats_res['x']['est'])[0]
        try:
            np.linalg.inv(hess)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                stats_res['x']['se'] = np.full(stats_res['x']['est'].size, np.nan)
                stats_res['res']['success'] = False
        if stats_res['res']['success']:
            stats_res['x']['se'] = np.sqrt(np.abs(np.diag(np.linalg.inv(hess))))     
        
        for es in stats_res['param']:
            stats_res['param'][es]['mu'] =  stats_res['x'][es][0]
            stats_res['param'][es]['omega'] =  stats_res['x'][es][1]
            for i in range(self._p):
                stats_res['param'][es][f'alpha_{1+i}'] =  stats_res['x'][es][2+i]
            for i in range(self._o):
                stats_res['param'][es][f'gamma_{1+i}'] =  stats_res['x'][es][2+self._p+i]
            for i in range(self._q):
                stats_res['param'][es][f'beta_{1+i}'] =  stats_res['x'][es][2+self._p+self._o+i]
            if self._dist=="t" or self._dist=="nct":
                stats_res['param'][es]['df'] =  stats_res['x'][es][2+self._p+self._o+self._q]
            if self._dist=="nct":
                stats_res['param'][es]['nc'] =  stats_res['x'][es][3+self._p+self._o+self._q]

        return stats_res


    def set_bdist(self):

        self._bdist = self.cvol_dist_s(params=self._stats_res['x']['est'], y=self._am._y_original.values, 
                            s2_0=self._am_res._volatility[0]**2, scale=self._scale)
        self.set_tdist()

    def set_bdist_evt(self, evt_a_u=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):


        self.set_evt_a_u(evt_a_u=evt_a_u, reset=True)
        self._bdist_evt = bulk_gp.bulk_gp(data=self._am.y.values, train_t=np.array([0, self._train_t_end]),
                                            bulk_dist=self._dist,
                                            bulk_loc=self._bdist.kwds['loc'],
                                            bulk_scale=self._bdist.kwds['scale'],
                                            gp_weight=self._evt_a_u)

        self._bdist_evt.set_y_dist() 
        if a_q is not None:
            self.set_vrs_evt(a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t) 

    def set_tdist(self):
        if self._dist=="norm":
            self._tdist = scipy.stats.norm()
        elif self._dist=="t":
            self._tdist = scipy.stats.t(df=self._bdist.kwds["df"])
        elif self._dist=="nct":
            self._tdist = scipy.stats.nct(df=self._bdist.kwds["df"], nc=self._bdist.kwds["nc"])

    def cvol_sigma2(self, params, y, s2_0, scale=None):
        return cvol_sigma2(params=params, y=y, s2_0=s2_0, vol=self._vol, p=self._p, o=self._o, q=self._q, scale=scale)
    def cvol_dist_s(self, params, y, s2_0, scale=None):
        return cvol_dist_s(params=params, y=y, s2_0=s2_0, vol=self._vol, p=self._p, o=self._o, q=self._q, dist=self._dist, a_u=self._a_u, scale=scale)
    def cvol_ll(self, params, y, s2_0, scale=None):
        return cvol_ll(params=params, y=y, s2_0=s2_0, vol=self._vol, p=self._p, o=self._o, q=self._q, dist=self._dist, a_u=self._a_u, scale=scale)
    def cvol_mll(self, params, y, s2_0, scale=None):
        return -self.cvol_ll(params=params, y=y, s2_0=s2_0, scale=scale)


    def set_F_X(self):

        self.F_X = self._bdist.cdf(self._am._y_original)
        self.f_X = self._bdist.pdf(self._am._y_original)
        self.log_L = calc_log_L(f_X=self.f_X, train_bool=~np.isnan(self._am_res._volatility))

 
    def set_vrs(self, a_q, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):

        self._bdist.set_vrs(a_q=a_q, y=self._am._y_original.values, train_t_end=self._train_t_end, 
                                            set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, 
                                            save_t=save_t)

    def set_vrs_eval(self, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):

        self._bdist.vrs.set_vrs_eval(y=self._am._y_original.values, train_t_end=self._train_t_end, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)


    def set_vrs_evt(self, a_q, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):

        self._bdist_evt.set_vrs(a_q=a_q, y=self._am._y_original.values, train_t_end=self._train_t_end, 
                                            set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, 
                                            save_t=save_t)

    def set_vrs_eval_evt(self, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):

        self._bdist_evt.vrs.set_vrs_eval(y=self._am._y_original.values, train_t_end=self._train_t_end, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)
    

    def set_vrs_2(self, a_q, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):

        self.vrs_2 = vares_3.vares_a(a_q=a_q, bdist=self._bdist, tdist=self._tdist, y=self._am._y_original.values, train_t_end=self._train_t_end, 
                                            set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, 
                                            save_t=save_t)

    def set_vrs_2_eval(self, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):

        self.vrs_2.set_vrs_eval(y=self._am._y_original.values, train_t_end=self._train_t_end, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)


    def calc_fit_cv_est_dict(self):

        cv_dict =   {
                        **self._stats_res['cv'],
                        **self._stats_res['param']['est'],
                        **{'log_L_G': self._stats_res['res']['log_L']},
                        **{f'opt_{t}': self._stats_res['opt_time'][t] for t in self._stats_res['opt_time']},
                    }
        return cv_dict

    def calc_fit_cv_se_dict(self):

        cv_dict =   {
                        **self._stats_res['cv'],
                        **self._stats_res['param']['se'],
                    }
        return cv_dict
    
    def calc_fit_cv_evt_meta(self):

        return {
                        **misc_fns.dict_key_rm(self._stats_res['cv'], key_rm='a_u'),
                        **{'a_u': self._evt_a_u},
                }

    def calc_fit_cv_evt_est_dict(self):

        cv_dict =   {
                        **self.calc_fit_cv_evt_meta(),
                        **self._bdist_evt._y_dist.calc_gp_dict(),
                    }
        return cv_dict

    def calc_fit_cv_evt_se_dict(self):

        cv_dict =   {
                        **self.calc_fit_cv_evt_meta(),
                    }
        return cv_dict


    def calc_fit_cv_dict(self, page=None):
        if page is None:
            page = 'est'
        if page=='est':
            return self.calc_fit_cv_est_dict()
        elif page=='se':
            return self.calc_fit_cv_se_dict()

    def calc_fit_cv_evt_dict(self, page=None):
        if page is None:
            page = 'est'
        if page=='est':
            return self.calc_fit_cv_evt_est_dict()
        elif page=='se':
            return self.calc_fit_cv_evt_se_dict()


    def calc_fit_cv_w(self, lead=None, page=None):
        if lead is None:
            lead = dict()

        cv_dict = self.calc_fit_cv_dict(page=page)
        columns = [     
                        [k for k in lead],
                        [k for k in cv_dict.keys()],
                ]
        df_dict = {c: [] for c in np.concatenate(columns)}
        
        for k in columns[-1]:
            df_dict[k].append(cv_dict[k])
        for k in columns[0]:
            df_dict[k] = np.repeat(lead[k], 1)

        return pd.DataFrame(df_dict)

    def calc_vrs_eval(self, lead=None, atf=None):
        if lead is None:
            lead = dict()
        return self._bdist.vrs.calc_vrs_eval(lead={**lead, **self.calc_cvol_dict()}, atf=atf)

    def calc_fit_cv_evt_w(self, lead=None, page=None):
        if lead is None:
            lead = dict()

        cv_dict = self.calc_fit_cv_evt_dict(page=page)
        columns = [     
                        [k for k in lead],
                        [k for k in cv_dict.keys()],
                ]
        df_dict = {c: [] for c in np.concatenate(columns)}
        
        for k in columns[-1]:
            df_dict[k].append(cv_dict[k])
        for k in columns[0]:
            df_dict[k] = np.repeat(lead[k], 1)

        return pd.DataFrame(df_dict)

    def calc_evt_vrs_eval(self, lead=None, atf=None):
        if lead is None:
            lead = dict()
        return self._bdist_evt.vrs.calc_vrs_eval(lead={**lead, **self.calc_cvol_dict(), **{'evt_a_u': self._evt_a_u}}, atf=atf)



    def set_tt(self, a_u=None, dt_dt_fine=None, meta_series=None, opt_bounds=None, set_fit=None, fix_a_lambda=None, opt_method=None):

        if set_fit is None:
            set_fit=False

        fit_fn_pre = 'cv_z_tt'
        fit_colval = {**self._stats_res['cv']}

        data = pd.Series(scipy.stats.norm().ppf(self.F_X), index=self._am._y_original.index)
        self.tt = hwks.tt_pot(data=data, 
                                    a_u=a_u, dt_dt_fine=dt_dt_fine, 
                                train_t_end=self._last_obs, meta_series=meta_series, fit_fn_pre=fit_fn_pre, fit_colval=fit_colval, fix_a_lambda=fix_a_lambda, opt_method=opt_method)

        if opt_bounds is not None:
            for mode in self.tt.tt:
                for mark in self.tt.tt[mode]:
                    for param in opt_bounds:
                        for bound in opt_bounds[param]:
                            self.tt.tt[mode][mark]._parameters.__dict__[param]._opt_range._values[bound] = opt_bounds[param][bound]
        if set_fit:
            self.fit_tt()



    def fit_tt(self):
        self.tt.set_tt_fit(append_w=True, fix_a_lambda=self._tt_sp_params['fix_a_lambda'], opt_method=self._tt_sp_params['opt_method'])

    def fit_tt_sp(self):
        self.tt.set_tt_fit(append_w=True, fix_a_lambda=self._tt_sp_params['fix_a_lambda'], opt_method=self._tt_sp_params['opt_method'])
        return self.tt

    def set_tt_bi_lr_sp(self):

        self.tt.set_tt_bi_lr(fit_bi_lr=self._tt_sp_params['fit_bi_lr'], fp0_bi=self._tt_sp_params['fp0_bi'], fix_a_lambda=self._tt_sp_params['fix_a_lambda'], opt_method=self._tt_sp_params['opt_method'])

        return self.tt

    
    def set_tt_sp_params(self, a_u=None, dt_dt_fine=None, meta_series=None, opt_bounds=None, set_fit=None, fix_a_lambda=None, opt_method=None, fit_bi_lr=None, fp0_bi=None):

        self._tt_sp_params = {
                                'a_u': a_u,
                                'dt_dt_fine': dt_dt_fine,
                                'meta_series': meta_series,
                                'opt_bounds': opt_bounds,
                                'set_fit': set_fit,
                                'fix_a_lambda': fix_a_lambda,
                                'opt_method': opt_method,
                                'fit_bi_lr': fit_bi_lr,
                                'fp0_bi': fp0_bi,
                            }

    def set_tt_sp(self):

        self.set_tt(a_u=self._tt_sp_params['a_u'], dt_dt_fine=self._tt_sp_params['dt_dt_fine'], meta_series=self._tt_sp_params['meta_series'], opt_bounds=self._tt_sp_params['opt_bounds'], set_fit=self._tt_sp_params['set_fit'], fix_a_lambda=self._tt_sp_params['fix_a_lambda'], opt_method=self._tt_sp_params['opt_method'])

        return self.tt


    
    def set_sim(self, metaseed=None):

        self._sim = cvol_sim(vol=self._vol, p=self._p, o=self._o, q=self._q, dist=self._dist, scale=self._am.scale, metaseed=metaseed)


    def set_sim_data(self, metaseed=None, N_sim=None, reset_sim_data=None):
        if self._sim is None:
            self.set_sim(metaseed=metaseed)
        
        self._sim.set_sim_data(params=self._am_res.params, N_sim=N_sim, index=self._am._y_original.index, reset_sim_data=reset_sim_data)



    def set_sim_tt(self, i=None, a_u=None, dt_dt_fine=None, meta_series=None, opt_bounds=None, set_fit=None, fix_a_lambda=None, opt_method=None, sim_number=None):


        if sim_number is None:
            sim_number = 0

        meta_series_sim =   {
                                **meta_series,
                                **{'metaseed': self._sim.get_metaseed()},
                                **{'sim': sim_number},
                            }

        self._sim.set_tt(i=i, a_u=a_u, dt_dt_fine=dt_dt_fine, train_t_end=self._last_obs, meta_series=meta_series_sim, opt_bounds=opt_bounds, set_fit=set_fit, fix_a_lambda=fix_a_lambda, opt_method=opt_method)

    
    def set_sim_sp(self):

        self.set_sim(metaseed=self._sim_tt_sp_params['metaseed'])
        if self._sim_tt_sp_params['set_sim_data']:
            self.set_sim_data(N_sim=self._sim_tt_sp_params['N_sim'])
        if self._sim_tt_sp_params['set_sim_tt']:
            self.set_sim_tt(i=self._sim_tt_sp_params['i'], a_u=self._sim_tt_sp_params['a_u'], dt_dt_fine=self._sim_tt_sp_params['dt_dt_fine'], meta_series=self._sim_tt_sp_params['meta_series'], opt_bounds=self._sim_tt_sp_params['opt_bounds'], set_fit=self._sim_tt_sp_params['set_fit'], fix_a_lambda=self._sim_tt_sp_params['fix_a_lambda'])

        return self._sim


    def set_sim_data_tt_fit(self, sim_number=None, fix_a_lambda=None, reset_sim_data=None):

        if reset_sim_data is None:
            reset_sim_data = True

        if sim_number is None:
            sim_number = 0
        if reset_sim_data:
            i=0
        elif reset_sim_data is None:
            i=0
        self.set_sim_data(N_sim=1, reset_sim_data=True)
        self._sim.tt._rw['meta']['series']['sim'] = sim_number
        self._sim.tt.set_new_data(data=self._sim.sim_data[0]['data'])
        self._sim.fit_tt(fix_a_lambda=fix_a_lambda)


    def set_sim_data_tt_fit_i(self, sim_number=None, fix_a_lambda=None, a_u=None, u=None, load_fit_w=None):


        if sim_number is None:
            sim_number = 0


        if a_u is not None:
            self._sim.tt._a_u = a_u
            for mode in self._sim.tt.tt:
                for mark in self._sim.tt.tt[mode]:
                    self._sim.tt.tt[mode][mark]._parameters.a_lambda.set_value(a_u)

        self._sim.tt._rw['meta']['series']['sim'] = sim_number
        self._sim.tt.set_new_data(data=self._sim.sim_data[sim_number]['data'])


        
        if u is not None:
            for mode in self._sim.tt.tt:
                for mark in self._sim.tt.tt[mode]:
                    self._sim.tt.tt[mode][mark]._parameters.u.set_value(u)
                    self._sim.tt.tt[mode][mark].detect_events()
        if load_fit_w is not None:
            if sim_number==0:
                self._sim.tt._rw['fit']['w'][load_fit_w] = {
                                                                fn: pd.DataFrame() for fn in self._sim.tt._rw['fit']['w']['tt']
                                                            }
            self._sim.tt._rw['fit']['w']['tt'] = copy.deepcopy(self._sim.tt._rw['fit']['w'][load_fit_w])
        self._sim.fit_tt(fix_a_lambda=fix_a_lambda)
        if load_fit_w is not None:
            self._sim.tt._rw['fit']['w'][load_fit_w] = copy.deepcopy(self._sim.tt._rw['fit']['w']['tt'])



    def set_sim_data_tt_fit_sp(self):

        self.set_sim_data_tt_fit(sim_number=self._sim_tt_sp_params['sim_number'], fix_a_lambda=self._sim_tt_sp_params['fix_a_lambda'], reset_sim_data=True)

        return self._sim

    def set_sim_data_tt_fit_cmu_old(self, sim_number=None, a_u=None):

        self.set_sim_data_tt_fit_i(sim_number=sim_number, fix_a_lambda=True, a_u=a_u, load_fit_w='tt_c')
        self.set_sim_data_tt_fit_i(sim_number=sim_number, fix_a_lambda=False, a_u=a_u, load_fit_w='tt_u')

    
    def set_sim_data_tt_fit_cmu(self, sim_number=None, a_u=None):

        self.set_sim_data_tt_fit_i(sim_number=sim_number, fix_a_lambda=True, a_u=a_u, load_fit_w=None)
        self.set_sim_data_tt_fit_i(sim_number=sim_number, fix_a_lambda=False, a_u=a_u, load_fit_w=None)


    def set_sim_data_tt_fit_cmu_sp_old(self):

        self.set_sim_data_tt_fit_cmu_old(sim_number=self._sim_tt_sp_params['sim_number'], a_u=self._sim_tt_sp_params['a_u'])

        return self._sim
    
    
    def set_sim_data_tt_fit_cmu_sp(self):

        self.set_sim_data_tt_fit_cmu(sim_number=self._sim_tt_sp_params['sim_number'], a_u=self._sim_tt_sp_params['a_u'])

        return self._sim

    


    def set_sim_tt_sp_params(self, metaseed=None, set_sim_data=None, set_sim_tt=None, N_sim=None, i=None, a_u=None, dt_dt_fine=None, meta_series=None, opt_bounds=None, set_fit=None, fix_a_lambda=None, opt_method=None, sim_number=None):

        self._sim_tt_sp_params = {
                                    'metaseed': metaseed,
                                    'set_sim_data': set_sim_data,
                                    'set_sim_tt': set_sim_tt,
                                    'N_sim': N_sim,
                                    'i': i,
                                    'a_u': a_u,
                                    'dt_dt_fine': dt_dt_fine,
                                    'meta_series': meta_series,
                                    'opt_bounds': opt_bounds,
                                    'set_fit': set_fit,
                                    'fix_a_lambda': fix_a_lambda,
                                    'opt_method': opt_method,
                                    'sim_number': sim_number,
                            }


class cvol_sim():
    """Class for conditional volatility simulations
    """
    def __init__(self, vol=None, p=None, o=None, q=None, dist=None, scale=None, metaseed=None):
        """Constructor for cvol_sim class
        """
        self.reset_am(vol=vol, p=p, o=o, q=q, dist=dist, scale=scale)
        self.set_metaseed(metaseed=metaseed, reset=True)


    def set_metaseed(self, metaseed=None, reset=None):

        if reset is None:
            reset = False
        
        if metaseed is not None:
            self._metaseed = metaseed
        elif reset:
            self._metaseed = 0

    def get_metaseed(self):
        
        return self._metaseed


    def reset_am(self, vol=None, p=None, o=None, q=None, dist=None, scale=None):
        self.am = arch.arch_model(None, vol=vol, p=p, o=o, q=q, dist=_arch_str[dist])
        self.scale = scale
        self.reset_sim_data()

    def reset_sim_data(self):
        self.sim_data = []

    def set_sim_data(self, params, N_sim=None, size=None, index=None, reset_sim_data=None):

        if N_sim is None:
            N_sim = 1
        if reset_sim_data or None:
            reset_sim_data = True

        if reset_sim_data:
            self.reset_sim_data()

        if size is None:
            if index is None:
                size = 1000
            else:
                size = index.size
        else:
            if index is not None:
                ind_size = np.size(index)
                if size < ind_size:
                    index = index[:size]
                elif size > ind_size:
                    index=None
            

        for i in range(N_sim):
            sim_data = self.am.simulate(params, size)
            if index is not None:
                sim_data = sim_data.set_index(index)
            self.sim_data.append(sim_data/self.scale)



    def set_tt(self, i=None, a_u=None, dt_dt_fine=None, train_t_end=None, meta_series=None, opt_bounds=None, set_fit=None, fix_a_lambda=None, opt_method=None):

        if set_fit is None:
            set_fit=False
        if i is None:
            i=0

        data = self.sim_data[i]['data']
        self.tt = hwks.tt_pot(data=data, 
                                    a_u=a_u, dt_dt_fine=dt_dt_fine, 
                                train_t_end=train_t_end, meta_series=meta_series, opt_method=opt_method)

        if opt_bounds is not None:
            for mode in self.tt.tt:
                for mark in self.tt.tt[mode]:
                    for param in opt_bounds:
                        for bound in opt_bounds[param]:
                            self.tt.tt[mode][mark]._parameters.__dict__[param]._opt_range._values[bound] = opt_bounds[param][bound]
        if set_fit:
            self.fit_tt(fix_a_lambda=fix_a_lambda)

    def fit_tt(self, fix_a_lambda=None, opt_method=None):
        self.tt.set_tt_fit(append_w=True, fix_a_lambda=fix_a_lambda, opt_method=opt_method)



    

        


#============
#============
#
# Functions
#
#============
#============

def calc_cv_v(y, vol, p=None, o=None, q=None, dist=None, a_u=None, last_obs=None, set_am_stats=None,
                                a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
                                
    return {str(v): cvol_poq(y=y, vol=str(v), p=p, o=o, q=q, a_u=a_u, dist=dist, last_obs=last_obs, set_am_stats=set_am_stats,
                            a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t) for v in misc_fns.make_iterable_array(vol)} 

def calc_cv_p(y, p, o, q, vol=None, dist=None, a_u=None, last_obs=None, set_am_stats=None,
                                a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
                                
    return  {p_i: 
                {o_i: 
                    {q_i: 
                            cvol_dist(y=y, vol=vol, p=int(p_i), o=int(o_i), q=int(q_i), dist=dist, a_u=a_u, last_obs=last_obs, set_am_stats=set_am_stats,
                                        a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t) 
                    for q_i in misc_fns.make_iterable_array(q, dtype=int)} 
                for o_i in misc_fns.make_iterable_array(o, dtype=int)} 
            for p_i in misc_fns.make_iterable_array(p, dtype=int)}

def calc_cv_d(y, dist, vol=None, p=None, o=None, q=None, a_u=None, last_obs=None, set_am_stats=None,
                    a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):

    return {str(d): cvol(y=y, vol=vol, p=p, o=o, q=q, dist=str(d), a_u=a_u, last_obs=last_obs, set_am_stats=set_am_stats,
                a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t) for d in misc_fns.make_iterable_array(dist)}

#--------------
#  cvol
#--------------
"""Object-independent functions for cvol class
"""

def nct_mu(df, nc=0):
    return nc * np.sqrt(df/2) * scipy.special.gamma((df-1)/2) / scipy.special.gamma(df/2)

def nct_var(df, nc=0):
    return (df*(1+(nc**2))/(df-2)) - ((nc**2)*df/2) * ((scipy.special.gamma((df-1)/2)/scipy.special.gamma(df/2))**2)


def cvol_sigma2(params, y, s2_0, vol=None, p=None, o=None, q=None, scale=None):

    if vol is None:
        vol="GARCH"
    if p is None:
        p = 1
    if o is None:
        o = 0
    if q is None:
        q = 1
    if scale is None:
        scale=1
    lead = np.max([p,o,q])

    mu = params[0]    
    omega = params[1]
    if p==0:
        a_q=0
    elif p==1:
        a_q = params[2]
    else:
        a_q = params[2:2+p]
    if o==0:
        gamma = 0
    elif o==1:
        gamma = params[2+p]
    else:
        gamma = params[2+p:2+p+o]
    if q==0:
        beta = 0
    elif q==1:
        beta = params[2+p+o]
    else:
        beta = params[2+p+o:2+p+o+q]


    resid = y*scale - mu
    sigma2 = np.full(resid.size, np.nan)
    sigma2[0:lead] = s2_0

    if lead==1:
        if vol=="EGARCH":
            e = resid/np.sqrt(sigma2)
            sigma2 = np.log(sigma2)
            for k in range(resid.size-1):
                sigma2[1+k] = omega + a_q*(np.abs(e[k]) - np.sqrt(2/np.pi)) + beta*sigma2[k] + gamma*e[k]
                e[1+k] = resid[1+k]/np.exp(sigma2[1+k]/2)
            return np.exp(sigma2)
        else:
            resid2 = resid**2
            resid2_neg = resid2 * (resid<0)
            for k in range(resid.size-1):
                sigma2[1+k] = omega + a_q*resid2[k] + beta*sigma2[k] + gamma*resid2_neg[k]
            return sigma2


def cvol_dist_s_2(params, y, s2_0, vol=None, p=None, o=None, q=None, dist=None, scale=None):

    if dist is None:
        dist = "norm"

    mu = params[0]
    if dist=="norm":
        sigma2 = cvol_sigma2(params=params, y=y, s2_0=s2_0, vol=vol, p=p, o=o, q=q, scale=scale)
        dist_s = scipy.stats.norm(loc=mu/scale,
                                scale=np.sqrt(sigma2)/scale)
    elif dist=="t":
        nu = params[-1]
        sigma2 = cvol_sigma2(params=params[:-1], y=y, s2_0=s2_0, vol=vol, p=p, o=o, q=q, scale=scale)
        dist_s = scipy.stats.t(df=nu,
                                loc=mu/scale,
                                scale=(np.sqrt(sigma2)/scale)/np.sqrt(nct_var(df=nu, nc=0)))
    elif dist=="nct":
        nu = params[-2]
        lambd = params[-1] 
        sigma2 = cvol_sigma2(params=params[:-2], y=y, s2_0=s2_0, vol=vol, p=p, o=o, q=q, scale=scale)
        dist_s = scipy.stats.nct(df=nu, nc=lambd, 
                                loc=mu/scale,
                                scale=(np.sqrt(sigma2)/scale)/np.sqrt(nct_var(df=nu, nc=lambd)))

    return dist_s

def cvol_dist_s(params, y, s2_0, vol=None, p=None, o=None, q=None, dist=None, a_u=None, scale=None):

    if dist is None:
        dist = "norm"
    bulk_dist = dist

    mu = params[0]
    nu=None
    lambd=None

    sigma2 = cvol_sigma2(params=params, y=y, s2_0=s2_0, vol=vol, p=p, o=o, q=q, scale=scale)

    bulk_loc = mu/scale
    bulk_scale = np.sqrt(sigma2)/scale

    if a_u is None:
        a_u=0

    if a_u==0:
        GP_N_param = 0
        gp_c = 0
        gp_loc = np.array([-np.infty, np.infty])
        gp_scale = 1
    else:
        if True:
            GP_N_param = 4
            gp_c = params[-4:-2]
            #gp_loc = bulk_gp.bulk_transform_x_y(params[-4:-2], bulk_loc=bulk_loc, bulk_scale=bulk_scale)
            gp_scale = bulk_gp.bulk_transform_x_y(params[-2:], bulk_loc=0, bulk_scale=np.array([bulk_scale]))
        else:
            GP_N_param = 6
            gp_c = params[-6:-4]
            gp_loc = bulk_gp.bulk_transform_x_y(params[-4:-2], bulk_loc=bulk_loc, bulk_scale=bulk_scale)
            gp_scale = bulk_gp.bulk_transform_x_y(params[-2:], bulk_loc=0, bulk_scale=bulk_scale)


    

        
    if dist=='t':
        nu = params[-(1+GP_N_param)]
    elif dist=='nct':
        nu = params[-(2+GP_N_param)]
        lambd = params[-(1+GP_N_param)]

    
    if a_u!=0:
        a_u_lr = np.array([a_u, 1-a_u])
        if True:
            if dist=='nct':
                scipy.stats.nct(df=nu, nc=lambd).ppf(a_u_lr)
            elif dist=='t':
                scipy.stats.t(df=nu).ppf(a_u_lr)
            else:
                t_u = scipy.stats.norm().ppf(a_u_lr)
            gp_loc = bulk_gp.bulk_transform_x_y(params[-4:-2], bulk_loc=bulk_loc, bulk_scale=bulk_scale)

        

    dist_s = bulk_gp.bulk_gp(bulk_dist=dist,
                                    bulk_loc=bulk_loc,
                                    bulk_scale=bulk_scale,
                                    bulk_df=nu,
                                    bulk_nc=lambd,
                                    gp_c=gp_c, 
                                    gp_loc=gp_loc, 
                                    gp_scale=gp_scale)


    return dist_s
    
def cvol_ll(params, y, s2_0, vol=None, p=None, o=None, q=None, dist=None, a_u=None, scale=None):
    dist = cvol_dist_s(params=params, y=y, s2_0=s2_0, vol=vol, p=p, o=o, q=q, dist=dist, a_u=a_u, scale=scale)
    return np.sum(np.log(dist.pdf(y)))

def cvol_mll(params, y, s2_0, vol=None, p=None, o=None, q=None, dist=None, a_u=None, scale=None):
    return -cvol_ll(params=params, y=y, s2_0=s2_0, vol=vol, p=p, o=o, q=q, dist=dist, a_u=a_u, scale=scale)

def cvol_vol_poq_dist_find(res):

    vols =  {
                "GARCH(p: 1, q: 1)": "GARCH",
                "GJR-GARCH(p: 1, o: 1, q: 1)": "GARCH",
                "EGARCH(p: 1, q: 1)": "EGARCH",
                "EGARCH(p: 1, o: 1, q: 1)": "EGARCH",
            }
    ps =    {
                "GARCH(p: 1, q: 1)": 1,
                "GJR-GARCH(p: 1, o: 1, q: 1)": 1,
                "EGARCH(p: 1, q: 1)": 1,
                "EGARCH(p: 1, o: 1, q: 1)": 1,
            }
    os =    {
                "GARCH(p: 1, q: 1)": 0,
                "GJR-GARCH(p: 1, o: 1, q: 1)": 1,
                "EGARCH(p: 1, q: 1)": 0,
                "EGARCH(p: 1, o: 1, q: 1)": 1,
            }
    qs =    {
                "GARCH(p: 1, q: 1)": 1,
                "GJR-GARCH(p: 1, o: 1, q: 1)": 1,
                "EGARCH(p: 1, q: 1)": 1,
                "EGARCH(p: 1, o: 1, q: 1)": 1,
            }
    dists = {
                "Normal distribution": "norm",
                "Standardized Student's t distribution": "t",
                "Standardized Skew Student's t distribution": "nct",

            }

    return vols[res._model._model_description()["volatility"]], ps[res._model._model_description()["volatility"]], os[res._model._model_description()["volatility"]], qs[res._model._model_description()["volatility"]], dists[res._model._model_description()["distribution"]]


def cvol_res_2(am, res):

    
    p_l = [param for param in res.params.keys()]
    b_a = np.array([_bound_stats_res[param] for param in p_l])

    vol, p, o, q, dist = cvol_vol_poq_dist_find(res)

    res_2 = scipy.optimize.minimize(cvol_mll, x0=res.params.values, 
                                args=(am._y[~np.isnan(res._volatility)], res._volatility[0]**2, vol, p, o, q, dist, 1),
                                        method='SLSQP', options={'disp': True},
                                        bounds=scipy.optimize.Bounds(lb=b_a[:,0], ub=b_a[:,1]))

    return res_2

def cvol_dist_s_res_2(am, res, res_2):

    vol, p, o, q, dist = cvol_vol_poq_dist_find(res)

    return cvol_dist_s(params=res_2.x, y=am._y_original.values, 
                        s2_0=res._volatility[0]**2, vol=vol, p=p, o=o, q=q, dist=dist, scale=am.scale)

                

def calc_log_L(f_X, train_bool=None, train_t_end=None):

    log_L = np.log(f_X)
    if train_bool is None and train_t_end is None:
        ll = np.sum(log_L)
    else:
        ll =    {
                    "all": np.sum(log_L),
                }
        if train_bool is not None:
            ll["train"] =  np.sum(log_L[train_bool])
            ll["forecast"] = np.sum(log_L[~train_bool])
        else:
            ll["train"] =  np.sum(log_L[:train_t_end])
            ll["forecast"] = np.sum(log_L[train_t_end:])

    return ll








#============
#============
#
# Variables
#
#============
#============


_arch_str = {
                        "norm": "normal",
                        "t": "t",
                        "nct": "skewt",
                    }


_bound_stats_res =  {
                        "mu": np.array([-10, 10]),
                        "omega": np.array([1E-4, 10]),
                        "alpha[1]": np.array([0, 1]),
                        "gamma[1]": np.array([-1, 1]),
                        "beta[1]": np.array([0, 1]),
                        "nu": np.array([0, 100]),
                        "lambda": np.array([-10, 10]),
                    }