import numpy as np
import pandas as pd
# Local files
import misc_functions as misc_fns
import plot_functions as plot_fns


class T23_Cfstmp(plot_fns.Cfstmp):

    def __init__(self, 
            root=None, fnsubs=None,
            dpi=None, thesis=None, an_mode=None,
            title=None, matter=None, n=None, n_post=None, initchr=None
            ):
        

        super().__init__(root=root, fnsubs=fnsubs,
                            dpi=dpi, thesis=thesis, an_mode=an_mode,
                            title=title, matter=matter, n=n, n_post=n_post, initchr=initchr)
        

    def calc_title_default(self):
        return 'T23'
    def calc_thesis_default(self):
        return True
    def calc_an_mode_default(self):
        return 'thesis'
    

class T23_misc():

    def __init__(self, 
                 sd_not_scale=None, app_dlrs=None, app_tex_num=None,
                 ):
        
        self.set_sd_not_scale(sd_not_scale=sd_not_scale, reset=True)
        self.set_app_dlrs(app_dlrs=app_dlrs, reset=True)
        self.set_app_tex_num(app_tex_num=app_tex_num, reset=True)

    def set_sd_not_scale(self, sd_not_scale=None, reset=None):
        if reset is None:
            reset = False
        if sd_not_scale is not None:
            self._sd_not_scale = sd_not_scale
        elif reset:
            self._sd_not_scale = False
    def get_sd_not_scale(self):
        return self._sd_not_scale
    
    def set_app_dlrs(self, app_dlrs=None, reset=None):
        if reset is None:
            reset = False
        if app_dlrs is not None:
            self._app_dlrs = app_dlrs
        elif reset:
            self._app_dlrs = True
    def get_app_dlrs(self):
        return self._app_dlrs
    
    def set_app_tex_num(self, app_tex_num=None, reset=None):
        if reset is None:
            reset = False
        if app_tex_num is not None:
            self._app_tex_num = app_tex_num
        elif reset:
            self._app_tex_num = True
    def get_app_tex_num(self):
        return self._app_tex_num
    

    def capp_dlrs(self, S, app_dlrs=None):

        if app_dlrs is None:
            app_dlrs = self.get_app_dlrs()

        return misc_fns.capp_dlrs(S=S, app_dlrs=app_dlrs)

    def capp_tex_num(self, S, app_tex_num=None):

        if app_tex_num is None:
            app_tex_num = self.get_app_tex_num()

        return misc_fns.capp_tex_num(S=S, app_tex_num=app_tex_num)


    def capp_dlrs_tex_num(self, S=None, app_dlrs=None, app_tex_num=None, app_null=None):

        if app_dlrs is None:
            app_dlrs = self.get_app_dlrs()
        if app_tex_num is None:
            app_tex_num = self.get_app_tex_num()

        return misc_fns.capp_dlrs(S=misc_fns.capp_tex_num(S=S, app_tex_num=app_tex_num, app_null=app_null), app_dlrs=app_dlrs, app_null=app_null)






def setup_T23_base(
                __fit__=None, __vares__=None,
                ticks=None,
                  all_start=None, all_end=None, train_end=None,  
                  tt_a_u=None, tt_fix_a_lambda=None, hb_dist=None, hb_mode_bound=None, 
                  cv_vol=None, cv_p=None, cv_o=None, cv_q=None, cv_dist=None,
                  a_q=None, CT_Q=None, CT_E=None,
                  save_t=None,
                  **kwargs):
    
    if __fit__ is None:
        __fit__ = False
    if __vares__ is None:
        __vares__ = False

    
    if ticks is None:
        ticks = ['SPX', 'DJI', 'DAX', 'CAC', 'NKX', 'HSI']
    if all_start is None:
        all_start = pd.Timestamp("1975-01-01")
    if all_end is None:
        all_end = pd.Timestamp("2023-09-16")
    if train_end is None:
        train_end = pd.Timestamp("2015-01-01")


    if tt_a_u is None:
        tt_a_u = np.array([float(f'{x:.5f}') for x in 0.0125*(1+np.array(range(20)))])
    if tt_fix_a_lambda is None:
        tt_fix_a_lambda = True
    if hb_dist is None:
        #hb_dist = ['norm', 't']
        hb_dist = ['norm']
    if hb_mode_bound is None:
        hb_mode_bound = {
                        'bi': 'both',
                        'bi_d': ['both', 'left', 'right'],
                        'ci': 'both',
                        'ci_s': 'both',
                    }
    
    if cv_vol is None:
         cv_vol = 'GARCH'
    if cv_p is None:
         cv_p = 1
    if cv_o is None:
         cv_o = [0,1]
    if cv_q is None:
         cv_q = 1
    if cv_dist is None:
         cv_dist = ['norm', 't']


    if a_q is None:
        if __vares__:
            a_q = np.array([float(f'{x:.5f}') for x in 0.0025*np.flip(1+np.array(range(60)))])
        else:
            a_q = np.array([float(f'{x:.5f}') for x in [0.025]])
    if CT_Q is None:
        if __vares__:
            CT_Q = np.concatenate([
                                    ['binom', 'UC', 'CC'],
                                    [f'DQ_{h}' for h in range(5)],
                                ])
        else:
            CT_Q = []
        
    if CT_E is None:
        if __vares__:
            CT_E = ['ZMD']
        else:
            CT_E = []

    if save_t is None:
        if __fit__:
            save_t = False
        else:
            save_t = True

    _locals = locals()

    return misc_fns.dict_collapse(d=misc_fns.dict_key_rm(_locals, keys_rm=['__fit__', '__vares__']), subdictkeys='kwargs')



def calc_T35_Cm2_all_start():

    all_start_dates_price_0 = {
                        "SPX": "1927-12-30",
                        "DJI": "1927-12-30",
                        "DAX": "1959-09-28",
                        "CAC": "1969-01-02",
                        "NKX": "1949-05-16",
                        "HSI": "1969-11-24",
                        }


    all_start_dates_r = {
                            "SPX": "1928-01-03",
                            "DJI": "1928-01-03",
                            "DAX": "1959-09-29",
                            "CAC": "1969-01-03",
                            "NKX": "1949-05-17",
                            "HSI": "1969-11-25",
                            }
    return {tick: pd.Timestamp(all_start_dates_r[tick]) for tick in all_start_dates_r}



def setup_T23_Cm1(**kwargs):
    
    _locals = locals()

    return setup_T23_Cm2(**misc_fns.dict_collapse(d=_locals, subdictkeys='kwargs'))



def setup_T23_Cm2(ticks=None,
                  all_start=None, all_end=None, train_end=None,  
                  **kwargs):
    
    if ticks is None:
        ticks = ['SPX', 'DJI', 'DAX', 'CAC', 'NKX', 'HSI']
    if all_start is None:
        all_start = calc_T35_Cm2_all_start()
    if all_end is None:
        all_end = pd.Timestamp("2023-05-20")
    if train_end is None:
        train_end = pd.Timestamp("2023-05-20")
    
    _locals = locals()

    return setup_T23_base(**misc_fns.dict_collapse(d=_locals, subdictkeys='kwargs'))



def setup_T23_Cm3(ticks=None,
                  all_start=None, all_end=None, train_end=None,  
                  **kwargs):
    
    if ticks is None:
        ticks = ['SPX', 'DJI', 'DAX', 'NKX']
    if all_start is None:
        all_start = pd.Timestamp("1959-10-02")
    if all_end is None:
        all_end = pd.Timestamp("2020-11-20")
    if train_end is None:
        train_end = pd.Timestamp("2008-09-01")
    
    _locals = locals()

    return setup_T23_base(**misc_fns.dict_collapse(d=_locals, subdictkeys='kwargs'))


def setup_T23_Cm4(ticks=None,
                  all_start=None, all_end=None, train_end=None,  
                  **kwargs):
    
    if ticks is None:
        ticks = ['SPX', 'DJI', 'DAX', 'CAC', 'NKX', 'HSI']
    if all_start is None:
        all_start = pd.Timestamp("1975-01-01")
    if all_end is None:
        all_end = pd.Timestamp("2022-09-10")
    if train_end is None:
        train_end = pd.Timestamp("2015-01-01")
    
    _locals = locals()

    return setup_T23_base(**misc_fns.dict_collapse(d=_locals, subdictkeys='kwargs'))


def setup_T23(__C__=None, **kwargs):

    _locals = locals()

    if __C__=='Cm1':
        return setup_T23_Cm1(**_locals['kwargs'])
    elif __C__=='Cm2':
        return setup_T23_Cm2(**_locals['kwargs'])
    elif __C__=='Cm3':
        return setup_T23_Cm3(**_locals['kwargs'])
    elif __C__=='Cm4':
        return setup_T23_Cm4(**_locals['kwargs'])
    else:
       return setup_T23_base(**_locals['kwargs'])