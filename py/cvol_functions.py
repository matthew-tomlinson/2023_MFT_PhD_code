import numpy as np

# Local files
import misc_functions as misc_fns
import plot_functions as plot_fns
import dist_functions as dist_fns

import hawkes_functions as hwks_fns



def calc_cv_vol_symb(val=None, tex=None):

    if tex is None:
        tex = False

    if tex:
        return calc_cv_vol_tex(val=val)

    else:
        _cv_vol_symb = {
                        'uv': f"W",
                        'GARCH': f'G',
                        }
        if val is None:
            return _cv_vol_symb
        else:
            return _cv_vol_symb[val]
    

def calc_cv_vol_tex(val=None):

    _cv_vol_tex = {
                    'uv': f'\\rsMdlPrtWlk',
                    'GARCH': f'\\rsMdlPrtGevtVolG',
                }
    if val is None:
        return _cv_vol_tex
    else:
        return _cv_vol_tex[val]
    


def calc_cv_poq_symb(val=None):
        
    if tex is None:
        tex = False

    if tex:
        return calc_cv_poq_tex(val=val)

    else:
        _cv_poq_symb = {
                        'p': f'p',
                        'o': f'o',
                        'q': f'q',
                        }
        if val is None:
            return _cv_poq_symb
        else:
            return _cv_poq_symb[val]
    

def calc_cv_poq_tex(val=None):
    _cv_poq_tex = {
                    {
                    'p': f'\\rsMdlPrtGevtPOQp',
                    'o': f'\\rsMdlPrtGevtPOQo',
                    'q': f'\\rsMdlPrtGevtPOQq',
                    }
                }
    if val is None:
        return _cv_poq_tex
    else:
        return _cv_poq_tex[val]





def calc_cv_poq_val_symb(p=None, o=None, q=None, poq_symb=None, tex=None):

    if poq_symb is None:
        poq_symb = False

    if tex is None:
        tex = False

    if poq_symb:
        if p is None:
            p = False
        if o is None:
            o = True
        if q is None:
            q = False

        if p:
            str_p = f"{calc_cv_poq_tex(val='p', tex=tex)},"
        else:
            str_p = ""
        if q:
            str_q = f",{calc_cv_poq_tex(val='q', tex=tex)}"
        else:
            str_q = ""
        
        if o:
            str_o = calc_cv_poq_tex(val='o', tex=tex)
        else:
            str_o = ""

        val_str = f"{str_p}{str_o}{str_q}"

    else:


        if p is None:
            p = 1
        if o is None:
            o = 0
        if q is None:
            q = 1

        if p==1:
            str_p = ""
        else:
            str_p = f"{p},"

        if q==1:
            str_q = ""
        else:
            str_q = f",{p}"

        val_str = f"{str_p}{o}{str_q}"

    return val_str



def calc_uv_symb(dist=None, a_u=None, show_a_u_symb=None, show_a_u_val=None, brac_type=None,
                 a_u_scif_pow10=None, a_u_dp=None, a_u_dp_buff=None, a_u_val_plus=None,
                 tex=None):

    if tex is None:
        tex=False

    if dist is None:
        dist = 'norm'

    if a_u is None:
        a_u = 0

    

    dist_symb = dist_fns.get_scipy_dist_symb(dist=dist, macro=True, tex=tex)
    vol_symb = calc_cv_vol_symb(val='uv', tex=tex)



    uv_str = f"{vol_symb}^{{{dist_symb}}}"

    a_u_str = ""
    if show_a_u_symb is None:
        show_a_u_symb = False
    if show_a_u_val is None:
        if a_u==0:
            show_a_u_val = False
        else:
            show_a_u_val = True

    if brac_type is None:
        brac_type = '()'
    if a_u_dp is None:
        if a_u==0:
            a_u_dp = 0

    if show_a_u_val or show_a_u_symb:
        a_u_str_val = hwks_fns.calc_str_val(a_u=np.repeat(a_u,2), show_symb=show_a_u_symb, show_val=show_a_u_val, brac_type=brac_type,
                                                a_u_scif_pow10=a_u_scif_pow10, a_u_dp=a_u_dp, a_u_dp_buff=a_u_dp_buff, a_u_val_plus=a_u_val_plus)['a_u']
        a_u_str = f"{{{a_u_str_val['str']}}}"
    

    return f"{uv_str}{a_u_str}"




def calc_cv_symb(vol=None, p=None, o=None, q=None, dist=None, a_u=None, show_a_u_symb=None, show_a_u_val=None, brac_type=None,
                 a_u_scif_pow10=None, a_u_dp=None, a_u_dp_buff=None, a_u_val_plus=None,
                 poq_symb=None,
                 tex=None):

    if tex is None:
        tex=False

    if vol is None:
        vol = "GARCH"

    if dist is None:
        dist = 'norm'

    if a_u is None:
        a_u = 0

    


    dist_symb = dist_fns.get_scipy_dist_symb(dist=dist, tex=tex)
    vol_symb = calc_cv_vol_symb(val=vol, tex=tex)
    poq_symb = calc_cv_poq_val_symb(p=p, o=o, q=q, poq_symb=poq_symb, tex=tex)


    cv_str = f"{vol_symb}_{{{poq_symb}}}^{{{dist_symb}}}"

    a_u_str = ""
    if show_a_u_symb is None:
        show_a_u_symb = False
    if show_a_u_val is None:
        if a_u==0:
            show_a_u_val = False
        else:
            show_a_u_val = True

    if brac_type is None:
        brac_type = '()'
    if a_u_dp is None:
        if a_u==0:
            a_u_dp = 0

    if show_a_u_val or show_a_u_symb:
        a_u_str_val = hwks_fns.calc_str_val(a_u=np.repeat(a_u,2), show_symb=show_a_u_symb, show_val=show_a_u_val, brac_type=brac_type,
                                                a_u_scif_pow10=a_u_scif_pow10, a_u_dp=a_u_dp, a_u_dp_buff=a_u_dp_buff, a_u_val_plus=a_u_val_plus)['a_u']
        a_u_str = f"{{{a_u_str_val['str']}}}"
    

    return f"{cv_str}{a_u_str}"

    







            



    