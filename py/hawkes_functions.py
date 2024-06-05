import numpy as np

# Local files
import misc_functions as misc_fns
import plot_functions as plot_fns




def model_slash(param, fill=None, model_unit=None):

    if model_unit is None:
        model_unit = get_model_unit()
    if model_unit[param] == "":
        return ""
    else:
        if fill is None:
            fill = f""
        return f"{fill}/{fill}"

def model_slash_unit(param, fill=None, model_unit=None):

    if model_unit is None:
        model_unit = get_model_unit()
    return f'{model_slash(param, fill, model_unit)}{model_unit[param]}'



def mode_translate(mode):
    
    return get_mode_translate()[mode] 

def mark_translate(mark):
    if mark:
        return "mark"
    else:
        return "unmark"




def create_dict_1T_POT(element=None):

    return  {
                tail: {mark: element for mark in get_marks()} for tail in get_tails()
            }
           
def create_dict_2T_POT(element=None):
    return {
                mode: {mark: element for mark in get_marks()} for mode in get_modes()
            } 




def flatten_parameters(a_lambda, gamma, beta, nu, omega, xi, varsigma, eta, alpha):
    return np.hstack([np.hstack(a_lambda), np.hstack(gamma), np.hstack(beta), np.hstack(nu), np.hstack(omega), np.hstack(xi), np.hstack(varsigma), np.hstack(eta), np.hstack(alpha)])


def unflatten_parameters(D_nodes, flat_parameters):


    stypes = ["vector", "matrix", "vector", "vector", "vector", "vector", "vector", "vector", "vector"]
    stypes_shape =  {
                        "scalar": (),
                        "vector": (D_nodes),
                        "matrix": (D_nodes, D_nodes),
                    }
    index_sizes = np.array([int(np.product(stypes_shape[i])) for i in stypes])
    index_sizes_cumsum = np.zeros(index_sizes.size+1, dtype=int)
    index_sizes_cumsum[1:] = np.cumsum(index_sizes)

    indicies = []
    parameters = []
    for i in range(index_sizes.size):
        indicies.append(range(index_sizes_cumsum[i], index_sizes_cumsum[i+1]))
        parameters.append(np.resize(flat_parameters[indicies[i]], stypes_shape[stypes[i]]))

    return parameters






#============
#============
#
# Variables
#
#===========
#===========

def get_u_dir_sign(val=None):

    _u_dir_sign =   {
                            "left": -1,
                            "right": 1,
                            "abs": 1,
                        }
    
    if val is None:
        return _u_dir_sign
    else:
        return _u_dir_sign[val]
    
def get_mode_translate(val=None):

    _mode_translate =   {
                            "nme": "bi",
                            "dc": "bi_d",
                            "ci": "ci",
                            "ci_s": "ci_s",
                        }
    
    if val is None:
        return _mode_translate
    else:
        return _mode_translate[val]

def get_unit_d_tr():

    _unit_d_tr = f'\\mathrm{{d}}_{{\\mathrm{{tr}}}}'

    return _unit_d_tr

def get_unit_y_tr():

    _unit_y_tr = f'\\mathrm{{y}}_{{\\mathrm{{tr}}}}'

    return _unit_y_tr

def get_model_unit(val=None):

    _model_unit =    {
                        "u": "",
                        "a_lambda": f"{get_unit_d_tr()}^{{-1}}",
                        "a_u": "",
                        "mu": f"{get_unit_d_tr()}^{{-1}}",
                        "gamma": "",
                        "beta": f"{get_unit_d_tr()}^{{-1}}",
                        "xi": "",
                        "varsigma": "",
                        "eta": "",
                        "alpha": "",
                        "df": "",
                        "nc": "",
                    }
    
    if val is None:
        return _model_unit
    else:
        return _model_unit[val]
    
def get_tails():
    _tails = ["left", "right", "abs"]
    return _tails
def get_modes():
    _modes = ["bi", "bi_d", "ci", "ci_s"]
    return _modes
def get_marks():
    _marks = ["mark", "unmark"]
    return _marks

def get_range_conditions():
    _range_conditions = ["permitted", "bound_incl_low", "bound_incl_high", "bound_excl_low", "bound_excl_high"]
    return _range_conditions

def get_tail_node(val=None):
    _tail_node = {
                'both': None,
                'left': 0,
                'right': 1,
                }
    if val is None:
        return _tail_node
    else:
        return _tail_node[val]

def get_tail_sign(val=None):
    _tail_sign = {
                    'left': -1,
                    'right': 1,
                }
    if val is None:
        return _tail_sign
    else:
        return _tail_sign[val]

def get_tail_symb(val=None):
    _tail_symb = {
                    'left': '\\leftarrow',
                    'right': '\\rightarrow',
                    'both': '\\leftrightarrow',
                    'and': '\\leftrightarrow',
                    'or': '\\leftrightharpoons',
                }
    if val is None:
        return _tail_symb
    else:
        return _tail_symb[val]

def get_tail_tex(val=None):
    _tail_tex = {
                    'left': '\\rsTL',
                    'right': '\\rsTR',
                    'both': '\\rsTA',
                    'and': '\\rsTA',
                    'or': '\\rsTO',
                }
    if val is None:
        return _tail_tex
    else:
        return _tail_tex[val]

def get_mode_symb(val=None):
    _mode_symb = {
                    'bi': f'\\mathrm{{bi}}',
                    'bi_d': f'\\mathrm{{bi}},\\mathrm{{d}}',
                    'ci': f'\\mathrm{{ci}}',
                    'ci_s': f'\\mathrm{{ci}},\\mathrm{{s}}',
                    'un': f'\\mathrm{{un}}',
                }
    if val is None:
        return _mode_symb
    else:
        return _mode_symb[val]

def get_mode_tex(val=None):
    _mode_tex = {
                    'bi': f'\\rsHwkModeBi',
                    'bi_d': f'\\rsHwkModeBiD',
                    'ci': f'\\rsHwkModeCi',
                    'ci_s': f'\\rsHwkModeCiS',
                    'un': f'\\rsHwkModeUn',
                }
    
    if val is None:
        return _mode_tex
    else:
        return _mode_tex[val]

def get_part_symb(val=None, tex=None):

    if tex is None:
        tex = False

    _part_symb = {
                    'Hwk': f'H',
                    'Pot': f'U',
                    'Thresh': f'u',
                    'Arrival': f'N',
                    'Intensity': f'\\lambda',
                    'Excess': f'M',
                    'Bulk': f'B',
                    'Pr': f'\\pi',
                    'Wlk': f'W',
                    'Param': f'\\mathbf{{\\theta}}',
                }
    
    if tex:
        _part_symb['Param'] = f'\\vect{{\\uptheta}}'

    if val is None:
        return _part_symb
    else:
        return _part_symb[val]


def get_part_tex(val=None):
    _part_symb = get_part_symb()
    _part_tex = {part: f"\\rsMdlPrtHwk{part}" for part in _part_symb}

    if val is None:
        return _part_tex
    else:
        return _part_tex[val]



def annotate_model_Hawkes(
                str_tick=None, str_mod=None, str_u=None, str_a_lambda=None, 
                plot_tick=None, plot_mod=None, plot_u=None, plot_a_lambda=None, 
                an_yM_mod=None, an_yM_tick=None, an_yM_u=None, an_yM_a_lambda=None,
                u_scif_pow10=None, u_dp=None, u_dp_buff=None, u_val_plus=None, 
                a_u_scif_pow10=None, a_u_dp=None, a_u_dp_buff=None, a_u_val_plus=None, 
                a_lambda_scif_pow10=None, a_lambda_dp=None, a_lambda_dp_buff=None, a_lambda_val_plus=None, 
                an_mode=None, an_xycoords=None, an_h=None, an_v=None, an_fs=None, an_x0=None, an_y0=None, an_xi=None, an_yi=None, an_xs=None, an_ys=None, flip_x0ry0t=None,
                u=None, a_u=None, a_lambda=None, 
                display_I_s=None, super_I_i=None, super_I_cat=None, 
                tf_mode=None, a_mode=None, a_subscript=None, plot_order=None,
                series=None, tick=None, 
                model=None, uncond=None, hwk_uncond=None, resid=None, mode=None, mark=None, 
                top_to_bottom=None, left_to_right=None, flip_i=None, flip_j=None,
                ax=None):
    
    


    if top_to_bottom is None:
        top_to_bottom = True


    if tf_mode is None:
        tf_mode=="u"



    if a_mode is None:
        if tf_mode=='u':
            a_mode = f"lambda"
        else:
            a_mode = f"u"

    if a_subscript is None:
        a_subscript = "\\lambda"
    

    if u_dp is None:
        if u_scif_pow10 is None:
            if tf_mode=='a_u':
                u_dp = 2


    if a_mode=="a_u":
        if a_u is not None:
            a_lambda = a_u



    _locals = locals()

    an_params = plot_fns.calc_an_params_dir(**{c: _locals[c] for c in plot_fns.calc_an_params_args(cat=['main', 'dir', 'flip'])})
    
    

    str_val = calc_str_val(
                            **{f"{v}{field}": _locals[f"{v}{field}"] for v in ['u', 'a_u', 'a_lambda'] for field in ['', '_scif_pow10', '_dp', '_dp_buff', '_val_plus']},
                                brac_type='', add_lr=False, super_I_i=super_I_i, super_I_cat=super_I_cat, v0_only=False, display_I_s=display_I_s,
                            )


    if _locals['str_tick'] is None:
        if series is None:
            series = f"r_{{t}}"
        if tick is not None:
            tick_str = f" [{tick}]"
        else:
            tick_str = f""

        _locals['str_tick'] = f"${series}${tick_str}"


    
    if _locals['str_mod'] is None:

        if str_val['a_u']['str_val']=="" and str_val['u']['str_val']=="":
            evt_str = ""
        else:
            evt_str_cat = 'str'
            if a_u is None or np.all(np.isnan(a_u)):
                evt_param='u'
                if plot_u:
                    evt_str_cat = 'str_symb'
            else:
                evt_param='a_u'

            evt_str = misc_fns.capp_brac(str_val[evt_param][evt_str_cat], brac_type='()', add_lr=True)

        
        if model is None:
            if uncond is None:
                uncond = False
            if hwk_uncond is None:
                hwk_uncond = False
            if uncond:
                if hwk_uncond:
                    model = f"{get_part_symb(val='Hwk')}^{{{get_mode_symb(val='un')}}}"
                else:
                    model = get_part_symb(val="Pot")
            else:
                model = get_part_symb(val="Hwk")
        
        if resid is not None:
            str_resid = f"{resid} | "
        else:
            str_resid = f""
        if model==get_part_symb(val="Hwk") and mode is not None:
            str_mode = f"^{{{get_mode_symb(val=mode)}}}"
        else:
            str_mode = ""

        

        _locals['str_mod'] = f"${str_resid}{model}{str_mode}{{{evt_str}}}$"



    if _locals['str_u'] is None:
        _locals['str_u'] = f"${str_val['u']['str']}$"

    if _locals['str_a_lambda'] is None:
        #_locals['str_a_lambda'] = f"${str_val[f'a_{a_mode}']['str']}$"
        if False:
            _locals['str_a_lambda'] = f"${str_val[f'a_lambda']['str']}$"
        else:
            ###### QUICK FIX
            str_val[f'a_lambda']['str'] = str_val[f'a_lambda']['str'].replace('lambda', 'pi')
            str_val[f'a_lambda']['str_symb'] = str_val[f'a_lambda']['str_symb'].replace('lambda', 'pi')
            if a_mode=="a_u":
                _locals['str_a_lambda'] = f"${str_val[f'a_lambda']['str_symb']} = {str_val[f'a_u']['str_symb']} = {str_val[f'a_u']['str_val']}$"
            else:
                _locals['str_a_lambda'] = f"${str_val[f'a_lambda']['str']}$"
            


    str_all, an_yM_all, plot_bool, plot_index = annotate_model_plot_order(**{c: _locals[c] for c in calc_annotate_model_args(cat=['plot', 'str', 'an_yM'])},
                                                                an_ys=an_params['an_ys'], tf_mode=tf_mode, a_mode=a_mode)


    if ax is not None:
        plot_fns.annotate_array(ax=ax, str_list=str_all[plot_index], an_yM=an_yM_all[plot_index],
                                top_to_bottom=top_to_bottom, left_to_right=left_to_right, flip_i=flip_i, flip_j=flip_j,
                                    **an_params)
        

    
    

    if False:
        np.zeros((2,2), dtype=int)
        xy_box[0,:] = an_params['an_x0']
        xy_box[1,:] = an_params['an_y0']

        if np.size(plot_index)>0:
            xy_box[1,1] = an_params['an_y0'] + an_params['an_ys']*(1+np.max(an_yM_all[plot_index]))*an_params['an_yi']



    if np.any(plot_bool):
        P_y = 1+np.max(an_yM_all[plot_index])
        # P_y = np.sum(plot_bool)
    else:
        P_y = 0


    xy_box = plot_fns.calc_xy_box(an_params=an_params, along_x=False, along_y=True,
                                    P_y=P_y)
    

    #return str_all, an_yM_all, plot_bool, plot_index, xy_box
    return xy_box
        

def annotate_model_plot_order(  
                                plot_tick=None, plot_mod=None, plot_u=None, plot_a_lambda=None,
                                str_tick=None, str_mod=None, str_u=None, str_a_lambda=None, 
                                an_yM_mod=None, an_yM_tick=None, an_yM_u=None, an_yM_a_lambda=None,
                                an_ys=None, 
                                tf_mode=None, a_mode=None, plot_order=None):




    if an_ys is None:
        an_ys = 1

    if plot_order is None:
        plot_order = ['tick', 'mod', 'u', 'a_lambda']
            
    P = len(plot_order)

    _locals = locals()
    plot_bool = np.full(P, False)
    for p in range(P):
        plot = plot_order[p]
        if _locals[f"plot_{plot}"] is not None:
            plot_bool[p] = _locals[f"plot_{plot}"]
    plot_index = np.where(plot_bool)[0]
    if False:
        if an_ys<0:
            an_yM_all = np.cumsum(plot_bool) - 1
        else:
            an_yM_all = np.flip(np.cumsum(np.flip(plot_bool))) - 1
    else:
        an_yM_all = np.cumsum(plot_bool) - 1
    
    for p in range(P):
        plot = plot_order[p]
        if _locals[f"an_yM_{plot}"] is not None:
            an_yM_all[p] = _locals[f"an_yM_{plot}"]


    str_all = np.array([_locals[f"str_{p}"] for p in plot_order])

    return str_all, an_yM_all, plot_bool, plot_index

    



def calc_annotate_model_args(cat=None, cat_rm=None, list_rm=None):

    p_list = ['tick', 'mod', 'u', 'a_lambda']
    field_list = ['plot', 'str', 'an_yM']
    scif_list = ['scif_pow10', 'dp', 'dp_buff', 'val_plus']
    val_list = ['u', 'a_u', 'a_lambda']

    args = {
                **{field: [f'{field}_{p}' for p in p_list] for field in field_list},
                **{f"{p}_scif": [f'{p}_{scif}' for scif in scif_list] for p in val_list},
                'an': plot_fns.calc_an_params_args(cat=['main', 'dir']),
                'val': val_list,
                'I': ['display_I_s', 'super_I_i', 'super_I_cat'],
                'meta': ['tf_mode', 'a_mode', 'plot_order'],
                'tick': ['series', 'tick'],
                'model': ['model', 'uncond', 'hwk_uncond', 'resid', 'mode', 'mark'],
                'ax': ['ax'],
            }
    
    return misc_fns.return_args(args=args, cat=cat, cat_rm=cat_rm, list_rm=list_rm)



def get_val_symb(val=None):

    _val_symbs =  {
                        'u': {'symb': 'u', 'sub': ''},
                        'a_u': {'symb': 'a', 'sub': 'u'},
                        'a_lambda': {'symb': 'a', 'sub': '\\lambda'},
                    }
    
    if val is None:
        return _val_symbs
    else:
        return misc_fns.dict_key_rm(dic=_val_symbs, keys=val, keys_rm=None, ret=None)
    




def calc_str_val(
                    val_names=None, v0_only=None,
                    val=None,
                    u=None, a_u=None, a_lambda=None,
                    scif_pow10=None, dp=None, dp_buff=None, val_plus=None,
                    u_scif_pow10=None, u_dp=None, u_dp_buff=None, u_val_plus=None,
                    a_u_scif_pow10=None, a_u_dp=None, a_u_dp_buff=None, a_u_val_plus=None,
                    a_lambda_scif_pow10=None, a_lambda_dp=None, a_lambda_dp_buff=None, a_lambda_val_plus=None,
                    display_I_s=None, brac_type=None, add_lr=None, super_I_i=None, super_I_cat=None,
                    force_vector=None, show_symb=None, show_val=None,
                ):
    



    if super_I_i is None:
        if display_I_s is not None:
            I_max = np.max(display_I_s)
        else:
            I_max = 10
        super_I_i = np.concatenate([[get_tail_symb()[d] for d in ['left', 'right']], [f"{n}" for n in np.arange(2,I_max)]])

    if super_I_cat is None:
        super_I_cat = get_tail_symb()['or']

    if v0_only is None:
        v0_only = False

    if u_val_plus is None:
        u_val_plus = True


    _locals = locals()

    args = get_val_symb(val=val_names)
    v_s = [v for v in args]
    for v in args:
        if _locals[v] is None:
            _locals[v] = _locals['val']
        args[v][f"val"] = _locals[v]
        for scif in ['scif_pow10', 'dp', 'dp_buff', 'val_plus']:
            if _locals[f"{v}_{scif}"] is None:
                _locals[f"{v}_{scif}"] = _locals[f"{scif}"]
            args[v][f"{scif}"] = _locals[f"{v}_{scif}"]

    common_args_list = ['display_I_s', 'brac_type', 'add_lr', 'super_I_i', 'super_I_cat',
                    'force_vector', 'show_symb', 'show_val']
    common_args = {a: _locals[a] for a in common_args_list}
    
    ret_dict = {v: misc_fns.calc_str_val_sub_sup(**args[v], **common_args) for v in v_s}

    if v0_only and len(v_s)==1:
        return ret_dict[v_s[0]]
    else:
        return ret_dict



def calc_opt_settings_dict(opt_method=None, fix_a_lambda=None):

    _locals = locals()
    return _locals


def calc_resets_opt_setting(reset=None, opt_method=None, fix_a_lambda=None):

    return misc_fns.calc_resets(reset=reset, resets=calc_opt_settings_dict(opt_method=opt_method, fix_a_lambda=fix_a_lambda))



