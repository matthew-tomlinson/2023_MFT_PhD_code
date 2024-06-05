import numpy as np
import matplotlib


import misc_functions as misc_fns




def get_arab_num(ps=None, pre=None, suff=None, 
                      app_mathrm=None, 
                      attr=None, attr_args_pos=None, attr_args_dict=None,
                      pattr=None, pattr_args_pos=None, pattr_args_dict=None,
                      ):
    

    
    _arab_num = misc_fns.app_val_attr(np.array(np.arange(10), dtype='<U1'),
                                           attr=attr, args_pos=attr_args_pos, args_dict=attr_args_dict)


    return misc_fns.app_val_attr(misc_fns.app_presuff(S=_arab_num,
                       ps=calc_ps_roman(ps=ps, pre=pre, suff=suff, app_mathrm=app_mathrm), reshape=None), 
                       attr=pattr, args_pos=pattr_args_pos, args_dict=pattr_args_dict)



def calc_ps_greek(ps=None, pre=None, suff=None, app_bs=None, app_up=None, app_var=None):

    if ps is None:
        if pre is None:
            pre = ""
        if suff is None:
            suff = ""
        ps = [pre, suff]
    
    pre_greek = misc_fns.str_concatenate([
                    misc_fns.calc_val_1_0(app_val_1=app_bs, app_val_1_default=False, val_1='\\', val_0=None),
                    misc_fns.calc_val_1_0(app_val_1=app_up, app_val_1_default=False, val_1=f'up', val_0=None),
                    misc_fns.calc_val_1_0(app_val_1=app_var, app_val_1_default=False, val_1=f'var', val_0=None),
                    ])
    
    ps[0] = f"{ps[0]}{pre_greek}"
        
    return ps

def calc_ps_roman(ps=None, pre=None, suff=None, app_mathrm=None):

    if ps is None:
        if pre is None:
            pre = ""
        if suff is None:
            suff = ""
        ps = [pre, suff]
    
    if app_mathrm is None:
        app_mathrm = False
    
    ps_roman = misc_fns.calc_tex_fn_ps(tex_fn='mathrm', app_tex_fn=app_mathrm)
    
    ps[0] = f"{ps[0]}{ps_roman[0]}"
    ps[1] = f"{ps_roman[1]}{ps[1]}"

    return ps


def get_roman_letters(ps=None, pre=None, suff=None, 
                      app_mathrm=None, 
                      attr=None, attr_args_pos=None, attr_args_dict=None,
                      pattr=None, pattr_args_pos=None, pattr_args_dict=None,
                      ):
    

    
    _roman_letters = misc_fns.app_val_attr(np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']), 
                                           attr=attr, args_pos=attr_args_pos, args_dict=attr_args_dict)


    return misc_fns.app_val_attr(misc_fns.app_presuff(S=_roman_letters, 
                       ps=calc_ps_roman(ps=ps, pre=pre, suff=suff, app_mathrm=app_mathrm), reshape=None), 
                       attr=pattr, args_pos=pattr_args_pos, args_dict=pattr_args_dict)


def calc_roman_letters_case_symb_dict(roman_app_mathrm=None, capital=None, 
                                      tex_vect=None, app_tex_vect=None, pltf=None, use_tex=None):
    
    if capital is None:
        capital = False

    if capital:
        attr = 'capitalize'
    else:
        attr = None

    if app_tex_vect is None:
        app_tex_vect = False
    if pltf is None:
        pltf = False
    if use_tex is None:
        use_tex = get_matplt_use_tex()
    
    if roman_app_mathrm is None:
        if pltf and not use_tex:
            roman_app_mathrm = False
        else:
            roman_app_mathrm = app_tex_vect

    _roman_letters = get_roman_letters(app_mathrm=False, attr=attr)
    _roman_letters_symb = misc_fns.capp_tex_vect(get_roman_letters(app_mathrm=roman_app_mathrm, attr=attr),
                                                 tex_vect=tex_vect, app_tex_vect=app_tex_vect, pltf=pltf, use_tex=use_tex, app_null=False)

    _roman_letters_symb_dict = {_roman_letters[i]: _roman_letters_symb[i] for i in range(_roman_letters.size)}

    return _roman_letters_symb_dict






def get_greek_letters(app_omicron=None, ow_omicron=None, 
                      ps=None, pre=None, suff=None, 
                      app_bs=None, app_up=None, app_var=None, 
                      incl_var=None, reorder_incl_var=None, incl_var_app_var=None,
                      attr=None, attr_args_pos=None, attr_args_dict=None,
                      pattr=None, pattr_args_pos=None, pattr_args_dict=None,
                      ):

    if incl_var is None:
        if reorder_incl_var==True:
            incl_var = True
        else:
            incl_var = False


    if app_omicron is None:
        app_omicron = False
    if ow_omicron is None:
        ow_omicron = False 
        
    if ow_omicron:
        omicron = ['omicron']
    elif app_omicron:
        omicron = ['o', 'omicron']
    else:
        omicron = ['o']

    _gl = np.concatenate([
                            np.array(['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi']),
                            omicron,
                            np.array(['pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']),
                        ])

    _greek_letters = misc_fns.app_val_attr(_gl, 
                                           attr=attr, args_pos=attr_args_pos, args_dict=attr_args_dict)

    greek_letters =   misc_fns.app_presuff(S=_greek_letters,
                       ps=calc_ps_greek(ps=ps, pre=pre, suff=suff, app_bs=app_bs, app_up=app_up, app_var=app_var), reshape=None)
    

    if incl_var:

        if incl_var_app_var is None:
            incl_var_app_var = True

        greek_letters_incl = np.concatenate([greek_letters, 
                                             get_greek_letters_var(ps=ps, pre=pre, suff=suff, app_bs=app_bs, app_up=app_up, app_var=incl_var_app_var,
                                                                    attr=attr, attr_args_pos=attr_args_pos, attr_args_dict=attr_args_dict)])

        if reorder_incl_var is None:
            reorder_incl_var = False

        if reorder_incl_var:
            _greek_letters_var = get_greek_letters_var(app_bs=False, app_up=False, app_var=False, attr=attr, attr_args_pos=attr_args_pos, attr_args_dict=attr_args_dict)
            order = np.argsort(np.concatenate([np.arange(_greek_letters.size), 0.01+np.array([float(np.where(_greek_letters_var[i]==_greek_letters)[0]) for i in range(_greek_letters_var.size)])]))

        else:
            order = np.arange(greek_letters_incl.size)

        greek_letters = greek_letters_incl[order]


    return misc_fns.app_val_attr(greek_letters, attr=pattr, args_pos=pattr_args_pos, args_dict=pattr_args_dict)

    


def get_greek_letters_var(ps=None, pre=None, suff=None, 
                          app_bs=None, app_up=None, app_var=None,
                            attr=None, attr_args_pos=None, attr_args_dict=None,
                            pattr=None, pattr_args_pos=None, pattr_args_dict=None):


    if app_var is None:
        app_var = True

    _greek_letters_var = misc_fns.app_val_attr(np.array(['epsilon', 'theta', 'pi', 'rho', 'sigma', 'phi']), attr=attr, args_pos=attr_args_pos, args_dict=attr_args_dict)

    return misc_fns.app_val_attr(misc_fns.app_presuff(S=_greek_letters_var, 
                       ps=calc_ps_greek(ps=ps, pre=pre, suff=suff, app_bs=app_bs, app_up=app_up, app_var=app_var), reshape=None), attr=pattr, args_pos=pattr_args_pos, args_dict=pattr_args_dict)


def get_greek_letters_var_bool(app_omicron=None, ow_omicron=None, incl_var=None, reorder_incl_var=None, incl_var_app_var=None):

    if incl_var is None:
        incl_var = False
    if reorder_incl_var is None:
        reorder_incl_var = True
    if incl_var_app_var is None:
        incl_var_app_var = False

    _greek_letters = get_greek_letters(app_omicron=app_omicron, ow_omicron=ow_omicron, incl_var=incl_var, reorder_incl_var=reorder_incl_var, incl_var_app_var=incl_var_app_var)
    _greek_letters_var = get_greek_letters_var(app_var=bool(incl_var*incl_var_app_var))

    return np.isin(_greek_letters, _greek_letters_var)




def get_greek_letters_nvar_blank(app_omicron=None, ow_omicron=None, 
                                        ps=None, pre=None, suff=None, 
                                        app_bs=None, app_up=None, app_var=None, 
                                        incl_var=None, reorder_incl_var=None, incl_var_app_var=None, 
                                        attr=None, attr_args_pos=None, attr_args_dict=None,
                                        ow_app_var=None, 
                                        pattr=None, pattr_args_pos=None, pattr_args_dict=None,
                                        blank_nvar=None, blank_var=None, 
                                        blank_str=None, blank_str_nvar=None, blank_str_var=None):

    if incl_var is None:
        incl_var = False
    if reorder_incl_var is None:
        reorder_incl_var = True
    if incl_var_app_var is None:
        incl_var_app_var = True

    if blank_str is None:
        blank_str = ''

    if blank_nvar is None:
        blank_nvar = True

    if blank_var is None:
        if blank_nvar==False or blank_str_var is not None:
            blank_var = True
        else:
            blank_var = False

    if ow_app_var is None:
        ow_app_var = True


    
    if blank_str_var is None:
        blank_str_var = blank_str
    
    greek_letters = get_greek_letters(
                                        app_omicron=app_omicron, ow_omicron=ow_omicron, 
                                        ps=ps, pre=pre, suff=suff, 
                                        app_bs=app_bs, app_up=app_up, app_var=app_var, 
                                        incl_var=incl_var, reorder_incl_var=reorder_incl_var, incl_var_app_var=incl_var_app_var,
                                        attr=attr, attr_args_pos=attr_args_pos, attr_args_dict=attr_args_dict,
                                        pattr=pattr, pattr_args_pos=pattr_args_pos, pattr_args_dict=pattr_args_dict)
    
    _greek_letters_var_bool = get_greek_letters_var_bool(app_omicron=app_omicron, ow_omicron=ow_omicron, incl_var=incl_var, reorder_incl_var=reorder_incl_var, incl_var_app_var=incl_var_app_var)

    if ow_app_var and not app_var:
        if not (incl_var and incl_var_app_var):

            greek_letters_var = get_greek_letters(app_omicron=app_omicron, ow_omicron=ow_omicron, 
                                                  ps=ps, pre=pre, suff=suff, 
                                                app_bs=app_bs, app_up=app_up, app_var=True, 
                                                incl_var=incl_var, reorder_incl_var=reorder_incl_var, incl_var_app_var=incl_var_app_var,
                                                attr=attr, attr_args_pos=attr_args_pos, attr_args_dict=attr_args_dict,
                                                pattr=pattr, pattr_args_pos=pattr_args_pos, pattr_args_dict=pattr_args_dict)
            

            greek_letters_var[~_greek_letters_var_bool] = greek_letters[~_greek_letters_var_bool]

            greek_letters = greek_letters_var

    if blank_nvar:
        if blank_str_nvar is None:
            blank_str_nvar = blank_str
        greek_letters[~_greek_letters_var_bool] = blank_str_nvar

    if blank_var:
        if blank_str_var is None:
            blank_str_var = blank_str
        greek_letters[_greek_letters_var_bool] = blank_str_var


    return greek_letters
    

def calc_greek_letters_roman_dict(roman_app_mathrm=None, 
                                  tex_vect=None, app_tex_vect=None, pltf=None, use_tex=None):

    if app_tex_vect is None:
        app_tex_vect = False
    if pltf is None:
        pltf = False
    if use_tex is None:
        use_tex = get_matplt_use_tex()

    if roman_app_mathrm is None:
        if pltf and not use_tex:
            roman_app_mathrm = False
        else:
            roman_app_mathrm = app_tex_vect
    
    _greek_letters_roman_dict = {
                                'Alpha': 'A',
                                'Beta': 'B',
                                'Epsilon': 'E',
                                'Zeta': 'Z',
                                'Eta': 'H',
                                'Iota': 'I',
                                'Kappa': 'K',
                                'Mu': 'M',
                                'Nu': 'N',
                                'Rho': 'P',
                                'Tau': 'T',
                                'Chi': 'X',
                            }
    
    for l in _greek_letters_roman_dict:
        _greek_letters_roman_dict[l] = misc_fns.capp_tex_vect(misc_fns.capp_tex_fn(_greek_letters_roman_dict[l], tex_fn='mathrm', app_tex_fn=roman_app_mathrm),
                                                                tex_vect=tex_vect, app_tex_vect=app_tex_vect, pltf=pltf, use_tex=use_tex, app_null=False)

    return _greek_letters_roman_dict

def calc_greek_letters_case_symb_dict(app_omicron=None, ow_omicron=None, safe_omicron=None, roman_sub=None, roman_app_mathrm=None, capital=None,var=None, blank_nvar=None, app_var_nvar=None, ow_app_var=None, 
                                      tex_vect=None, app_tex_vect=None, pltf=None, use_tex=None):
    
    if capital is None:
        capital = False

    if capital:
        attr = 'capitalize'
    else:
        attr = None

    if var is None:
        var = False

    if app_omicron is None:
        app_omicron = False
    if ow_omicron is None:
        ow_omicron = False


    if app_tex_vect is None:
        app_tex_vect = False
    if pltf is None:
        pltf = False
    if use_tex is None:
        use_tex = get_matplt_use_tex()
    
    if roman_app_mathrm is None:
        if pltf and not use_tex:
            roman_app_mathrm = False
        else:
            roman_app_mathrm = app_tex_vect



    if var and not capital:
        if blank_nvar is None:
            blank_nvar = True
        if ow_app_var is None:
            ow_app_var = True
        if app_var_nvar is None:
            app_var_nvar = False
        _greek_letters = get_greek_letters(app_omicron=app_omicron, ow_omicron=ow_omicron, app_bs=False, app_var=True, attr=attr)
        _greek_letters_symb = misc_fns.capp_tex_vect(get_greek_letters_nvar_blank(app_omicron=app_omicron, ow_omicron=ow_omicron, app_bs=True, app_var=app_var_nvar, blank_nvar=blank_nvar, blank_var=False, ow_app_var=ow_app_var),
                                                 tex_vect=tex_vect, app_tex_vect=app_tex_vect, pltf=pltf, use_tex=use_tex, app_null=False)
    else:
        _greek_letters = get_greek_letters(app_omicron=app_omicron, ow_omicron=ow_omicron, app_bs=False, attr=attr)
        _greek_letters_symb = misc_fns.capp_tex_vect(get_greek_letters(app_omicron=app_omicron, ow_omicron=ow_omicron, app_bs=True, attr=attr),
                                                 tex_vect=tex_vect, app_tex_vect=app_tex_vect, pltf=pltf, use_tex=use_tex, app_null=False)

    _greek_letters_symb_dict = {_greek_letters[i]: _greek_letters_symb[i] for i in range(_greek_letters.size)}

    if capital:
        if roman_sub is None:
            roman_sub = True


            _greek_letters_roman_dict = calc_greek_letters_roman_dict(roman_app_mathrm=roman_app_mathrm, tex_vect=tex_vect, app_tex_vect=app_tex_vect, pltf=pltf, use_tex=use_tex)
            

            for l in _greek_letters_roman_dict:
                _greek_letters_symb_dict[l] = _greek_letters_roman_dict[l]


    if app_omicron or ow_omicron:
        if safe_omicron is None:
            safe_omicron = True


        if safe_omicron:
            o_keys = ['o', 'O', 'varo']
            omicron_keys = [f"{o}micron" for o in o_keys]
            om_keys = misc_fns.A_rm(omicron_keys, A_in=misc_fns.d_keys(_greek_letters_symb_dict))
            
            for om in om_keys:
                _greek_letters_symb_dict[om] = _greek_letters_symb_dict[om].replace('micron', '')


    return _greek_letters_symb_dict





def calc_arab_num_symb_dict(tex_vect=None, app_tex_vect=None, pltf=None, use_tex=None):

    if app_tex_vect is None:
        app_tex_vect = False
    if pltf is None:
        pltf = False
    if use_tex is None:
        use_tex = get_matplt_use_tex()

    _arab_num = get_arab_num()
    _arab_num_symb = misc_fns.capp_tex_vect(get_arab_num(), tex_vect=tex_vect, app_tex_vect=app_tex_vect, pltf=pltf, use_tex=use_tex, app_null=False)

    return {_arab_num[n]: _arab_num_symb[n] for n in range(np.size(_arab_num))}


def calc_roman_letters_symb_dict(roman_app_mathrm=None, tex_vect=None, app_tex_vect=None, pltf=None, use_tex=None):

    if app_tex_vect is None:
        app_tex_vect = False
    if pltf is None:
        pltf = False
    if use_tex is None:
        use_tex = get_matplt_use_tex()

    return {
                **calc_roman_letters_case_symb_dict(roman_app_mathrm=roman_app_mathrm, capital=False, tex_vect=tex_vect, app_tex_vect=app_tex_vect, pltf=pltf, use_tex=use_tex),
                **calc_roman_letters_case_symb_dict(roman_app_mathrm=roman_app_mathrm, capital=True, tex_vect=tex_vect, app_tex_vect=app_tex_vect, pltf=pltf, use_tex=use_tex),
            }



def calc_greek_letters_symb_dict(app_omicron=None, ow_omicron=None, safe_omicron=None, roman_sub=None, roman_app_mathrm=None, blank_nvar=None, app_var_nvar=None,
                                 tex_vect=None, app_tex_vect=None, pltf=None, use_tex=None):

    
    if blank_nvar is None:
        blank_nvar = False


    return {
                **calc_greek_letters_case_symb_dict(app_omicron=app_omicron, ow_omicron=ow_omicron, safe_omicron=safe_omicron, roman_sub=roman_sub, roman_app_mathrm=roman_app_mathrm, capital=False, var=False, blank_nvar=blank_nvar, app_var_nvar=app_var_nvar, tex_vect=tex_vect, app_tex_vect=app_tex_vect, pltf=pltf, use_tex=use_tex),
                **calc_greek_letters_case_symb_dict(app_omicron=app_omicron, ow_omicron=ow_omicron, safe_omicron=safe_omicron, roman_sub=roman_sub, roman_app_mathrm=roman_app_mathrm, capital=True, var=False, blank_nvar=blank_nvar, app_var_nvar=app_var_nvar, tex_vect=tex_vect, app_tex_vect=app_tex_vect, pltf=pltf, use_tex=use_tex),
                **calc_greek_letters_case_symb_dict(app_omicron=app_omicron, ow_omicron=ow_omicron, safe_omicron=safe_omicron, roman_sub=roman_sub, roman_app_mathrm=roman_app_mathrm, capital=False, var=True, blank_nvar=blank_nvar, app_var_nvar=app_var_nvar, tex_vect=tex_vect, app_tex_vect=app_tex_vect, pltf=pltf, use_tex=use_tex),
            }




def calc_arg_numletters_symb_dict(safe_omicron=None, roman_sub=None, roman_app_mathrm=None, blank_nvar=None, app_var_nvar=None,
                                  tex_vect=None, app_tex_vect=None, pltf=None, use_tex=None):


    
    return {
                **calc_arab_num_symb_dict(tex_vect=tex_vect, app_tex_vect=app_tex_vect, pltf=pltf, use_tex=use_tex),
                **calc_greek_letters_symb_dict(app_omicron=False, ow_omicron=True, safe_omicron=safe_omicron, roman_sub=roman_sub, roman_app_mathrm=roman_app_mathrm, blank_nvar=blank_nvar, app_var_nvar=app_var_nvar, 
                                               tex_vect=tex_vect, app_tex_vect=app_tex_vect, pltf=pltf, use_tex=use_tex),
                **calc_roman_letters_symb_dict(roman_app_mathrm=roman_app_mathrm, 
                                               tex_vect=tex_vect, app_tex_vect=app_tex_vect, pltf=pltf, use_tex=use_tex),
            }


def sb(l=None, v=None, safe_omicron=None, roman_sub=None, roman_app_mathrm=None, blank_nvar=None,
       tex_vect=None, pltf=None, use_tex=None):

    if v is None:
        v = False

    if use_tex is None:
        use_tex = get_matplt_use_tex()

    if l is None:
        s = ''
    else:
        s = calc_arg_numletters_symb_dict(safe_omicron=safe_omicron, roman_sub=roman_sub, roman_app_mathrm=roman_app_mathrm, blank_nvar=blank_nvar,
                                          tex_vect=tex_vect, app_tex_vect=v, pltf=pltf, use_tex=use_tex)[l]

    return s


def sbv(l=None, v=None, safe_omicron=None, roman_sub=None, roman_app_mathrm=None, blank_nvar=None,
       tex_vect=None, pltf=None, use_tex=None):
    
    if v is None:
        v = True
    
    return sb(**locals())

def pltsb(l=None, v=None, safe_omicron=None, roman_sub=None, roman_app_mathrm=None, blank_nvar=None,
       tex_vect=None, pltf=None, use_tex=None):
    
    if pltf is None:
        pltf = True
    
    return sb(**locals())

def pltsbv(l=None, v=None, safe_omicron=None, roman_sub=None, roman_app_mathrm=None, blank_nvar=None,
       tex_vect=None, pltf=None, use_tex=None):
    
    if v is None:
        v = True
    
    return pltsb(**locals())



def calc_arab_num_array(tex_vect=None, pltf=None, use_tex=None):

    if pltf is None:
        pltf = True
    if use_tex is None:
        use_tex = get_matplt_use_tex()

    
    num_array = np.transpose(np.array([
            misc_fns.capp_dlrs(misc_fns.d_vals(d=calc_arab_num_symb_dict(tex_vect=tex_vect, app_tex_vect=bool(i%2), pltf=pltf, use_tex=use_tex), array=True))
    for i in range(2)]))

    return num_array


def calc_roman_letters_array(roman_app_mathrm=None,
                                tex_vect=None, pltf=None, use_tex=None):
    
    

    if pltf is None:
        pltf = True
    if use_tex is None:
        use_tex = get_matplt_use_tex()

    
    letters_array = np.transpose(np.array([
            misc_fns.capp_dlrs(misc_fns.d_vals(d=calc_roman_letters_case_symb_dict(roman_app_mathrm=roman_app_mathrm, capital=bool(i//2), tex_vect=tex_vect, app_tex_vect=bool(i%2), pltf=pltf, use_tex=use_tex), array=True))
    for i in range(4)]))

    return letters_array


def calc_greek_letters_array(app_omicron=None, ow_omicron=None, 
                                roman_app_mathrm=None, blank_nvar=None,
                                tex_vect=None, pltf=None, use_tex=None):
    

    if pltf is None:
        pltf = True
    if use_tex is None:
        use_tex = get_matplt_use_tex()

    
    letters_array = np.transpose(np.array([
            misc_fns.capp_dlrs(misc_fns.d_vals(d=calc_greek_letters_case_symb_dict(app_omicron=app_omicron, ow_omicron=ow_omicron, 
                                                                                                         roman_sub=None, roman_app_mathrm=roman_app_mathrm, capital=bool((i//2)==1), var=bool((i//2)==2),
                                                                                                         blank_nvar=blank_nvar,
                                                                                                         tex_vect=tex_vect, app_tex_vect=bool(i%2), pltf=pltf, use_tex=use_tex), array=True))
    for i in range(6)]))

    return letters_array


def calc_arg_numletters_array(
                                app_omicron=None, ow_omicron=None, roman_app_mathrm=None, blank_nvar=None,
                                tex_vect=None, pltf=None, use_tex=None,
                                order_arg=None):
    
    if pltf is None:
        pltf = True
    if use_tex is None:
        use_tex = get_matplt_use_tex()
    
    
    if order_arg is None:
        order_arg = ['arab', 'roman', 'greek']
    order_arg = misc_fns.make_iterable_array(order_arg)
    
    _arg_array = {
                        'arab': calc_arab_num_array(tex_vect=tex_vect, pltf=pltf, use_tex=use_tex),
                        'roman': calc_roman_letters_array(roman_app_mathrm=roman_app_mathrm,
                                    tex_vect=tex_vect, pltf=pltf, use_tex=use_tex),
                        'greek': calc_greek_letters_array(app_omicron=app_omicron, ow_omicron=ow_omicron, roman_app_mathrm=roman_app_mathrm, blank_nvar=blank_nvar, 
                                                    tex_vect=tex_vect, pltf=pltf, use_tex=use_tex)
                    }

    return misc_fns.merge_arrays(arrays=[_arg_array[o] for o in order_arg], axis=1)




def calc_greek_roman_tex_preamble_print_lines(app_omicron=None, def_Greek=None):


    if app_omicron is None:
        app_omicron = True
    if def_Greek is None:
        def_Greek = True
        

    _greek_letters = get_greek_letters(app_omicron=app_omicron, ow_omicron=False, incl_var=True)
    _greek_letters_roman_dict = calc_greek_letters_roman_dict(roman_app_mathrm=True)
    _roman_letters_symb_dict = calc_roman_letters_symb_dict(roman_app_mathrm=True)

    tex_preamble_pack = [
    r"\usepackage{bm} % bold math",
    r"\usepackage{upgreek} % non-italic greek",
    "",
    ]
    tex_preamble_vect = [
        r"\def\vect#1{{",
        r"\bm{#1}}}",
        "",
    ]

    tex_preamble_vect_greek = [f"\\let\\{l}\\up{l}" for l in misc_fns.A_rm(_greek_letters, A_rm=['o', 'omicron'])]
    #tex_preamble_vect_roman = [f"\\let{l}{_roman_letters_symb_dict[l]}" for l in _roman_letters_symb_dict]

    if app_omicron:
        tex_preamble_omicron = np.concatenate([[
                                f"\\def\\{o}micron{{{{\\{o}}}}}" for o in ['o', 'O']
                            ], ['']])
    else:
        tex_preamble_omicron = []

    
    if def_Greek:
        tex_preamble_Greek = np.concatenate([[
                                f"\\def\\{l}{{{{{_greek_letters_roman_dict[l]}}}}}" for l in _greek_letters_roman_dict
                            ], ['']])
    else:
        tex_preamble_Greek = []


    tex_preamble = np.concatenate([
    tex_preamble_pack,
    tex_preamble_omicron,
    tex_preamble_Greek,
    misc_fns.make_iterable_array(tex_preamble_vect[0]),
    tex_preamble_vect_greek,
    #tex_preamble_vect_roman,
    misc_fns.make_iterable_array(tex_preamble_vect[1:]),
    ])


    return tex_preamble


def calc_greek_roman_tex_preamble(app_omicron=None, def_Greek=None):

    return misc_fns.str_concatenate(calc_greek_roman_tex_preamble_print_lines(app_omicron=app_omicron, def_Greek=def_Greek), fill_str="\n")



def get_matplt_use_tex():
    return matplotlib.rcParams['text.usetex']