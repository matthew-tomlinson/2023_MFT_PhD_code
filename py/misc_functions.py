
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
#import misc_functions as misc_fns
#import plot_functions as plot_fns




class timer():
    def __init__(self, print_timer=None, print_tab=None, print_head=None, write_timer=None, write_dir=None,  write_fn=None, write_preamble=None, reset_start=None):

        
        self.set_print_params(print_timer=print_timer, print_tab=print_tab, print_head=print_head)
        self.set_write_params(write_timer=write_timer, write_dir=write_dir,  write_fn=write_fn, preamble=write_preamble)

        if reset_start is None:
            reset_start=False
        if reset_start:
            self.reset()
        else:
            self.start()
        

    def set_print_params(self, print_timer=None, print_tab=None, print_head=None):
        self.set_print_timer(print_timer=print_timer)
        self.set_print_tab(print_tab=print_tab)
        self.set_print_head(print_head=print_head)

   

    def set_print_timer(self, print_timer=None):
        if print_timer is None:
            print_timer = True
        self._print_timer = print_timer

    def set_print_tab(self, print_tab=None):
        if print_tab is None:
            print_tab = ""
        self._print_tab = print_tab

    def set_print_head(self, print_head=None):

        self._print_head = print_head

    def set_write_params(self, write_timer=None, write_dir=None,  write_fn=None, preamble=None):
        self.set_write_timer(write_timer=write_timer)
        self.set_write_dir(write_dir=write_dir)
        self.set_write_fn(write_fn=write_fn)
        self.write_preamble(preamble=preamble)

    def write_preamble(self, preamble=None):

        if self._write_timer and preamble is not None:
            self._file = open(self.get_write_dir_fn(), 'a')
            self._file.write(preamble)
            self._file.write("\n")
            self._file.close()

        

    def set_write_timer(self, write_timer=None):
        if write_timer is None:
            write_timer = False
        self._write_timer = write_timer

    def set_write_dir(self, write_dir=None):
        if write_dir is None:
            write_dir = "./"
        self._write_dir = write_dir

    def set_write_fn(self, write_fn=None):
        if write_fn is None:
            write_fn = "write_timer.txt"
        self._write_fn = write_fn

    def get_write_dir_fn(self):
        
        return f"{self._write_dir}{self._write_fn}"


    def get_utc_time(self):
        return pd.Timestamp(datetime.datetime.utcnow(), tz="UTC")

    def get_cpu_time(self):
        return time.perf_counter()
    


    def set_lap(self, reset=None, utc_time=None, cpu_time=None):

        if reset is None:
            reset = False
        if utc_time is None:
            utc_time = self.get_utc_time()
        if cpu_time is None:
            cpu_time = self.get_cpu_time()

        if reset:
            self._utc_lap = [utc_time for i in range(2)]
            self._cpu_lap = [cpu_time for i in range(2)]
        else:
            self._utc_lap[0] = self._utc_lap[1]
            self._utc_lap[1] = utc_time

            self._cpu_lap[0] = self._cpu_lap[1]
            self._cpu_lap[1] = cpu_time

        self._utc_lap_delta = self._utc_lap[1] - self._utc_lap[0]
        self._cpu_lap_delta = self._cpu_lap[1] - self._cpu_lap[0]



    def get_lap_delta(self, val=None):

        if val is None:
            val = 'cpu'


        _lap_delta = {
                        'utc': self._utc_lap_delta,
                        'cpu': self._cpu_lap_delta*pd.Timedelta(seconds=1),
                    }
        
        return _lap_delta[val]

            

    def reset(self):
        self._utc_start = self.get_utc_time()
        self._cpu_start = self.get_cpu_time()

        self._utc_stop = self._utc_start
        self._cpu_stop = self._cpu_start

        self.set_delta()

        self.set_lap(reset=True, utc_time=self._utc_start, cpu_time=self._cpu_start)



    def start(self):

        self.reset()
        if self._print_timer:
            if self._print_head is not None:
                print(f"{self._print_tab}{self._print_head}")
            
            print(f"{self._print_tab}start: {self._utc_start}")

        if self._write_timer:
            self._file = open(self.get_write_dir_fn(), 'a')
            if self._print_head is not None:
                self._file.write(f"{self._print_tab}{self._print_head}")
                self._file.write("\n")
            self._file.write(f"{self._print_tab}start: {self._utc_start}")
            self._file.write("\n")
            self._file.close()

    def stop(self, print_timer=None):
        self._cpu_stop = self.get_cpu_time()
        self._utc_stop = self.get_utc_time()
        self.set_delta()
        if self._print_timer:
             print(f"{self._print_tab}  end: {self._utc_stop} | delta: {self._cpu_delta*pd.Timedelta(seconds=1)}")
        if self._write_timer:
            self._file = open(self.get_write_dir_fn(), 'a')
            self._file.write(f"{self._print_tab}  end: {self._utc_stop} | delta: {self._cpu_delta*pd.Timedelta(seconds=1)}")
            self._file.write("\n")
            self._file.close()
        

    def set_delta(self):
        self._cpu_delta = self._cpu_stop - self._cpu_start
        self._utc_delta = self._utc_stop - self._utc_start


    def get_timer_dict(self, pref=None):

        if pref is None:
            pref = ""

        timer =  {   
                    f"{pref}start": self._utc_start,
                    f"{pref}end": self._utc_stop,
                    f"{pref}delta": self._cpu_delta*pd.Timedelta(seconds=1),
                }

        return timer
    



    

#============
#============
#
# Multiprocess
#
#============
#============

def smap(f):
    return f()

def mp_funs(funs):

    pool = Pool(processes=len(funs))

    res = pool.map(smap, funs)

    return res

def mp_funs_factor(funs, factor=None, print_mp_time=None, write_mp_time=None, write_mp_dir=None, write_mp_fn=None):

    if print_mp_time is None:
        print_mp_time = True
    if write_mp_time is None:
        write_mp_time = False

    if write_mp_time:
        if write_mp_fn is None:
            write_mp_fn = f"write_mp_timer.txt"

    fn = calc_rlist(funs, factor=factor)

    n_runs = len(fn)
    res = [None for i in range(n_runs)]
    time_run = [None for i in range(n_runs)]

    for i in range(n_runs):
        preamble = f"\t{'mp run'} {1+i}/{n_runs} (n_proc = {len(fn[i])})"
        print(preamble)
        timer_run = timer(print_timer=print_mp_time, print_tab="\t", write_timer=write_mp_time, write_dir=write_mp_dir, write_fn=write_mp_fn, write_preamble=preamble)
        res[i] = mp_funs(funs=fn[i])
        timer_run.stop()
        time_run[i] = timer_run.get_timer_dict()


    return calc_flist(res), time_run

def calc_rlist(flist, factor=None):

    n_funs = len(flist)
    if factor is None:
        factor = calc_factor(n_funs=n_funs)

    n_full_runs = n_funs//factor
    n_rem = n_funs%factor

    nlist = [[flist[i*factor+j] for j in range(factor)] for i in range(n_full_runs)]
    if n_rem != 0:
        nlist.append([flist[n_full_runs*factor+j] for j in range(n_rem)])

    return nlist

def calc_factor(n_funs, max_factor=None):

    if max_factor is None:
        max_factor = multiprocessing.cpu_count()

    factors = (1+np.array(range(max_factor)))
    n_runs = (n_funs//factors)  + (n_funs%factors!=0)
    
    return factors[np.where(n_runs==n_runs[-1])[0][0]]



def calc_flist(rlist):

    return [rlist[i][j] for i in range(len(rlist)) for j in range(len(rlist[i]))]


#============
#============
#
# Functions
#
#============
#============

def return_x(x=None):

    return x

def array_func(a=None, f=None, args_pos=None, args_dict=None, reshape=None):


    if f is None:
        return a 
    else:
        if args_pos is None:
            args_pos = ()
        if args_dict is None:
            args_dict = {}

            
        if reshape is None:
            reshape = True

        if a is None:
            a = []
        a = np.array(a)
        a_shape = np.array(np.shape(a))

        N_i = np.product(a_shape)

        a = np.reshape(a, (N_i))

        r = np.array([f(a[i], *args_pos, **args_dict) for i in range(N_i)])

        if reshape:
            return np.reshape(r, a_shape)
        else:
            return r

def array_attr(a=None, attr=None, args_pos=None, args_dict=None, reshape=None):


    if attr is None:
        return a 
    else:
        if args_pos is None:
            args_pos = ()
        if args_dict is None:
            args_dict = {}


        if reshape is None:
            reshape = True

        if a is None:
            a = []
        a = np.array(a)
        a_shape = np.array(np.shape(a))

        N_i = np.product(a_shape)

        a = np.reshape(a, (N_i))

        r = np.array([a[i].__getattribute__(attr)(*args_pos, **args_dict) for i in range(N_i)])

        if reshape:
            return np.reshape(r, a_shape)
        else:
            return r

def d_attr(d=None, attr=None, args_pos=None, args_dict=None, array=None):

    if d is None:
        d = {}

    if attr is None:
        ls = [None for v in d.__getattribute__('keys')()]
    else:
        if args_pos is None:
            args_pos = ()
        if args_dict is None:
            args_dict = {}

        ls = [v for v in d.__getattribute__(attr)(*args_pos, **args_dict)]

    if array is None:
        array = True
    
    if array:
        return np.array(ls)
    else:
        return ls
    

def d_keys(d=None, array=None):
    return d_attr(d=d, attr='keys', array=array)

def d_vals(d=None, array=None):
    return d_attr(d=d, attr='values', array=array)


def d_sort_keys(d=None, array=None):

    if array is None:
        array = True

    _sort_keys = d_keys(d=d)[np.argsort(d_vals(d=d))]
    if array:
        return _sort_keys
    else:
        
        return [k for k in _sort_keys]
    
    
def d_sort_d(d=None, ret=None):

    sort_keys =  d_sort_keys(d=d, array=True)

    if ret=='list' or ret=='array':
        l =  [d[k] for k in sort_keys]
        if ret=='array':
            return np.array(l)
        else:
            return l
    else:
        return {k: d[k] for k in sort_keys}



def calc_val_1_0(app_val_1=None, app_val_1_default=None, val_1=None, val_0=None):


    if app_val_1 is None:
        if app_val_1_default is None:
            app_val_1_default = True
        app_val_1 = app_val_1_default

    if val_0 is None:
        val_0 = ""
    if val_1 is None:
        val_1 = ""

    if app_val_1:
        return val_1
    else:
        return val_0
    

def calc_val_1_0_N(app_val_1=None, app_val_1_default=None, val_1=None, val_0=None, N=None):


    if app_val_1 is None:
        if app_val_1_default is None:
            app_val_1_default = True
        app_val_1 = app_val_1_default

    if val_0 is None:
        val_0 = ""
    if val_1 is None:
        val_1 = ""

    app_val_1 = list_1D(app_val_1, I_max=N)
    val_0 = list_1D(val_0, I_max=N)
    val_1 = list_1D(val_1, I_max=N)

    val_1[~app_val_1] = val_0[~app_val_1]

    return val_1


def S_meta(S_func_i, S=None, args_pos=None, args_dict=None, reshape=None, ):

    if S is None:
        S = ""
    S = np.array(S)

    if args_pos is None:
        args_pos = []
    if args_dict is None:
        args_dict = {}

    S_shape = np.array(np.shape(S), dtype=int)
    N_i = np.product(S_shape, dtype=int)

    if len(S_shape)==0 or N_i==0:
        return make_iterable_array(S_func_i(S=S, *args_pos, **args_dict))[0]
    else:
        if reshape is None:
            reshape = True
        S = np.reshape(S, (N_i))
        
    R = np.array([S_func_i(S=S[i], *args_pos, **args_dict) for i in range(N_i)])

    if reshape:
        R_shape = np.concatenate([S_shape, np.array(np.shape(R), dtype=int)[1:]])
        R = np.reshape(R, R_shape)

    return R


def app_val_attr(S=None, attr=None, args_pos=None, args_dict=None, reshape=None):

    if S is None:
        S = ""

    if args_pos is None:
        args_pos = []
    if args_dict is None:
        args_dict = {}

    if attr is None:
        return S
    else:
        return S_meta(S_func_i=app_val_attr_i, S=S, args_dict=dict_key_rm(locals(), keys_rm=['S', 'reshape']), reshape=reshape)


def app_val_attr_i(S=None, attr=None, args_pos=None, args_dict=None):

    if S is None:
        S = ""
    if args_pos is None:
        args_pos = []
    if args_dict is None:
        args_dict = {}

    if attr is None:
        return S
    else:
        return S.__getattribute__(attr)(*args_pos, **args_dict)




def app_presuff(S=None, ps=None, pre=None, suff=None, app_null=None, reshape=None):

    if app_null is None:
        app_null = True

    if S is None:
        S = ""
    if ps is None:
        if pre is None:
            pre = ""
        if suff is None:
            suff = ""
        ps = [pre, suff]

    return S_meta(S_func_i=app_presuff_i, S=S, args_dict=dict_key_rm(locals(), keys_rm=['S', 'reshape']), reshape=reshape)



def check_presuff(S=None, ps=None, pre=None, suff=None, reshape=None):

    if S is None:
        S = ""
    if ps is None:
        if pre is None:
            pre = ""
        if suff is None:
            suff = ""
        ps = [pre, suff]

    return S_meta(S_func_i=check_presuff_i, S=S, args_dict=dict_key_rm(locals(), keys_rm=['S', 'reshape']), reshape=reshape)


def rm_presuff(S=None, ps=None, pre=None, suff=None, require=None, reshape=None):

    if require is None:
        require = np.repeat(False,2)
    if not np.array_equal(np.shape(require), np.array([2])):
        require = list_1D(require, I_max=2)

    if S is None:
        S = ""
    if ps is None:
        if pre is None:
            pre = ""
        if suff is None:
            suff = ""
        ps = [pre, suff]

    return S_meta(S_func_i=rm_presuff_i, S=S, args_dict=dict_key_rm(locals(), keys_rm=['S', 'reshape']), reshape=reshape)


def app_presuff_i(S=None, ps=None, pre=None, suff=None, app_null=None):

    if app_null is None:
        app_null = True

    if S is None:
        S = ""

    if not app_null and S=="":
        return S
    else:
        if ps is None:
            if pre is None:
                pre = ""
            if suff is None:
                suff = ""
            ps = [pre, suff]

        return f"{ps[0]}{S}{ps[1]}"


def check_presuff_i(S=None, ps=None, pre=None, suff=None):

    if S is None:
        S = ""
    if ps is None:
        if pre is None:
            pre = ""
        if suff is None:
            suff = ""
        ps = [pre, suff]

    return np.array([S.startswith(ps[0]), S.endswith(ps[1])])


def rm_presuff_i(S=None, ps=None, pre=None, suff=None, require=None):

    if require is None:
        require = np.repeat(False,2)
    if not np.array_equal(np.shape(require), np.array([2])):
        require = list_1D(require, I_max=2)

    if S is None:
        S = ""
    if ps is None:
        if pre is None:
            pre = ""
        if suff is None:
            suff = ""
        ps = [pre, suff]

    check = check_presuff_i(S=S, ps=ps)

    ps_i = np.array([0, len(S)], dtype=int) + np.all(check[require]) * check * np.array([len(ps[0]), -len(ps[1])], dtype=int)

    return S[ps_i[0]:ps_i[1]]


def calc_index_N(shape, N, reshape=None):

    shape = np.array(shape, dtype=int)

    if reshape is None:
        reshape = True

    N_shape = np.array(np.shape(N), dtype=int)
    N_i = np.product(N_shape, dtype=int)

    N = np.reshape(N, N_i)

    cp = np.flip(np.cumproduct(np.flip(shape)))
    cp_shift = np.concatenate([cp, [1]])[1:]

    
    index = np.remainder.outer(N, cp)//cp_shift

    if reshape:
        return np.reshape(index,  np.concatenate([N_shape, [shape.size]]))
    else:
        return index

def calc_N_index(shape, index):


    shape = np.array(shape, dtype=int)

    cp = np.flip(np.cumproduct(np.flip(shape)))
    cp_shift = np.concatenate([cp, [1]])[1:]

    return index @ cp_shift


def calc_shape_reshapes(shape=None, reshape=None, end_dims=None):


    if shape is None:
        if reshape is not None:
            shape = np.product(reshape)
        else:
            shape = np.array([1])
    if reshape is None:
        reshape = np.product(shape)

    if end_dims is None:
        end_dims = {}

    shapes = {
                'shape': {'': make_iterable_array(shape, dtype=int)},
                'reshape': {'': make_iterable_array(reshape, dtype=int)},
              }

    for shape in shapes:
        for D in end_dims:
            shapes[shape][D] = np.concatenate([shapes[shape][''], make_iterable_array(end_dims[D], dtype=int)])

    return shapes


def calc_reshape_val(val, shapes=None, shape=None, reshape=None, end_dims=None, do_reshape=None):

    if do_reshape is None:
        do_reshape = True

    if do_reshape:
        if shapes is None:
            shapes = calc_shape_reshapes(shape=shape, reshape=reshape, end_dims=end_dims)

        reshaped = False
        val_shape = make_iterable_array(np.shape(val), dtype=int)
        for D in shapes['shape']:
            if not reshaped:
                if np.array_equal(val_shape, shapes['shape'][D]):
                    val = np.reshape(val, shapes['reshape'][D])
                    reshaped = True
            if reshaped:
                break

    return val


def calc_reshape_dict(dic, shapes=None, shape=None, reshape=None, end_dims=None, no_reshape=None):

    if no_reshape is None:
        no_reshape = []
    no_reshape = make_iterable_array(no_reshape)
    
    if shapes is None:
        shapes = calc_shape_reshapes(shape=shape, reshape=reshape, end_dims=end_dims)

    return {field: calc_reshape_val(val=dic[field], shapes=shapes, do_reshape=~np.isin(field, no_reshape)) for field in dic}


def calc_train_t(train_t=None, train_T=None, X=None):

    train_t = make_iterable_array(train_t)[:2]

    if train_T is None:
        if X is not None:
            train_T = np.shape(X)[-1]
        else:
            train_T = np.max(np.abs(train_t))

    base_train_t = np.array([0,train_T])



    if np.size(train_t)==1:
        if train_t[0] is not None:
            if train_t[0]<0:
                base_train_t[0] = np.max([0, train_T+train_t[0]])
            else:
                base_train_t[1] = np.min([train_T, train_t[0]])
    else:
        train_t[train_t<0] += train_T
        base_train_t[0] = np.max([0, train_t[0]])
        base_train_t[1] = np.min([train_T, train_t[1]])

    return base_train_t


def calc_samp_t(X, train_t=None, train_T=None):

    train_t = calc_train_t(train_t=train_t, train_T=train_T, X=X)
    return X[...,train_t[0]:train_t[1]]


def calc_resets(reset=None, resets=None):

    if resets is None:
        resets = {}


    if reset is not None:
        for r in resets:
            resets[r] = reset

    return resets



def get_index(df, colval):

   return pd.Series(np.all(np.array([np.any(np.array([df[col].values == val for val in make_iterable_array(colval[col])]), axis=0) for col in colval]), axis=0), index=df.index)

def index_df(df, colval):
    return df[get_index(df=df, colval=colval)]

def dt_str(dt):
    return str(dt.strftime("%Y-%m-%d_%H%M%S"))


def invert_dict_lists(odict):

    idict = dict()
    for o in odict:
        for i in odict[o]:
            idict[i] = o
    return idict



def calc_brac(brac_type=None, add_lr=None, repeat=None, app_brac=None, phantom=None, plot=None):

    if plot is None:
        plot = False

    if phantom is None:
        phantom = False

    if brac_type is None:
        bracket_type = 'parentheses'

    if add_lr is None:
        add_lr = False

    if repeat is None:
        repeat = 1

    
    if np.isin(brac_type, ['']):
        brac = ['', '']
        add_lr = False
    elif np.isin(brac_type, ['parentheses', '()', '(', ')']):
        brac = ['(', ')']
    elif np.isin(brac_type, ['square', '[]', '[', ']']):
        brac = ['[', ']']
    elif np.isin(brac_type, ['angular', '<>', '<', '>']):
        brac = ['<', '>']
    elif np.isin(brac_type, ['curly', '{}', '{', '}']):
        brac = [f'\\{{', f'\\}}']
    else:
        brac = ['(', ')']

    if phantom:
        add_lr = False
        if plot:
            brac = np.repeat(f"\\:", 2)
        else:
            brac = capp_tex_fn(S=brac, tex_fn='phantom', app_tex_fn=True, app_null=None, reshape=None)

    if add_lr:
        lr = ['\\left', '\\right']
    else: 
        lr = ['', '']


    return calc_val_1_0_N(app_val_1=app_brac, app_val_1_default=True, val_1=[str_concatenate(np.repeat(f"{lr[k]}{brac[k]}", repeat)) for k in range(2)], val_0='', N=2)

 
 
    

def calc_brac_ps(brac_type=None, add_lr=None, app_brac=None, repeat=None):
    return calc_brac(brac_type=brac_type, add_lr=add_lr, app_brac=app_brac, repeat=repeat)



def calc_tex_fn_ps(tex_fn=None, app_tex_fn=None):

    if tex_fn is None:
        tex = ""

    if tex_fn=="":
        app_tex_fn = False

    return calc_val_1_0_N(app_val_1=app_tex_fn, app_val_1_default=True, val_1=[f"\\{tex_fn}{{", f"}}"], val_0='', N=2)


def calc_tex_vect_fn(pltf=None, use_tex=None, tex_vect=None):


    if tex_vect is None:
        if pltf is None:
            pltf = False
        if use_tex is None:
            use_tex = False

        if pltf and not use_tex:
            tex_vect = 'mathbf'
        else:
            tex_vect = "vect"

    return tex_vect



def calc_tex_vect_ps(tex_vect=None, app_tex_vect=None, pltf=None, use_tex=None):

    tex_vect = calc_tex_vect_fn(pltf=pltf, use_tex=use_tex, tex_vect=tex_vect)

    return calc_tex_fn_ps(tex_fn=tex_vect, app_tex_fn=app_tex_vect)


def calc_tex_num_fn(pltf=None, use_tex=None, tex_num=None):


    if tex_num is None:
        if pltf is None:
            pltf = False
        if use_tex is None:
            use_tex = False

        if pltf and not use_tex:
            tex_num = 'num'
        else:
            tex_num = 'num'

    return tex_num


def calc_tex_num_ps(tex_num=None, app_tex_num=None, pltf=None, use_tex=None):

    if app_tex_num is None:
        if pltf is None:
            pltf = False
        if use_tex is None:
            use_tex = False

        if pltf and not use_tex:
            app_tex_num = False
        else:
            app_tex_num = True

    tex_num = calc_tex_num_fn(pltf=pltf, use_tex=use_tex, tex_num=tex_num)

    return calc_tex_fn_ps(tex_fn=tex_num, app_tex_fn=app_tex_num)


def calc_tex_hphantom_fn(pltf=None, use_tex=None, tex_hphantom=None):


    if tex_hphantom is None:
        if pltf is None:
            pltf = False
        if use_tex is None:
            use_tex = False

        if pltf and not use_tex:
            tex_hphantom = 'hphantom'
        else:
            tex_hphantom = 'hphantom'

    return tex_hphantom


def calc_tex_hphantom_ps(tex_hphantom=None, app_tex_hphantom=None, pltf=None, use_tex=None):

    if app_tex_hphantom is None:
        if pltf is None:
            pltf = False
        if use_tex is None:
            use_tex = False

        if pltf and not use_tex:
            app_tex_hphantom = False
        else:
            app_tex_hphantom = True

    tex_hphantom = calc_tex_hphantom_fn(pltf=pltf, use_tex=use_tex, tex_hphantom=tex_hphantom)

    return calc_tex_fn_ps(tex_fn=tex_hphantom, app_tex_fn=app_tex_hphantom)



def calc_dlrs_ps(app_dlrs=None):

    return calc_val_1_0_N(app_val_1=app_dlrs, app_val_1_default=True, val_1='$', val_0='', N=2)



def capp_brac(S=None, brac_type=None, add_lr=None, app_brac=None, repeat=None, app_null=None, reshape=None):
    if app_null is None:
        app_null = False
    return app_presuff(S=S, ps=calc_brac_ps(**dict_key_rm(locals(), keys_rm=['S', 'app_null', 'reshape'])), app_null=app_null, reshape=reshape)
def check_brac(S=None, brac_type=None, add_lr=None, app_brac=None, reshape=None):
    return check_presuff(S=S, ps=calc_brac_ps(**dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)
def rm_brac(S=None, brac_type=None, add_lr=None, app_brac=None, reshape=None):
    return rm_presuff(S=S, ps=calc_brac_ps(**dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)


def capp_tex_fn(S=None, tex_fn=None, app_tex_fn=None, app_null=None, reshape=None):
    if app_null is None:
        app_null = False
    return app_presuff(S=S, ps=calc_tex_fn_ps(**dict_key_rm(locals(), keys_rm=['S', 'app_null', 'reshape'])), app_null=app_null, reshape=reshape)
def check_tex_fn(S=None, tex_fn=None, app_tex_fn=None, reshape=None):
    return check_presuff(S=S, ps=calc_tex_fn_ps(**dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)
def rm_tex_fn(S=None, tex_fn=None, app_tex_fn=None, reshape=None):
    return rm_presuff(S=S, ps=calc_tex_fn_ps(**dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)


def capp_tex_vect(S=None, tex_vect=None, app_tex_vect=None, app_null=None, reshape=None, pltf=None, use_tex=None):
    if app_null is None:
        app_null = False
    return app_presuff(S=S, ps=calc_tex_vect_ps(**dict_key_rm(locals(), keys_rm=['S', 'app_null', 'reshape'])), app_null=app_null, reshape=reshape)
def check_tex_vect(S=None, tex_vect=None, app_tex_vect=None, reshape=None, pltf=None, use_tex=None):
    return check_presuff(S=S, ps=calc_tex_vect_ps(**dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)
def rm_tex_vect(S=None, tex_vect=None, app_tex_vect=None, reshape=None, pltf=None, use_tex=None):
    return rm_presuff(S=S, ps=calc_tex_vect_ps(**dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)


def capp_tex_num(S=None, tex_num=None, app_tex_num=None, app_null=None, reshape=None, pltf=None, use_tex=None):
    if app_null is None:
        app_null = False
    return app_presuff(S=S, ps=calc_tex_num_ps(**dict_key_rm(locals(), keys_rm=['S', 'app_null', 'reshape'])), app_null=app_null, reshape=reshape)
def check_tex_num(S=None, tex_num=None, app_tex_num=None, reshape=None, pltf=None, use_tex=None):
    return check_presuff(S=S, ps=calc_tex_num_ps(**dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)
def rm_tex_num(S=None, tex_num=None, app_tex_num=None, reshape=None, pltf=None, use_tex=None):
    return rm_presuff(S=S, ps=calc_tex_num_ps(**dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)

def capp_tex_hphantom(S=None, tex_hphantom=None, app_tex_hphantom=None, app_null=None, reshape=None, pltf=None, use_tex=None):
    if app_null is None:
        app_null = False
    return app_presuff(S=S, ps=calc_tex_hphantom_ps(**dict_key_rm(locals(), keys_rm=['S', 'app_null', 'reshape'])), app_null=app_null, reshape=reshape)
def check_tex_hphantom(S=None, tex_hphantom=None, app_tex_hphantom=None, reshape=None, pltf=None, use_tex=None):
    return check_presuff(S=S, ps=calc_tex_hphantom_ps(**dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)
def rm_tex_hphantom(S=None, tex_hphantom=None, app_tex_hphantom=None, reshape=None, pltf=None, use_tex=None):
    return rm_presuff(S=S, ps=calc_tex_hphantom_ps(**dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)


def capp_dlrs(S=None, app_dlrs=None, app_null=None, reshape=None):
    if app_null is None:
        app_null = False
    return app_presuff(S=S, ps=calc_dlrs_ps(**dict_key_rm(locals(), keys_rm=['S', 'app_null', 'reshape'])), app_null=app_null, reshape=reshape)
def check_dlrs(S=None, app_dlrs=None, reshape=None):
    return check_presuff(S=S, ps=calc_dlrs_ps(**dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)
def rm_dlrs(S=None, app_dlrs=None, reshape=None):
    return rm_presuff(S=S, ps=calc_dlrs_ps(**dict_key_rm(locals(), keys_rm=['S', 'reshape'])), reshape=reshape)


def capp_dlrs_tex_num(S=None, app_dlrs=None, app_tex_num=None, app_null=None):

    return capp_dlrs(S=capp_tex_num(S=S, app_tex_num=app_tex_num, app_null=app_null), app_dlrs=app_dlrs, app_null=app_null)




def calc_fsf(val):

    return np.floor(np.log(np.abs(val)) / np.log(10))


def scif_convert(val, se=None, scif_pow10=None, dp=None):

    if scif_pow10 is None:
        force_pow10=False
        if val==0 or np.isnan(val) or np.isinf(val):
            scif_pow10 = 0
        else:
            scif_pow10 = int(calc_fsf(val=val))
    else:
        force_pow10=True
        scif_pow10 = int(scif_pow10)

    if scif_pow10<-307:
        if dp is None:
            dp_buff = 10
        else:
            dp_buff = np.min([dp])
        scif_val = float(f"{val:+.{dp_buff}E}"[:dp_buff+3])

    else:
        scif_val = val*10**(-scif_pow10)
    if dp is not None:
        if not force_pow10:
            #if f"{scif_val:.{dp}f}"==f"{10:.{dp}f}":
            #if f"{scif_val:.{dp}f}"[:2]==f"10":
            if scif_val + 0.5*10**(-dp) >= 10:
                scif_val /= 10
                scif_pow10 += 1
    if se==None:
        return [scif_val, scif_pow10]
    else:
        scif_se = se*10**(-scif_pow10)
        return [scif_val, scif_pow10, scif_se]


def scif_string(val, se=None, scif_pow10=None, dp=None, dp_buff=None, app_dlrs=None, app_tex_num=None, val_plus=None, exp_plus=None, exp_zeros=None, brac_type=None, add_lr=None, return_dict=None, se_space=None, plot=None, phantom_sign_str=None, phantom_sign=None, phantom_brac=None, sign_inside_brac=None, scif_string_format=None, mathmode=None):
    
    
    if plot is None:
        plot = False

    if mathmode is None:
        mathmode = True

    if phantom_sign_str is None:
        if plot:
            phantom_sign_str = "\\:\\:"
        else:
            phantom_sign_str = f"\\phantom{{+}}"


    if dp_buff is None:
        dp_buff = 1

    if dp is None:
        dp_buff = 1
        if scif_pow10 is None:
            dp = dp_buff
        else:
            fsf = calc_fsf(val)
            if np.isnan(fsf):
                fsf = 0
            dp = dp_buff + int(np.max([-fsf, -dp_buff]))
    if val_plus is None:
        val_plus = False

    if phantom_sign is None:
        phantom_sign = False

    if phantom_brac is None:
        phantom_brac = False

    if val_plus:
        phantom_sign = False

    if sign_inside_brac is None:
        if phantom_sign: 
            sign_inside_brac = False
        else:
            sign_inside_brac = False

    if exp_plus is None:
        exp_plus = True

    if exp_zeros is None:
        if np.isin(scif_string_format, ['e', 'E']):
            exp_zeros = 2
        else:
            exp_zeros = 1

    if app_dlrs is None:
        if mathmode:
            app_dlrs = True
        else:
            app_dlrs = False
    if app_tex_num is None:
        app_tex_num = False

    if val_plus:
        val_plus_str = f"+"
    else:
        val_plus_str = f""

    if exp_plus:
        #exp_zeros += 1
        exp_plus_str = f"+"
    else:
        exp_plus_str = f""

    
    

    scif_comp = scif_convert(val, se=se, scif_pow10=scif_pow10, dp=dp)

    show_se = True
    if se is None:
        show_se = False
    elif np.isnan(se) or np.isnan(scif_comp[2]):
        show_se = False


    str_dict = {v: None for v in ['sign', 'val', 'abs_val', 'se', 'exp_sign' 'exp', 'brac', 'dollar', 'full']}


    if add_lr is None:
        if mathmode:
            add_lr = True
        else:
            add_lr = False

    if brac_type is None:
        brac_type = "()"

    if scif_comp[1] == 0 or not show_se:
        if phantom_brac:
            str_dict['brac'] = calc_brac(brac_type=brac_type, add_lr=add_lr, phantom=phantom_brac, plot=plot)
        else:
            str_dict['brac'] = ["", ""]
    else:
        str_dict['brac'] = calc_brac(brac_type=brac_type, add_lr=add_lr)



    
    
        
    if scif_comp[1] == 0:
        str_dict['exp_sign'] = ""
        str_dict['exp'] = ""
    else:
        if scif_comp[1]<0:
            str_dict['exp_sign'] = '-'
        else:
            if exp_plus:
                str_dict['exp_sign'] = '+'
            else:
                str_dict['sign'] = f""


        abs_exp = np.abs(scif_comp[1])
        if np.isin(scif_string_format, ['e', 'E']):
            if mathmode:
                str_dict['exp'] =  f"\\mathrm{{{scif_string_format}{str_dict['exp_sign']}}}{{{abs_exp:0{exp_zeros}d}}}"
            else:
                str_dict['exp'] =  f"{scif_string_format}{str_dict['exp_sign']}{{{abs_exp:0{exp_zeros}d}}}"
        else:
            str_dict['exp'] =  f" \\times 10^{{{str_dict['exp_sign']}{abs_exp:0{exp_zeros}d}}}"


    if not show_se:
        str_dict['se'] = ""
    else: 
        if se_space is None:
            if mathmode:
                se_space = " "
            else:
                se_space = ""
        if se_space=="":
            post_se_pm = f"{{}}"
        else:
            post_se_pm = "" 

        if mathmode:
            se_pm = '\\pm'
        else:
            se_pm = '\\textpm'

        str_dict['se'] =  f"{se_space}{se_pm}{post_se_pm}{se_space}{scif_comp[2]:.{dp}f}"


    str_dict['val'] = f"{scif_comp[0]:{val_plus_str}.{dp}f}"
    str_dict['abs_val'] = f"{np.abs(scif_comp[0]):.{dp}f}"
    if val<0:
        str_dict['sign'] = '-'
    else:
        if val_plus:
            str_dict['sign'] = '+'
        elif phantom_sign:
            str_dict['sign'] = phantom_sign_str
        else:
            str_dict['sign'] = f""

    
    str_dict['app_dlrs'] = app_dlrs
    str_dict['app_tex_num'] = app_tex_num

    if sign_inside_brac:
        s = f"{str_dict['brac'][0]}{str_dict['sign']}{str_dict['abs_val']}{str_dict['se']}{str_dict['brac'][1]}{str_dict['exp']}"
    else:
        s = f"{str_dict['sign']}{str_dict['brac'][0]}{str_dict['abs_val']}{str_dict['se']}{str_dict['brac'][1]}{str_dict['exp']}"


    str_dict['full'] = capp_dlrs_tex_num(S=s,  app_dlrs=str_dict['app_dlrs'], app_tex_num=str_dict['app_tex_num'])


    if return_dict:
        return str_dict
    else:
        return str_dict['full']


    
def vector_string(val, scif_pow10=None, dp=None, dp_buff=None, app_dlrs=None, app_tex_num=None, val_plus=None, exp_plus=None, exp_zeros=None, brac_type=None, add_lr=None, fill_str=None, fill_str_void=None, simple_scalar=None, scif_vector_simple=None):

    val = make_iterable_array(val)
    if np.size(val)==1:
        scalar = True
    else:
        scalar = False

    if fill_str is None:
        fill_str = ','
    if fill_str_void is None:
        fill_str_void = -1

    if brac_type is None:
        brac_type = "[]"

    if simple_scalar is None:
        simple_scalar = True

    if scalar and simple_scalar:
        brac_type = ""


    if scif_vector_simple is None:
        scif_vector_simple = True

    if dp_buff is None:
        dp_buff = 1

    if app_dlrs is None:
        app_dlrs = False
    if app_tex_num is None:
        app_tex_num = False

    if not scalar and scif_vector_simple:
        fsf = calc_fsf(val)
        fsf_nonan_bool = ~np.isnan(fsf)
        fsf_int = np.array(fsf[fsf_nonan_bool], dtype=int)

        if np.sum(fsf_nonan_bool)>0:
            fsf_minmax = [np.min(fsf_int), np.max(fsf_int)]
        else:
            fsf_minmax = np.zeros(2, dtype=int)

        force_pow10 = True
        if scif_pow10 is None:
            force_pow10 = False
            scif_pow10 = fsf_minmax[-1]
        if dp is None:
            #dp = np.max([dp_buff, scif_pow10-fsf_minmax[0]])
            dp = dp_buff + int(np.max([(scif_pow10-fsf_minmax[0]), -dp_buff]))

        if not force_pow10:
            if np.any(val*10**(-scif_pow10) + 0.5*10**(-dp) >= 10):
                scif_pow10 += 1

        str_dict_list = [scif_string(val=v, se=None, scif_pow10=scif_pow10, dp=dp, app_dlrs=False, val_plus=val_plus, exp_plus=exp_plus, exp_zeros=exp_zeros, brac_type='none', return_dict=True) for v in val]
        str_list = [s['val'] for s in str_dict_list]
        post = str_dict_list[0]['exp']

    else:
        str_list = [scif_string(val=v, se=None, scif_pow10=scif_pow10, dp=dp, app_dlrs=False, app_tex_num=app_tex_num, val_plus=val_plus, exp_plus=exp_plus, exp_zeros=exp_zeros, brac_type='none', return_dict=False) for v in val]
        post = ""

        
    components = str_concatenate(str_list=str_list, fill_str=fill_str, fill_str_void=fill_str_void)
                                       
    brac = calc_brac(brac_type=brac_type, add_lr=add_lr)

    return capp_dlrs(S=f"{brac[0]}{components}{brac[1]}{post}", app_dlrs=app_dlrs)



def calc_str_val_sub_sup(   val=None, display_I_s=None, brac_type=None, add_lr=None, 
                            symb=None, sub=None, sup=None, super_I_i=None, super_I_cat=None, super_I_i_pre=None, super_I_i_post=None,
                            scif_pow10=None, dp=None, dp_buff=None, val_plus=None,
                            force_vector=None, show_symb=None, show_val=None):

    
    if show_symb is None:
        show_symb = True
    if show_val is None:
        show_val = True

    if val_plus is None:
        val_plus = False
    
    
    
    if brac_type is None:
        brac_type = ''
    if add_lr is None:
        add_lr = False



    if val is None:
        val = np.array([])

    

    if display_I_s is None:
        display_I_s = np.arange(np.size(val))


    ret_dict = {
                    'sign': f"",
                    'val_plus': val_plus,
                    'display_I_s': display_I_s,
                    '=': '=',
                }
    

    ret_dict['brac'] = calc_brac(brac_type=brac_type, add_lr=add_lr)

    



    if symb is None:
        symb = "a"
    ret_dict['symb'] = symb

    


    if ret_dict['symb']=="":
        show_symb = False


    if show_symb and show_val:
        ret_dict['='] = '='
    else:
        ret_dict['='] = ''

    


    if sub is None:
        sub = f""
    if sup is None:
        sup = f""


    if sub=="":
        ret_dict['sub'] = f""
    else:
        ret_dict['sub'] = f"_{{{sub}}}"
    if sup=="":
        ret_dict['sup'] = f""
    else:
        ret_dict['sup'] = f"^{{{sup}}}"


    if super_I_i is None:
        super_I_i = 1+np.arange(np.max(display_I_s))
    super_I_i = make_iterable_array(super_I_i)
    
    if super_I_cat is None:
        super_I_cat = "i"

    


    if super_I_i_pre is None:
        if sup=="" or np.all(super_I_i[display_I_s]=="") or super_I_cat=="":
            super_I_i_pre = ""
        else:
            super_I_i_pre = ","
    if super_I_i_post is None:
        if super_I_i_pre==f"^{{":
            super_I_i_post = f"}}"
        else:
            super_I_i_post = ""

    ret_dict['super_I_i'] = super_I_i
    ret_dict['super_I_cat'] = super_I_cat



    if force_vector is None:
        force_vector = False


    ret_dict['sign'] = f""
    if np.size(display_I_s)==0:
        ret_dict['='] = ''
        ret_dict['str_val'] = ''
    else:
        if np.size(display_I_s)>1:
            if np.all(val[display_I_s]==val[display_I_s[0]]):        
                ret_dict['val'] = val[display_I_s[0]]
            elif val[display_I_s[0]] == -val[display_I_s[1]]:
                ret_dict['sup'] = f"^{{{sup}{super_I_i_pre}{super_I_cat}{super_I_i_post}}}"
                ret_dict['sign'] = f"\\mp"
                ret_dict['val_plus'] = False
                ret_dict['val'] = -val[display_I_s[0]]
            else:
                force_vector = True
                ret_dict['val'] = val[display_I_s]
        else:
            ret_dict['val'] = val[display_I_s[0]]
            ret_dict['sup'] = f"^{{{sup}{super_I_i_pre}{super_I_i[display_I_s[0]]}{super_I_i_post}}}"


        ret_dict['str_val'] = vector_string(val=ret_dict['val'], scif_pow10=scif_pow10, dp=dp, dp_buff=dp_buff, val_plus=ret_dict['val_plus'], simple_scalar=True, app_dlrs=False)



    if force_vector:
        if ret_dict['symb']!="":
            ret_dict['symb'] = f"\\mathbf{{{ret_dict['symb']}}}"


    ret_dict['str_symb'] = f"{ret_dict['symb']}{ret_dict['sub']}{ret_dict['sup']}"

    if show_symb:
        str_symb = ret_dict['str_symb']
    else:
        str_symb = ""
    if show_val:
        str_val = ret_dict['str_val']
    else:
        str_val = ""


    ret_dict['str'] = f"{ret_dict['brac'][0]}{str_symb}{ret_dict['=']}{ret_dict['sign']}{str_val}{ret_dict['brac'][1]}"
        
        

    return ret_dict







def str_concatenate(str_list, fill_str=None, fill_str_void=None):

    S = len(str_list)

    if fill_str is None:
        fill_str = f""
    if fill_str_void is None:
        fill_str_void = []
    fill_str_void = make_iterable_array(fill_str_void)
    fill_str_void[fill_str_void<0] = fill_str_void[fill_str_void<0] + S

    outstr = f""

    for s in range(S):
        outstr += str(str_list[s])
        if not np.isin(s, fill_str_void):
            outstr += fill_str
            
    return outstr


def multiple_round(val=None, factor=None, buffer=None, buffer_sign=None, round_type=None):

    if val is None:
        val = 0
    if factor is None:
        factor = 1
    if buffer is None:
        buffer = 0
    if round_type is None:
        round_type = 'round'

    if buffer_sign is None:
        if round_type=='floor':
            buffer_sign=-1
        else:
            buffer_sign=1

    if round_type == 'round_math':
        return factor*int(np.floor((val/factor) + buffer + 0.5))
    else:
        return factor*int(np.__dict__[round_type]((val/factor)+buffer_sign*buffer))



#### hawkes


def dropint(array, N_0=True, N_s=1000):

    array_0 = np.copy(array)
    check = np.full(np.shape(array), True)
    for k in range(N_s):
        array[check] -= 1
        check = np.isin(array, array_0)
        if not np.any(check):
            break
        if k == N_s-1:
            print("Limit reached!")

    if N_0:
        subzero = array < 0
        if np.any(subzero):
            array[subzero] = np.where(~np.isin(np.array(range(np.max(array))), array))[0][0]
        
    return array





def xint_dist(dist=None, x=None, loc=None, scale=None, df=None, nc=None):

    if dist is None:
        dist='norm'
    if x is None:
        x=0
    if loc is None:
        loc=0
    if scale is None:
        scale=1


    if dist=="norm":
        return -(loc*scipy.special.erf((loc - x)/(np.sqrt(2)*scale))/2) - (scale*np.exp(-(((loc - x)/scale)**2)/2)/np.sqrt(2*np.pi))
    else:
        if df is None:
            df=100
        t = ((x-loc)/scale)
        if type(x) == type(np.array([])):
            inf = np.abs(x) == np.infty
            t[inf] = np.sign(x[inf])*1E+3
        else:
            if np.abs(x)==np.infty:
                t = np.sign(x)*1E+3
        if dist=="t":
            gamma_coef = (scipy.special.gamma((1+df)/2)/(np.sqrt(df*np.pi)*scipy.special.gamma(df/2)))
            return gamma_coef*(loc*t*scipy.special.hyp2f1(1/2, (1+df)/2, 3/2, -(t**2)/df) + scale*(df/(1-df))*(((((t**2)+df)/df)**(-(df-1)/2)) - 1))
        else:
            if nc is None:
                nc=0
            return np.full(np.size(x), np.nan)

def bound_xint_dist(dist=None, lb=None, ub=None, loc=None, scale=None, df=None, nc=None):
    if lb is None:
        lb = -np.infty
    if ub is None:
        ub = np.infty
    return xint_dist(dist=dist, x=ub, loc=loc, scale=scale, df=df, nc=nc) - xint_dist(dist=dist, x=lb, loc=loc, scale=scale, df=df, nc=nc)



def calc_AIC(log_L, N_params):
    return 2*(N_params - log_L)

def calc_BIC(log_L, N_params, N_data):
    return np.log(N_data)*N_params - 2*log_L

def calc_IC(log_L, N_params, N_data=None, IC=None):

    if IC is None:
        IC = 'AIC'

    if IC=='BIC':
        return calc_BIC(log_L=log_L, N_params=N_params, N_data=N_data)
    else:
        return calc_AIC(log_L=log_L, N_params=N_params)
    

def calc_IC_list(log_L, N_params, N_data):
        return log_L, N_params, N_data, calc_AIC(log_L, N_params), calc_BIC(log_L, N_params, N_data)

def IC_dict(log_L=None, N_params=None, N_data=None):

    if N_params is None or log_L is None:
        AIC = None
        BIC = None
    else:
        AIC = calc_AIC(log_L, N_params)
        if N_data is None:
            BIC = None
        else:
            BIC = calc_BIC(log_L, N_params, N_data)

    IC =   {
                "log_L": log_L,
                "N_params": N_params,
                "N_data": N_data,
                "AIC": AIC,
                "BIC": BIC,
            }
    return IC


def exceedance_transform(data, u, u_dir="right", u_zero=0):
    if u_dir=="left":
        return -((data-u_zero)-u)
    elif u_dir=="right":
        return ((data-u_zero)-u)
    elif u_dir=="abs":
        return (np.abs(data-u_zero)-u)

def exceedance_transform_inv(data, u, u_dir="right", u_zero=0, sign=None):
    if u_dir=="left":
        return (u-data) + u_zero 
    elif u_dir=="right":
        return (u+data) + u_zero 
    elif u_dir=="abs":
        if sign is None:
            sign=np.ones_like(data)
        return sign*(u+data) + u_zero

def stack(data_1, data_0=0, size=None):

    if size is None:
        size = data_1.size
    stack = np.full((size, 4), np.nan)
    stack[:, 1] = data_0
    stack[:, 2] = data_1

    return stack

def make_iterable_array(values, shape=None, dtype=None, as_list=None):

    if as_list is None:
        as_list = False

    if np.isin([type(values)], [np.ndarray, np.matrix]) == False:
        values = np.array(values)
    if type(shape) == type(None):
        shape = values.size
    if np.product(shape) != values.size:
            raise ValueError("Mismatch between size of values and specified output shape")

    if dtype is not None:
        values = np.array(np.resize(values, shape), dtype=dtype)
    else:
        values = np.resize(values, shape)

    if as_list:
        return [v for v in values]
    else:
        return values

def calc_length(val=None, shape=None):

    if shape is None:
        shape = np.shape(val)
    if len(shape)==0:
        length = 1
    else:
        length = shape[0]

    return length



def fill_master_shape(val=None, master_shape=None):

    if val is None:
        val = 0

    if master_shape is None:
        master_shape = np.array([1])

    val_shape = np.array(np.shape(val))

    master_dim = len(master_shape)
    val_dim = len(val_shape)
    


    dim_diff = val_dim - master_dim

    ret_val = None
    repeat_fill = True
    if val_dim<=0:
        repeat_fill = False
        if np.size(val)>1:
            ret_val = np.full(master_shape, list_1D(val, I_max=master_shape[-1]))
        else:
            ret_val = np.full(master_shape, val)

    elif dim_diff==0:
        if np.array_equal(val_shape, master_shape):
            repeat_fill = False
            ret_val = val


    elif dim_diff < 0:

        if np.array_equal(val_shape, master_shape[np.abs(dim_diff):]):
            repeat_fill = False
            I = np.product(master_shape[:np.abs(dim_diff)])
            ret_val = np.reshape(np.array([val for i in range(I)]), master_shape)

        elif np.array_equal(val_shape, master_shape[:np.abs(dim_diff)]):
            repeat_fill = False
            I = np.product(master_shape[np.abs(dim_diff):])
            ret_val = np.reshape(np.repeat(val, I), master_shape)


    elif dim_diff > 0:

        if np.array_equal(val_shape[np.abs(dim_diff):], master_shape):
            repeat_fill = False
            I = np.product(val_shape[:np.abs(dim_diff)])
            ret_val = np.reshape(val, (I, master_shape))[0,:]

        elif np.array_equal(val_shape, master_shape[:np.abs(dim_diff)]):
            repeat_fill = False
            I = np.product(val_shape[np.abs(dim_diff):])
            ret_val = np.reshape(val, (master_shape,I))[...,0]



    if repeat_fill:
        I_master = np.product(master_shape)
        I_val = np.product(val_shape)
        I_eff = 1 + (I_master-1)//I_val

        ret_val = np.reshape(np.hstack(np.array([val for i in range(I_eff)]))[:I_master], master_shape)

            
    
    return ret_val




def list_1D(val, I_max=None):


    val = make_iterable_array(val)
    val_size = np.size(val)

    if I_max is None:
        I_max = val_size

    if I_max > 0 and val_size > 0:
        I = 1 + (I_max-1)//val_size
        return np.concatenate([val for i in range(I)])[:I_max]
    else:
        return np.array([], dtype=val.dtype)
    



def is_types(val, types=None):

    if types is None:
        types = []

    type_val = type(val)
    #return np.isin(type_val, types)
    return np.any([type_val==types[i] for i in range(len(types))])

def is_types_index(val):

    types_index = [list, np.ndarray, np.matrix]

    if True:
        return is_types(val=val, types=types_index)
    else:
        type_val = type(val)
        return type_val==list or type_val==np.ndarray or type_val==np.matrix


def index_I_max(i=None, I_max=None, flip=None, circle=None):

    if flip is None:
        flip = False

    if circle is None:
        circle = True

    if flip:
        i = -(1+i)

    if circle and I_max is not None:
        return i%I_max
    else:
        return i
        



def list_2D(val, I_max=None, J_max=None, set_J_max=None, mode_1D=None, as_array=None, flip_i=None, flip_j=None):

    if flip_i is None:
        flip_i = False
    if flip_j is None:
        flip_j = False

    if mode_1D is None:
        mode_1D = 'i'

    shape = np.shape(val)
    _I_max_val = calc_length(shape=shape)

    if I_max is None:
        I_max = _I_max_val


    if set_J_max is None:
        if J_max is None:
            set_J_max = True
        else:
            set_J_max = False

    if J_max is None:
        if len(shape)>=2:
            J_max = shape[1]
            #set_J_max = False

    J_max = make_iterable_array(J_max)

    _I_max_J = np.size(J_max)
    _J_max = np.zeros(I_max, dtype=int)

    if is_types_index(val=val):
        val_eff = [None for i in range(I_max)]
        for i in range(I_max):
            shape_i = np.shape(val[index_I_max(i=i, I_max=_I_max_val, flip=flip_i, circle=True)])
            _J_max[i] = calc_length(shape=shape_i)
            if set_J_max:
                J_max[index_I_max(i=i, I_max=_I_max_J, flip=False, circle=True)] = _J_max[i]
            if is_types_index(val=val[index_I_max(i=i, I_max=_I_max_val, flip=flip_i, circle=True)]):
                if np.size(val[index_I_max(i=i, I_max=_I_max_val, flip=flip_i, circle=True)])==J_max[index_I_max(i=i, I_max=_I_max_J, flip=False, circle=True)]:
                    if flip_j:
                        val_eff[i] = np.flip(val[index_I_max(i=i, I_max=_I_max_val, flip=flip_i, circle=True)])
                    else:
                        val_eff[i] = val[index_I_max(i=i, I_max=_I_max_val, flip=flip_i, circle=True)]
                else:
                    val_eff[i] = [val[index_I_max(i=i, I_max=_I_max_val, flip=flip_i, circle=True)][index_I_max(i=j, I_max=_J_max[i], flip=flip_j, circle=True)] for j in range(J_max[index_I_max(i=i, I_max=_I_max_J, flip=False, circle=True)])]
            else:
                if np.isin(mode_1D, ['column', 'columns', 'j']):
                    val_eff[i] = [val[index_I_max(i=j, I_max=_J_max[i], flip=flip_j, circle=True)] for j in range(J_max[index_I_max(i=i, I_max=_I_max_J, flip=False, circle=True)])] 
                else:
                    val_eff[i] = [val[index_I_max(i=i, I_max=_I_max_val, flip=flip_i, circle=True)] for j in range(J_max[index_I_max(i=i, I_max=_I_max_J, flip=False, circle=True)])] 
    else:
        if set_J_max:
            J_max[0] = 1 
        val_eff = [[val for j in range(J_max[index_I_max(i=i, I_max=_I_max_J, flip=False, circle=True)])] for i in range(I_max)]


    if as_array is None:
        if np.isin(type(val), [np.ndarray, np.matrix]):
            as_array = True
        else:
            as_array = False

    if as_array:
        return np.array(val_eff)
    else:
        return val_eff
    
    
    
def merge_arrays(arrays=None, axis=None, string=None, null=None, dtype=None):

    if axis is None:
        axis = 0

    if arrays is None:
        arrays = [np.array([])]

    N_a = len(arrays)
    
    
    dims = np.array([len(np.shape(arrays[a])) for a in range(N_a)])
    shapes = np.zeros((N_a, np.max(dims)), dtype=int)
    for a in range(N_a):
        if dims[a]<2:
            arrays[a] = list_2D(arrays[a], as_array=True, I_max=1, J_max=1)
            dims[a] = 2
    
        arrays[a] = np.moveaxis(arrays[a], source=axis, destination=-1)
        shapes[a,:dims[a]] = np.array(np.shape(arrays[a]))

    dtypes = np.array([arrays[a].dtype for a in range(N_a)])


    shapes_axis  = np.cumsum(np.concatenate([[0], shapes[:,-1]]))

    shape_merge = np.max(shapes, axis=0)
    shape_merge[-1] = np.sum(shapes[:,-1])

    N_i = np.product(shape_merge[:-1])
    ar_N = np.arange(N_i)

    if string is None:
        if np.all('<U'==np.array(dtypes, dtype='<U2')):
            string = True
        else:
            string = False

    

    if dtype is None:
        if string:
            dtype = np.max(dtypes)
        elif np.all(dtypes)==int:
            dtype = int
        else:
            dtype = float

    if null is None:
        if string:
            null = ''
        elif dtype==int:
            null = 0
        else:
            null = np.nan
            

    merge_array = np.full(shape=shape_merge, fill_value=null, dtype=dtype)
    axis_intervals = np.cumsum(np.concatenate([[0], shapes[:,-1]]))

    for a in range(len(arrays)):
        a_bool = np.all(calc_index_N(shape=shape_merge[:-1], N=ar_N) < shapes[a,:-1], axis=1)
        merge_array[a_bool,axis_intervals[a]:axis_intervals[1+a]] = arrays[a]


    return np.moveaxis(merge_array, source=-1, destination=axis)



def A_rm(A, A_in=None, A_rm=None):

    A = make_iterable_array(A)

    if A_in is not None:
        A_in = make_iterable_array(A_in)
        A_in_bool = np.isin(A, A_in)
    else:
        A_in_bool = np.repeat(True, A.size)

    if A_rm is not None:
        A_rm = make_iterable_array(A_rm)
        A_rm_bool = ~np.isin(A, A_rm)
    else:
        A_rm_bool = np.repeat(True, A.size)

    return A[np.logical_and(A_in_bool, A_rm_bool)]




def dict_key_rm(dic, keys=None, keys_in=None, keys_rm=None, None_rm=None, ret=None, repeat=None, make_iterable=None, ignore_index=None, index=None, force_index=None):

    dic_keys = d_keys(d=dic, array=True)
    if keys is None:
        keys = dic_keys
    if keys_in is None:
        keys_in = dic_keys

    if None_rm is None:
        None_rm = True

    if None_rm:
        None_keys = dic_keys[np.array([dic[k] is None for k in dic_keys])]
        if None_keys.size>0:
            if keys_rm is None:
                keys_rm = None_keys
            else:
                keys_rm = np.concatenate([keys_rm, None_keys])

    keys = A_rm(A=keys, A_in=keys_in, A_rm=keys_rm)

    if np.size(keys)==0:
        if ret=='keys':
            return keys
        elif ret=='concat':
            return np.concatenate([[]])
        else:
            ret_dict = {}
            if ret=='DataFrame':
                return pd.DataFrame(ret_dict)
            else:
                return ret_dict
    else:
        df_size = np.size(dic[keys[0]])

        if ret=='keys':
            return keys
        elif ret=='concat':
            return np.concatenate([dic[k] for k in keys])
        else:

            if ignore_index is None:
                ignore_index = False
                if type(dic)==pd.core.frame.DataFrame or type(dic)==pd.core.frame.Series:
                    ignore_index = True

            if make_iterable is None:
                make_iterable = False
                if (type(dic)==pd.core.frame.DataFrame or type(dic)==pd.core.frame.Series):
                    if ignore_index:
                        make_iterable = True

            if repeat is not None:
                ret_dict = {k: make_iterable_array(np.repeat(dic[k], repeats=repeat)) for k in keys}
            elif make_iterable:
                ret_dict = {k: make_iterable_array(dic[k]) for k in keys}
            else:
                ret_dict = {k: dic[k] for k in keys}


            if ret=='DataFrame':
                if index is None:
                    index = np.arange(df_size)
                    if not ignore_index:
                        if type(dic)==pd.core.frame.DataFrame or type(dic)==pd.core.frame.Series:
                            if np.isin(type(dic.index[0]), [str, np.str_]):
                                index = [dic.name]
                            else:
                                index = dic.index
                    
                if force_index is None:
                    force_index = False
                    if not ignore_index:
                        if df_size>1:
                            force_index = True

                if force_index:
                    return pd.DataFrame(ret_dict, index=index)
                else:
                    return pd.DataFrame(ret_dict)
            else:
                return ret_dict
            

def dict_collapse(d=None, subdictkeys=None):

    if d is None:
        d = dict()

    if subdictkeys is None:
        subdictkeys = []
    subdictkeys = make_iterable_array(subdictkeys)

    return {**dict_key_rm(d, keys_rm=subdictkeys), **{b: d[a][b] for a in subdictkeys for b in d[a]}}





def dict_pre_suff(dic, pre=None, suff=None):

    if pre is None and suff is None:
        return dic
    else:
        if pre is None:
            pre = ''
        if suff is None:
            suff = ''
        return {f"{pre}{f}{suff}": dic[f] for f in dic}


def return_args(args, cat=None, cat_rm=None, list_rm=None):

    if cat is None:
        cat = True

    if cat==True:
        return np.concatenate([args[c] for c in args])
    elif cat==False:
        return args
    else:
        return A_rm(A=dict_key_rm(dic=args, keys=cat, keys_rm=cat_rm, ret='concat'), A_rm=list_rm)
    

def calc_log_L_eval_dict(N_data=None, N_params=None, N_params_free=None, log_L=None, AIC=None, AIC_free=None, BIC=None, BIC_free=None):

    _locals = locals()
    return _locals

def calc_log_L_eval_dict_complete(N_data=None, N_params=None, N_params_free=None, log_L=None, AIC=None, AIC_free=None, BIC=None, BIC_free=None,
                                  complete=None, log_L_eval_dict=None):

    if log_L_eval_dict is None:
        log_L_eval_dict = calc_log_L_eval_dict(N_data=N_data, N_params=N_params, N_params_free=N_params_free, log_L=log_L, AIC=AIC, AIC_free=AIC_free, BIC=BIC, BIC_free=BIC_free)

    if complete is None:
        complete = True

    if complete:
        if log_L_eval_dict[f'log_L'] is not None:
            free_suff = ['', '_free']
            if log_L_eval_dict[f'N_data'] is None:
                IC_list = ['AIC']
            else:
                IC_list = ['AIC', 'BIC']
        
            for free in free_suff:
                if log_L_eval_dict[f'N_params{free}'] is not None:
                    for IC in IC_list: 
                        if log_L_eval_dict[f'{IC}{free}'] is None:
                            log_L_eval_dict[f'{IC}{free}'] = calc_IC(IC=IC, log_L=log_L_eval_dict[f'log_L'], N_params=log_L_eval_dict[f'N_params{free}'], N_data=log_L_eval_dict[f'N_data'])

    return log_L_eval_dict



def calc_log_L_eval_dict_pre_suff(N_data=None, N_params=None, N_params_free=None, log_L=None, AIC=None, AIC_free=None, BIC=None, BIC_free=None,
                                  log_L_eval_dict=None, complete=None, push_free=None, free_rm=None, None_rm=None, pre=None, suff=None, keys_rm=None):
    
    free_list = ['N_params', 'AIC', 'BIC']

    if log_L_eval_dict is None:
        log_L_eval_dict = calc_log_L_eval_dict_complete(N_data=N_data, N_params=N_params, N_params_free=N_params_free, log_L=log_L, AIC=AIC, AIC_free=AIC_free, BIC=BIC, BIC_free=BIC_free, complete=complete)

    if push_free is None:
        push_free = False

    if free_rm is None:
        if push_free:
            free_rm = True
        else:
            free_rm = False

    if push_free:
        for field in free_list:
            if log_L_eval_dict[f"{field}_free"] is not None:
                log_L_eval_dict[f"{field}"] = log_L_eval_dict[f"{field}_free"]

    if free_rm:
        free_rm_list = [f"{field}_free" for field in free_list]
    else:
        free_rm_list = []

    if keys_rm is None:
        keys_rm = []

    keys_rm = np.concatenate([keys_rm, free_rm_list])


    return dict_pre_suff(dic=dict_key_rm(log_L_eval_dict, keys_rm=keys_rm, None_rm=None_rm), pre=pre, suff=suff)




def check_type(values, types=None, elementwise=None):

        if type(elementwise) != bool:
            elementwise = True

        if type(types) == type(None):
            types = np.array([int, float])
        else:
            types = make_iterable_array(types)

        types_dict =    {
                            int: [int, np.int32],
                            float: [float, np.float64],
                            str: [str],
                            bool: [bool, np.bool_],
                            type(None): [type(None)],
                        }

        types_all = []
        for i in types:
            types_all.append(types_dict[i][:])
        types_all_flat = [j for i in types_all for j in i]
        types_all_flat = make_iterable_array(types_all_flat)

        if elementwise == True:
            values = make_iterable_array(values)
            type_elements = np.empty((types_all_flat.size, values.size), dtype=bool)
            for i in range(types_all_flat.size):
                type_elements[i,:] = [isinstance(j, types_all_flat[i]) for j in values]
            type_elements = np.any(type_elements, axis=0)
            passed = np.all(type_elements == True)
        else:
            passed = np.isin([type(values)], types_all_flat)

        return passed



def clean_D_nodes(D_nodes):
    """Check function for D_nodes 
            -- the number of point processes
    """
    if type(D_nodes) != int:
        raise TypeError("D_nodes must be of type int")
    if D_nodes < 1:
        raise ValueError("D_nodes must be greater than or equal to 1")
    return D_nodes


def clean_scalar(scalar=None, name="scalar"):

    if type(scalar) == type(None):
        scalar = 0
    scalar = np.squeeze(scalar)

    if scalar.ndim == 0:
        scalar = np.resize(scalar, 1)[0]
    else:
        raise TypeError("{} must be of ndim 0".format(name))

    if type(scalar) == np.int32 or type(scalar) == np.int64:
        scalar = int(scalar)
    elif type(scalar) == np.float64:
        scalar = float(scalar)
    elif type(scalar) == np.str_:
        scalar = str(scalar)
    else:
        raise TypeError("Unsupported type")

    return scalar


def clean_vector(D_nodes, vector=None, nodes=None, unique=True, name="vector"):

    if type(vector) == type(None):
        vector = 0
    vector = np.squeeze(vector)

    nodes = clean_nodes(D_nodes=D_nodes, nodes=nodes, unique=unique)

    if vector.ndim == 0:
        vector = np.full(nodes.size, vector)
    elif vector.ndim == 1:
        if vector.size != nodes.size:
            raise TypeError("{} must be of size nodes.size={} or 1".format(name, nodes.size))
    else:
        raise TypeError("{} must be of ndim 0 or 1".format(name))

    return vector, nodes

def clean_matrix(D_nodes, matrix=None, nodes_i=None, nodes_j=None, name="matrix"):

    if type(matrix) == type(None):
        matrix = 0
    matrix = np.squeeze(matrix)

    shape = (D_nodes, D_nodes)

    if matrix.ndim == 2 or type(nodes_i) ==  type(nodes_j) == type(None):
        if matrix.ndim == 0:
            matrix = matrix * np.ones(shape)
        elif matrix.ndim == 2:
            if matrix.shape != shape:
                raise TypeError("{} must be of shape (D_nodes, D_nodes) = {} or (1,1)".format(name, shape))
        else:
            raise TypeError("{} must be of ndim 0 or 2".format(name))
        nodes_i = None
        nodes_j = None
        pairs = None
    else:
        if type(nodes_j) == None:
            nodes_j = nodes_i
        elif type(nodes_i) == None:
            nodes_i = nodes_j
        nodes_i = clean_nodes(D_nodes, nodes=nodes_i, unique=False)
        nodes_j = clean_nodes(D_nodes, nodes=nodes_j, unique=False)

        if nodes_i.size != nodes_j.size:
            if nodes_i.size == 1:
                nodes_i = np.full(nodes_j.shape, nodes_i)
            elif nodes_j.size == 1:
                nodes_j = np.full(nodes_i.shape, nodes_j)
            else:
                raise ValueError("nodes_i and nodes_j must be of equal length or one must be of ndim 1")
        if nodes_i.size > D_nodes**2:
            raise ValueError("nodes_i and nodes_j must be of length less than or equal to D_nodes^2 = {}".format(D_nodes**2))
    
        if matrix.ndim == 0:
            matrix = matrix * np.ones(nodes_i.size)
        elif matrix.ndim == 1:
            if matrix.size != nodes_i.size:
                raise TypeError("{} must be of size nodes_i.size = nodes_j.size = {} or 1".format(name, shape))
        if matrix.size != nodes_i.size:
            raise TypeError("{} must be of size nodes_i.size = nodes_j.size = {} or 1".format(name, shape))
        else:
            raise TypeError("{} must be of ndim 0 or 2".format(name))
        
        pairs = np.transpose(np.squeeze([[nodes_i], [nodes_j]]))
        duplicates = np.full((nodes_i.size, nodes_j.size), False)
        for j in range(nodes_j.size):
            for i in range(j):
                duplicates[i,j] = memoryview(pairs[i]) == memoryview(pairs[j])
        if np.any(duplicates):
            raise ValueError("All pairs of node indices (nodes_i[k], nodes_j[k]) must be unique")

    return matrix, nodes_i, nodes_j, pairs


def clean_nodes(D_nodes, nodes=None, unique=True):

    if type(nodes) == type(None):
        nodes = np.array(range(D_nodes))
    else:
        nodes = np.squeeze(nodes)
        if nodes.dtype != int:
            raise TypeError("All elements of nodes must of type int")
        
        if nodes.ndim > 1:
            raise TypeError("nodes must be of ndim 0 or 1")
        if nodes.size > D_nodes:
            raise TypeError("nodes must be of size D_nodes or smaller")
        if np.all(np.isin(nodes, range(-D_nodes, D_nodes))) == False:
            raise ValueError("All elements of nodes must be valid indices within the range of the number of nodes")

        #nodes[nodes<0] = nodes[nodes<0] + D_nodes

        if unique:
            if nodes.size != np.unique(nodes).size:
                raise ValueError("All elements of nodes must be unique")

    return nodes


def diff_zero_old(vector, u_zero=None, v_zero=None, dtype=None):
    """
    """
    vector_diff = np.copy(vector)
    if v_zero is None:
        if u_zero is None:
            u_zero = 0
        v_zero = vector[0] - u_zero
    vector_diff[0] = v_zero
    vector_diff[1:] = np.diff(vector)

    if dtype is not None:
        vector_diff = np.array(vector_diff, dtype=dtype)

    return vector_diff

def diff_zero(vector, u_zero=None, v_zero=None, dtype=None, axis=None, reshape=None, make_copy=None):
    """
    """
    if make_copy is None:
        make_copy = True

    if reshape is None:
        reshape = True

    if axis is None:
        axis = -1

    vector_shape = make_iterable_array(np.shape(vector))
    if vector_shape[axis]==0:
        return np.zeros(vector_shape)
    else:

        V_shape = np.array(np.shape(vector))
        V_dim = len(V_shape)

        if make_copy:
            vector_c = np.copy(vector)
        else:
            vector_c = vector

        move = bool(1 - np.isin(axis, np.array([0,V_dim])-1))
        if move:
            vector_c = np.moveaxis(vector_c, source=axis, destination=-1)

        if v_zero is None:
            if u_zero is None:
                u_zero = 0
            v_zero = vector_c[...,0] - u_zero
        
        vector_c[...,1:] = np.diff(vector_c, axis=-1)
        vector_c[...,0] = v_zero

        if move:
            if reshape:
                vector_c = np.moveaxis(vector_c, source=-1, destination=axis)

        if dtype is not None:
            vector_c = np.array(vector_c, dtype=dtype)


        return vector_c




def M_sum(D_nodes, D_ci=1):

    D_full = D_nodes//D_ci
    D_rem = D_nodes % D_ci
    D_eff = (D_nodes+D_ci-1)//D_ci

    M = np.eye(D_ci*D_eff)
    #M[:D_ci*D_full, :D_ci*D_full] /= D_ci
    #if D_rem != 0:
    #    M[-(1+D_rem):, -(1+D_rem):] /= D_rem
    rng = D_ci*np.array(range(D_eff))
    

    return np.sum([M[rng+i] for i in range(D_ci)], axis=0)[:,:D_nodes]




def get_blr_atf():

    return {tail: {period: None for period in ['all', 'train', 'forecast']} for tail in ['both', 'left', 'right']}

def get_abblr_atf():

    return {tail: {period: None for period in ['all', 'train', 'forecast']} for tail in ['all', 'bulk', 'both', 'left', 'right']}


def calc_eval_default(measure_name=None, measure_inp=None, val=None):

    if measure_inp is None and measure_name is not None:
        if measure_name=='delta':
            if np.isin(val, ['t', 'tilde_t']):
                measure_inp = True
            else:
                measure_inp = False
        elif measure_name=='set_zero':
            if np.isin(val, ['t', 'tilde_t']):
                measure_inp = True
            else:
                measure_inp = False
            
    return measure_inp

def calc_val_head(val_head=None, val=None, delta=None, normalize=None):

    if val_head is None:
        val_head = False

    if type(val_head)!=bool:
        return val_head
    elif val_head==False or val is None:
        return f''
    else:
        if delta is None:
            if np.isin(val, ['t', 'tilde_t']):
                delta = True
            else:
                delta = False
        if normalize is None:
            normalize = False
        
        if delta:
            delta_str = f"delta_"
        else:
            delta_str = f''
        if normalize:
            normalize_str = f'Norm_'
        else:
            normalize_str = f''

        return f"{normalize_str}{delta_str}{val}_"



def KS_Poisson_p(tilde_t):
    N = tilde_t.size
    tilde_t_dist = (tilde_t - (1+np.arange(N)))/N
    tilde_t_sup = np.max(np.abs(tilde_t_dist))
    p = scipy.special.kolmogorov(tilde_t_sup*np.sqrt(N))
    return p

def KS_Poisson_N_t_p(N_t, tilde_t, N_t_0=None, tilde_t_0=None):

    if N_t_0 is None:
        N_t_0 = 0
    if tilde_t_0 is None:
        tilde_t_0 = 0.0
    
    N = N_t[-1] - N_t_0
    tilde_t_dist = ((N_t-N_t_0) - (tilde_t-tilde_t_0))/N
    tilde_t_sup = np.max(np.abs(tilde_t_dist))
    KS_stat = tilde_t_sup*np.sqrt(N)
    KS_p = scipy.special.kolmogorov(KS_stat)
    
    return np.array([KS_stat, KS_p])

def calc_Poisson_cdf(N):

    def cdf(x):
        return x/N

    return cdf


def KS_test(x, cdf=None, sort=None, print_full=None, return_full=None):

    if sort is None:
        sort = True
    if print_full is None:
        print_full = False

    if return_full is None:
        return_full = False

    if cdf is None:
        cdf = scipy.stats.norm().cdf


    x = make_iterable_array(x)
    if sort:
        x = np.sort(x)

    N = np.size(x)
    if cdf=="Poisson":

        cdf = calc_Poisson_cdf(N=N)


    F_emp = np.arange(N+1)/N
    F_dist = cdf(x)


    KS_sup = np.abs(np.array([F_dist-F_emp[i:(N+i)] for i in range(2)]))

    KS_argmax = np.argmax(KS_sup)


    

    KS_stat = KS_sup[KS_argmax//N, KS_argmax%N]
    KS_p = scipy.special.kolmogorov(KS_stat*np.sqrt(N))
    KS_loc = x[KS_argmax%N]
    KS_sign = int((-1)**(1+KS_argmax//N))

    full = np.array([KS_stat, KS_p, KS_loc, KS_sign, KS_sup, KS_argmax])

    if print_full:
        print(full)


    if return_full:
        return full
    else:
        return full[:2]



def KS_test_Poisson(tilde_t, print_full=None):

    return KS_test(x=tilde_t, cdf="Poisson", sort=False, print_full=print_full)


def KS_test_Poisson_scipy(tilde_t, method=None):

    if method is None:
        method = 'auto'

    N = np.size(tilde_t)
    if N<1:
        tilde_t = [np.nan]

    args = {
            'rvs': tilde_t,
            'cdf': calc_Poisson_cdf(N=N),
            #'method': method,
            }
    

    return scipy.stats.kstest(**args)


def calc_residual_KS(tilde_t):
    """
    """
    tilde_t_delta = diff_zero(vector=tilde_t)
    return stats.kstest(tilde_t_delta, stats.expon(loc=0, scale=1).cdf)




def calc_KSS(c_mat):

    return (c_mat[0,0]/(c_mat[0,0] + c_mat[1,0])) - (c_mat[0,1]/(c_mat[0,1] + c_mat[1,1]))

def calc_ETS(c_mat):

    a_r = ((c_mat[0,0] + c_mat[1,0]) * (c_mat[0,0] + c_mat[0,1])) / (c_mat[0,0] + c_mat[1,0] + c_mat[0,1] + c_mat[1,1])

    return (c_mat[0,0] - a_r) / (c_mat[0,0] + c_mat[1,0] + c_mat[0,1] - a_r)

def calc_FBS(c_mat):

    return (c_mat[0,0] + c_mat[0,1]) / (c_mat[0,0] + c_mat[1,0])


def p_test_dict(p=None, incl_p_str=None, p_str_thresh_format=None, p_thresh=None, 
                phantom_sign=None, phantom_brac=None, sign_inside_brac=None, scif_string_format=None, mathmode=None):

    if mathmode is None:
        mathmode = True

    if p is None:
        p = np.nan
    
    
    if incl_p_str is None:
        incl_p_str = True


    if p_thresh is None:
        p_thresh = [0.01, 0.05]
    p_thresh = np.sort(make_iterable_array(p_thresh))

    p_dict = dict()
    p_dict['p'] = p
    

    if incl_p_str:


        if mathmode:
            app_dlrs = True
        else:
            app_dlrs = False

        if phantom_sign is None:
            phantom_sign = False
        if phantom_brac is None:
            phantom_brac = False
        

        p_dict['p_str'] = f"{scif_string(p_dict['p'], dp=1, app_dlrs=app_dlrs, phantom_sign=phantom_sign, phantom_brac=phantom_brac, sign_inside_brac=sign_inside_brac, scif_string_format=scif_string_format, mathmode=mathmode)}"

        p_dict['p_str_thresh'] = p_dict['p_str']
        if p_str_thresh_format=='bold':
            if mathmode:
                tex_fn = 'mathbf'
            else:
                tex_fn = 'textbf'
            p_dict['p_str_thresh'] = capp_tex_fn(p_dict['p_str'], tex_fn=tex_fn)
        elif p_str_thresh_format=='HL':
            p_less = np.sum(p_dict['p'] < p_thresh)
            if mathmode:
                space = ''
            else:
                space = ''
            if p_less > 0:
                p_dict['p_str_thresh'] = f"\\cellcolor{{HL{p_less}}}{space}{p_dict['p_str']}"


    return p_dict


def LR_test_dict(df=None, D_0=None, D_1=None, dev=None, dev_0=None, dev_1=None, log_L_0=None, log_L_1=None, incl_p_str=None, p_str_thresh_format=None, p_thresh=None, 
                 phantom_sign=None, phantom_brac=None, sign_inside_brac=None, scif_string_format=None, mathmode=None):

    
    
    
    if incl_p_str is None:
        incl_p_str = True


    if p_thresh is None:
        p_thresh = [0.01, 0.05]
    p_thresh = np.sort(make_iterable_array(p_thresh))

    if df is None:
        if D_0 is None or D_1 is None:
            df = 0
        else:
            df = np.abs(D_0 - D_1)

    if dev is None:
        if dev_0 is None or dev_1 is None:
            if log_L_0 is None or log_L_1 is None:
                dev = np.nan
            else:
                dev = -2*(log_L_1 - log_L_0)
        else:
            dev = dev_0 - dev_1

    LR_dict = dict()

    LR_dict['df'] = df
    LR_dict['dev'] = dev
    
    p_dict = p_test_dict(p=scipy.stats.chi2(df=df).sf(dev), incl_p_str=incl_p_str, p_str_thresh_format=p_str_thresh_format, p_thresh=p_thresh, 
                         phantom_sign=phantom_sign, phantom_brac=phantom_brac,  sign_inside_brac=sign_inside_brac, scif_string_format=scif_string_format, mathmode=mathmode)

    return {**LR_dict, **p_dict}

def PD_dict(log_L=None, N_params=None, N_data=None, T=None, dp=None, dp_intr=None, sci_pow10=None, sci_pow10_intr=None, symb_pre=None, symb_post=None, intr_symb=None, incl_str=None, 
            phantom_sign=None, phantom_brac=None, sign_inside_brac=None, scif_string_format=None, mathmode=None):

    if mathmode is None:
        mathmode = True

    if T is None:
        T = N_data

    if incl_str is None:
        incl_str = True
        

    PD_dict = dict()


    PD_dict['val'] = {
                        'dev': -2*log_L,
                        'AIC': calc_AIC(log_L=log_L, N_params=N_params),
                        'BIC': calc_BIC(log_L=log_L, N_params=N_params, N_data=N_data),
                    }
    PD_dict['val_intr'] = {val: PD_dict['val'][val]/T for val in PD_dict['val']}


    if incl_str:
        
        if symb_pre is None:
            symb_pre = ''
        if symb_post is None:
            symb_post = ''
        if intr_symb is None:
            intr_symb = 'T'

        if phantom_brac is None:
            phantom_brac = False
        


        PD_dict['symb'] = {
                            'dev': f"-2 {symb_pre}\\ell{symb_post}",
                            'AIC': f"{symb_pre}\\mathrm{{AIC}}{symb_post}",
                            'BIC': f"{symb_pre}\\mathrm{{BIC}}{symb_post}",
                        }
        PD_dict['symb_intr'] = {val: f"{PD_dict['symb'][val]}/{intr_symb}" for val in PD_dict['val']}


        if mathmode:
            app_dlrs = True
        else:
            app_dlrs = False

        if dp is None:
            dp = 3
        if dp_intr is None:
            dp_intr = dp

        PD_dict['val_str'] = {val: f"{scif_string(PD_dict['val'][val], dp=dp, scif_pow10=sci_pow10, app_dlrs=app_dlrs, phantom_sign=phantom_sign, phantom_brac=phantom_brac,  sign_inside_brac=sign_inside_brac, scif_string_format=scif_string_format, mathmode=mathmode)}" for val in PD_dict['val']}
        PD_dict['val_intr_str'] = {val: f"{scif_string(PD_dict['val_intr'][val], dp=dp_intr, scif_pow10=sci_pow10_intr, app_dlrs=app_dlrs, phantom_sign=phantom_sign, phantom_brac=phantom_brac,  sign_inside_brac=sign_inside_brac, scif_string_format=scif_string_format, mathmode=mathmode)}" for val in PD_dict['val_intr']}

        


    return PD_dict