import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
import pandas as pd
# Local files
import misc_functions as misc_fns




class vares_a():
    """Class for all Value-at-Risk (Q) and Expected Shortfall (E) forecasts across range of a_q
    """
    def __init__(self, a_q, bdist, set_eval=None, y=None, train_t_end=None, CT_Q=None,  CT_E=None, S_Q=None, S_E=None, save_t=None, t_start=None, t_end=None, period=None):

        
        self.set_vrs_a(a_q=a_q, bdist=bdist,
                                    set_eval=set_eval, y=y, train_t_end=train_t_end, 
                                    CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E,
                                    save_t=save_t, t_start=t_start, t_end=t_end, period=period)

    def set_vrs_a(self, bdist, a_q=None, set_eval=None, y=None, train_t_end=None, CT_Q=None,  CT_E=None, S_Q=None, S_E=None, save_t=None, t_start=None, t_end=None, period=None):

        if a_q is not None:
            self._a_q = misc_fns.make_iterable_array(a_q)
        self._t_train_end = train_t_end
        self.vrs_a = calc_vrs_a(a_q=self._a_q, bdist=bdist,
                                                set_eval=set_eval, y=y, train_t_end=train_t_end, 
                                                CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E,
                                                save_t=save_t, t_start=t_start, t_end=t_end, period=period)


    def set_vrs_eval(self, y=None, train_t_end=None, CT_Q=None,  CT_E=None, S_Q=None, S_E=None, save_t=None, t_start=None, t_end=None, period=None):
        
        for a_q in self.vrs_a:
            self.vrs_a[a_q].set_vares_eval(y=y, train_t_end=train_t_end, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t, t_start=t_start, t_end=t_end, period=period)


    def calc_vrs_eval(self, lead=None, atf=None):

        if atf is None:
            if self._t_train_end is None:
                atf=False
            else:
                atf=True
        if atf:
            return self.calc_vrs_eval_atf(lead=lead)
        else:
            return self.calc_vrs_eval_a(lead=lead)


    def calc_vrs_eval_a(self, lead=None):

        if lead is None:
            lead = dict()

        columns = [     
                        [k for k in lead],
                        ["a_q"],
                        [k for k in self.vrs_a[str(self._a_q[0])].eval.get_Q_E_TtS_flat().keys()],
                ]
        df_dict = {c: [] for c in np.concatenate(columns)}
        for a_q in self.vrs_a:
            df_dict["a_q"].append(self.vrs_a[a_q].eval._a_q)
            for k in columns[-1]:
                df_dict[k].append(self.vrs_a[a_q].eval.get_Q_E_TtS_flat()[k])
        for k in columns[0]:
            df_dict[k] = np.repeat(lead[k], len(df_dict["a_q"]))

        return pd.DataFrame(df_dict)

    def calc_vrs_eval_atf(self, lead=None):
        if lead is None:
            lead = dict()

        columns = [     
                        [k for k in lead],
                        ["a_q", "tail", "period"],
                        [k for k in self.vrs_a[str(self._a_q[0])].eval_atf[self.vrs_a[str(self._a_q[0])]._tails[0]][_atf[0]].get_Q_E_TtS_flat().keys()],
                ]
        df_dict = {c: [] for c in np.concatenate(columns)}
        for a_q in self.vrs_a:
            for tail in self.vrs_a[a_q].eval_atf:
                for period in self.vrs_a[a_q].eval_atf[tail]:
                    df_dict["a_q"].append(self.vrs_a[a_q].eval_atf[tail][period]._a_q)
                    df_dict["tail"].append(self.vrs_a[a_q].eval_atf[tail][period]._tail)
                    df_dict["period"].append(self.vrs_a[a_q].eval_atf[tail][period]._period)
                    for k in columns[-1]:
                        df_dict[k].append(self.vrs_a[a_q].eval_atf[tail][period].get_Q_E_TtS_flat()[k])
        for k in columns[0]:
            df_dict[k] = np.repeat(lead[k], len(df_dict["a_q"]))

        return pd.DataFrame(df_dict)


class vares():
    """Class for all Value-at-Risk (Q) and Expected Shortfall (E) forecasts
    """
    def __init__(self, a_q, bdist, set_eval=None, y=None, train_t_end=None, CT_Q=None,  CT_E=None, S_Q=None, S_E=None, save_t=None, t_start=None, t_end=None, period=None):
        """Constructor for vares class
        """
        self.set_vares(a_q=a_q, bdist=bdist,
                        set_eval=set_eval, y=y, train_t_end=train_t_end,
                        CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E,
                        save_t=save_t, t_start=t_start, t_end=t_end, period=period)

        
    def set_vares(self, bdist, a_q=None, set_eval=None, y=None, train_t_end=None, CT_Q=None,  CT_E=None, S_Q=None, S_E=None, tails=None, save_t=None, t_start=None, t_end=None, period=None):
        if a_q is not None:
            self._a_q = a_q
        self.Q = bdist.Q(self._a_q)
        self.E = bdist.E(self._a_q)
        self.Q_med = bdist.Q(0.5)[0,:]
        if set_eval is None:
            set_eval = True
        if set_eval:
            self.set_vares_eval(y=y, train_t_end=train_t_end, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, tails=tails, save_t=save_t, t_start=t_start, t_end=t_end, period=period)
        


    def set_vares_eval(self, y=None, train_t_end=None, CT_Q=None,  CT_E=None, S_Q=None, S_E=None, tails=None, save_t=None, t_start=None, t_end=None, period=None):

        if y is not None:
            if train_t_end is None:
                self.set_eval(y=y, 
                                CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E,
                                tails=tails, save_t=save_t, t_start=t_start, t_end=t_end, period=period)
            else:
                self.set_eval_atf(y=y, train_t_end=train_t_end, 
                                    CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E,
                                    tails=tails, save_t=save_t)


    def set_eval(self, y, CT_Q=None,  CT_E=None, S_Q=None, S_E=None, tails=None, save_t=None, t_start=None, t_end=None, period=None):

        if save_t is None:
            save_t = False
        self.eval = self.calc_eval(y=y, 
                                    CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E,
                                    tails=tails, save_t=save_t, t_start=t_start, t_end=t_end, period=period)

    def set_eval_atf(self, y, train_t_end, CT_Q=None, CT_E=None, S_Q=None, S_E=None, tails=None, save_t=None):
        if save_t is None:
            save_t = False
        self.eval_atf = self.calc_eval_atf(y=y, train_t_end=train_t_end,  
                                            CT_Q=CT_Q, CT_E=CT_E, S_Q=S_Q, S_E=S_E,
                                            tails=tails, save_t=save_t)


    def calc_eval(self, y, CT_Q=None, CT_E=None, S_Q=None, S_E=None, tails=None, save_t=None, t_start=None, t_end=None, period=None):

        if tails is not None:
            self._tails = misc_fns.make_iterable_array(tails)
        else:
            self._tails = ['left', 'right']

        return {self._tails[i]: vares_eval(a_q=self._a_q, y=y, Q=self.Q[i,:], E=self.E[i,:], tail=self._tails[i], 
                            CT_Q=CT_Q, CT_E=CT_E, S_Q=S_Q, S_E=S_E,
                            save_t=save_t, t_start=t_start, t_end=t_end, period=period,) for i in range(len(self._tails))}

    def calc_eval_atf(self, y, train_t_end, CT_Q=None,  CT_E=None, S_Q=None, S_E=None, tails=None, save_t=None):

        if tails is not None:
            self._tails = misc_fns.make_iterable_array(tails)
        else:
            self._tails = ['left', 'right']

        return calc_eval_atf(a_q=self._a_q, y=y, train_t_end=train_t_end, Q=self.Q, E=self.E, Q_med=self.Q_med,
                                    CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E,
                                    tails=self._tails, save_t=save_t)



    
class vares_eval():
    """Class for all Value-at-Risk (Q) and Expected Shortfall (E) forecast evaluations
    """
    def __init__(self, a_q, y, Q, E, Q_med=None, tail=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, t_start=None, t_end=None, period=None):
        """Constructor for vares_eval class
        """
        self._a_q = a_q

        self
        if tail is not None:
            self._tail = tail
        else:
            self._tail = 'left'

        self._sign = -1
        if self._tail=='right':
            self._sign = 1
            
        if save_t is None:
            save_t = True
        if period is None:
            period = "all"
        self._save_t = save_t 
        self._period = period   


        self._t_max = Q.size
        self.set_t_bounds(t_start=t_start, t_end=t_end, reset=True)

        self._y, self._Q, self._Q_I, self._E, self._Q_med = self.get_vares(y=y, Q=Q, E=E, Q_med=Q_med, sign=self._sign, save_t=self._save_t)
        _y, _Q, _Q_I, _E, _Q_med = self.get_vares(y=y, Q=Q, E=E, Q_med=Q_med, sign=self._sign, save_t=~self._save_t)
            

        self.set_Q_I_T(Q_I=_Q_I)
        self.set_CT_QE(CT_Q=CT_Q, CT_E=CT_E, y=_y, E=_E, Q=_Q, Q_I=_Q_I, Q_med=_Q_med)
        #self.set_CT_Q(CT_Q=CT_Q, Q=_Q, Q_I=_Q_I)
        self.set_S(S_Q=S_Q, S_E=S_E, y=_y, Q=_Q, Q_I=_Q_I, E=_E)



    def get_vares(self, y, Q, E, Q_med=None, sign=None, save_t=True):

        ret = [None for i in range(5)]
        if save_t:
            ret[0] = y[self._t_bounds[0]:self._t_bounds[1]]
            ret[1] = Q[self._t_bounds[0]:self._t_bounds[1]]
            ret[2] = calc_Q_I(y=ret[0], Q=ret[1], sign=sign)
            ret[3] = E[self._t_bounds[0]:self._t_bounds[1]]
            if Q_med is not None:
                ret[4] = Q_med[self._t_bounds[0]:self._t_bounds[1]]
            
        return ret[0], ret[1], ret[2], ret[3], ret[4]




    def set_t_bounds(self, t_start=None, t_end=None, reset=False):

        if reset:
            self._t_bounds = np.array([0, self._t_max])
        if t_start is not None:
            self._t_bounds[0] = t_start
        if t_end is not None:
            self._t_bounds[1] = t_end



    def set_Q_I_T(self, Q_I=None):
        if Q_I is None:
            Q_I = self._Q_I

        self.set_T(Q_I)
        self.set_il1(Q_I)


    def set_T(self, Q_I=None):
        if Q_I is None:
            Q_I = self._Q_I
        self._T, self._T_i = calc_T(Q_I)
        self._a_q_T_i = self._T_i/self._T

    def set_il1(self, Q_I=None):
        if Q_I is None:
            Q_I = self._Q_I
        self._T_i1l, self._pi_il1 = calc_il1(Q_I)



    def set_CT_QE(self, CT_Q=None, CT_E=None, y=None, E=None, Q=None, Q_I=None, Q_med=None):
        self.set_CT_Q(CT_Q=CT_Q, Q=Q, Q_I=Q_I)
        self.set_CT_E(CT_E=CT_E, y=y, E=E, Q=Q, Q_I=Q_I, Q_med=Q_med)
    

    def set_CT_Q(self, CT_Q, Q=None, Q_I=None):
        if CT_Q is not None:
            self._CT_Q = CT_Q
        else:
            self._CT_Q = np.concatenate([
                                                ["binom", "UC", "CC"],
                                                [f"DQ_{i}" for i in range(5)], 
                                                ["DL"],
                                            ])
        if Q is None:
            Q = self._Q
        if Q_I is None:
            Q_I = self._Q_I

        self.CT_Q = {test: calc_CT_Q(a_q=self._a_q, Q=Q, Q_I=Q_I, T_i=self._T_i, 
                                                T_il1=self._T_i1l, pi_il1=self._pi_il1, CT_Q=test) for test in self._CT_Q}


    def set_CT_E(self, CT_E, y=None, E=None, Q=None, Q_I=None, Q_med=None):
        if CT_E is not None:
            self._CT_E = CT_E
        else:
            self._CT_E = [f'ZMD']
        if y is None:
            y = self._y
        if E is None:
            E = self._E
        if Q is None:
            Q = self._Q
        if Q_I is None:
            Q_I = self._Q_I
        if Q_med is None:
            Q_med = self._Q_med


        CT_E_list = [calc_CT_E(y=y, E=E, Q=Q, Q_I=Q_I, Q_med=Q_med, CT_E=test) for test in self._CT_E]

        self.CT_E = {
                        test: CT_E_list[k][test] for k in range(len(CT_E_list)) for test in CT_E_list [k]
                    }
    

    def set_S(self, S_Q=None, S_E=None, y=None, Q=None, Q_I=None, E=None):
        self.set_S_Q(S_Q=S_Q, y=y, Q=Q, Q_I=Q_I)
        self.set_S_E(S_E=S_E, y=y, Q=Q, Q_I=Q_I, E=E)

    def set_S_Q(self, S_Q=None, y=None, Q=None, sign=None, Q_I=None):
        if S_Q is not None:
            self._S_Q = S_Q
        else:
            self._S_Q = ["G"]
        if y is None:
            y = self._y
        if Q is None:
            Q = self._Q
        if sign is None:
            sign = self._sign
        if Q_I is None:
            Q_I = self._Q_I
        self.S_Q = {S: calc_S_Q(a_q=self._a_q, y=y, Q=Q, sign=sign, Q_I=Q_I, S_Q=S, save_t=self._save_t) for S in self._S_Q}
    
    def set_S_E(self, S_E=None, y=None, Q=None, sign=None, Q_I=None, E=None):
        if S_E is not None:
            self._S_E = S_E
        else:
            self._S_E = ["AL", "NZ", "FZG"]
        if y is None:
            y = self._y
        if Q is None:
            Q = self._Q
        if sign is None:
            sign = self._sign
        if Q_I is None:
            Q_I = self._Q_I
        if E is None:
            E = self._E
        
        self.S_E = {S: calc_S_E(a_q=self._a_q, y=y, Q=Q, sign=sign, Q_I=Q_I, E=E, S_E=S, save_t=self._save_t) for S in self._S_E}


    def get_Q_E_TtS_flat(self):
        return {**self.get_Q_T_flat(), **self.get_CT_QE_flat(), **self.get_S_QE_flat()}


    def get_Q_T_flat(self):
        return  {
                    "Q_T": self._T,
                    "Q_T_1": self._T_i[1],
                    "Q_a_q_T_1": self._a_q_T_i[1]  
                }

    def get_QE_tS_flat(self):
        return {**self.get_CT_Q_flat(), **self.get_S_QE_flat()}

    def get_CT_QE_flat(self):
        return {**self.get_CT_Q_flat(), **self.get_CT_E_flat()}

    def get_S_QE_flat(self):
        return {**self.get_S_Q_flat(), **self.get_S_E_flat()}

    def get_Q_tS_flat(self):
        return {**self.get_CT_Q_flat(), **self.get_S_Q_flat()}

    def get_CT_Q_flat(self):
        return {f"CT_Q_{test}_{v}": self.CT_Q[test][v] for test in self.CT_Q for v in self.CT_Q[test]}

    def get_CT_E_flat(self):
        return {f"CT_E_{test}_{v}": self.CT_E[test][v] for test in self.CT_E for v in self.CT_E[test]}

    def get_S_Q_flat(self):
        return {f"S_Q_{S}": self.S_Q[S]["sum"] for S in self.S_Q}

    def get_S_E_flat(self):
        return {f"S_E_{S}": self.S_E[S]["sum"] for S in self.S_E}

    

#============
#============
#
# Functions
#
#============
#============

#--------------
#  vares_dict
#--------------



def calc_vrs_a(a_q, bdist, set_eval=None, y=None, train_t_end=None, CT_Q=None,  CT_E=None, S_Q=None, S_E=None, save_t=None, t_start=None, t_end=None, period=None):

        return {str(a): vares(a_q=a, bdist=bdist,
                                    set_eval=set_eval, y=y, train_t_end=train_t_end,
                                    CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E,
                                    save_t=save_t, t_start=t_start, t_end=t_end, period=period) for a in a_q}

#--------------
#  vares
#--------------
"""Object-independent functions for vares class
"""


def calc_atf_t(train_t_end=None):

    atf_t = {t: [None, None] for t in _atf}
    for t in atf_t:
        if t=='train':
            atf_t[t] = [None, train_t_end]
        elif t=='forecast':
            atf_t[t] = [train_t_end, None]

    return atf_t


def calc_eval_atf(a_q, y, Q, E, train_t_end, Q_med=None, CT_Q=None,  CT_E=None, S_Q=None, S_E=None, tails=None, save_t=None):

    atf_t = calc_atf_t(train_t_end=train_t_end)

    if tails is not None:
        tails = misc_fns.make_iterable_array(tails)
    else:
        tails = ['left', 'right']


    eval_atf = {tails[i]:
                    {period: vares_eval(a_q=a_q, y=y, Q=Q[i,:], E=E[i,:], tail=tails[i], Q_med=Q_med,
                                            CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E,
                                            save_t=save_t, t_start=atf_t[period][0], t_end=atf_t[period][1], period=period) for period in atf_t}
                for i in range(len(tails))}

    return eval_atf

#--------------
#  vares_eval
#--------------
"""Object-independent functions for vares_eval class
"""

def log_e(base, exp):

    #type_exp = type(exp)
    #if type_exp==int or type_exp==float:
    if np.size(exp)==1:
        if exp == 0:
            return np.zeros(np.shape(base))
        else:
            return exp*np.log(base)
    else:
        #type_base = type(base)
        log = np.zeros(np.shape(exp))
        nz = exp != 0
        #if type_base==int or type_base==float:
        if np.size(base)==1:
            log[nz] = exp[nz]*np.log(base)
        else:
            log[nz] = exp[nz]*np.log(base[nz])
        return log


def calc_Q_I(y, Q, sign=None):

    if sign is None:
        sign = -1
    return sign*y > sign*Q

def calc_Q_I_2(y, Q):
    return np.array([y<Q[0,:], y>Q[1,:]], dtype=bool)



def calc_T(Q_I):

    T = Q_I.size
    T_i = np.zeros(2, dtype=int)
    T_i[1] = np.sum(Q_I)
    T_i[0] = T - T_i[1]

    return T, T_i

def calc_Q_binom(a_q, Q_I=None, T_i=None):
    if Q_I is not None:
        T, T_i = calc_T(Q_I)
    else:
        T = np.sum(T_i)
    return {"p": scipy.stats.binom_test(x=T_i[1], n=T, p=a_q)}

def calc_Q_UC(a_q, Q_I=None, T_i=None):

    if Q_I is not None:
        T, T_i = calc_T(Q_I)
    else:
        T = np.sum(T_i)

    log_T_T_i = log_e(T, T_i)
    log_a_T_i = log_e(a_q, T_i)
    log_1ma_T_i = log_e(1-a_q, T_i)
    log_T_i_T_i = log_e(T_i, T_i)

    res = dict()
    res["LR"] = -2 * (log_a_T_i[1]+log_1ma_T_i[0] + np.sum(log_T_T_i-log_T_i_T_i))
    #res["LR"] = -2 * (log_a_T_i[1]+log_T_T_i[1]-log_T_i_T_i[1] + log_1ma_T_i[0]+log_T_T_i[0]-log_T_i_T_i[0])
    #res["LR"] = -2 * (log_e(a_q,T_i[1])+log_e(T,T_i[1])-log_e(T_i[1],T_i[1]) + log_e(1-a_q,T_i[0])+log_e(T,T_i[0])-log_e(T_i[0],T_i[0]))
    res["p"] = 1 - scipy.stats.chi2(df=1).cdf(res["LR"])

    return res


def calc_il1(Q_I):
    T_il1 = np.array([
                    [np.sum(~Q_I[1+np.where(~Q_I[:-1])[0]]), np.sum(Q_I[1+np.where(~Q_I[:-1])[0]])],
                    [np.sum(~Q_I[1+np.where(Q_I[:-1])[0]]), np.sum(Q_I[1+np.where(Q_I[:-1])[0]])],
                ])
    pi_il1 = np.array([[T_il1[i,j]/(np.sum(T_il1[i,:])) for j in range(2)] for i in range(2)])

    return T_il1, pi_il1 

def calc_Q_CC(a_q, Q_I=None, T_il1=None, pi_il1=None):

    if Q_I is not None:
        T_il1, pi_il1 = calc_il1(Q_I)

    res = dict()
    res["LR"] = -2 * (log_e(a_q, np.sum(T_il1[1,:])) + log_e(1-a_q, np.sum(T_il1[0,:])) - np.sum([log_e(1-pi_il1[i,1], T_il1[i,0])  + log_e(pi_il1[i,1], T_il1[i,1]) for i in range(2)]))
    res["p"] = 1 - scipy.stats.chi2(df=2).cdf(res["LR"])
    return res


def calc_Q_DQ(a_q, Q, Q_I, DQ_Hlags=None, output_res=False):
    # Dynamic Quantile Conditional Coverage Test
    # LR_CC ~ xi^2(d.f.= 2+DQ_Hlags)
    res = dict()
    if np.any(np.isnan(Q_I)) or np.any(np.isnan(Q)):
        res["LR"] = np.nan
        res["p"] = np.nan
        if output_res:
            res["res"] = None
    else:
        if DQ_Hlags is None:
            DQ_Hlags = 4

        Hit_t = Q_I - a_q

        Y = Hit_t[DQ_Hlags:]
        X = [None for i in range(2+DQ_Hlags)]
        X[0] = np.ones(Y.size)
        X[1] = Q[DQ_Hlags:]
        for i in range(DQ_Hlags):      
            X[2+i] = Hit_t[DQ_Hlags-1-i:-1-i]
        X = np.transpose(np.array(X))

        reg = LinearRegression(fit_intercept=False).fit(X, Y)

        
        res["LR"] = (np.transpose(reg.coef_) @ np.transpose(X) @ X @ reg.coef_)/(a_q*(1-a_q))
        res["p"] = 1 - scipy.stats.chi2(df=2+DQ_Hlags).cdf(res["LR"])
        if output_res:
            res["res"] = reg

    return res

def calc_Q_DL_a_t(phi, X):

    a_t = X @ phi
    for i in range(a_t.size-1):
        X[1+i, 3] = a_t[i]
        a_t[1+i] = X[1+i,:] @ phi

    return a_t, X

def calc_Q_DL_log_L(a_t, Q_I):

    #return np.sum(np.log(scipy.stats.logistic.cdf(a_t[Q_I]))) + np.sum(np.log(1-scipy.stats.logistic.cdf(a_t[~Q_I])))
    return -np.sum(np.log(1 + np.exp(-a_t[Q_I]))) - np.sum(np.log(1 + np.exp(a_t[~Q_I])))


def calc_Q_DL_log_L_a_t_alt(phi, args):

    Q_I = args[0]
    Q = args[1]

    X = [None for i in range(4)]
    X[0] = np.ones(Q_I.size-1)
    X[1] = Q_I[:-1]
    X[2] = Q[:-1]
    X[3] = np.zeros(Q_I.size-1)
    X = np.transpose(np.array(X))
    a_t, X = calc_Q_DL_a_t(phi, X)

    return calc_Q_DL_log_L(a_t, Q_I[1:])

def calc_Q_DL_log_L_a_t(phi, args):

    Q_I = args[0]
    Q = args[1]

    X = [None for i in range(3)]
    X[0] = np.ones(Q_I.size-1)
    X[1] = Q_I[:-1]
    X[2] = Q[:-1]
    X = np.transpose(np.array(X))

    a_t = X @ phi[:3]
    for i in range(a_t.size-1):
        a_t[1+i] += a_t[i]*phi[3]

    return calc_Q_DL_log_L(a_t, Q_I[1:])


def calc_Q_DL_mlog_L_a_t(phi, args):

    return -calc_Q_DL_log_L_a_t(phi, args)


def calc_Q_DL(a_q, Q, Q_I, output_res=False):

    res = dict()
    if np.any(np.isnan(Q_I)) or np.any(np.isnan(Q)):
        res["LR"] = np.nan
        res["p"] = np.nan
        if output_res:
            res["res"] = None
    else:
        args = [Q_I, Q]
        phi_H0 = np.zeros(4)
        phi_H0[0] = scipy.stats.logistic().ppf(a_q)

        rS_ELSQP = scipy.optimize.minimize(calc_Q_DL_mlog_L_a_t, phi_H0, args=args, method='SLSQP')

        res = dict()
        res["LR"] = -2 * (calc_Q_DL_log_L_a_t(phi_H0, args=args) - calc_Q_DL_log_L_a_t(rS_ELSQP.x, args=args))
        res["p"] = 1 - scipy.stats.chi2(df=4).cdf(res["LR"])
        if output_res:
            res["res"] = rS_ELSQP

    return res

def calc_CT_Q(a_q, Q=None, Q_I=None, T_i=None, T_il1=None, pi_il1=None, CT_Q=None):
    if CT_Q is None:
        CT_Q=="binom"
    if CT_Q=="binom":
        return calc_Q_binom(a_q=a_q, T_i=T_i)
    elif CT_Q=="UC":
        return calc_Q_UC(a_q=a_q, T_i=T_i)
    elif CT_Q=="CC":
        return calc_Q_CC(a_q=a_q, T_il1=T_il1, pi_il1=pi_il1)
    elif CT_Q[:2]=="DQ":
        DQ_Hlags = int(CT_Q[3:])
        return calc_Q_DQ(a_q=a_q, Q=Q, Q_I=Q_I, DQ_Hlags=DQ_Hlags)
    elif CT_Q=="DL":
        return calc_Q_DL(a_q=a_q, Q=Q, Q_I=Q_I)


def calc_E_ZMD_t(y, E, Q, Q_I, Q_med=None, ZMD_b_opt=None):
    # Dynamic Violation Expectation Zero Mean Deviation Convergence Test
    # Circular bootstrap
    if ZMD_b_opt is None:
        ZMD_b_opt = True

    res = dict()
    fail = False
    if np.any(np.isnan(Q_I)) or np.any(np.isnan(Q)) or np.any(np.isnan(E)) or np.any(np.isnan(y)):
        fail = True
    elif np.sum(Q_I) < 2:
        fail = True
    if fail:
        res["t"] = np.nan
        if ZMD_b_opt:
            res["b_opt_CB"] = np.nan
    else:
        use_Q_med=True
        if Q_med is None:
            use_Q_med=False
        elif np.any(np.isnan(Q_med)):
            use_Q_med=False
        if use_Q_med:
            R = ((y-E)/(Q-Q_med))[Q_I]
        else:
            R = ((y-E)/Q)[Q_I]

        N_R=R.size
        Rbar = np.mean(R)
        S_R = np.std(R) * np.sqrt((N_R/(N_R-1)))
        t_R = (Rbar - 0)/(S_R/np.sqrt(N_R))


        if ZMD_b_opt:
            b_opt_CB = ZMD_b_opt_CB(R=R)
            #ZMD_L_samp = int(np.round(b_opt_CB))
        
        res["t"] = t_R
        if ZMD_b_opt:
            res["b_opt_CB"] = b_opt_CB


    return res


def calc_E_ZMD_F(y, E, Q, Q_I, Q_med=None, t_R=None, ZMD_b_opt=None, ZMD_N_boot=None, ZMD_L_samp=None):
    # Dynamic Violation Expectation Zero Mean Deviation Convergence Test
    # Circular bootstrap

    if t_R is None:
        get_t_R = True
    else:
        get_t_R = False
    if ZMD_L_samp is None and ZMD_b_opt:
        get_b_opt_CB = True
    else:
        get_b_opt_CB = False



    res = dict()
    fail = False

    if np.any(np.isnan(Q_I)) or np.any(np.isnan(Q)) or np.any(np.isnan(E)) or np.any(np.isnan(y)):
        fail = True
    elif np.sum(Q_I) < 2:
        fail = True
    elif ZMD_L_samp is not None:
        if np.isnan(ZMD_L_samp):
            fail = True
    if fail:
        if get_t_R:
            res["t"] = np.nan
        res["F"] = np.nan
        if get_b_opt_CB:
            res["b_opt_CB"] = np.nan
    else:
        use_Q_med=True
        if Q_med is None:
            use_Q_med=False
        elif np.any(np.isnan(Q_med)):
            use_Q_med=False
        if use_Q_med:
            R = ((y-E)/(Q-Q_med))[Q_I]
        else:
            R = ((y-E)/Q)[Q_I]
            
        N_R=R.size
        Rbar = np.mean(R)
        if get_t_R:
            S_R = np.std(R) * np.sqrt((N_R/(N_R-1)))
            t_R = (Rbar - 0)/(S_R/np.sqrt(N_R))


        if ZMD_N_boot is None:
            ZMD_N_boot = 1000
        if ZMD_L_samp is None:
            if ZMD_b_opt:
                b_opt_CB = ZMD_b_opt_CB(R=R)
                if np.isnan(b_opt_CB):
                    fail=True
                else:
                    ZMD_L_samp = np.max([1, int(np.round(b_opt_CB))])
            else:
                b_opt_CB = np.nan
                ZMD_L_samp=1
        if fail:
            if get_t_R:
                res["t"] = np.nan
            res["F"] = np.nan
            if get_b_opt_CB:
                res["b_opt_CB"] = np.nan


        N_samp=N_R
        N_samp_L= (N_samp+(ZMD_L_samp-1)) // ZMD_L_samp
        t_samps = np.full(ZMD_N_boot, np.nan)
        for k in range(ZMD_N_boot):
            samp_index = np.random.randint(low=0, high=N_R, size=N_samp_L)
            samp_index_c  = np.concatenate(np.outer(np.ones(N_samp_L, dtype=int), np.array(range(ZMD_L_samp), dtype=int)) + np.outer(samp_index, np.ones(ZMD_L_samp, dtype=int))) % N_R
            R_samp = R[samp_index_c[:ZMD_N_boot]]
            Rbar_samp = np.mean(R_samp)
            S_samp = np.std(R_samp) * np.sqrt((R_samp.size/(R_samp.size-1)))
            t_samps[k] = (Rbar_samp - Rbar)/(S_samp/np.sqrt(N_R))


        if get_t_R:
            res["t"] = t_R
        res["F"] = np.sum(t_R > t_samps)/t_samps.size
        if get_b_opt_CB:
            res["b_opt_CB"] = b_opt_CB


    return res


def ZMD_lambda_trap(t):
    mod_t = np.abs(misc_fns.make_iterable_array(t))
    lambd = np.zeros(mod_t.shape, dtype=float)

    intv_1 = np.logical_and(mod_t>=0, mod_t<0.5)
    intv_2 = np.logical_and(mod_t>=0.5, mod_t<1)

    lambd[intv_1] = 1
    lambd[intv_2] = 2*(1-mod_t[intv_2])

    return lambd

def ZMD_R_hat(R, k=None, M=None):

    N = R.size
    Rbar = np.mean(R)

    if M is not None:
        M, k = ZMD_get_M_k(M=M)
    else:
        if k is None:
            k=0

    mod_k = np.abs(misc_fns.make_iterable_array(k))
    


    return (1/N)*np.array([np.sum((R[:(N-mod_k[k_i])] - Rbar)*(R[mod_k[k_i]:] - Rbar)) for k_i in range(mod_k.size)])


def ZMD_get_M_k(M=None):

    if M is None:
        M=1
    k = np.array(range(-M, M+1), dtype=int)
    return M, k


def ZMD_g_hat(R, M=None, w=0):

    M, k = ZMD_get_M_k(M=M)
    return np.sum(ZMD_lambda_trap(k/M) * ZMD_R_hat(R=R, k=k) * np.cos(w*k))


def ZMD_G_hat(R, M=None):

    M, k = ZMD_get_M_k(M=M)
    return np.sum(ZMD_lambda_trap(k/M) * np.abs(k) * ZMD_R_hat(R=R, k=k))


def ZMD_D_CB(R, M=None):
    return (4/3)*ZMD_g_hat(R=R, M=M, w=0)**2


def ZMD_m_hat(R, c=None):

    if c is None:
        c = 2

    N = R.size
    K_N = int(np.max([5, np.sqrt(np.log10(N))]))

    m_min = 1
    m_max_1 = ((N-1)//2)
    m_max_2 = N - K_N

    m_max = np.min([m_max_1, m_max_2])
    if m_max<=m_min:
        return m_min

    cutoff = c * np.sqrt(np.log10(N)/N)

    for m in range(m_min, m_max):
        rho = np.abs(ZMD_R_hat(R=R, k=m+1+np.array(range(K_N)))/ZMD_R_hat(R=R, k=0))
        if np.all(rho < cutoff):
            break

    return m


def ZMD_b_opt_CB(R, M=None, c=None):

    N = R.size
    if N < 3:
        return np.nan
    else:
        if M is None:
            M = 2*ZMD_m_hat(R=R, c=c)

        return ((2*(ZMD_G_hat(R, M=M)**2)/ZMD_D_CB(R, M=M))**(1/3)) * N**(1/3)


def where_str_char(str, char):
    return np.where([str[k]==char for k in range(len(str))])[0]



def decode_ZMD_str(ZMD_str):
    hyph = where_str_char(ZMD_str, '_')
    colon = where_str_char(ZMD_str, ':')
    if len(colon)==0:
        trunc = len(ZMD_str)
        N_iteratons = 1
        
    else:
        trunc = colon[0]
        N_iteratons = int(ZMD_str[1+trunc:])


    if len(hyph)==2:
        N_boot_str_rng = 1+np.array(range(hyph[0],hyph[1]), dtype=int)

        L_samp_str_rng = 1+np.array(range(hyph[1],trunc), dtype=int)
        L_samp_str = ZMD_str[L_samp_str_rng[0]:L_samp_str_rng[-1]]
        L_samp_comma = np.unique(np.concatenate([[-1], where_str_char(L_samp_str, ','), [len(L_samp_str)]]))
        L_samp_list_size = L_samp_comma.size-1
        L_samp_list = np.array([L_samp_str[1+L_samp_comma[k]:L_samp_comma[k+1]] for k in range(L_samp_list_size)], dtype=int)

    else:
        N_boot_str_rng = 1+np.array(range(hyph[0],trunc), dtype=int)
        L_samp_list = np.zeros(1, dtype=int)


    N_boot_str = ZMD_str[N_boot_str_rng[0]:N_boot_str_rng[-1]]
    N_boot_comma = np.unique(np.concatenate([[-1], where_str_char(N_boot_str, ','), [len(N_boot_str)]]))
    N_boot_list_size = N_boot_comma.size-1
    N_boot_list = np.array([N_boot_str[1+N_boot_comma[k]:N_boot_comma[k+1]] for k in range(N_boot_list_size)], dtype=int)

    return N_boot_list, L_samp_list, N_iteratons



def calc_CT_E(y, E, Q, Q_I=None, Q_med=None, CT_E=None):
    if CT_E is None:
        CT_E=="ZMD"
    if CT_E[:3]=="ZMD":
        E_ZMD = {'ZMD': calc_E_ZMD_t(y=y, E=E, Q=Q, Q_I=Q_I, Q_med=Q_med, ZMD_b_opt=True)}
        t_R = E_ZMD['ZMD']['t']
        b_opt_CB = E_ZMD['ZMD']['b_opt_CB']
        if len(CT_E)==3:

            N_boot=1000
            L_samp=0
            if np.isnan(b_opt_CB):
                L_samp_inp = np.nan
            else:
                L_samp_inp = np.max([1, int(np.round(b_opt_CB))])
            i=0
            E_ZMD[f"ZMD_{N_boot}_{L_samp}:{i}"] = calc_E_ZMD_F(y=y, E=E, Q=Q, Q_I=Q_I, Q_med=Q_med, t_R=t_R, ZMD_b_opt=False, ZMD_N_boot=N_boot, ZMD_L_samp=L_samp_inp)

        else:
            N_boot_list, L_samp_list, N_iteratons = decode_ZMD_str(CT_E)

            for N_boot in N_boot_list:
                for L_samp in L_samp_list:
                    if L_samp==0:
                        L_samp_inp = np.max([1, int(np.round(b_opt_CB))])
                    else:
                        L_samp_inp = L_samp
                    for i in range(N_iteratons):
                        E_ZMD[f"ZMD_{N_boot}_{L_samp}:{i}"] = calc_E_ZMD_F(y=y, E=E, Q=Q, Q_I=Q_I, Q_med=Q_med, t_R=t_R, ZMD_b_opt=False, ZMD_N_boot=N_boot, ZMD_L_samp=L_samp_inp)

        return E_ZMD







def set_S_Q_fn(S_Q=None):

    if S_Q is None:
        S_Q="G"

    if S_Q=="G":
        G = lambda x : x
        #def G(x):
        #    return x
        #fn = {
        #        "G": G,
        #    }
    return G

def calc_S_Q(a_q, y, Q, sign=None, Q_I=None, S_Q=None, save_t=None):

    G = set_S_Q_fn(S_Q)
    if sign is None:
        sign = -1
    y_s = sign * y
    Q_s = sign * Q

    if Q_I is None:
        Q_I = y_s > Q_s
    if save_t is None:
        save_t = True

    res = dict()
    res["t"] = (Q_I - a_q)*(G(y_s) - G(Q_s))
    res["sum"] = np.sum(res["t"])

    if save_t:
        return res
    else:
        return {v: res[v] for v in ["sum"]}



def set_S_E_fn(S_E=None):

    if S_E is None:
        S_E = "AL"

    
    if S_E=="AL":
        G_1 = lambda x : 0
        G_2 = lambda x : 1/x
        zeta_2 = lambda x : -np.log(x)
        a = lambda y, a_q : 1 - np.log(1-a_q)

    elif S_E=="NZ":
        G_1 = lambda x : 0
        G_2 = lambda x : (1/2)*(1/np.sqrt(x))
        zeta_2 = lambda x : -np.sqrt(x)
        a = lambda y, a_q : 0

    elif S_E=="FZG":
        G_1 = lambda x : -x
        G_2 = lambda x : np.exp(-x)/(1+np.exp(-x))
        zeta_2 = lambda x : np.log(1+np.exp(-x))
        a = lambda y, a_q : np.log(2)
       

    return G_1, G_2, zeta_2, a 

def calc_S_E(a_q, y, E, Q, sign=None, Q_I=None, S_E=None, save_t=None):

    
    G_1, G_2, zeta_2, a = set_S_E_fn(S_E)
    if sign is None:
        sign = -1
    y_s = sign * y
    Q_s = sign * Q
    E_s = sign * E

    if Q_I is None:
        Q_I = y_s > Q_s
    if save_t is None:
        save_t = True

    res = dict()
    res["t"] = (Q_I - a_q)*G_1(Q_s) - Q_I*G_1(y_s) + G_2(E_s)*(Q_s - E_s + Q_I*(y_s - Q_s)/a_q) - zeta_2(E_s) + a(y_s, a_q)
    res["sum"] = np.sum(res["t"])

    if save_t:
        return res
    else:
        return {v: res[v] for v in ["sum"]}




#============
#============
#
# Variables
#
#============
#============

_atf = ["all", "train", "forecast"]
#_atf = ["forecast"]