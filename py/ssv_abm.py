import numpy as np
from numpy.lib.function_base import diff
import scipy
import scipy.stats as stats
from scipy.stats import genpareto
import statsmodels
from statsmodels.stats.diagnostic import acorr_ljungbox
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import copy
import functools
import financial_data_handling as fin_data_handl
import numexpr as ne
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool
import time
import numba as nb
from numba import jit, njit
import numdifftools
import datetime
import os
# Local files
import misc_functions as misc_fns
import plot_functions as plot_fns
import parameters

#import hawkes as hwks
#import cvol as cvl
#import vares
#import vares_3
#import log_ret
#import bulk_gp
import ssv_abm_nb


class pruna_parameters():
    """Class for all Pruna parameters
    """
    def __init__(self, N_s=None, insert=None, opt_method=None, 
                    rbe_str=None, rmode=None, bmode=None, emode=None,
                    mu_F=None, sigma_F=None, p_0_F=None, p_0=None,
                    N_s_a_bool=None, N_s_e_bool=None, beta=None,
                    alpha_0_s=None, n_0_s=None,
                    alpha_n=None, alpha_w=None,
                    eta_w=None, lambda_w_s=None, 
                    G_sigma=None,
                    psi=None, delta_s=None, gamma_1_s=None, gamma_2_s=None, eta_s=None, sigma_s=None,
                    rho_type_s=None, 
                    eta_varsigma=None,
					):
        """Constructor for pruna_parameters class
        """
        if N_s is None:
            N_s = 3     # A_n mod

        self._N_s = parameters.clean_D_nodes(D_nodes=N_s)
        self.reset()

        self.set_mode(rbe_str=rbe_str, rmode=rmode, bmode=bmode, emode=emode, reset=True)

        self.set_opt_method(opt_method, reset=True)

        
        self.set_all(rmode=rmode, bmode=bmode, emode=emode,
                    mu_F=mu_F, sigma_F=sigma_F, p_0_F=p_0_F, p_0=p_0,
                    N_s_a_bool=N_s_a_bool, N_s_e_bool=N_s_e_bool, beta=beta,
                    alpha_0_s=alpha_0_s, n_0_s=n_0_s,
                    alpha_n=alpha_n, alpha_w=alpha_w,
                    eta_w=eta_w, lambda_w_s=lambda_w_s, 
                    G_sigma=G_sigma,
                    psi=psi, delta_s=delta_s, gamma_1_s=gamma_1_s, gamma_2_s=gamma_2_s, eta_s=eta_s, sigma_s=sigma_s,
                    rho_type_s=rho_type_s, 
                    eta_varsigma=eta_varsigma,
                    fix=False)


        


        


        # Parameter estimation
        self.res = None
        self.res_time = {step: dict() for step in ["res", 'opt']}
        self.res_time['opt']['method'] = None
        for step in self.res_time:
            for t in ["start", "end", "delta"]:
                self.res_time[step][t] = None

        self.res_fp0 = None
        self.res_fp1 = None
        self.res_fp2 = None

        self.ftol = 2.220446049250313e-09

        self._fit_est = {param_name: None for param_name in self._fp_names}
        self._fit_se = self._fit_est.copy()
        self._fit_fp1 = self._fit_est.copy()
        self._fit_scores = misc_fns.IC_dict()

        self._log_L_dict = None
        self._log_L_dict_track = None


        self._train_t = None



    def get_N_s(self):
        return self._N_s


    def reset(self):

        self.reset_params()
        self.reset_flat()
        


    def reset_params(self):

        
        
        #----------------------------
        # Fundamental price parameters
        #----------------------------
        # Fundemental price drift
        self.mu_F = parameters.p_scalar(D_nodes=self._N_s, name="mu_F", etype=[int, float], family="p_F", default=0.0)
        
        # Fundemental price volatility
        self.sigma_F = parameters.p_scalar(D_nodes=self._N_s, name="sigma_F", etype=[int, float], family="p_F", default=1.0, excl_low=0)
        
        # Intitial fundamental price
        self.p_0_F = parameters.p_scalar(D_nodes=self._N_s, name="p_0_F", etype=[int, float], family="p_F", default=0.0)

        # Intitial observed price
        self.p_0 = parameters.p_scalar(D_nodes=self._N_s, name="p_0", etype=[int, float], family="p_F", default=0.0)
        
        
        
        #----------------------------
        # Fraction parameters
        #----------------------------
        # Active agent types
        self.N_s_a_bool = parameters.p_vector(D_nodes=self._N_s, name="N_s_a_bool", etype=[bool], family="fraction", default=True)
       
        # Evolving agent types
        self.N_s_e_bool = parameters.p_vector(D_nodes=self._N_s, name="N_s_e_bool", etype=[bool], family="fraction", default=True)
        
        # Evolution inverse temperature
        self.beta = parameters.p_scalar(D_nodes=self._N_s, name="beta", etype=[int, float], family="fraction", default=1.0, incl_low=0)
        
        # Intitial relative log fraction
        self.alpha_0_s = parameters.p_vector(D_nodes=self._N_s, name="alpha_0_s", etype=[int, float], family="fraction", default=0.0)
        
        # Intitial fraction
        self.n_0_s = parameters.p_vector(D_nodes=self._N_s, name="n_0_s", etype=[int, float], family="fraction", default=0.0, incl_low=0, incl_high=1)
        
        # Last fraction weight
        self.alpha_n = parameters.p_scalar(D_nodes=self._N_s, name="alpha_n", etype=[int, float], family="fraction", default=0.0, opt_incl_high=2000, opt_excl_low=0)
        
        # Wealth weight
        self.alpha_w = parameters.p_scalar(D_nodes=self._N_s, name="alpha_w", etype=[int, float], family="fraction", default=0.0, opt_incl_high=2000, opt_excl_low=0)
        
        # Wealth memory decay
        self.eta_w = parameters.p_scalar(D_nodes=self._N_s, name="eta_w", etype=[int, float], family="fraction", default=0.05, excl_low=0, incl_high=1) ##### eta_w_test
        
        # Wealth loss aversion
        self.lambda_w_s = parameters.p_vector(D_nodes=self._N_s, name="lambda_w_s", etype=[int, float], family="fraction", default=0.0, opt_incl_low=-np.log(10), opt_incl_high=np.log(10))

        # Stochastic demand in gains
        self.G_sigma = parameters.p_scalar(D_nodes=self._N_s, name="G_sigma", etype=[int], family="fraction", default=0, permitted=[0,1])
        
 
 
        #----------------------------
        # Demand parameters
        #----------------------------
        # Expected intensity
        self.psi = parameters.p_scalar(D_nodes=self._N_s, name="psi", etype=[int, float], family="demand", default=0.01, incl_low=0)

        # Demand coefficients
        self.delta_s = parameters.p_vector(D_nodes=self._N_s, name="delta_s", etype=[int, float], family="demand", default=0.0, opt_incl_low=1E-3, opt_incl_high=1/np.sqrt(self._N_s)) # delta_s_mod  

        # Noise coefficients
        self.sigma_s = parameters.p_vector(D_nodes=self._N_s, name="sigma_s", etype=[int, float], family="demand", default=1.0, incl_low=0, opt_incl_high=2) # A_n mod
        
        # Demand higher-order coefficients
        self.gamma_1_s = parameters.p_vector(D_nodes=self._N_s, name="gamma_1_s", etype=[int, float], family="demand", default=0.0, opt_incl_low=0, opt_incl_high=2)

        self.gamma_2_s = parameters.p_vector(D_nodes=self._N_s, name="gamma_2_s", etype=[int, float], family="demand", default=0.0, opt_incl_low=np.log(np.sqrt(2/np.pi)), opt_incl_high=np.log(2))

        # Memory decay coefficients
        self.eta_s = parameters.p_vector(D_nodes=self._N_s, name="eta_s", etype=[int, float], family="demand", default=1.0, excl_low=0, incl_high=1)
        
        # rho_type
        #self.rho_type_s = parameters.p_vector(D_nodes=self._N_s, name="rho_type_s", etype=[str], family="demand",  default=["cubic", "tanh", "tanh", "linear"], permitted=["zero", "linear", "cubic", "tanh", "wtar"]) # A_n mod
        self.rho_type_s = parameters.p_vector(D_nodes=self._N_s, name="rho_type_s", etype=[str], family="demand",  default=["linear", "linear", "wtar"], permitted=["zero", "linear", "cubic", "tanh", "wtar"]) # A_n mod
            
            
            
        #----------------------------
        # Volatility parameters
        #----------------------------
        
        # Volatility estimate decay
        self.eta_varsigma = parameters.p_scalar(D_nodes=self._N_s, name="eta_varsigma", etype=[int, float], family="volatility", default=0.1, excl_low=0, incl_high=1)

        # Volatility estimate initial value
        self.varsigma_0 = parameters.p_scalar(D_nodes=self._N_s, name="varsigma_0", etype=[int, float], family="volatility", default=1.0, excl_low=0)
        
       


    def calc_gamma_2_s_default(self, gamma_2_s2=None):

        if gamma_2_s2 is None:
            gamma_2_s2 = 0.0

        gamma_2_default = np.zeros(self.get_N_s())
        if self.get_N_s()>2:
            gamma_2_default[2] = gamma_2_s2

        return gamma_2_default
    

    def calc_names(self, name=None, name_rm=None):
        name = misc_fns.A_rm(A=name, A_rm=name_rm)
        return [element for lis in [[i for i in misc_fns.make_iterable_array(self.__dict__[param]._names)] for param in name] for element in lis]
    
    
    def get_fp_name(self, name=None, name_rm=None):
        if name is None:
            name = self._fp_name
        return misc_fns.A_rm(A=name, A_rm=name_rm)
    
    def calc_fp_names(self, name=None, name_rm=None):

        return self.calc_names(self, name=self.get_fp_name(name=name, name_rm=name_rm))

    def get_fp_names(self, names=None, names_rm=None):
        if names is None:
            names = self._fp_names
        return misc_fns.A_rm(A=names, A_rm=names_rm)

    def reset_fp_names(self):

        self._fp_name = ["mu_F", "sigma_F", "p_0_F", "p_0", "beta", "alpha_0_s", "alpha_n", "alpha_w", "eta_w", "lambda_w_s", "psi", "delta_s", "gamma_1_s", "gamma_2_s", "eta_s", "sigma_s", "eta_varsigma", "varsigma_0"]
        self._fp_names = self.calc_fp_names()


    def reset_flat(self):

        self.reset_fp_names()


        self._flat_index = {struct: {index: None for index in ["size", "start", "range"]} for struct in ["list", "dict"]}
        self._flat_index["list"]["size"] = np.array([np.size(self.__dict__[param].get_value()) for param in self._fp_name])
        self._flat_index["list"]["start"] = np.zeros(self._flat_index["list"]["size"].size, dtype=int)
        self._flat_index["list"]["start"][1:] = np.cumsum(self._flat_index["list"]["size"])[:-1]
        self._flat_index["list"]["range"] = [range(self._flat_index["list"]["start"][i], self._flat_index["list"]["start"][i]+self._flat_index["list"]["size"][i]) for i in range(len(self._fp_name))]
        for index in self._flat_index["list"]:
            self._flat_index["dict"][index] = {self._fp_name[i]: self._flat_index["list"][index][i] for i in range(len(self._fp_name))}

        self._flat_size = np.sum(np.array(self._flat_index["list"]["size"]))


    def set_opt_method(self, opt_method=None, reset=False):

        if opt_method is not None:
            self._opt_method = opt_method
        else:
            if reset:
                #self._opt_method = "SLSQP"
                self._opt_method = "L-BFGS-B"


    def get_opt_method(self):
        return self._opt_method


#################################################################


    def set_mode(self, rbe_str=None, rmode=None, bmode=None, emode=None, reset=None):

        if reset is None:
            reset = False

        if rbe_str is not None:
            rmode, bmode, emode = unpack_rbe_str(rbe_str=rbe_str)

       
        self.set_rmode(rmode=rmode, reset=reset, push=False)
        self.set_bmode(bmode=bmode, reset=reset, push=False)
        self.set_emode(emode=emode, reset=reset, push=False)

        self.push_mode()

    def get_mode(self):

        return self._mode


    
    def push_mode(self):

        self.set_N_s_mode()
        self.set_flat_subordinate(free=False)
        self.set_flat_fix(reset=True, free=False)
        self.set_flat_fix_mode(free=False)
        self.set_flat_free()

        self.set_mode_defaults()





    def set_rmode(self, rmode=None, push=None, reset=None):

        if push is None:
            push = True

        if reset is None:
            reset = False
        if reset:
            self._rmode = "llz"
        if rmode is not None:
            self._rmode = rmode

        self.push_rmode()


        if push:
            self.push_mode()


    def calc_rmode_array_bool(self):

        rmode_array = np.array(self.rho_type_s.get_value())
        return np.array([rmode_array == n for n in _rmode_names]) 


    def set_rmode_array_bool(self):

        
        self._rmode_array_bool = self.calc_rmode_array_bool()


    def push_rmode(self):


        #rho_type_s = np.concatenate([[_rmode_dict[self._rmode[k]] for k in range(len(self._rmode))], ["zero"]]) # A_n mod
        rho_type_s = np.array([_rmode_dict[self._rmode[k]] for k in range(len(self._rmode))]) 
        self.rho_type_s.set_value(rho_type_s) 

        self.set_rmode_array_bool()

        #delta_s = self.delta_s.get_value()
        #delta_s[0] = 0.2/np.sqrt(np.sum(self.N_s_a_bool.get_value()))  # delta_f mod
        delta_s = np.zeros(self.get_N_s())
        delta_s[self.N_s_a_bool.get_value()] = 0.1/np.sqrt(np.sum(self.N_s_a_bool.get_value()))


        gamma_1_s = self.gamma_1_s.get_value()
        gamma_2_s = self.gamma_2_s.get_value()

        if False:
            delta_s[np.array(self._rmode_array_bool[0,:], dtype=bool)] = 0
            gamma_1_s[np.array(np.sum(self._rmode_array_bool[:2,:], axis=0), dtype=bool)] = 0
        else:
            delta_s[np.where(self._rmode_array_bool[0,:])[0]] = 0.0
            gamma_1_s[np.where(np.sum(self._rmode_array_bool[:2,:], axis=0))[0]] = 0.0
            gamma_2_s[np.where(np.sum(self._rmode_array_bool[:3,:], axis=0))[0]] = 0.0

        self.delta_s.set_value(delta_s)
        self.gamma_1_s.set_value(gamma_1_s)
        self.gamma_2_s.set_value(gamma_2_s)


    def set_bmode(self, bmode=None, push=None, reset=None):

        if push is None:
            push = True

        if reset is None:
            reset = False
        if reset:
            self._bmode = False
        if bmode is not None:
            self._bmode = int(bmode)


        self.push_bmode()

        if push:
            self.push_mode()


    def push_bmode(self):

        N_s_a_bool = self.N_s_a_bool.get_value()
        N_s_e_bool = self.N_s_e_bool.get_value()

        bmode_n0 = self.get_bmode()!=0
        bmode_1 = self.get_bmode()==1
        bmode_2 = self.get_bmode()==2


        bmode_n0_fix_s = np.array(1 - (bmode_n0 * N_s_e_bool * N_s_a_bool), dtype=bool)
        bmode_2_fix_s = np.array(1 - (bmode_2 * N_s_e_bool * N_s_a_bool), dtype=bool)


        sigma_s = self.sigma_s.get_value()
        alpha_0_s = self.alpha_0_s.get_value()
        lambda_w_s = self.lambda_w_s.get_value()


        alpha_0_s[bmode_n0_fix_s] = 0
        if np.any(bmode_n0_fix_s==False):
            alpha_0_s[np.where(bmode_n0_fix_s==False)[0][0]] = 0

        lambda_w_s[bmode_2_fix_s] = 0

        if bmode_n0:
            self.beta.set_value(1)
            self.alpha_n.set_value(0)
            #sigma_s[3] = 0     # A_n mod
        else:
            self.beta.set_value(0)

            self.alpha_n.set_value(0)
            self.alpha_w.set_value(0)
            self.eta_w.set_value(1)

            #sigma_s[:3] = 0     # A_n mod
            sigma_s[:] = np.sqrt(np.mean(sigma_s[N_s_a_bool]**2))
        
        sigma_s[~N_s_a_bool] = 0.0
        

        self.sigma_s.set_value(sigma_s)
        self.alpha_0_s.set_value(alpha_0_s)
        self.lambda_w_s.set_value(lambda_w_s)
    

    def set_emode(self, emode=None, push=None, reset=None):

        if push is None:
            push = True

        if reset is None:
            reset = False
        if reset:
            self._emode = False
        if emode is not None:
            self._emode = int(emode)

        self.push_emode()

        if push:
            self.push_mode()


    def push_emode(self):

        if self.get_emode()==0:
            eta_s = 1
        else:
            eta_s = self.eta_s.get_value()
            eta_s[0] = 1
        
        self.eta_s.set_value(eta_s)


    def get_rmode(self):
        return self._rmode

    def get_bmode(self):
        return self._bmode

    def get_emode(self):
        return self._emode


    def calc_mode_dict(self):
        mode_dict =   {
                        "rbe_str": self.calc_rbe_str(),
                        "rmode": self.get_rmode(),
                        "bmode": int(self.get_bmode()),
                        "emode": int(self.get_emode()),
                    }
        return mode_dict


    def calc_fixed_param_dict(self):

        #return {"tbar_v": self.tbar_v.get_value(),}
        return {}
    
    

    def calc_rbe_str(self):

        return pack_rbe_str(rmode=self.get_rmode(), bmode=self.get_bmode(), emode=self.get_emode())



    def calc_A_n(self):

        #return np.sqrt(np.sum(self.N_s_a_bool.get_value()))

        return ssv_abm_nb.calc_A_n(N_s_a_bool=self.N_s_a_bool.get_value())




    def calc_A_G(self):

        N_s_a_bool = self.N_s_a_bool.get_value()
        N_s_e_bool = self.N_s_e_bool.get_value()

        if False:
            if self.get_bmode()==0:     
                A_n = 1
            else:
                A_n = 1/np.sqrt(np.sum(N_s_e_bool))
        
        
        delta_D_2 = np.sum(self.delta_s.get_value()[N_s_a_bool]**2)
        sigma_D_2 = np.sum(self.sigma_s.get_value()[N_s_a_bool]**2)

        if delta_D_2==0:
            return 1.0
        else:
            #return A_n * np.sqrt((delta_D_2 + sigma_D_2)*(delta_D_2 + self.G_sigma.get_value()*sigma_D_2)) # A_n mod

            return np.sqrt((delta_D_2 + sigma_D_2)*(delta_D_2 + self.G_sigma.get_value()*sigma_D_2))        



    def calc_Q_q(self):
        
        #return self.alpha_w.get_value() * self.delta_s.get_value()[0] * self.eta_w.get_value()
        return self.alpha_w.get_value() * self.delta_s.get_value()[0] * self.eta_w.get_value() / (self.psi.get_value()*self.calc_A_G())
        #return self.alpha_w.get_value() * self.delta_s.get_value()[0] * self.eta_w.get_value() / self.calc_A_G()



    def set_N_s_mode(self):

        N_s_a_bool = np.array(np.sum(self._rmode_array_bool[1:,:], axis=0), dtype=bool)


        if self.get_bmode()==0:     # A_n mod
            #N_s_a_bool[3] = True
            self.set_N_s_e_bool(N_s_e_bool=False)
        else:
            #N_s_a_bool[3] = False
            self.set_N_s_e_bool(N_s_e_bool=N_s_a_bool) 

        self.set_N_s_a_bool(N_s_a_bool=N_s_a_bool)



    def set_flat_fix(self, reset=False, fix=None, unfix=None, flip=None, free=True):

        if reset:
            self._flat_fix = np.full(self._flat_size, False)
        if fix is not None:
            self._flat_fix[fix] = True
        if unfix is not None:
            self._flat_fix[unfix] = False
        if flip is not None:
            self._flat_fix[flip] = ~self._flat_fix[flip]

        if free:
            self.set_flat_free()







    def set_flat_fix_mode(self, free=None):

        if free is None:
            free = True



        self._flat_fix[self._flat_index["dict"]["range"][f"psi"]] = True
        self._flat_fix[self._flat_index["dict"]["range"][f"beta"]] = True
        self._flat_fix[self._flat_index["dict"]["range"][f"p_0"]] = True
        self._flat_fix[self._flat_index["dict"]["range"][f"varsigma_0"]] = True

        
        

        vec_param = ["delta", "gamma_1", "gamma_2", "eta", "sigma"]

        for vec in vec_param:
            self._flat_fix[self._flat_index["dict"]["range"][f"{vec}_s"]] = True

        N_s_a_bool = self.N_s_a_bool.get_value()
        N_s_e_bool = self.N_s_a_bool.get_value()

        bmode_n0 = self.get_bmode()!=0
        bmode_1 = self.get_bmode()==1
        bmode_2 = self.get_bmode()==2

 
        bmode_n0_fix_s = np.array(1 - (bmode_n0 * N_s_e_bool * N_s_a_bool), dtype=bool)
        bmode_2_fix_s = np.array(1 - (bmode_2 * N_s_e_bool * N_s_a_bool), dtype=bool)

        #self._flat_fix[self._flat_index["dict"]["range"][f"alpha_n"]] = not bmode_n0
        self._flat_fix[self._flat_index["dict"]["range"][f"alpha_n"]] = True




        self._flat_fix[self._flat_index["dict"]["range"][f"alpha_0_s"]] = bmode_n0_fix_s
        if np.any(bmode_n0_fix_s==False):
            self._flat_fix[self._flat_index["dict"]["range"][f"alpha_0_s"][np.where(bmode_n0_fix_s==False)[0][0]]] = True

        self._flat_fix[self._flat_index["dict"]["range"][f"alpha_w"]] = not bmode_n0
        self._flat_fix[self._flat_index["dict"]["range"][f"eta_w"]] = not bmode_n0
        #self._flat_fix[self._flat_index["dict"]["range"][f"eta_w"]] = True ##### eta_w_test

        self._flat_fix[self._flat_index["dict"]["range"][f"lambda_w_s"]] = bmode_2_fix_s
        #self._flat_fix[self._flat_index["dict"]["range"][f"lambda_w_s"][3]] = True


        emode_n0 = self.get_emode()!=0
        emode_fix_s = np.array(1 - (emode_n0 * N_s_a_bool), dtype=bool)

        self._flat_fix[self._flat_index["dict"]["range"][f"eta_s"][1:]] = emode_fix_s[1:]


        self._flat_fix[self._flat_index["dict"]["range"][f"delta_s"]] = np.array(1 - (N_s_a_bool * np.sum(self._rmode_array_bool[1:,:], axis=0)), dtype=bool)
        self._flat_fix[self._flat_index["dict"]["range"][f"gamma_1_s"]] = np.array(1 - (N_s_a_bool * np.sum(self._rmode_array_bool[2:,:], axis=0)), dtype=bool)

        self._flat_fix[self._flat_index["dict"]["range"][f"gamma_2_s"]] = np.array(1 - (N_s_a_bool * self._rmode_array_bool[4,:]), dtype=bool)

        if False:
            if self.get_N_s()>2:
                if N_s_a_bool[2]:
                    self._flat_fix[self._flat_index["dict"]["range"][f"gamma_2_s"][2]] = False




        #self._flat_fix[self._flat_index["dict"]["range"][f"sigma_s"]] = np.array(1 - (bmode_n0 * N_s_a_bool * np.sum(self._rmode_array_bool[1:,:], axis=0)), dtype=bool)     # A_n mod
        #self._flat_fix[self._flat_index["dict"]["range"][f"sigma_s"][3]] = bool(1 - ((1-bmode_n0) * N_s_a_bool[3]))

        if True:
            self._flat_fix[self._flat_index["dict"]["range"][f"sigma_s"]] = np.array(1 - (N_s_a_bool * np.sum(self._rmode_array_bool[1:,:], axis=0)), dtype=bool)     # A_n mod
        else:
            self._flat_fix[self._flat_index["dict"]["range"][f"sigma_s"]] = np.array(1 - (bmode_n0 * N_s_a_bool * np.sum(self._rmode_array_bool[1:,:], axis=0)), dtype=bool)     # A_n mod
            if self._bmode==0:
                self._flat_fix[np.array(self._flat_index["dict"]["range"][f"sigma_s"])[self.calc_N_s_0()]] = False


        self._flat_fix[self._flat_index["dict"]["range"][f"eta_varsigma"]] = True
        #self._flat_fix[self._flat_index["dict"]["range"][f"eta_varsigma"]] = bool(1 - N_s_a_bool[2])

        #self._flat_fix[self._flat_index["dict"]["range"][f"delta_s"][0]] = True # delta_f mod


        self._flat_fix[self._flat_index["dict"]["range"][f"alpha_0_s"]] = True


        if free:
            self.set_flat_free()





    def set_mode_defaults(self, p_0_F=None, p_0=None, mu_F=None, sigma_F=None, sigma_D=None, gamma_2=None):


        if p_0_F is None:
            p_0_F = 0
        if p_0 is None:
            p_0 = 0
        if mu_F is None:
            mu_F = 0
        if sigma_F is None:
            sigma_F = 1.0
        if sigma_D is None:
            sigma_D = np.copy(sigma_F)
        if gamma_2 is None:
            gamma_2 = sigma_D

        

        sigma_s_vec = np.zeros(self._N_s)

        if self.get_bmode()==0:     # A_n mod
            #sigma_s_vec[3] = sigma_D / 2 # sigma_D mod
            beta = 0
            alpha_n = 0
            alpha_w = 0
        else:
            beta = 1
            alpha_n = 0
            #alpha_w = 50
            alpha_w = 5.0   # alpha_w mod
            #sigma_s_vec[self.N_s_a_bool.get_value()] = sigma_D * np.sqrt(self._N_s_a) / 2 # sigma_D mod
            
        sigma_s_vec[self.N_s_a_bool.get_value()] = sigma_D #/ 2 # sigma_D mod


        delta_s = np.zeros(self._N_s)
        gamma_1_s = np.zeros(self._N_s)
        #gamma_2_s = self.calc_gamma_2_s_default()
        gamma_2_s = np.zeros(self._N_s)

        if False:
            delta_s[self.rho_type_s.get_value()=="linear"] = 0.25
            gamma_1_s[self.rho_type_s.get_value()=="linear"] = 0

            delta_s[self.rho_type_s.get_value()=="cubic"] = 0.25
            gamma_1_s[self.rho_type_s.get_value()=="cubic"] = 0.25

            delta_s[self.rho_type_s.get_value()=="tanh"] = 0.01
            gamma_1_s[self.rho_type_s.get_value()=="tanh"] = 1/0.01

            delta_s[self.rho_type_s.get_value()=="wtar"] = 0.01
            gamma_1_s[self.rho_type_s.get_value()=="wtar"] = 1
            gamma_2_s[self.rho_type_s.get_value()=="wtar"] = gamma_2

            delta_s[self.rho_type_s.get_value()=="zero"] = 0
            gamma_1_s[self.rho_type_s.get_value()=="zero"] = 0
        else:
            #delta_s[self.rho_type_s.get_value()=="linear"] = 0.1
            delta_s[self.rho_type_s.get_value()=="linear"] = 0.1/np.sqrt(np.sum(self.N_s_a_bool.get_value()))     # delta_f mod
            #delta_s[0] = 0.2/np.sqrt(np.sum(self.N_s_a_bool.get_value()))     # delta_f mod

            gamma_1_s[self.rho_type_s.get_value()=="linear"] = 0.0

            delta_s[self.rho_type_s.get_value()=="cubic"] = 0.1
            gamma_1_s[self.rho_type_s.get_value()=="cubic"] = 1E-3





            if False:
                delta_s[self.rho_type_s.get_value()=="tanh"] = 0.1
                gamma_1_s[self.rho_type_s.get_value()=="tanh"] = 1

                delta_s[self.rho_type_s.get_value()=="wtar"] = 0.1
                gamma_1_s[self.rho_type_s.get_value()=="wtar"] = 1
                gamma_2_s[self.rho_type_s.get_value()=="wtar"] = gamma_2
            else:
                delta_s[self.rho_type_s.get_value()=="tanh"] = np.sqrt(0.1)
                gamma_1_s[self.rho_type_s.get_value()=="tanh"] = np.sqrt(0.1)

                delta_s[self.rho_type_s.get_value()=="wtar"] = np.sqrt(0.1)
                gamma_1_s[self.rho_type_s.get_value()=="wtar"] = np.sqrt(0.1)
                gamma_2_s[self.rho_type_s.get_value()=="wtar"] = gamma_2


            delta_s[self.rho_type_s.get_value()=="zero"] = 0.0
            gamma_1_s[self.rho_type_s.get_value()=="zero"] = 0.0




        self.p_0_F.set_default(p_0_F)
        self.p_0.set_default(p_0)

        self.mu_F.set_default(mu_F)
        self.sigma_F.set_default(sigma_F)

        
        self.beta.set_default(beta)
        self.alpha_0_s.set_default(0.0)
        self.n_0_s.set_default(0.0)
        self.alpha_n.set_default(alpha_n)
        self.alpha_w.set_default(alpha_w)
        self.eta_w.set_default(1.0)
        self.eta_w.set_default(0.05)  ##### eta_w_test
        #self.lambda_w_s.set_default(0)

        self.psi.set_default(1.0)
        self.delta_s.set_default(delta_s)
        self.gamma_1_s.set_default(gamma_1_s)
        self.gamma_2_s.set_default(gamma_2_s)
        self.eta_s.set_default(1.0)
        
        self.sigma_s.set_default(sigma_s_vec)


        


    def calc_rho_type_int_s(self):

        return np.array([_rho_type_int[rho_type] for rho_type in self.rho_type_s.get_value()], dtype=np.int32)




    
     

###################################################################


    def calc_N_s_0(self):

        where_N_s_a = np.where(self.N_s_a_bool.get_value())[0]
        where_N_s_e = np.where(self.N_s_e_bool.get_value())[0]
        if len(where_N_s_e) > 0:
            N_s_0 = where_N_s_e[0]
        elif len(where_N_s_a) > 0:
            N_s_0 = where_N_s_a[0]
        else:
            N_s_0 = 0

        return N_s_0

    

    def set_flat_subordinate(self, free=True):      # A_n mod

        self._flat_subordinate = np.reshape(np.repeat(range(self._flat_size),2), (self._flat_size,2))

        N_s_0 = self.calc_N_s_0()

        self._flat_subordinate[self._flat_index["dict"]["range"]["lambda_w_s"], 1] = self._flat_subordinate[self._flat_index["dict"]["range"]["lambda_w_s"][N_s_0], 0]
        self._flat_subordinate[np.array(self._flat_index["dict"]["range"]["delta_s"])[self.N_s_a_bool.get_value()], 1] = self._flat_subordinate[self._flat_index["dict"]["range"]["delta_s"][N_s_0], 0]

        if self.get_bmode()==0:
            self._flat_subordinate[self._flat_index["dict"]["range"]["sigma_s"], 1] = self._flat_subordinate[self._flat_index["dict"]["range"]["sigma_s"][N_s_0], 0]

        if free:
            self.set_flat_free()


    def get_flat_subordinate(self):

        return self._flat_fix


    def set_flat_fix(self, reset=False, fix=None, unfix=None, flip=None, free=None):

        if free is None:
            free = True

        if reset:
            self._flat_fix = np.full(self._flat_size, False)
        if fix is not None:
            self._flat_fix[fix] = True
        if unfix is not None:
            self._flat_fix[unfix] = False
        if flip is not None:
            self._flat_fix[flip] = ~self._flat_fix[flip]

        if free:
            self.set_flat_free()


     
    def get_flat_fix(self):

        return self._flat_fix

    def set_flat_free(self):
        
        self._flat_free = np.logical_and(~self._flat_fix, self._flat_subordinate[:,0]==self._flat_subordinate[:,1])
        self._N_params = np.sum(self._flat_free)

    def get_flat_free(self):

        return self._flat_free
        
        
        

    def flatten(self):
        return flatten_parameters(self.mu_F._value, self.sigma_F._value, self.p_0_F._value, self.p_0._value,
                                        self.beta._value, self.alpha_0_s._value, self.alpha_n._value, self.alpha_w._value,
                                        self.eta_w._value, self.lambda_w_s._value,
                                        self.psi._value, self.delta_s._value, self.gamma_1_s._value, self.gamma_2_s._value, self.eta_s._value, self.sigma_s._value, 
                                        self.eta_varsigma._value, self.varsigma_0._value,
                                        )

    def flatten_defaults(self):
        return flatten_parameters(self.mu_F._default, self.sigma_F._default, self.p_0_F._default, self.p_0._default,
                                        self.beta._default, self.alpha_0_s._default, self.alpha_n._default, self.alpha_w._default,
                                        self.eta_w._default, self.lambda_w_s._default,
                                        self.psi._default, self.delta_s._default, self.gamma_1_s._default, self.gamma_2_s._default, self.eta_s._default, self.sigma_s._default, 
                                        self.eta_varsigma._default, self.varsigma_0._default,
                                        )


    def unflatten_parameters(self, flat_parameters):
        
        return [np.resize(flat_parameters[self._flat_index["dict"]["range"][name]], self.__dict__[name]._shape) for name in self._fp_name]


    def set_flat_parameters(self, flat_parameters, subordinate=None, push=None):

        if subordinate is None:
            subordinate = True
        if push is None:
            push = True

        if subordinate:
            self._flat_parameters[self._flat_subordinate[~self._flat_fix,0]] = flat_parameters[self._flat_subordinate[~self._flat_fix,1]]
        if push:
            self.push_flat_parameters()

    def set_flat_parameters_free(self, flat_parameters_free, flat_free):

        self._flat_parameters[flat_free] = flat_parameters_free
        self._flat_parameters[self._flat_subordinate[~self._flat_fix,0]] = self._flat_parameters[self._flat_subordinate[~self._flat_fix,1]]
        self.push_flat_parameters()


    def get_flat_parameters(self, pull=None):

        if pull is None:
            pull=True

        if pull:
            self.pull_flat_parameters()
            
        return self._flat_parameters

    


    def push_flat_parameters_orig(self):

        unflat_parameters = self.unflatten_parameters(self._flat_parameters)
        

        self.mu_F._value = unflat_parameters[0][0]
        self.sigma_F._value = unflat_parameters[1][0]
        self.p_0_F._value = unflat_parameters[2][0]
       
        self.beta._value = unflat_parameters[3][0]
        self.alpha_0_s._value = unflat_parameters[4]
        self.alpha_n._value = unflat_parameters[5][0]
        self.alpha_w._value = unflat_parameters[6][0]
                                        
        self.eta_w._value = unflat_parameters[7][0]
        self.lambda_w_s._value = unflat_parameters[8]
        
        self.psi._value = unflat_parameters[9][0]
        self.delta_s._value = unflat_parameters[10]
        self.gamma_1_s._value = unflat_parameters[11]
        self.gamma_1_s._value = unflat_parameters[12]
        self.eta_s._value = unflat_parameters[13]
        self.sigma_s._value = unflat_parameters[14]

        self.eta_varsigma._value = unflat_parameters[15]
        self.varsigma_0._value = unflat_parameters[16]
                                                          


    def push_flat_parameters(self):

        unflat_parameters = self.unflatten_parameters(self._flat_parameters)
        
        for k in range(len(self._fp_name)):
            name = self._fp_name[k]
            if self.__dict__[name]._stype=="scalar":
                self.__dict__[name]._value = np.float64(unflat_parameters[k])
            else:
                self.__dict__[name]._value = unflat_parameters[k]


       

    def pull_flat_parameters(self):

        self._flat_parameters = self.flatten()




        



    def set_all(self, rbe_str=None, rmode=None, bmode=None, emode=None,
                    mu_F=None, sigma_F=None, p_0_F=None, p_0=None,
                    N_s_a_bool=None, N_s_e_bool=None, beta=None,  
                    alpha_0_s=None, n_0_s=None,
                    alpha_n=None, alpha_w=None,
                    eta_w=None, lambda_w_s=None,
                    G_sigma=None,
                    psi=None, delta_s=None, gamma_1_s=None, gamma_2_s=None, eta_s=None, sigma_s=None,
                    rho_type_s=None, 
                    eta_varsigma=None,
                    varsigma_0=None,
                    pull=True, fix=True):
        """
        """
        
        
        self.set_mode(rbe_str=rbe_str, rmode=rmode, bmode=bmode, emode=emode)

        
        self.set_p_F_parameters(mu_F=mu_F, sigma_F=sigma_F, p_0_F=p_0_F, p_0=p_0, 
                                pull=False)
        self.set_fraction_parameters(N_s_a_bool=N_s_a_bool, N_s_e_bool=N_s_e_bool, beta=beta, 
                                        alpha_0_s=alpha_0_s, n_0_s=n_0_s,
                                        alpha_n=alpha_n, alpha_w=alpha_w,
                                        eta_w=eta_w, lambda_w_s=lambda_w_s,
                                        G_sigma=G_sigma,
                                        pull=False)

        self.set_demand_parameters(psi=psi, delta_s=delta_s, gamma_1_s=gamma_1_s, gamma_2_s=gamma_2_s, eta_s=eta_s, sigma_s=sigma_s,
                                        rho_type_s=rho_type_s,
                                        eta_varsigma=eta_varsigma,
                                        varsigma_0=varsigma_0,
                                        pull=False)
        
        
        
        if fix:
            self.set_flat_fix_mode()
        if pull:
            self.pull_flat_parameters()


    def set_p_F_parameters(self, mu_F=None, sigma_F=None, p_0_F=None, p_0=None, pull=None):
        """
        """
        if pull is None:
            pull = True


        if mu_F is not None:
            self.mu_F.set_value(value=mu_F)
        if sigma_F is not None:
            self.sigma_F.set_value(value=sigma_F) 
        if p_0_F is not None:
            self.p_0_F.set_value(value=p_0_F)   
        if p_0 is not None:
            self.p_0.set_value(value=p_0)     
        if pull:
            self.pull_flat_parameters()   


    def set_fraction_parameters(self,
                                N_s_a_bool=None, N_s_e_bool=None, beta=None, 
                                        alpha_0_s=None, n_0_s=None,
                                        alpha_n=None, alpha_w=None,
                                        eta_w=None, lambda_w_s=None,
                                        G_sigma=None,
                                        pull=None
                                    ):

        if pull is None:
            pull = True

        self.set_N_s_a_bool(N_s_a_bool=N_s_a_bool)
        self.set_N_s_e_bool(N_s_e_bool=N_s_e_bool) 

        if beta is not None:
            self.beta.set_value(value=beta)  

        if n_0_s is not None:
            self.set_n_0_s(n_0_s=n_0_s)
        elif alpha_0_s is not None:
            self.set_alpha_0_s(alpha_0_s=alpha_0_s)

        if alpha_n is not None:
            self.alpha_n.set_value(value=alpha_n)

        if alpha_w is not None:
            self.alpha_w.set_value(value=alpha_w)

        if eta_w is not None:
            self.eta_w.set_value(value=eta_w)

        if lambda_w_s is not None:
            self.lambda_w_s.set_value(value=lambda_w_s)

        if G_sigma is not None:
            self.G_sigma.set_value(value=G_sigma)

        
        if pull:
            self.pull_flat_parameters()  




    def set_N_s_a_bool(self, N_s_a_bool=None):

        self.N_s_a_bool.set_value(value=N_s_a_bool)
        self.set_N_s_a()

    def set_N_s_a(self):

        self._N_s_a = np.sum(self.N_s_a_bool.get_value())


    def set_N_s_e_bool(self, N_s_e_bool=None):

        self.N_s_e_bool.set_value(value=N_s_e_bool)
        self.set_N_s_e()

    def set_N_s_e(self):

        self._N_s_e = np.sum(self.N_s_e_bool.get_value())
        
            
    def set_n_0_s(self, n_0_s=None):

        self.n_0_s.set_value(value=n_0_s)
        self.update_alpha_0_s(from_n=True)
        self.update_n_0_s()
    
    def set_alpha_0_s(self, alpha_0_s=None):

        self.alpha_0_s.set_value(value=alpha_0_s)
        self.update_n_0_s(from_alpha=True)
        self.update_alpha_0_s()


    def update_n_0_s(self, from_alpha=None):

        if from_alpha is None:
            from_alpha=False
        if from_alpha:
            self.n_0_s._value = np.exp(self.beta.get_value()*self.alpha_0_s.get_value())
            n_0_s_e = self.n_0_s._value[self.N_s_e_bool.get_value()]
            self.n_0_s._value[self.N_s_e_bool.get_value()] = n_0_s_e/np.sum(n_0_s_e)


        self.n_0_s._value[~self.N_s_a_bool.get_value()] = np.nan



    def update_alpha_0_s(self, from_n=None):

        if from_n is None:
            from_n=False
        if from_n:
            self.alpha_0_s._value = np.log(self.n_0_s.get_value())/self.beta.get_value()
            self.alpha_0_s._value[self.n_0_s.get_value()==0] = -np.inf

        self.alpha_0_s._value[~self.N_s_a_bool.get_value()] = np.nan

    
    def set_demand_parameters(self,
                                psi=None, delta_s=None, gamma_1_s=None, gamma_2_s=None, eta_s=None, sigma_s=None,
                                rho_type_s=None,
                                eta_varsigma=None,
                                varsigma_0=None,
                                pull=None):

        if pull is None:
            pull = True

        if psi is not None:
            self.psi.set_value(value=psi)

        if delta_s is not None:
            self.delta_s.set_value(value=delta_s)

        if gamma_1_s is not None:
            self.gamma_1_s.set_value(value=gamma_1_s)

        if gamma_2_s is not None:
            self.gamma_2_s.set_value(value=gamma_2_s)

        if eta_s is not None:
            self.eta_s.set_value(value=eta_s)

        if sigma_s is not None:
            self.sigma_s.set_value(value=sigma_s)

        if rho_type_s is not None:
            self.rho_type_s.set_value(value=rho_type_s)
        

        if eta_varsigma is not None:
            self.eta_varsigma.set_value(value=eta_varsigma)

        if varsigma_0 is not None:
            self.varsigma_0.set_value(value=varsigma_0)

        if pull:
            self.pull_flat_parameters()  


            
    

    def rho(self, x=None, s=None):

        if s is None:
            s=0

        return rho(x=x, type=self.rho_type_s.get_value()[s], gamma_1=self.gamma_1_s.get_value()[s], gamma_2=self.gamma_2_s.get_value()[s])
            


    def rho_njit(self, x=None, s=None):

        if s is None:
            s=0

        return ssv_abm_nb.rho_njit(x=x, type_int=_rho_type_int[self.rho_type_s.get_value()[s]], gamma_1=self.gamma_1_s.get_value()[s], gamma_2=self.gamma_2_s.get_value()[s])



    def print(self):
        """
        """

        
        
        
        print("")
        print("==================")
        print("PRUNA-WHITEHOUSE PARAMETERS")
        print("==================")
        print("")
        print("Fundamental price parameters:")
        print(f"mu_F = {self.mu_F.get_value()}")
        print(f"sigma_F = {self.sigma_F.get_value()}")
        print(f"p_0_F = {self.p_0_F.get_value()}")
        print(f"p_0 = {self.p_0.get_value()}")
        print("")
        print("Fraction parameters:")
        print(f"N_s_a_bool = {self.N_s_a_bool.get_value()}")
        print(f"N_s_e_bool = {self.N_s_e_bool.get_value()}")
        print(f"beta = {self.beta.get_value()}")
        print(f"n_0_s = {self.n_0_s.get_value()}")
        print(f"alpha_0_s = {self.alpha_0_s.get_value()}")
        print(f"alpha_n = {self.alpha_n.get_value()}")
        print(f"alpha_w = {self.alpha_w.get_value()}")
        print(f"eta_w = {self.eta_w.get_value()}")
        print(f"lambda_w_s = {self.lambda_w_s.get_value()}")
        print("")
        print("Demand parameters:")
        print(f"psi = {self.psi.get_value()}")
        print(f"delta_s = {self.delta_s.get_value()}")
        print(f"gamma_1_s = {self.gamma_1_s.get_value()}")
        print(f"gamma_2_s = {self.gamma_2_s.get_value()}")
        print(f"eta_s = {self.eta_s.get_value()}")
        print(f"sigma_s = {self.sigma_s.get_value()}")
        print("")
        print("Volatility parameters:")
        print(f"eta_varsigma = {self.eta_varsigma.get_value()}")
        print(f"varsigma_0 = {self.varsigma_0.get_value()}")
        print("")

    def print_fit(self):

        for param in self._fit_est:
            print(f"{param} = {self._fit_est[param]} +/- {self._fit_se[param]}")



    def set_Bounds(self, delta_excl=None):

        self._Bounds = np.zeros((np.sum(self._flat_index["list"]["size"]), 2))
        for name in self._fp_name:
            self._Bounds[self._flat_index["dict"]["range"][name],:] = self.__dict__[name]._p_range.get_Bounds(delta_excl=delta_excl)


    def set_opt_Bounds(self, delta_excl=None):

        self._opt_Bounds = np.zeros((np.sum(self._flat_index["list"]["size"]), 2))
        for name in self._fp_name:
            self._opt_Bounds[self._flat_index["dict"]["range"][name],:] = self.__dict__[name]._opt_range.get_Bounds(delta_excl=delta_excl)


    def stypes_shapes_sizes(self):
    


        stypes = ["scalar", "scalar", "scalar", "scalar", "scalar", "vector", "scalar", "scalar", "scalar", "vector", "scalar", "vector", "vector", "vector", "vector", "vector", "scalar"]
        stypes_shapes = {
                            "scalar": (),
                            "vector": (self._N_s),
                            "matrix": (self._N_s, self._N_s),
                        }
        stypes_sizes = np.array([int(np.product(stypes_shapes[i])) for i in stypes])
        index_sizes_cumsum = np.zeros(stypes_sizes.size+1, dtype=int)
        index_sizes_cumsum[1:] = np.cumsum(stypes_sizes)

        return stypes, stypes_shapes, stypes_sizes, index_sizes_cumsum


   



    def set_fit(self, se=None, fp1=None):

        est = self.flatten()
        N_params = np.size(est)
        if se is None:
            se = [None for i in range(N_params)]
        if fp1 is None:
            fp1 = [None for i in range(N_params)]
        for i in range(N_params):
            name = self._fp_names[i]
            if est[i] != 0:
                self._fit_est[name] = est[i]
                self._fit_se[name] = se[i]
                self._fit_fp1[name] = fp1[i]
            else:
                self._fit_est[name] = None
                self._fit_se[name] = None
                self._fit_fp1[name] = None



    def calc_N_params(self, component=None):

        if component is None:
            component = "LA"

        flat_free = np.copy(self._flat_free)
        select = np.full(flat_free.size, True)

        if component=="F":
            select[:] = False
            for param in ["p_0_F", "mu_F", "sigma_F"]:
                 select[self._flat_index["dict"]["range"][param]] = True
     

        return np.sum(np.logical_and(flat_free,select))



    def calc_fit_df(self, fdict=None, lead=None, tail=None):

        if fdict is None:
            fdict=="est"

        if fdict=="est":
            return self.calc_fit_est_df(lead=lead, tail=tail)
        elif fdict=="se":
            return self.calc_fit_se_df(lead=lead, tail=tail)
        elif fdict=="fp1":
            return self.calc_fit_fp1_df(lead=lead, tail=tail)




    def calc_fit_est_df(self, lead=None, tail=None):
        if lead is None:
            lead = dict()
        if tail is None:
            tail = dict()


        fit_dict = {
                        **lead,
                        **self.res_log_L_dict["spec"],
                        **self.calc_mode_dict(),
                        **self.calc_fixed_param_dict(),
                        **self._fit_est,
                        **self._fit_scores,
                        **{f"{cat}_{time}": self.res_time[cat][time] for cat in self.res_time for time in self.res_time[cat]},
                        **tail,
                    }

        return pd.DataFrame(fit_dict, index=[0])

    def calc_fit_se_df(self, lead=None, tail=None):
        if lead is None:
            lead = dict()
        if tail is None:
            tail = dict()

        fit_dict = {
                        **lead,
                        **self.res_log_L_dict["spec"],
                        **self.calc_mode_dict(),
                        **self.calc_fixed_param_dict(),
                        **self._fit_se,
                        **tail,
                    }

        return pd.DataFrame(fit_dict, index=[0])

    def calc_fit_fp1_df(self, lead=None, tail=None):
        if lead is None:
            lead = dict()
        if tail is None:
            tail = dict()

        fit_dict = {
                        **lead,
                        **self.res_log_L_dict["spec"],
                        **self.calc_mode_dict(),
                        **self.calc_fixed_param_dict(),
                        **self._fit_fp1,
                        **tail,
                    }

        return pd.DataFrame(fit_dict, index=[0])
    
    





class pruna_process:
    """Class for pruna process
    """

    def __init__(self, N_s=None,
                    p_t=None, r_t=None, p_0=None, 
                    meta_series=None, fit_dir_r=None, datetime_w_str=None,
                    insert=None, opt_method=None,
                    rmode_s0s=None, rmode_s1s=None, rmode_s2s=None, bmodes=None, emodes=None, 
                    T=None, T_burn=None,
                    train_t_start=None, train_t_end=None,
                    use_fp0=None, check_opt_last=None,
                    epsilon=None, step_norm_cutoff=None, N_k=None, N_k_trunc=None, gauss_sigma=None,
                ):
        """
        """
        if N_s is None:
            N_s=3   # A_n mod

        self._N_s = parameters.clean_D_nodes(D_nodes=N_s)
        self._parameters = pruna_parameters(N_s=self._N_s, 
                                            insert=insert, opt_method=opt_method, 
                                            #rmode=rmode, bmode=bmode, emode=emode,
                                            #mu_F=None, sigma_F=None, p_0_F=None, p_0=None,
                                            #N_s_a_bool=None, N_s_e_bool=None, beta=None,
                                            #alpha_0_s=None, n_0_s=None,
                                            #alpha_n=None, alpha_w=None,
                                            #eta_w=None, lambda_w_s=None,
                                            #psi=None, delta_s=None, gamma_1_s=None, gamma_2_s=None, eta_s=None, sigma_s=None,
                                            #rho_type_s=None, 
                                            #eta_varsigma=None,
                                            ###a_u=a_u, u_dir=u_dir, mode=mode, mark=mark, fix_a_lambda=fix_a_lambda, insert=insert, opt_method=opt_method
                                            )

        self._T_buff = 2

        T, T_burn = calc_T_T_burn(p_t=p_t, r_t=r_t, p_0=p_0, T=T, T_burn=T_burn)
        self.set_T(T=T, reset=True)
        self.set_T_burn(T_burn=T_burn, reset=True)



        self.set_use_fp0(use_fp0=use_fp0, reset=True)
        self.set_check_opt_last(check_opt_last=check_opt_last, reset=True)


        self.set_N_k_trunc(N_k_trunc=N_k_trunc, reset=True)
        self.set_gauss_sigma(gauss_sigma=gauss_sigma, reset=True)
        self.set_epsilon(epsilon=epsilon, reset=True)
        self.set_step_norm_cutoff(step_norm_cutoff=step_norm_cutoff, reset=True)

            

        self.set_train_t(start=train_t_start, end=train_t_end, reset=True)


        self.reset_process()


        self._fit_scores = misc_fns.IC_dict()

        self._Hcomp = dict()
        self.set_Hcomp_new(reset=True)

        self.reset_Hess_Jac_log_L()

        self.reset_log_L_dict(N_k=N_k, reset=True, keep_opt_last=False)


        self.reset_rw(meta_series=meta_series)
        self.load_p_t_r_t(p_t=p_t, r_t=r_t, p_0=p_0, run=False)

        self.set_fit_dir_r(fit_dir_r=fit_dir_r, reset=True)
        self.set_datetime_w_str(datetime_w_str=datetime_w_str, set_fit_dir_w=False, reset=True)
        self._fit_dir_w = None

        self.reset_modes(init=True)
        self.set_modes(rmode_s0=rmode_s0s, rmode_s1=rmode_s1s, rmode_s2=rmode_s2s, bmode=bmodes, emode=emodes, reset=True, set_rbe_0=True, set_defaults=True)

        self.reset_fit_df_dict()





#--#===========================
    #--------------------------
    # Meta
    #--------------------------
    #===========================

     


    def reset_rw(self, meta_series=None):
        self._rw = dict()
        self._rw['meta'] = dict()
        self.set_meta_series(meta_series=meta_series)

        for cat in ['fit']:
            self._rw[cat] = {rw: dict() for rw in ['r', 'w']}

    
        self._rw['fit']['w']['pabm'] = empty_fit_df_dict(incl_meta=True)

    def set_meta_series(self, meta_series=None):
        if meta_series is not None:
            self._rw['meta']['series'] = meta_series


    
    def load_p_t_r_t_old(self, p_t=None, r_t=None, p_0=None, run=None):

        if run is None:
            run = False

        
        self.sim_p_t_reset(reset_t_series=False)



        if p_t is not None or r_t is not None:
            if p_t is not None:
                if p_0 is None:
                    p_0 = p_t[0]
                if p_t.size==self.get_T()+1:
                    r_t = np.diff(p_t)
                    p_t = p_t[1:]
                else:
                    r_t = np.concatanate([[0.0], np.diff(p_t)])
            else:
                if p_0 is None:
                    p_0 = 0.0
                p_t = p_0 + np.cumsum(r_t)
        


            self._parameters.p_0.set_value(p_0)

            varsigma_0 = np.std(r_t)/self._parameters.psi.get_value()
            self._parameters.varsigma_0.set_value(varsigma_0)
            self._parameters.varsigma_0.set_default(varsigma_0)


            self._t_series["p"][:2] = p_0
            self._t_series["r"][:2] = 0.0

            self._t_series["p"][2:] = p_t
            self._t_series["r"][2:] = r_t


            p_F_0 = self.calc_p_F_0()
            
            self._t_series["p_F"][2:] = p_F_0
            self._t_series["r_F"][2:] = self._parameters.psi.get_value() * self._parameters.mu_F.get_value()

            self._t_series["p_F"][:2] = self._parameters.p_0_F.get_value()
            self._t_series["r_F"][:2] = 0.0

        if run:
            self.sim_p_t_nb(sim=False)








    def load_p_t_r_t(self, p_t=None, r_t=None, p_0=None, update_p_F_param=None, gauss_sigma=None, run=None):

        if run is None:
            run = False

        if update_p_F_param is None:
            update_p_F_param = True

        
        self.sim_p_t_reset(reset_t_series=False)



        if p_t is not None or r_t is not None:
            if p_t is not None:
                if p_0 is None:
                    p_0 = p_t[0]
                if p_t.size==self.get_T()+1:
                    r_t = np.diff(p_t)
                    p_t = p_t[1:]
                else:
                    r_t = np.concatanate([[0.0], np.diff(p_t)])
            else:
                if p_0 is None:
                    p_0 = 0.0
                p_t = p_0 + np.cumsum(r_t)
        


            self._parameters.p_0.set_value(p_0)
            self._parameters.p_0.set_default(p_0)
            


            varsigma_0 = np.std(r_t)/self._parameters.psi.get_value()
            self._parameters.varsigma_0.set_value(varsigma_0)
            self._parameters.varsigma_0.set_default(varsigma_0)


            self._t_series["p"][:2] = p_0
            self._t_series["r"][:2] = 0.0

            self._t_series["p"][2:] = p_t
            self._t_series["r"][2:] = r_t

            if update_p_F_param:
                mu_F, p_0_F, p_0, sigma_F = self.calc_theta_p_F_endpoints(est_p_0=False, p_0=p_0)

                self._parameters.p_0_F.set_value(p_0_F)
                self._parameters.p_0_F.set_default(p_0_F)

                self._parameters.mu_F.set_value(mu_F)
                self._parameters.mu_F.set_default(mu_F)

            self.set_gauss_sigma(gauss_sigma=gauss_sigma, reset=False)
            p_F_0 = self.calc_p_F_0_gauss()
            
            self._t_series["p_F"][2:] = p_F_0
            self._t_series["p_F"][:2] = self._parameters.p_0_F.get_value()

            self._t_series["r_F"][:1] = 0.0
            self._t_series["r_F"][1:] = np.diff(self._t_series["p_F"])

        if run:
            self.sim_p_t_nb(sim=False)






    def set_fit_dir_r(self, fit_dir_r=None, reset=None):

        if reset is None:
            reset = False

        if reset:
            if fit_dir_r is None:
                fit_dir_r = f"../results/fit/"

        if fit_dir_r is not None:
            self._fit_dir_r = plot_fns.mkdir_export(fit_dir_r)
        else:
            if self._fit_dir_r is not None:
                self._fit_dir_r = plot_fns.mkdir_export(self._fit_dir_r)


    def set_datetime_w_str(self, datetime_w_str=None, set_fit_dir_w=None, reset=None):

        if set_fit_dir_w is None:
            set_fit_dir_w = True

        if datetime_w_str is None:
            if reset:
                datetime_w_str = misc_fns.dt_str(datetime.datetime.utcnow())
        
        if datetime_w_str is not None:
            self._datetime_w_str = datetime_w_str

        if set_fit_dir_w:
            self.set_fit_dir_w()


    def set_fit_dir_w(self):

        self._fit_dir_w = plot_fns.mkdir_export(f"{self._fit_dir_r}{self._datetime_w_str}/")







#--#===========================
    #--------------------------
    # Setup
    #--------------------------
    #===========================
    
    def set_T(self, T=None, reset=None):

        if reset is None:
            reset = False
        if reset:
            self._T = 10000
        if T is not None:
            self._T = T
            self.reset_ts()

    def reset_ts(self):

        self._ts = np.arange(self.get_T())


    def set_T_burn(self, T_burn=None, reset=None):

        if reset is None:
            reset = False
        if reset:
            self._T_burn = 10000
        if T_burn is not None:
            self._T_burn = T_burn


    def get_T_buff(self):
        return self._T_buff

    def get_T(self):
        return self._T

    def get_ts(self):
        return self._ts

    def get_T_burn(self):
        return self._T_burn

    def get_T_tot(self):
        return self.get_T_burn() + self.get_T()

    def get_T_tot_buff(self):
        return self.get_T_buff() + self.get_T_tot()

    def get_T_0(self):
        return self.get_T_buff() + self.get_T_burn()


    def get_t_series(self, series=None, T_0=None):

        if T_0 is None:
            T_0 = self.get_T_0()

        if series is not None:
            return self._t_series[series][T_0:]


    def get_t_s_series(self, series=None, T_0=None):

        if T_0 is None:
            T_0 = self.get_T_0()

        if series is not None:
            return self._t_s_series[series][:, T_0:]


    def reset_Hess_Jac_log_L_old(self):

        T = self.get_T()

        self._Hess = np.full((T,T), np.nan)
        self._Jac = np.full((T), np.nan)

        self._log_L_FD = np.nan
        self._log_L_H = np.nan
        self._log_L_LA = np.nan

    def reset_Hess_Jac_log_L(self):

        T = self.get_T()

        self._Hess = np.full((T,T), np.nan)
        self._Jac = np.full((T), np.nan)

        self._log_L = {
                        "LA": np.nan,
                        "FDH": np.repeat(np.nan, 3),
                    }

    
    def set_epsilon(self, epsilon=None, reset=None):

        if reset is None:
            reset = False
        if reset:
            self._epsilon = 1E-7
        if epsilon is not None:
            self._epsilon = epsilon


    def get_epsilon(self):

        return self._epsilon



    def set_N_k_trunc(self, N_k_trunc=None, reset=None):

        if reset is None:
            reset = False
        if reset:
            self._N_k_trunc = 4
        if N_k_trunc is not None:
            self._N_k_trunc = N_k_trunc


    def get_N_k_trunc(self):

        return self._N_k_trunc
    

    def set_gauss_sigma(self, gauss_sigma=None, reset=None):

        if reset is None:
            reset = False
        if reset:
            self._gauss_sigma = 1.0
        if gauss_sigma is not None:
            self._gauss_sigma = gauss_sigma


    def get_gauss_sigma(self):

        return self._gauss_sigma
    



    def set_step_norm_cutoff(self, step_norm_cutoff=None, reset=None):

        if reset is None:
            reset = False
        if reset:
            #self._step_norm_cutoff = 1E-3   # step_norm_cutoff mod
            #self._step_norm_cutoff = 2E-4   
            self._step_norm_cutoff = 2E-4   
        if step_norm_cutoff is not None:
            self._step_norm_cutoff = step_norm_cutoff


    def get_step_norm_cutoff(self):

        return self._step_norm_cutoff
    


    def set_check_opt_last(self, check_opt_last=None, reset=None):

        if reset is None:
            reset = False
        if reset:
            self._check_opt_last = False
        if check_opt_last is not None:
            self._check_opt_last = check_opt_last
    

    def get_check_opt_last(self):

        return self._check_opt_last



    def set_p_F_np(self, p_F, T_0=None, run_p_t=None, set_Hess_adj=None, set_log_L=None):

        if run_p_t is None:
            run_p_t = True


        if T_0 is None:
            T_0 = self.get_T_0()

        self._t_series["p_F"][T_0:] = p_F
        self._t_series["p_F"][:T_0] = self._parameters.p_0_F.get_value()

        self._t_series["r_F"] = misc_fns.diff_zero(self._t_series["p_F"], v_zero=0)

        if run_p_t:
            self.sim_p_t(sim=False, set_Hess_adj=set_Hess_adj, set_log_L=set_log_L)


    def set_p_F_return(self, p_F, T_0=None):

        self.set_p_F_np(p_F=p_F, T_0=T_0, run_p_t=True, set_log_L=True)
        return self.get_log_L_LA(), self.get_Hess(), self.get_Jac()


    def set_p_F_return_minus(self, p_F, T_0=None):

        log_L, Hess, Jac = self.set_p_F_return(p_F=p_F, T_0=T_0)
        return -log_L, -Hess, -Jac



    def sim_p_t_reset(self, reset_t_series=None, force_n_0_e=None):


        T_tot = self.get_T_tot()

        if reset_t_series is None:
            reset_t_series = False
        if force_n_0_e is None:
            force_n_0_e = False

        T_buff = self.get_T_buff()


        if reset_t_series:
            self.reset_t_series()


        self.reset_t_s_series()

        self._t_s_series["nepsilon"] = scipy.stats.norm().rvs((self.get_N_s(),self.get_T_tot_buff()))

        #self._t_s_series["wrho"][:,:T_buff] = 0
        self._t_s_series["rho"][:,:T_buff] = 0.0
        self._t_s_series["D"][:,:T_buff] = np.outer(self._parameters.sigma_s.get_value(), np.ones(2)) * self._t_s_series["nepsilon"][:,:T_buff]

        self._t_series["p"][:T_buff] = self._parameters.p_0.get_value() - (self.get_T_burn() * self._parameters.psi.get_value() * self._parameters.mu_F.get_value())
        self._t_series["r"][:T_buff] = 0.0
        self._t_series["p_F"][:T_buff] = self._parameters.p_0_F.get_value() - (self.get_T_burn() * self._parameters.psi.get_value() * self._parameters.mu_F.get_value())

        self._t_series["r_F"][:T_buff] = 0.0

        self._t_series["varsigma"][:T_buff] = 0.0



        self._t_series["r_F"][T_buff:] = self._parameters.psi.get_value() * (self._parameters.mu_F.get_value() + self._parameters.sigma_F.get_value()*scipy.stats.norm().rvs(T_tot))
        self._t_series["p_F"][T_buff:] = self._parameters.p_0_F.get_value() + np.cumsum(self._t_series["r_F"][T_buff:])

        self._t_s_series["S"][:,:T_buff] = 0
        self._t_s_series["M"][:,:T_buff] = 0


        if False:
            if self.get_N_s()>=4:   # A_n mod
                self._t_s_series["S"][3,T_buff:] = 0
                self._t_s_series["M"][3,T_buff:] = 0


        N_s_a_bool = self._parameters.N_s_a_bool.get_value()
        N_s_e_bool = self._parameters.N_s_e_bool.get_value()

        self._t_s_series["a"][:,:T_buff] = np.outer(self._parameters.alpha_0_s.get_value(), np.ones(2))
        #self._t_s_series["n"][self._parameters.N_s_e_bool.get_value(),:T_buff]  = np.exp(self._parameters.beta.get_value()*self._t_s_series["a"][self._parameters.N_s_e_bool.get_value(),:T_buff])/np.sum(np.exp(self._parameters.beta.get_value()*self._t_s_series["a"][self._parameters.N_s_e_bool.get_value(),:T_buff]), axis=0)
        
        if np.any(N_s_e_bool):
            #self._t_s_series["n"][self._parameters.N_s_e_bool.get_value(),:T_buff]  = ssv_abm_nb.calc_n_t_s_2d(a_t_s=self._t_s_series["a"][self._parameters.N_s_e_bool.get_value(),:T_buff], beta=self._parameters.beta.get_value()) # A_n mod
            self._t_s_series["n"][N_s_e_bool,:T_buff]  = ssv_abm_nb.calc_A_n(N_s_a_bool=N_s_a_bool) * ssv_abm_nb.calc_n_t_s_2d(a_t_s=self._t_s_series["a"][N_s_e_bool,:T_buff], beta=self._parameters.beta.get_value())
        
        
        if force_n_0_e: # A_n mod
            self._t_s_series["n"][~self._parameters.N_s_e_bool.get_value(),:] = self._parameters.n_0_s.get_value()[~self._parameters.N_s_e_bool.get_value()]
        else:
            #self._t_s_series["n"][~self._parameters.N_s_e_bool.get_value(),:] = 1  # A_n mod
            
            self._t_s_series["n"][~self._parameters.N_s_e_bool.get_value(),:] = ssv_abm_nb.calc_A_n(N_s_a_bool=N_s_a_bool)/np.sum(N_s_a_bool)

        self._t_s_series["n"][~self._parameters.N_s_a_bool.get_value(),:] = 0



        self._t_s_series["g"][:,:T_buff] = 0
        self._t_s_series["w"][:,:T_buff] = 0
        self._t_s_series["nu"][:,:T_buff] = 0


        self._t_series["p_F_sim"] = np.copy(self._t_series["p_F"])
        self._t_series["r_F_sim"] = np.copy(self._t_series["r_F"])

        self.reset_log_L_dict_track()





    def sim_p_t(self, sim=None, set_Hess_adj=None, set_log_L=None):

        if sim is None:
            sim = True

        if set_Hess_adj==None:
            set_Hess_adj=False

        if set_log_L is None:
            set_log_L = False


        rho_type_int_s = self._parameters.calc_rho_type_int_s()



        T_buff = self.get_T_buff()

        T_0 = self.get_T_0()
        T_1 = self.get_T_tot_buff()

        self._t_series["p"], self._t_series["r"], self._t_s_series["D"], self._t_s_series["a"], self._t_s_series["n"], self._t_s_series["S"], self._t_s_series["M"], self._t_s_series["rho"], self._t_s_series["g"], self._t_s_series["w"], self._t_s_series["nu"] = ssv_abm_nb.run_p_t_njit(sim=sim, T_buff=T_buff, T_1=T_1, 
                                                                                                    psi=np.float64(self._parameters.psi.get_value()), 
                                                                                                    delta_s=self._parameters.delta_s.get_value(), 
                                                                                                    gamma_1_s=self._parameters.gamma_1_s.get_value(), 
                                                                                                    gamma_2_s=self._parameters.gamma_2_s.get_value(), 
                                                                                                    sigma_s=self._parameters.sigma_s.get_value(), 
                                                                                                    eta_s=self._parameters.eta_s.get_value(), 
                                                                                                    eta_varsigma=np.float64(self._parameters.eta_varsigma.get_value()), 
                                                                                                    varsigma_0=np.float64(self._parameters.varsigma_0.get_value()), 
                                                                                                    beta=np.float64(self._parameters.beta.get_value()), 
                                                                                                    eta_w=np.float64(self._parameters.eta_w.get_value()), 
                                                                                                    alpha_0_s=self._parameters.alpha_0_s.get_value(), 
                                                                                                    alpha_n=np.float64(self._parameters.alpha_n.get_value()), 
                                                                                                    alpha_w=np.float64(self._parameters.alpha_w.get_value()), 
                                                                                                    lambda_w_s=self._parameters.lambda_w_s.get_value(), 
                                                                                                    G_sigma=np.float64(self._parameters.G_sigma.get_value()),
                                                                                                    A_G=np.float64(self._parameters.calc_A_G()),  
                                                                                                    N_s_a_bool=self._parameters.N_s_a_bool.get_value(), 
                                                                                                    N_s_e_bool=self._parameters.N_s_e_bool.get_value(), 
                                                                                                    rho_type_int_s=rho_type_int_s, 
                                                                                                    nepsilon_t_s=self._t_s_series["nepsilon"], 
                                                                                                    p_F=self._t_series["p_F"], 
                                                                                                    p=self._t_series["p"], 
                                                                                                    r=self._t_series["r"], 
                                                                                                    varsigma_t=self._t_series["varsigma"], 
                                                                                                    D_t_s=self._t_s_series["D"], 
                                                                                                    a_t_s=self._t_s_series["a"], 
                                                                                                    n_t_s=self._t_s_series["n"], 
                                                                                                    S_t_s=self._t_s_series["S"], 
                                                                                                    M_t_s=self._t_s_series["M"], 
                                                                                                    rho_t_s=self._t_s_series["rho"], 
                                                                                                    g_t_s=self._t_s_series["g"], 
                                                                                                    w_t_s=self._t_s_series["w"], 
                                                                                                    nu_t_s=self._t_s_series["nu"]
                                                                                                    )


        self.set_Hcomp_new(reset=False, reset_T=False, reset_param=False, reset_p_p_F=True)

        
        if set_log_L:
            self.set_log_L_LA()
            if set_Hess_adj:
                self.set_Hess_adj_copy()
            self.set_Jac()


    def sim_p_t_nb(self, sim=None, calc_log_L=None, Hess_F_D_N=None):

        if sim is None:
            sim = True
        if calc_log_L is None:
            calc_log_L = True
        if Hess_F_D_N is None:
            Hess_F_D_N = False

        rho_type_int_s = self._parameters.calc_rho_type_int_s()
        A_G = self._parameters.calc_A_G()
        Q_q = self._parameters.calc_Q_q()

        T = self.get_T()

        T_buff = self.get_T_buff()

        T_0 = self.get_T_0()
        T_1 = self.get_T_tot_buff()

        (
            self._Hess, self._Jac, 
            self._log_L["LA"], self._log_L["FDH"],
            self._t_series["p"], self._t_series["r"],
            self._t_series["varsigma"],
            self._t_s_series["D"], self._t_s_series["a"], self._t_s_series["n"], self._t_s_series["S"], self._t_s_series["M"], self._t_s_series["rho"], self._t_s_series["g"], self._t_s_series["w"], self._t_s_series["nu"], self._t_s_series["nepsilon"], 
            self._Hcomp["mu_t_D"], self._Hcomp["sigma_t_D_2"], self._Hcomp["rho_prime_k"], self._Hcomp["rho_prime2_k"], self._Hcomp["F_mu"], self._Hcomp["D_mu"], self._Hcomp["dr_mu"], self._Hcomp["ns2_m_1"], self._Hcomp["nu_w"], self._Hcomp["r_1"],
            ) = ssv_abm_nb.sim_p_t(
                                                    T=T, T_0=T_0, T_1=T_1, T_buff=T_buff,
                                                    Hess=self._Hess, Jac=self._Jac, 
                                                    log_L_LA=self._log_L["LA"], log_L_FDH=self._log_L["FDH"],
                                                    p=self._t_series["p"], r=self._t_series["r"],
                                                    p_F=self._t_series["p_F"], r_F=self._t_series["r_F"],
                                                    varsigma_t=self._t_series["varsigma"],
                                                    N_s_a_bool=self._parameters.N_s_a_bool.get_value(), N_s_e_bool=self._parameters.N_s_e_bool.get_value(), rho_type_int_s=rho_type_int_s,
                                                    psi=self._parameters.psi.get_value(), mu_F=self._parameters.mu_F.get_value(), sigma_F=self._parameters.sigma_F.get_value(), delta_s=self._parameters.delta_s.get_value(), gamma_1_s=self._parameters.gamma_1_s.get_value(), gamma_2_s=self._parameters.gamma_2_s.get_value(), sigma_s=self._parameters.sigma_s.get_value(), eta_s=self._parameters.eta_s.get_value(), eta_varsigma=self._parameters.eta_varsigma.get_value(), varsigma_0=self._parameters.varsigma_0.get_value(),
                                                    beta=self._parameters.beta.get_value(), eta_w=self._parameters.eta_w.get_value(), alpha_0_s=self._parameters.alpha_0_s.get_value(), alpha_n=self._parameters.alpha_n.get_value(), alpha_w=self._parameters.alpha_w.get_value(), lambda_w_s=self._parameters.lambda_w_s.get_value(), 
                                                    G_sigma=np.float64(self._parameters.G_sigma.get_value()), A_G=A_G, Q_q=Q_q, 
                                                    D_t_s=self._t_s_series["D"], a_t_s=self._t_s_series["a"], n_t_s=self._t_s_series["n"], S_t_s=self._t_s_series["S"], M_t_s=self._t_s_series["M"], rho_t_s=self._t_s_series["rho"], g_t_s=self._t_s_series["g"], w_t_s=self._t_s_series["w"], nu_t_s=self._t_s_series["nu"], nepsilon_t_s=self._t_s_series["nepsilon"], 
                                                    mu_t_D=self._Hcomp["mu_t_D"], sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], rho_prime2_k=self._Hcomp["rho_prime2_k"], F_mu=self._Hcomp["F_mu"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                                    sim=sim, calc_log_L=calc_log_L, Hess_F_D_N=Hess_F_D_N)
        

        





    def set_Hcomp_T(self):


        T = self.get_T()

        #self.get_ts() = np.arange(T)


        ks = np.reshape(np.repeat(self.get_ts(), T), (T,T))
        ls = np.transpose(ks)

        self._Hcomp["klmax"] = np.max([ks, ls], axis=0)


        ones = np.ones(T)
        self._Hcomp["diag"] = np.eye(T)
        self._Hcomp["diag_1"] = np.diag(ones[:-1],1) + np.diag(ones[:-1],-1)


        self._Hcomp["nu_w"] = np.zeros(T)
        self._Hcomp["r_1"] = np.zeros(T)

    
    def set_Hcomp_param(self):

        T = self.get_T()
        eta_w = self._parameters.eta_w.get_value()

        self._Hcomp["Hess_F"] = self.calc_Hess_F_np()

        self._Hcomp["Q_q"] = self._parameters.calc_Q_q()
        if eta_w==1:
            self._Hcomp["eta_l"] = np.zeros(T) #
            self._Hcomp["eta_k"] = np.zeros(T) #
            #self._Hcomp["eta_l"][:3] = 0
            #self._Hcomp["eta_k"][-3:] = 0
            self._Hcomp["Reta_w_lt_0"] = np.ascontiguousarray(np.eye(T-2))
        else:
            self._Hcomp["eta_l"] = (1-eta_w)**self.get_ts()
            self._Hcomp["eta_k"] = ((1-eta_w)**(-2))/self._Hcomp["eta_l"]
            self._Hcomp["eta_l"][:2] = 0
            self._Hcomp["eta_k"][-2:] = 0

            ln_Reta = np.log(1-eta_w)

            self._Hcomp["Reta_w_lt_0"] = np.ascontiguousarray(np.triu(np.exp(np.add.outer(-self.get_ts()[:-2]*ln_Reta, 2*self.get_ts()[:-2]*ln_Reta))))
            


    def set_Hcomp_p_p_F(self):


        T = self.get_T()
        psi = self._parameters.psi.get_value()

        self._Hcomp["mu_t_D"] = self.calc_mu_t_D()
        self._Hcomp["sigma_t_D_2"] = self.calc_sigma_t_D_2()

        self._Hcomp["rho_prime_k"] = self.calc_rho_prime_k()
        self._Hcomp["rho_prime2_k"] = self.calc_rho_prime2_k()

        self._Hcomp["F_mu"] = self.get_t_series("r_F")/psi - self._parameters.mu_F.get_value()
        self._Hcomp["D_mu"] = self.get_t_series("r")/psi - self._Hcomp["mu_t_D"] 
        self._Hcomp["dr_mu"] = self._parameters.delta_s.get_value()[0]*self.get_t_s_series("rho")[0,:] - self._Hcomp["mu_t_D"]


        if np.all(self._parameters.sigma_s.get_value()==0):
            #self._Hcomp["ns2_m_1"] = self.get_t_s_series("n")[0,:] - 1
            self._Hcomp["ns2_m_1"] = np.repeat(-1.0, T)
        else:
            self._Hcomp["ns2_m_1"] = self.get_t_s_series("n")[0,:]*((self._parameters.sigma_s.get_value()[0]**2)/self._Hcomp["sigma_t_D_2"]) - 1.0



        #self._Hcomp["nu_w"][:1] = 0
        self._Hcomp["nu_w"] = np.zeros(T, dtype=float)
        self._Hcomp["nu_w"][1:] = np.exp(self._parameters.lambda_w_s.get_value()[0]*(self.get_t_s_series("w")[0,:-1] < 0))

        #self._Hcomp["r_1"][-1:] = 0
        self._Hcomp["r_1"] = np.zeros(T, dtype=float)
        self._Hcomp["r_1"][:-1] = self.get_t_series("r")[1:]




    def set_Hcomp_new_old(self, reset=None, reset_T=None, reset_param=None, reset_p_p_F=None):

        if reset is None:
            reset = False
        if reset:
            self._Hcomp = dict()

            reset_T = True
            reset_param = True
            reset_p_p_F = True
        else:
            if reset_T is None:
                reset_T = False
            if reset_param is None:
                reset_param = True
            if reset_p_p_F is False:
                reset_p_p_F = True


        if reset_T:
            self.set_Hcomp_T()
        if reset_param:
            self.set_Hcomp_param()
        if reset_p_p_F:
            self.set_Hcomp_p_p_F()


    def set_Hcomp_new(self, reset=None, reset_T=None, reset_param=None, reset_p_p_F=None):

        if reset is None:
            reset = False
        if reset:
            self._Hcomp = dict()

            reset_T = False
            reset_param = False
            reset_p_p_F = True
        else:
            if reset_T is None:
                reset_T = False
            if reset_param is None:
                reset_param = False
            if reset_p_p_F is False:
                reset_p_p_F = True


        if reset_T:
            self.set_Hcomp_T()
        if reset_param:
            self.set_Hcomp_param()
        if reset_p_p_F:
            self.set_Hcomp_p_p_F()







    def set_Hcomp(self, reset=None, reset_T=None, reset_param=None, reset_shape=None, surplus=None, PRN=None):

        if surplus is None:
            surplus = False
        if PRN is None:
            PRN = True

        if reset is None:
            reset = False
        if reset:
            self._Hcomp = dict()

            reset_T = True
            reset_param = True
            reset_shape = True
        else:
            if reset_T is None:
                reset_T = False
            if reset_param is None:
                reset_param = False
            if reset_shape is None:
                reset_shape = False

        T = self.get_T()
        psi = self._parameters.psi.get_value()
        eta = self._parameters.eta_w.get_value()


        if reset_T:

            #self._Hcomp["zeros_1"] = np.zeros(self.get_T())
            #self._Hcomp["zeros_2"] = np.zeros((self.get_T(),self.get_T()))

            self._Hcomp["ks"] = np.reshape(np.repeat(self.get_ts(), T), (T,T))
            self._Hcomp["ls"] = np.transpose(self._Hcomp["ks"])
            self._Hcomp["klmax"] = np.max([self._Hcomp["ks"], self._Hcomp["ls"]], axis=0)


            self._Hcomp["diag"] = np.eye(T)

            ones = np.ones(T)
            self._Hcomp["diag_1"] = np.diag(ones[:-1],1) + np.diag(ones[:-1],-1)
        

        if reset_param:

            if False:
                sigma_F_m2 = self._parameters.sigma_F.get_value()**(-2)
                

                sigma_F_m2_ones = np.diag(np.repeat(sigma_F_m2, T-1),1)

                self._Hcomp["Hess_diag_1"] = sigma_F_m2_ones  + sigma_F_m2_ones.T
                self._Hcomp["Hess_F"] = np.copy(self._Hcomp["Hess_diag_1"])
                np.fill_diagonal(self._Hcomp["Hess_F"], np.repeat(-2.0*sigma_F_m2, T))
            else:
                self._Hcomp["Hess_F"] = self.calc_Hess_F_np()



            self._Hcomp["Q_q"] = self._parameters.alpha_w.get_value() * self._parameters.delta_s.get_value()[0] * eta / psi
            if eta==1:
                self._Hcomp["eta_l"] = np.zeros(T) #
                self._Hcomp["eta_k"] = np.zeros(T) #
                #self._Hcomp["eta_l"][:3] = 0
                #self._Hcomp["eta_k"][-3:] = 0
            else:
                self._Hcomp["eta_l"] = (1-eta)**self.get_ts()
                self._Hcomp["eta_k"] = ((1-eta)**(-2))/self._Hcomp["eta_l"]
                self._Hcomp["eta_l"][:2] = 0
                self._Hcomp["eta_k"][-2:] = 0

            

        if reset_shape:

            if PRN:
                self._Hcomp["nu_w"] = np.zeros(T)
                self._Hcomp["r_1"] = np.zeros(T)

                if surplus:
                    self._Hcomp["nu_w_2"] = np.zeros((T,T))
                    self._Hcomp["r_1_2"] = np.zeros((T,T))
                else:
                    self._Hcomp["nu_w_2"] = None
                    self._Hcomp["r_1_2"] = None



        self._Hcomp["mu_t_D"] = self.calc_mu_t_D()
        self._Hcomp["D_mu"] = self.get_t_series("r")/psi - self._Hcomp["mu_t_D"] 

        self._Hcomp["sigma_t_D_2"] = self.calc_sigma_t_D_2()

        self._Hcomp["rho_prime_k"] = self.calc_rho_prime_k()
        self._Hcomp["rho_prime2_k"] = self.calc_rho_prime2_k()


        if PRN: 

            if np.all(self._parameters.sigma_s.get_value()==0):
                #self._Hcomp["ns2_m_1"] = self.get_t_s_series("n")[0,:] - 1
                self._Hcomp["ns2_m_1"] = np.repeat(-1, T)
            else:
                self._Hcomp["ns2_m_1"] = self.get_t_s_series("n")[0,:]*((self._parameters.sigma_s.get_value()[0]**2)/self._Hcomp["sigma_t_D_2"]) - 1

            self._Hcomp["dr_mu"] = self._parameters.delta_s.get_value()[0]*self.get_t_s_series("rho")[0,:] - self._Hcomp["mu_t_D"]


            self._Hcomp["nu_w"][:1] = 0
            self._Hcomp["nu_w"][1:] = np.exp(self._parameters.lambda_w_s.get_value()[0]*(self.get_t_s_series("w")[0,:-1] < 0))

            self._Hcomp["r_1"][-1:] = 0
            self._Hcomp["r_1"][:-1] = self.get_t_series("r")[1:]


            if eta==1:
                Q_k_l1 = np.zeros((T,T))
                Q_k_l1[:-2,2:] = np.diag(self._Hcomp["r_1"][:-2]*self._Hcomp["nu_w"][2:])

                self._Hcomp["Q_k_l"] = self._Hcomp["Q_q"] * (self._Hcomp["ls"] >= self._Hcomp["ks"]+2) *(Q_k_l1)

            else:
                self._Hcomp["Q_k_l"] = self._Hcomp["Q_q"] * (self._Hcomp["ls"] >= self._Hcomp["ks"]+2) * np.outer(self._Hcomp["eta_k"]*self._Hcomp["r_1"], self._Hcomp["eta_l"]*self._Hcomp["nu_w"])
            #self._Hcomp["Q_k_l"] = self._Hcomp["Q_q"] * np.transpose(self._Hcomp["ls"] >= self._Hcomp["ks"]+2) * np.outer(self._Hcomp["eta_k"]*self._Hcomp["r_1"], self._Hcomp["eta_l"]*self._Hcomp["nu_w"])

            #self._Hcomp["Q_k_l"] = self._Hcomp["Q_q"] * (self._Hcomp["ls"] >= self._Hcomp["ks"]+2) * np.outer(self._Hcomp["eta_l"]*self._Hcomp["nu_w"], self._Hcomp["eta_k"]*self._Hcomp["r_1"])


        if surplus:
            self._Hcomp["sigma_t_D"] = np.sqrt(self._Hcomp["sigma_t_D_2"])

            if PRN:

                if self._Hcomp["nu_w_2"] is None:
                    self._Hcomp["nu_w_2"] = np.zeros((T,T))
                self._Hcomp["nu_w_2"][:,:1] = 0
                self._Hcomp["nu_w_2"][:,1:] = np.reshape(np.repeat(np.exp(self._parameters.lambda_w_s.get_value()[0]*(self.get_t_s_series("w")[0,:-1] < 0)), T), (T,T-1))


                if self._Hcomp["r_1_2"] is None:
                    self._Hcomp["r_1_2"] = np.zeros((T,T))
                self._Hcomp["r_1_2"][-1:, :] = 0
                self._Hcomp["r_1_2"][:-1, :] = self.get_t_series("r")[self._Hcomp["ks"][1:,:]]

                if eta==1:
                    Q_k_l_2_eta = np.zeros((T,T))
                    Q_k_l_2_eta[:-2,2:] = np.diag(np.ones(T-2))

                    self._Hcomp["Q_k_l_2"] = (self._Hcomp["ls"] >= self._Hcomp["ks"]+2) * self._Hcomp["Q_q"] * self._Hcomp["nu_w_2"] * Q_k_l_2_eta * self._Hcomp["r_1_2"]

                else:
                    self._Hcomp["Q_k_l_2"] = (self._Hcomp["ls"] >= self._Hcomp["ks"]+2) * self._Hcomp["Q_q"] * self._Hcomp["nu_w_2"] * ((1-eta)**(self._Hcomp["ls"]-(self._Hcomp["ks"]+2))) * self._Hcomp["r_1_2"]






                


    def calc_log_L_FD_t(self):

        T = self.get_T()

        psi = self._parameters.psi.get_value()
        mu_F = self._parameters.mu_F.get_value()
        sigma_F_2 = self._parameters.sigma_F.get_value()**2

        mu_t_D = self.calc_mu_t_D()
        sigma_t_D_2 = self.calc_sigma_t_D_2()
        D_mu = (self.get_t_series("r")/psi) - mu_t_D

        F_mu = (self.get_t_series("r_F")/psi) - mu_F

        
        ll_t = np.full((3,T), np.nan)

        ll_t[0,:] = -(1/2)*(F_mu**2)/sigma_F_2
        ll_t[1,:] = -(1/2)*np.log(sigma_t_D_2)
        ll_t[2,:] = -(1/2)*(D_mu**2)/sigma_t_D_2

        return np.sum(ll_t)


    def calc_log_L_FD_c(self):

        T = self.get_T()
        sigma_F = self._parameters.sigma_F.get_value()
        psi = self._parameters.psi.get_value()

        
        ll_c = -T * np.array([1,1,2]) * np.log(np.array([2*np.pi, sigma_F, psi]))

        return np.sum(ll_c)
    

    def calc_log_L_F(self):
            
        T = self.get_T()
        psi = self._parameters.psi.get_value()
        mu_F = self._parameters.mu_F.get_value()
        sigma_F = self._parameters.sigma_F.get_value()
        F_mu = (self.get_t_series("r_F")/psi) - mu_F

        return  -(1/2) * (T*np.sum(np.array([1,2,2]) * np.log(np.array([2*np.pi, psi, sigma_F]))) + np.sum(((F_mu**2)/(sigma_F**2))))



    def calc_log_L_D(self):
            
        T = self.get_T()
        psi = self._parameters.psi.get_value()
        mu_t_D = self.calc_mu_t_D()
        sigma_t_D_2 = self.calc_sigma_t_D_2()
        D_mu = (self.get_t_series("r")/psi) - mu_t_D

        return  -(1/2) * (T*np.sum(np.array([1,2]) * np.log(np.array([2*np.pi, psi]))) + np.sum(((D_mu**2)/sigma_t_D_2) + np.log(sigma_t_D_2)))


    def calc_log_L_H(self):

        self.set_Hess()

        return -(1/2)*np.linalg.slogdet(self._Hess)[1]
    


    

    def calc_log_L_FDH(self):

       

        return np.array([
                            self.calc_log_L_F(),
                            self.calc_log_L_D(),
                            self.calc_log_L_H(),
                        ])
    
    def calc_log_L_LA(self):

        self.set_log_L_FDH()

        return np.sum(self._log_L["FDH"])
    

    
    def calc_log_L_all(self):

        log_L_LA = self.calc_log_LA()

        return log_L_LA, self._log_L["FDH"]

    

    def calc_log_L_FD(self):

        #return np.sum(self.calc_log_L_FD_c()) + np.sum(self.calc_log_L_FD())
        return self.calc_log_L_FD_c() + self.calc_log_L_FD_t()





    def set_log_L_F(self):

        self._log_L["FDH"][0] = self.calc_log_L_F_nb()

    def set_log_L_D(self):

        self._log_L["FDH"][1] = self.calc_log_L_D_nb()

    def set_log_L_H(self):

        self._log_L["FDH"][2] = self.calc_log_L_H_nb()

    def set_log_L_FDH(self):

        self._log_L["FDH"] = self.calc_log_L_FDH_nb()

    def set_log_L_LA(self):

        self._log_L["LA"] = self.calc_log_L_LA_nb()

    def set_log_L_all(self):

        self._log_L["LA"], self._log_L["FDH"]  = self.calc_log_L_all_nb()



    
    

    

    def get_log_L_LA(self):

        return self._log_L["LA"]
    

    
    def calc_log_L_FD_t_nb(self):


        return ssv_abm_nb.calc_log_L_FD_t(sigma_F=self._parameters.sigma_F.get_value(), 
                                            sigma_t_D_2=self._Hcomp["sigma_t_D_2"], D_mu=self._Hcomp["D_mu"], F_mu=self._Hcomp["F_mu"])



    def calc_log_L_FD_c_nb(self):



        return ssv_abm_nb.calc_log_L_FD_c(T=self.get_T(), 
                                            psi=self._parameters.psi.get_value(), sigma_F=self._parameters.sigma_F.get_value())


    def calc_log_L_F_nb(self):

        return ssv_abm_nb.calc_log_L_F(T=self.get_T(), 
                                            psi=self._parameters.psi.get_value(), sigma_F=self._parameters.sigma_F.get_value(), 
                                            F_mu=self._Hcomp["F_mu"]
                                        )
    
    def calc_log_L_D_nb(self):

        return ssv_abm_nb.calc_log_L_D(T=self.get_T(), 
                                            psi=self._parameters.psi.get_value(),
                                            sigma_t_D_2=self._Hcomp["sigma_t_D_2"], D_mu=self._Hcomp["D_mu"],
                                        )


    def calc_log_L_FD_nb(self):

        return ssv_abm_nb.calc_log_L_FD(T=self.get_T(), 
                                            psi=self._parameters.psi.get_value(), sigma_F=self._parameters.sigma_F.get_value(), 
                                            sigma_t_D_2=self._Hcomp["sigma_t_D_2"], D_mu=self._Hcomp["D_mu"], F_mu=self._Hcomp["F_mu"]
                                        )
    
    def calc_log_L_FD_alt_nb(self):

        return ssv_abm_nb.calc_log_L_FD_alt(T=self.get_T(), 
                                            psi=self._parameters.psi.get_value(), sigma_F=self._parameters.sigma_F.get_value(), 
                                            sigma_t_D_2=self._Hcomp["sigma_t_D_2"], D_mu=self._Hcomp["D_mu"], F_mu=self._Hcomp["F_mu"]
                                        )


    def calc_log_L_H_nb(self):

        self.set_Hess()

        return ssv_abm_nb.calc_log_L_H(Hess=self._Hess)


    def calc_log_L_FDH_nb(self):

        self.set_Hess()

        return ssv_abm_nb.calc_log_L_FDH(T=self.get_T(), 
                                            Hess=self._Hess,
                                            psi=self._parameters.psi.get_value(), sigma_F=self._parameters.sigma_F.get_value(), 
                                            sigma_t_D_2=self._Hcomp["sigma_t_D_2"], D_mu=self._Hcomp["D_mu"], F_mu=self._Hcomp["F_mu"], 
                                )



    def calc_log_L_LA_nb(self):

        self.set_Hess()

        return ssv_abm_nb.calc_log_L_LA(T=self.get_T(), 
                                            Hess=self._Hess,
                                            psi=self._parameters.psi.get_value(), sigma_F=self._parameters.sigma_F.get_value(), 
                                            sigma_t_D_2=self._Hcomp["sigma_t_D_2"], D_mu=self._Hcomp["D_mu"], F_mu=self._Hcomp["F_mu"], 
                                )
    


    def calc_log_L_all_nb(self):

        self.set_Hess()

        return ssv_abm_nb.calc_log_L_all(T=self.get_T(), 
                                            Hess=self._Hess,
                                            psi=self._parameters.psi.get_value(), sigma_F=self._parameters.sigma_F.get_value(), 
                                            sigma_t_D_2=self._Hcomp["sigma_t_D_2"], D_mu=self._Hcomp["D_mu"], F_mu=self._Hcomp["F_mu"], 
                                )


    def calc_log_L_all_dict(self):


        log_L_LA, log_L_FDH = self.calc_log_L_all_nb()

        return {
                    "LA": log_L_LA,
                    "F": log_L_FDH[0],
                    "D": log_L_FDH[1],
                    "H": log_L_FDH[2],
                }




    def calc_Hess_np(self, epsilon=None):

        Hess = self.calc_Hess_F_D_K_np(epsilon=epsilon)
        if self._parameters.beta.get_value()!=0:
            Hess += self.calc_Hess_D_B_L_np()
            Hess += self.calc_Hess_D_B_Sigma_np()

        return Hess
   



    def calc_Hess_adj_copy(self):

        Hess_adj = np.copy(self._Hess)
        np.fill_diagonal(Hess_adj, np.diag(Hess_adj)+self._epsilon)

        return Hess_adj


    def set_Hess_adj_copy(self):

        self._Hess_adj = self.calc_Hess_adj_copy()


    def set_Hess_adj(self):

        self._Hess_adj = self.calc_Hess_nb(epsilon=self._epsilon)



    def set_Hess(self):

        self._Hess = self.calc_Hess_nb(epsilon=0.0)


    def get_Hess(self):

        return self._Hess


    def calc_Jac_np(self):

        return self.calc_Jac_F_np() + self.calc_Jac_D_np()

    def set_Jac(self):

        self._Jac = self.calc_Jac_nb()


    def get_Jac(self):

        return self._Jac



    def calc_Jac_F_np(self):

        T = self.get_T()

        sigma_F = self._parameters.psi.get_value()*self._parameters.sigma_F.get_value()


        p_F = self._t_series["p_F"][self.get_T_0()-1:]

            

        D1 = np.zeros(T)
        D1[:-1] = -(2*p_F[1:-1] - p_F[:-2] - p_F[2:])
        D1[-1] = -(p_F[-1] - p_F[-2])


        return D1/(sigma_F**2)


    def calc_Jac_D_np(self):

        T = self.get_T()
        beta = self._parameters.beta.get_value()


        delta_f = self._parameters.delta_s.get_value()[0]

        n_t_f = self.get_t_s_series("n")[0,:]




        eta_w = self._parameters.eta_w.get_value()



        Q_q = self._Hcomp["Q_q"]
        eta_l = self._Hcomp["eta_l"]
        eta_k = self._Hcomp["eta_k"]
        nu_w = self._Hcomp["nu_w"]

        D_mu  = self._Hcomp["D_mu"]
        sigma_t_D_2 = self._Hcomp["sigma_t_D_2"]

        rho_prime_k = self._Hcomp["rho_prime_k"]

        ns2_m_1 = self._Hcomp["ns2_m_1"]
        dr_mu = self._Hcomp["dr_mu"]
        r_1 = self._Hcomp["r_1"]


            

        D1 = np.zeros((2,T))
        D1[0,:-1] = (delta_f*D_mu*n_t_f*rho_prime_k/sigma_t_D_2)[1:]

        if beta==0:
            return D1[0,:]
        else:
            if eta_w==1:
                D1[1,:-2] = beta * rho_prime_k[1:-1] * Q_q * (r_1)[:-2] * (nu_w*n_t_f*((D_mu*dr_mu/sigma_t_D_2) + ns2_m_1*(((D_mu**2)/sigma_t_D_2)-1)))[2:]
            else:
                D1[1,:-2] = beta * rho_prime_k[1:-1] * Q_q * (eta_k*r_1)[:-2] * np.flip(np.cumsum(np.flip((eta_l*nu_w*n_t_f*((D_mu*dr_mu/sigma_t_D_2) + ns2_m_1*(((D_mu**2)/sigma_t_D_2)-1))))))[2:]

            return np.sum(D1, axis=0)



    
    

    def calc_Hess_F_np(self):

        return ssv_abm_nb.calc_Hess_F_fast_2(T=self.get_T(), sigma_F=self._parameters.psi.get_value()*self._parameters.sigma_F.get_value(), diag=self._Hcomp["diag"], diag_1=self._Hcomp["diag_1"])

    def calc_Hess_F_D_K_np(self, epsilon=None):

        if epsilon is None:
            epsilon = 0.0

        T = self.get_T()
        beta = self._parameters.beta.get_value()
        delta_f = self._parameters.delta_s.get_value()[0]

        n_t_f = self.get_t_s_series("n")[0,:]


        Q_q = self._Hcomp["Q_q"]
        eta_l = self._Hcomp["eta_l"]
        eta_k = self._Hcomp["eta_k"]
        nu_w = self._Hcomp["nu_w"]

        D_mu  = self._Hcomp["D_mu"]
        sigma_t_D_2 = self._Hcomp["sigma_t_D_2"]

        rho_prime_k = self._Hcomp["rho_prime_k"]
        rho_prime2_k = self._Hcomp["rho_prime2_k"]
        ns2_m_1 = self._Hcomp["ns2_m_1"]
        dr_mu = self._Hcomp["dr_mu"]
        r_1 = self._Hcomp["r_1"]


        
        
        D1 = np.zeros((2,T))
        D1[0,:-1] = -(((rho_prime_k*delta_f*n_t_f)**2 - rho_prime2_k*delta_f*D_mu)/ sigma_t_D_2)[1:]
        if beta!=0:
            D1[1,:-2] = -beta * Q_q * (rho_prime2_k*eta_k*r_1)[:-2] * np.flip(np.cumsum(np.flip(eta_l*nu_w * (ns2_m_1*(1 - (D_mu**2)/sigma_t_D_2) - ((D_mu*dr_mu)/sigma_t_D_2)) * n_t_f)))[2:]


        sigma_F_2 = (self._parameters.psi.get_value()*self._parameters.sigma_F.get_value())**2

        M = np.copy(self._Hcomp["Hess_F"])

        np.fill_diagonal(M, (epsilon-(2.0/sigma_F_2)) + np.sum(D1, axis=0))
        
        return M

    def calc_Hess_k(self, k=None, epsilon=None):

        if k is None:
            k = 0
        if epsilon is None:
            epsilon = 0.0

        return ssv_abm_nb.calc_Hess_k(k=k, 
                                    T=self.get_T(), 
                                    beta=self._parameters.beta.get_value(), 
                                    psi=self._parameters.psi.get_value(),
                                    sigma_F=self._parameters.sigma_F.get_value(), 
                                    delta_f=self._parameters.delta_s.get_value()[0], 
                                    sigma_f=self._parameters.sigma_s.get_value()[0], 
                                    n_t_f=self.get_t_s_series("n")[0,:], 
                                    Q_q=self._parameters.calc_Q_q(), 
                                    nu_w=self._Hcomp["nu_w"], 
                                    D_mu=self._Hcomp["D_mu"], 
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], 
                                    rho_prime_k=self._Hcomp["rho_prime_k"], 
                                    rho_prime2_k=self._Hcomp["rho_prime2_k"], 
                                    ns2_m_1=self._Hcomp["ns2_m_1"], 
                                    dr_mu=self._Hcomp["dr_mu"], 
                                    r_1=self._Hcomp["r_1"], 
                                    eta_w=self._parameters.eta_w.get_value(), 
                                    epsilon=epsilon
                                )

    

    def calc_Hess_F_D_K_k(self, k=None, epsilon=None):

        if k is None:
            k = 0
        if epsilon is None:
            epsilon = 0.0

        return ssv_abm_nb.calc_Hess_F_D_K_k(k=k, 
                                    T=self.get_T(), 
                                    beta=self._parameters.beta.get_value(), 
                                    psi=self._parameters.psi.get_value(),
                                    sigma_F=self._parameters.sigma_F.get_value(), 
                                    delta_f=self._parameters.delta_s.get_value()[0], 
                                    n_t_f=self.get_t_s_series("n")[0,:], 
                                    Q_q=self._parameters.calc_Q_q(), 
                                    nu_w=self._Hcomp["nu_w"], 
                                    D_mu=self._Hcomp["D_mu"], 
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], 
                                    rho_prime_k=self._Hcomp["rho_prime_k"], 
                                    rho_prime2_k=self._Hcomp["rho_prime2_k"], 
                                    ns2_m_1=self._Hcomp["ns2_m_1"], 
                                    dr_mu=self._Hcomp["dr_mu"], 
                                    r_1=self._Hcomp["r_1"], 
                                    eta_w=self._parameters.eta_w.get_value(), 
                                    epsilon=epsilon
                                )

    def calc_Hess_F_D_K_nb(self, epsilon=None):

        if epsilon is None:
            epsilon = 0.0

        return ssv_abm_nb.calc_Hess_F_D_K_nb( 
                                    T=self.get_T(), 
                                    beta=self._parameters.beta.get_value(), 
                                    psi=self._parameters.psi.get_value(),
                                    sigma_F=self._parameters.sigma_F.get_value(), 
                                    delta_f=self._parameters.delta_s.get_value()[0], 
                                    n_t_f=self.get_t_s_series("n")[0,:], 
                                    Q_q=self._parameters.calc_Q_q(), 
                                    nu_w=self._Hcomp["nu_w"], 
                                    D_mu=self._Hcomp["D_mu"], 
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], 
                                    rho_prime_k=self._Hcomp["rho_prime_k"], 
                                    rho_prime2_k=self._Hcomp["rho_prime2_k"], 
                                    ns2_m_1=self._Hcomp["ns2_m_1"], 
                                    dr_mu=self._Hcomp["dr_mu"], 
                                    r_1=self._Hcomp["r_1"], 
                                    eta_w=self._parameters.eta_w.get_value(), 
                                    epsilon=epsilon
                                )


    def calc_Hess_D_B_L_k(self, k=None):

        if k is None:
            k = 0

        return ssv_abm_nb.calc_Hess_D_B_L_k(k=k, 
                                    T=self.get_T(), 
                                    beta=self._parameters.beta.get_value(), 
                                    delta_f=self._parameters.delta_s.get_value()[0], 
                                    n_t_f=self.get_t_s_series("n")[0,:], 
                                    Q_q=self._parameters.calc_Q_q(), 
                                    nu_w=self._Hcomp["nu_w"], 
                                    D_mu=self._Hcomp["D_mu"], 
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], 
                                    rho_prime_k=self._Hcomp["rho_prime_k"], 
                                    ns2_m_1=self._Hcomp["ns2_m_1"], 
                                    dr_mu=self._Hcomp["dr_mu"], 
                                    r_1=self._Hcomp["r_1"], 
                                    eta_w=self._parameters.eta_w.get_value(), 
                                )

    def calc_Hess_D_B_L_nb(self):

        return ssv_abm_nb.calc_Hess_D_B_L_nb( 
                                    T=self.get_T(), 
                                    beta=self._parameters.beta.get_value(), 
                                    delta_f=self._parameters.delta_s.get_value()[0], 
                                    n_t_f=self.get_t_s_series("n")[0,:], 
                                    Q_q=self._parameters.calc_Q_q(), 
                                    nu_w=self._Hcomp["nu_w"], 
                                    D_mu=self._Hcomp["D_mu"], 
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], 
                                    rho_prime_k=self._Hcomp["rho_prime_k"], 
                                    ns2_m_1=self._Hcomp["ns2_m_1"], 
                                    dr_mu=self._Hcomp["dr_mu"], 
                                    r_1=self._Hcomp["r_1"], 
                                    eta_w=self._parameters.eta_w.get_value(), 
                                )





    def calc_Hess_D_N_np(self):

        T = self.get_T()
        beta = self._parameters.beta.get_value()
        delta_f = self._parameters.delta_s.get_value()[0]

        n_t_f = self.get_t_s_series("n")[0,:]


        Q_q = self._Hcomp["Q_q"]
        eta_l = self._Hcomp["eta_l"]
        eta_k = self._Hcomp["eta_k"]
        nu_w = self._Hcomp["nu_w"]

        D_mu  = self._Hcomp["D_mu"]
        sigma_t_D_2 = self._Hcomp["sigma_t_D_2"]

        rho_prime_k = self._Hcomp["rho_prime_k"]
        rho_prime2_k = self._Hcomp["rho_prime2_k"]
        ns2_m_1 = self._Hcomp["ns2_m_1"]
        dr_mu = self._Hcomp["dr_mu"]
        r_1 = self._Hcomp["r_1"]


        
        
        D1 = np.zeros((2,T))
        D1[0,:-1] = -(((rho_prime_k*delta_f*n_t_f)**2 - rho_prime2_k*delta_f*D_mu)/ sigma_t_D_2)[1:]
        if beta!=0:
            D1[1,:-2] = -beta * Q_q * (rho_prime2_k*eta_k*r_1)[:-2] * np.flip(np.cumsum(np.flip(eta_l*nu_w * (ns2_m_1*(1 - (D_mu**2)/sigma_t_D_2) - ((D_mu*dr_mu)/sigma_t_D_2)) * n_t_f)))[2:]

        
        D = np.diag(np.sum(D1, axis=0))
        
        return D



        



    def calc_Hess_D_B_L_np(self):

        T = self.get_T()
        beta = self._parameters.beta.get_value()
        if beta==0:
            return np.zeros((T,T))
        else:

            delta_f = self._parameters.delta_s.get_value()[0]

            n_t_f = self.get_t_s_series("n")[0,:]

            eta_w = self._parameters.eta_w.get_value()



            Q_q = self._Hcomp["Q_q"]
            eta_l = self._Hcomp["eta_l"]
            eta_k = self._Hcomp["eta_k"]
            nu_w = self._Hcomp["nu_w"]

            D_mu  = self._Hcomp["D_mu"]
            sigma_t_D_2 = self._Hcomp["sigma_t_D_2"]

            rho_prime_k = self._Hcomp["rho_prime_k"]

            ns2_m_1 = self._Hcomp["ns2_m_1"]
            dr_mu = self._Hcomp["dr_mu"]
            r_1 = self._Hcomp["r_1"]





            P1 = np.zeros((T,T))
            if eta_w==1:
                P1[:-2,1:-1] = np.diag(-(beta * delta_f * Q_q) * (r_1)[:-2]*rho_prime_k[1:-1]*(nu_w)[2:]*((1/sigma_t_D_2) * ((n_t_f**2)*(2*beta*D_mu*ns2_m_1 + rho_prime_k*dr_mu) - n_t_f*(1-n_t_f)*rho_prime_k*D_mu))[2:])
            else:
                P1[:-2,1:-1] = np.triu(np.outer((eta_k*r_1)[:-2]*rho_prime_k[1:-1], -(beta * delta_f * Q_q) * (eta_l*nu_w)[2:]*((1/sigma_t_D_2) * ((n_t_f**2)*(2*beta*D_mu*ns2_m_1 + rho_prime_k*dr_mu) - n_t_f*(1-n_t_f)*rho_prime_k*D_mu))[2:]))

            return ssv_abm_nb.diag_mirror(P1)
    

    


    





    def calc_Hess_D_B_Sigma_sumSl1S12_fast(self):

        return (ssv_abm_nb.calc_Hess_D_B_Sigma_sumSl1_fast(
                                                T=self.get_T(),
                                                sigma_f=self._parameters.sigma_s.get_value()[0],
                                                eta_w=self._parameters.eta_w.get_value(),
                                                n_t_f=self.get_t_s_series("n")[0,:],
                                                eta_l=self._Hcomp["eta_l"],
                                                nu_w=self._Hcomp["nu_w"],
                                                D_mu=self._Hcomp["D_mu"],
                                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"],
                                                dr_mu=self._Hcomp["dr_mu"],
                                            ),
                ssv_abm_nb.calc_Hess_D_B_Sigma_sumSl2_fast(
                                                eta_w=self._parameters.eta_w.get_value(),
                                                n_t_f=self.get_t_s_series("n")[0,:],
                                                eta_l=self._Hcomp["eta_l"],
                                                nu_w=self._Hcomp["nu_w"],
                                                D_mu=self._Hcomp["D_mu"],
                                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"],
                                                ns2_m_1=self._Hcomp["ns2_m_1"],
                                                dr_mu=self._Hcomp["dr_mu"],
                                            ))


    def calc_Hess_D_B_Sigma_nb_Sl1Sl2_t_factor(self, factor=1.0):

        return ssv_abm_nb.calc_Hess_D_B_Sigma_nb_Sl1Sl2_t_factor(
                                                    sigma_f=self._parameters.sigma_s.get_value()[0],
                                                    n_t_f=self.get_t_s_series("n")[0,:],
                                                    D_mu=self._Hcomp["D_mu"],
                                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"],
                                                    ns2_m_1=self._Hcomp["ns2_m_1"],
                                                    dr_mu=self._Hcomp["dr_mu"],
                                                    factor=factor
                                                )


    def calc_Hess_D_B_Sigma_nb_Sl1Sl2_t_nu_w_n_t_f_factor(self, factor=1.0):

        return ssv_abm_nb.calc_Hess_D_B_Sigma_nb_Sl1Sl2_t_nu_w_n_t_f_factor(
                                                    sigma_f=self._parameters.sigma_s.get_value()[0],
                                                    n_t_f=self.get_t_s_series("n")[0,:],
                                                    D_mu=self._Hcomp["D_mu"],
                                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"],
                                                    ns2_m_1=self._Hcomp["ns2_m_1"],
                                                    dr_mu=self._Hcomp["dr_mu"],
                                                    nu_w=self._Hcomp["nu_w"],
                                                    factor=factor,
                                                )
    

    def calc_Hess_D_B_Sigma_nb_Sl1Sl2_t_nu_w_n_t_f_bfactor(self, factor=1.0):

        bfactor = -(self._parameters.beta.get_value()*self._parameters.calc_Q_q())**2

        return ssv_abm_nb.calc_Hess_D_B_Sigma_nb_Sl1Sl2_t_nu_w_n_t_f_factor(
                                                    sigma_f=self._parameters.sigma_s.get_value()[0],
                                                    n_t_f=self.get_t_s_series("n")[0,:],
                                                    D_mu=self._Hcomp["D_mu"],
                                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"],
                                                    ns2_m_1=self._Hcomp["ns2_m_1"],
                                                    dr_mu=self._Hcomp["dr_mu"],
                                                    nu_w=self._Hcomp["nu_w"],
                                                    factor=factor*bfactor,
                                                )



    def calc_Hess_D_B_Sigma_np(self):

        T = self.get_T()
        beta = self._parameters.beta.get_value()
        sigma_s = self._parameters.sigma_s.get_value()

        if beta==0:
            return np.zeros((T,T))



        sigma_f = sigma_s[0]
        n_t_f = self.get_t_s_series("n")[0,:]

        eta_w = self._parameters.eta_w.get_value()


        
        Q_q = self._Hcomp["Q_q"]
        eta_l = self._Hcomp["eta_l"]
        eta_k = self._Hcomp["eta_k"]
        nu_w = self._Hcomp["nu_w"]

        D_mu  = self._Hcomp["D_mu"]
        sigma_t_D_2 = self._Hcomp["sigma_t_D_2"]

        rho_prime_k = self._Hcomp["rho_prime_k"]

        ns2_m_1 = self._Hcomp["ns2_m_1"]
        dr_mu = self._Hcomp["dr_mu"]
        r_1 = self._Hcomp["r_1"]




       

        S = np.zeros((T,T))
        sumSl1 = ssv_abm_nb.calc_Hess_D_B_Sigma_sumSl1_fast(T, sigma_f, eta_w, n_t_f, eta_l, nu_w, D_mu, sigma_t_D_2, dr_mu)
        sumSl2 = ssv_abm_nb.calc_Hess_D_B_Sigma_sumSl2_fast(eta_w, n_t_f, eta_l, nu_w, D_mu, sigma_t_D_2, ns2_m_1, dr_mu)

        if eta_w==1: 


            S[:-2,:-2] =  np.diag(- ((beta**2) * (Q_q**2)) * (r_1**2)[:-2] * ((rho_prime_k**2)[1:-1]*sumSl1[2:] + (2*rho_prime_k)[1:-1]*sumSl2[2:]))


        else:
            klmax = self._Hcomp["klmax"]




            S[:-2,:-2] = np.outer((eta_k*r_1)[:-2], - ((beta**2) * (Q_q**2)) * (eta_k*r_1)[:-2]) * (np.multiply.outer(rho_prime_k[1:-1], rho_prime_k[1:-1]) * (sumSl1[2:])[klmax[:-2,:-2]]
                                                                                                     + np.add.outer(rho_prime_k[1:-1], rho_prime_k[1:-1]) * (sumSl2[2:])[klmax[:-2,:-2]]
                                                                                                    )



        return S

    def calc_Hess_D_B_Sigma_kl(self, k=None, l=None):

        if k is None:
            k = 0
        if l is None:
            l = 0
    
        return ssv_abm_nb.calc_Hess_D_B_Sigma_kl(k=k, l=l,
                                    T=self.get_T(), 
                                    beta=self._parameters.beta.get_value(), 
                                    sigma_f=self._parameters.sigma_s.get_value()[0], 
                                    n_t_f=self.get_t_s_series("n")[0,:], 
                                    Q_q=self._parameters.calc_Q_q(), 
                                    nu_w=self._Hcomp["nu_w"], 
                                    D_mu=self._Hcomp["D_mu"], 
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], 
                                    rho_prime_k=self._Hcomp["rho_prime_k"], 
                                    ns2_m_1=self._Hcomp["ns2_m_1"], 
                                    dr_mu=self._Hcomp["dr_mu"], 
                                    r_1=self._Hcomp["r_1"], 
                                    eta_w=self._parameters.eta_w.get_value(), 
                                )

    def calc_Hess_D_B_Sigma_kl_k(self, k=None):

        if k is None:
            k = 0
    
        return ssv_abm_nb.calc_Hess_D_B_Sigma_kl_k(k=k, 
                                    T=self.get_T(), 
                                    beta=self._parameters.beta.get_value(), 
                                    sigma_f=self._parameters.sigma_s.get_value()[0], 
                                    n_t_f=self.get_t_s_series("n")[0,:], 
                                    Q_q=self._parameters.calc_Q_q(), 
                                    nu_w=self._Hcomp["nu_w"], 
                                    D_mu=self._Hcomp["D_mu"], 
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], 
                                    rho_prime_k=self._Hcomp["rho_prime_k"], 
                                    ns2_m_1=self._Hcomp["ns2_m_1"], 
                                    dr_mu=self._Hcomp["dr_mu"], 
                                    r_1=self._Hcomp["r_1"], 
                                    eta_w=self._parameters.eta_w.get_value(), 
                                )

    def calc_Hess_D_B_Sigma_kl_nb(self):

    
        return ssv_abm_nb.calc_Hess_D_B_Sigma_kl_nb(
                                    T=self.get_T(), 
                                    beta=self._parameters.beta.get_value(), 
                                    sigma_f=self._parameters.sigma_s.get_value()[0], 
                                    n_t_f=self.get_t_s_series("n")[0,:], 
                                    Q_q=self._parameters.calc_Q_q(), 
                                    nu_w=self._Hcomp["nu_w"], 
                                    D_mu=self._Hcomp["D_mu"], 
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], 
                                    rho_prime_k=self._Hcomp["rho_prime_k"], 
                                    ns2_m_1=self._Hcomp["ns2_m_1"], 
                                    dr_mu=self._Hcomp["dr_mu"], 
                                    r_1=self._Hcomp["r_1"], 
                                    eta_w=self._parameters.eta_w.get_value(), 
                                )

    def calc_Hess_D_B_Sigma_k(self, k=None):

        if k is None:
            k = 0
    
        return ssv_abm_nb.calc_Hess_D_B_Sigma_k(k=k, 
                                    T=self.get_T(), 
                                    beta=self._parameters.beta.get_value(), 
                                    sigma_f=self._parameters.sigma_s.get_value()[0], 
                                    n_t_f=self.get_t_s_series("n")[0,:], 
                                    Q_q=self._parameters.calc_Q_q(), 
                                    nu_w=self._Hcomp["nu_w"], 
                                    D_mu=self._Hcomp["D_mu"], 
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], 
                                    rho_prime_k=self._Hcomp["rho_prime_k"], 
                                    ns2_m_1=self._Hcomp["ns2_m_1"], 
                                    dr_mu=self._Hcomp["dr_mu"], 
                                    r_1=self._Hcomp["r_1"], 
                                    eta_w=self._parameters.eta_w.get_value(), 
                                )




    def calc_Hess_D_B_Sigma_k_alt(self, k=None):

        if k is None:
            k = 0
    
        return ssv_abm_nb.calc_Hess_D_B_Sigma_k_alt(k=k, 
                                    T=self.get_T(), 
                                    beta=self._parameters.beta.get_value(), 
                                    sigma_f=self._parameters.sigma_s.get_value()[0], 
                                    n_t_f=self.get_t_s_series("n")[0,:], 
                                    Q_q=self._parameters.calc_Q_q(), 
                                    nu_w=self._Hcomp["nu_w"], 
                                    D_mu=self._Hcomp["D_mu"], 
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], 
                                    rho_prime_k=self._Hcomp["rho_prime_k"], 
                                    ns2_m_1=self._Hcomp["ns2_m_1"], 
                                    dr_mu=self._Hcomp["dr_mu"], 
                                    Reta_w_lt_0 = self._Hcomp["Reta_w_lt_0"],
                                    r_1=self._Hcomp["r_1"], 
                                    eta_w=self._parameters.eta_w.get_value(), 
                                )

    def calc_Hess_D_B_Sigma_nb_alt(self):

    
        return ssv_abm_nb.calc_Hess_D_B_Sigma_nb_alt(
                                    T=self.get_T(), 
                                    beta=self._parameters.beta.get_value(), 
                                    sigma_f=self._parameters.sigma_s.get_value()[0], 
                                    n_t_f=self.get_t_s_series("n")[0,:], 
                                    Q_q=self._parameters.calc_Q_q(), 
                                    nu_w=self._Hcomp["nu_w"], 
                                    D_mu=self._Hcomp["D_mu"], 
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], 
                                    rho_prime_k=self._Hcomp["rho_prime_k"], 
                                    ns2_m_1=self._Hcomp["ns2_m_1"], 
                                    dr_mu=self._Hcomp["dr_mu"], 
                                    r_1=self._Hcomp["r_1"], 
                                    eta_w=self._parameters.eta_w.get_value(), 
                                )
    

    def calc_Hess_nb(self, epsilon=None):

        if epsilon is None:
            epsilon = 0.0

        return ssv_abm_nb.calc_Hess_nb( 
                                    T=self.get_T(), 
                                    psi=self._parameters.psi.get_value(), sigma_F=self._parameters.sigma_F.get_value(), delta_f=self._parameters.delta_s.get_value()[0], sigma_f=self._parameters.sigma_s.get_value()[0], 
                                    beta=self._parameters.beta.get_value(), 
                                    Q_q=self._parameters.calc_Q_q(), 
                                    n_t_f=self.get_t_s_series("n")[0,:], 
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], rho_prime2_k=self._Hcomp["rho_prime2_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"], 
                                    eta_w=self._parameters.eta_w.get_value(), 
                                    epsilon=epsilon
                                )
    
    def append_Hess_nb(self, Hess, epsilon=None):

        if epsilon is None:
            epsilon = 0.0

        return ssv_abm_nb.calc_Hess_nb(Hess=Hess,
                                    T=self.get_T(), 
                                    psi=self._parameters.psi.get_value(), sigma_F=self._parameters.sigma_F.get_value(), delta_f=self._parameters.delta_s.get_value()[0], sigma_f=self._parameters.sigma_s.get_value()[0], 
                                    beta=self._parameters.beta.get_value(), 
                                    Q_q=self._parameters.calc_Q_q(), 
                                    n_t_f=self.get_t_s_series("n")[0,:], 
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], rho_prime2_k=self._Hcomp["rho_prime2_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"], 
                                    eta_w=self._parameters.eta_w.get_value(), 
                                    epsilon=epsilon
                                )
    

    def calc_Hess_D_B_nb(self, epsilon=None):

        if epsilon is None:
            epsilon = 0.0

        return ssv_abm_nb.calc_Hess_D_B_nb( 
                                    T=self.get_T(), 
                                    delta_f=self._parameters.delta_s.get_value()[0], sigma_f=self._parameters.sigma_s.get_value()[0], 
                                    beta=self._parameters.beta.get_value(), 
                                    Q_q=self._parameters.calc_Q_q(), 
                                    n_t_f=self.get_t_s_series("n")[0,:], 
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], rho_prime2_k=self._Hcomp["rho_prime2_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"], 
                                    eta_w=self._parameters.eta_w.get_value(), 
                                )

    def append_Hess_D_B_nb(self, Hess, epsilon=None):

        if epsilon is None:
            epsilon = 0.0

        return ssv_abm_nb.calc_Hess_D_B_nb(Hess=Hess,
                                    T=self.get_T(), 
                                    delta_f=self._parameters.delta_s.get_value()[0], sigma_f=self._parameters.sigma_s.get_value()[0], 
                                    beta=self._parameters.beta.get_value(), 
                                    Q_q=self._parameters.calc_Q_q(), 
                                    n_t_f=self.get_t_s_series("n")[0,:], 
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], rho_prime2_k=self._Hcomp["rho_prime2_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"], 
                                    eta_w=self._parameters.eta_w.get_value(), 
                                )


    def calc_Hess_F_nb(self, epsilon=None):

        if epsilon is None:
            epsilon = 0.0


        return ssv_abm_nb.calc_Hess_F_nb(

                                    T=self.get_T(),
                                    psi=self._parameters.psi.get_value(),
                                    sigma_F=self._parameters.sigma_F.get_value(),
                                    epsilon=epsilon,
                                    )


    def append_Hess_F_nb(self, Hess, epsilon=None):

        if epsilon is None:
            epsilon = 0.0

        return ssv_abm_nb.append_Hess_F_nb(
                                    Hess=Hess,
                                    T=self.get_T(),
                                    psi=self._parameters.psi.get_value(),
                                    sigma_F=self._parameters.sigma_F.get_value(),
                                    epsilon=epsilon,
                                    )



    def calc_Hess_D_N_nb(self):


        return ssv_abm_nb.calc_Hess_D_N_nb(
                                    
                                    T=self.get_T(),
                                    delta_f=self._parameters.delta_s.get_value()[0], 
                                    n_t_f=self.get_t_s_series("n")[0,:],
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], rho_prime2_k=self._Hcomp["rho_prime2_k"], D_mu=self._Hcomp["D_mu"], 
                                    )
    

    def append_Hess_D_N_nb(self, Hess):


        return ssv_abm_nb.append_Hess_D_N_nb(Hess=Hess,
                                    T=self.get_T(),
                                    delta_f=self._parameters.delta_s.get_value()[0], 
                                    n_t_f=self.get_t_s_series("n")[0,:],
                                    )
    

    def calc_Hess_D_B_K_nb(self):


        return ssv_abm_nb.calc_Hess_D_B_K_nb(
                                    T=self.get_T(),
                                    beta=self._parameters.beta.get_value(), 
                                    Q_q=self._parameters.calc_Q_q(),
                                    n_t_f=self.get_t_s_series("n")[0,:],
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime2_k=self._Hcomp["rho_prime2_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                    eta_w=self._parameters.eta_w.get_value(),
                                    )


    def append_Hess_D_B_K_nb(self, Hess):


        return ssv_abm_nb.calc_Hess_D_B_K_nb(Hess=Hess,
                                    T=self.get_T(),
                                    beta=self._parameters.beta.get_value(), 
                                    Q_q=self._parameters.calc_Q_q(),
                                    n_t_f=self.get_t_s_series("n")[0,:],
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime2_k=self._Hcomp["rho_prime2_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                    eta_w=self._parameters.eta_w.get_value(),
                                    )


    def calc_Hess_D_K_nb(self):


        return ssv_abm_nb.calc_Hess_D_K_nb(
                                    T=self.get_T(),
                                    delta_f=self._parameters.delta_s.get_value()[0], 
                                    beta=self._parameters.beta.get_value(), 
                                    Q_q=self._parameters.calc_Q_q(),
                                    n_t_f=self.get_t_s_series("n")[0,:],
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], rho_prime2_k=self._Hcomp["rho_prime2_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                    eta_w=self._parameters.eta_w.get_value(),
                                    )


    def append_Hess_D_K_nb(self, Hess):


        return ssv_abm_nb.append_Hess_D_K_nb(
                                    T=self.get_T(),
                                    delta_f=self._parameters.delta_s.get_value()[0], 
                                    beta=self._parameters.beta.get_value(), 
                                    Q_q=self._parameters.calc_Q_q(),
                                    n_t_f=self.get_t_s_series("n")[0,:],
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], rho_prime2_k=self._Hcomp["rho_prime2_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                    eta_w=self._parameters.eta_w.get_value(),
                                    )

    def calc_Hess_D_K_nb_alt(self):


        return ssv_abm_nb.calc_Hess_D_K_nb_alt(

                                    T=self.get_T(),
                                    beta=self._parameters.beta.get_value(),
                                    delta_f=self._parameters.delta_s.get_value()[0],  
                                    n_t_f=self.get_t_s_series("n")[0,:],
                                    Q_q=self._parameters.calc_Q_q(),
                                    nu_w=self._Hcomp["nu_w"], 
                                    D_mu=self._Hcomp["D_mu"], 
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], 
                                    rho_prime_k=self._Hcomp["rho_prime_k"], 
                                    rho_prime2_k=self._Hcomp["rho_prime2_k"], 
                                    ns2_m_1=self._Hcomp["ns2_m_1"], 
                                    dr_mu=self._Hcomp["dr_mu"], 
                                    r_1=self._Hcomp["r_1"],
                                    eta_w=self._parameters.eta_w.get_value(), 
                                    )


    def append_Hess_D_K_nb_alt(self, Hess):


        return ssv_abm_nb.append_Hess_D_K_nb_alt(
                                    Hess=Hess,
                                    T=self.get_T(),
                                    beta=self._parameters.beta.get_value(),
                                    delta_f=self._parameters.delta_s.get_value()[0],  
                                    n_t_f=self.get_t_s_series("n")[0,:],
                                    Q_q=self._parameters.calc_Q_q(),
                                    nu_w=self._Hcomp["nu_w"], 
                                    D_mu=self._Hcomp["D_mu"], 
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], 
                                    rho_prime_k=self._Hcomp["rho_prime_k"], 
                                    rho_prime2_k=self._Hcomp["rho_prime2_k"], 
                                    ns2_m_1=self._Hcomp["ns2_m_1"], 
                                    dr_mu=self._Hcomp["dr_mu"], 
                                    r_1=self._Hcomp["r_1"],
                                    eta_w=self._parameters.eta_w.get_value(),
                                    )
    
    


    def calc_Hess_F_D_N_nb(self, epsilon=None):

        if epsilon is None:
            epsilon = 0.0


        return ssv_abm_nb.calc_Hess_F_D_N_nb(
                                    T=self.get_T(),
                                    psi=self._parameters.psi.get_value(), sigma_F=self._parameters.sigma_F.get_value(), delta_f=self._parameters.delta_s.get_value()[0], 
                                    n_t_f=self.get_t_s_series("n")[0,:],
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], rho_prime2_k=self._Hcomp["rho_prime2_k"], D_mu=self._Hcomp["D_mu"], 
                                    epsilon=epsilon,
                                    )

    def append_Hess_F_D_N_nb(self, Hess, epsilon=None):

        if epsilon is None:
            epsilon = 0.0


        return ssv_abm_nb.append_Hess_F_D_N_nb(Hess=Hess,
                                    T=self.get_T(),
                                    psi=self._parameters.psi.get_value(), sigma_F=self._parameters.sigma_F.get_value(), delta_f=self._parameters.delta_s.get_value()[0], 
                                    n_t_f=self.get_t_s_series("n")[0,:],
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], rho_prime2_k=self._Hcomp["rho_prime2_k"], D_mu=self._Hcomp["D_mu"], 
                                    epsilon=epsilon,
                                    )



    def calc_Hess_F_D_K_nb(self, epsilon=None):

        if epsilon is None:
            epsilon = 0.0


        return ssv_abm_nb.calc_Hess_F_D_K_nb(
                                    T=self.get_T(),
                                    psi=self._parameters.psi.get_value(), sigma_F=self._parameters.sigma_F.get_value(), delta_f=self._parameters.delta_s.get_value()[0], 
                                    beta=self._parameters.beta.get_value(), 
                                    Q_q=self._parameters.calc_Q_q(),
                                    n_t_f=self.get_t_s_series("n")[0,:],
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], rho_prime2_k=self._Hcomp["rho_prime2_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                    eta_w=self._parameters.eta_w.get_value(), epsilon=epsilon,
                                    )


    def append_Hess_F_D_K_nb(self, Hess, epsilon=None):

        if epsilon is None:
            epsilon = 0.0

        return ssv_abm_nb.append_Hess_F_D_K_nb(
                                    Hess=Hess,
                                    T=self.get_T(),
                                    psi=self._parameters.psi.get_value(), sigma_F=self._parameters.sigma_F.get_value(), delta_f=self._parameters.delta_s.get_value()[0], 
                                    beta=self._parameters.beta.get_value(), 
                                    Q_q=self._parameters.calc_Q_q(),
                                    n_t_f=self.get_t_s_series("n")[0,:],
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], rho_prime2_k=self._Hcomp["rho_prime2_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                    eta_w=self._parameters.eta_w.get_value(), epsilon=epsilon,
                                    )


    def calc_Hess_D_B_L_nb(self):


        return ssv_abm_nb.calc_Hess_D_B_L_nb(
                                    T=self.get_T(),
                                    delta_f=self._parameters.delta_s.get_value()[0],  
                                    beta=self._parameters.beta.get_value(),
                                    Q_q=self._parameters.calc_Q_q(),
                                    n_t_f=self.get_t_s_series("n")[0,:],
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                    eta_w=self._parameters.eta_w.get_value(), 
                                    )


    def append_Hess_D_B_L_nb(self, Hess):


        return ssv_abm_nb.append_Hess_D_B_L_nb(
                                    Hess=Hess,
                                    T=self.get_T(),
                                    delta_f=self._parameters.delta_s.get_value()[0],  
                                    beta=self._parameters.beta.get_value(),
                                    Q_q=self._parameters.calc_Q_q(),
                                    n_t_f=self.get_t_s_series("n")[0,:],
                                    sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                    eta_w=self._parameters.eta_w.get_value(),
                                    )


    def calc_Hess_D_B_Sigma_nb_sumSl1Sl2_factor(self, factor=1.0):
    
    
        return ssv_abm_nb.calc_Hess_D_B_Sigma_nb_sumSl1Sl2_factor(
                                                T=self.get_T(), 
                                                sigma_f=self._parameters.sigma_s.get_value()[0], 
                                                n_t_f=self.get_t_s_series("n")[0,:], 
                                                nu_w=self._Hcomp["nu_w"], 
                                                D_mu=self._Hcomp["D_mu"], 
                                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], 
                                                ns2_m_1=self._Hcomp["ns2_m_1"], 
                                                dr_mu=self._Hcomp["dr_mu"], 
                                                eta_w=self._parameters.eta_w.get_value(), 
                                                factor=factor
                                                )


    def calc_Hess_D_B_Sigma_nb(self):

        return ssv_abm_nb.calc_Hess_D_B_Sigma_nb(
                                                T=self.get_T(),
                                                beta=self._parameters.beta.get_value(),
                                                sigma_f=self._parameters.sigma_s.get_value()[0], 
                                                n_t_f=self.get_t_s_series("n")[0,:],
                                                Q_q=self._parameters.calc_Q_q(),
                                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                                eta_w=self._parameters.eta_w.get_value(), 
                                            )

    def append_Hess_D_B_Sigma_nb(self, Hess):

        return ssv_abm_nb.append_Hess_D_B_Sigma_nb(
                                                Hess=Hess,
                                                T=self.get_T(),
                                                sigma_f=self._parameters.sigma_s.get_value()[0], 
                                                beta=self._parameters.beta.get_value(),
                                                Q_q=self._parameters.calc_Q_q(),
                                                n_t_f=self.get_t_s_series("n")[0,:],
                                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                                eta_w=self._parameters.eta_w.get_value(), 
                                            )

    def calc_Hess_D_B_Sigma_nb_alt(self):

        return ssv_abm_nb.calc_Hess_D_B_Sigma_nb_alt(
                                                T=self.get_T(),
                                                sigma_f=self._parameters.sigma_s.get_value()[0], 
                                                beta=self._parameters.beta.get_value(),                           
                                                Q_q=self._parameters.calc_Q_q(),
                                                n_t_f=self.get_t_s_series("n")[0,:],
                                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                                eta_w=self._parameters.eta_w.get_value(), 
                                            )

    def append_Hess_D_B_Sigma_nb_alt(self, Hess):

        return ssv_abm_nb.append_Hess_D_B_Sigma_nb_alt(
                                                Hess=Hess,
                                                T=self.get_T(),
                                                sigma_f=self._parameters.sigma_s.get_value()[0], 
                                                beta=self._parameters.beta.get_value(),
                                                n_t_f=self.get_t_s_series("n")[0,:],
                                                Q_q=self._parameters.calc_Q_q(),
                                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                                eta_w=self._parameters.eta_w.get_value(), 
                                            )


    
    def calc_Hess_D_B_Sigma_nb_factor(self, factor=1.0):

        f2 = - (self._parameters.beta.get_value()*self._parameters.calc_Q_q())**2

        return ssv_abm_nb.calc_Hess_D_B_Sigma_nb_factor(
                                                T=self.get_T(), 
                                                sigma_f=self._parameters.sigma_s.get_value()[0], 
                                                n_t_f=self.get_t_s_series("n")[0,:], 
                                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                                eta_w=self._parameters.eta_w.get_value(), factor=factor*f2
                                            )

    def append_Hess_D_B_Sigma_nb_factor(self, Hess, factor=1.0):

        f2 = - (self._parameters.beta.get_value()*self._parameters.calc_Q_q())**2

        return ssv_abm_nb.append_Hess_D_B_Sigma_nb_factor(
                                                Hess,
                                                T=self.get_T(), 
                                                sigma_f=self._parameters.sigma_s.get_value()[0], 
                                                n_t_f=self.get_t_s_series("n")[0,:], 
                                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                                eta_w=self._parameters.eta_w.get_value(), factor=factor*f2
                                            )


    def calc_Hess_D_B_Sigma_nb_np_factor(self, factor=1.0):

        f2 = - (self._parameters.beta.get_value()*self._parameters.calc_Q_q())**2

        return ssv_abm_nb.calc_Hess_D_B_Sigma_nb_np_factor(
                                                T=self.get_T(), 
                                                sigma_f=self._parameters.sigma_s.get_value()[0], 
                                                n_t_f=self.get_t_s_series("n")[0,:], 
                                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                                eta_w=self._parameters.eta_w.get_value(), factor=factor*f2
                                            )


    def calc_Jac_F_nb(self):

        return ssv_abm_nb.calc_Jac_F_nb(
                                T=self.get_T(),
                                p_F=self._t_series["p_F"][self.get_T_0()-1:], 
                                psi=self._parameters.psi.get_value(),
                                sigma_F=self._parameters.sigma_F.get_value(),  
                            )

    def append_Jac_F_nb(self, Jac):

        return ssv_abm_nb.append_Jac_F_nb(
                                Jac=Jac,
                                p_F=self._t_series["p_F"][self.get_T_0()-1:],
                                T=self.get_T(), 
                                psi=self._parameters.psi.get_value(),
                                sigma_F=self._parameters.sigma_F.get_value(),
                            )
    

    def calc_Jac_D_N_nb(self):

        return ssv_abm_nb.calc_Jac_D_N_nb(
                                T=self.get_T(), 
                                delta_f=self._parameters.delta_s.get_value()[0], 
                                n_t_f=self.get_t_s_series("n")[0,:],
                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], 
                            )
    

    def append_Jac_D_N_nb(self, Jac):

        return ssv_abm_nb.append_Jac_D_N_nb(
                                Jac=Jac,
                                T=self.get_T(), 
                                delta_f=self._parameters.delta_s.get_value()[0], 
                                n_t_f=self.get_t_s_series("n")[0,:],
                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], 
                            )
    

    
    

    def calc_Jac_F_D_N_nb(self):

        return ssv_abm_nb.calc_Jac_F_D_N_nb(
                                T=self.get_T(), 
                                p_F=self._t_series["p_F"][self.get_T_0()-1:], 
                                psi=self._parameters.psi.get_value(), sigma_F=self._parameters.sigma_F.get_value(),  delta_f=self._parameters.delta_s.get_value()[0], 
                                n_t_f=self.get_t_s_series("n")[0,:],
                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], 
                            )

    def append_Jac_F_D_N_nb(self, Jac):

        return ssv_abm_nb.append_Jac_F_D_N_nb(
                                Jac=Jac,
                                T=self.get_T(), 
                                p_F=self._t_series["p_F"][self.get_T_0()-1:], 
                                psi=self._parameters.psi.get_value(), sigma_F=self._parameters.sigma_F.get_value(),  delta_f=self._parameters.delta_s.get_value()[0], 
                                n_t_f=self.get_t_s_series("n")[0,:],
                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], 
                            )



    def calc_Jac_D_B_nb(self):

        return ssv_abm_nb.calc_Jac_D_B_nb(
                                T=self.get_T(), 
                                beta=self._parameters.beta.get_value(), 
                                Q_q=self._parameters.calc_Q_q(), 
                                n_t_f=self.get_t_s_series("n")[0,:],
                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                eta_w=self._parameters.eta_w.get_value(),  
                            )

    def append_Jac_D_B_nb(self, Jac):

        return ssv_abm_nb.append_Jac_D_B_nb(
                                Jac=Jac,
                                T=self.get_T(), 
                                beta=self._parameters.beta.get_value(), 
                                Q_q=self._parameters.calc_Q_q(), 
                                n_t_f=self.get_t_s_series("n")[0,:],
                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                eta_w=self._parameters.eta_w.get_value(),  
                            )


    def calc_Jac_D_nb(self):

        return ssv_abm_nb.calc_Jac_D_nb(
                                T=self.get_T(), 
                                delta_f=self._parameters.delta_s.get_value()[0], 
                                beta=self._parameters.beta.get_value(), 
                                Q_q=self._parameters.calc_Q_q(), 
                                n_t_f=self.get_t_s_series("n")[0,:],
                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                eta_w=self._parameters.eta_w.get_value(),  
                            )

    def append_Jac_D_nb(self, Jac):

        return ssv_abm_nb.append_Jac_D_nb(
                                Jac=Jac,
                                T=self.get_T(), 
                                delta_f=self._parameters.delta_s.get_value()[0], 
                                beta=self._parameters.beta.get_value(), 
                                Q_q=self._parameters.calc_Q_q(), 
                                n_t_f=self.get_t_s_series("n")[0,:],
                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                eta_w=self._parameters.eta_w.get_value(),  
                            )



    def calc_Jac_nb(self):

        return ssv_abm_nb.calc_Jac_nb(
                                T=self.get_T(), 
                                p_F=self._t_series["p_F"][self.get_T_0()-1:],
                                psi=self._parameters.psi.get_value(),
                                sigma_F=self._parameters.sigma_F.get_value(),
                                delta_f=self._parameters.delta_s.get_value()[0], 
                                beta=self._parameters.beta.get_value(), 
                                Q_q=self._parameters.calc_Q_q(), 
                                n_t_f=self.get_t_s_series("n")[0,:],
                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                eta_w=self._parameters.eta_w.get_value(),
                            )

    def append_Jac_nb(self, Jac):

        return ssv_abm_nb.append_Jac_nb(
                                Jac=Jac,
                                T=self.get_T(), 
                                p_F=self._t_series["p_F"][self.get_T_0()-1:],
                                psi=self._parameters.psi.get_value(),
                                sigma_F=self._parameters.sigma_F.get_value(),
                                delta_f=self._parameters.delta_s.get_value()[0], 
                                beta=self._parameters.beta.get_value(), 
                                Q_q=self._parameters.calc_Q_q(), 
                                n_t_f=self.get_t_s_series("n")[0,:],
                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                eta_w=self._parameters.eta_w.get_value(),
                            )
    

    def calc_Jac_nb_old(self):

        return ssv_abm_nb.calc_Jac_nb(
                                T=self.get_T(), 
                                p_F=self._t_series["p_F"][self.get_T_0()-1:],
                                psi=self._parameters.psi.get_value(),
                                sigma_F=self._parameters.sigma_F.get_value(),
                                delta_f=self._parameters.delta_s.get_value()[0], 
                                beta=self._parameters.beta.get_value(), 
                                Q_q=self._parameters.calc_Q_q(), 
                                n_t_f=self.get_t_s_series("n")[0,:],
                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                eta_w=self._parameters.eta_w.get_value(),
                            )

    def append_Jac_nb_old(self, Jac):

        return ssv_abm_nb.append_Jac_nb(
                                Jac=Jac,
                                T=self.get_T(), 
                                p_F=self._t_series["p_F"][self.get_T_0()-1:],
                                psi=self._parameters.psi.get_value(),
                                sigma_F=self._parameters.sigma_F.get_value(),
                                delta_f=self._parameters.delta_s.get_value()[0], 
                                beta=self._parameters.beta.get_value(), 
                                Q_q=self._parameters.calc_Q_q(), 
                                n_t_f=self.get_t_s_series("n")[0,:],
                                sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                eta_w=self._parameters.eta_w.get_value(),
                            )


    def calc_mu_t_D(self, T_0=None): # A_n mod

        N_s_a_bool = self._parameters.N_s_a_bool.get_value()

        return self._parameters.delta_s.get_value()[N_s_a_bool] @ (self.get_t_s_series("rho", T_0=T_0)[N_s_a_bool,:] * self.get_t_s_series("n", T_0=T_0)[N_s_a_bool,:])

    def calc_sigma_t_D_2(self, T_0=None): # A_n mod

        N_s_a_bool = self._parameters.N_s_a_bool.get_value()

        return (self._parameters.sigma_s.get_value()[N_s_a_bool]**2) @ (self.get_t_s_series("n", T_0=T_0)[N_s_a_bool,:]**2)

    def calc_sigma_t_D(self, T_0=None): # A_n mod

        return np.sqrt(self.calc_sigma_t_D_2(T_0=T_0))


    def calc_rho_prime_k(self):

        gamma_1_f = self._parameters.gamma_1_s.get_value()[0]
        gamma_2_f = self._parameters.gamma_2_s.get_value()[0]


        T = self.get_T()
        psi = self._parameters.psi.get_value()


        if psi==0:
            rho_prime_k = np.zeros(T)
        else:
            inv_psi = psi**(-1)
            if self._parameters.rho_type_s.get_value()[0]=="linear":
                rho_prime_k =  np.repeat(inv_psi, T)
            elif self._parameters.rho_type_s.get_value()[0]=="cubic":
                rho_prime_k = inv_psi*(1 + 3*gamma_1_f*(self._t_s_series["M"][0,self.get_T_0():]**2))
            elif self._parameters.rho_type_s.get_value()[0]=="tanh":
                rho_prime_k = (inv_psi*gamma_1_f)*(1 - (self._t_s_series["rho"][0,self.get_T_0():]**2))
            elif self._parameters.rho_type_s.get_value()[0]=="wtar":
                rho_prime_k = (-inv_psi*gamma_1_f/gamma_2_f)*(1 - (self._t_s_series["rho"][0,self.get_T_0():]**2))/(2*np.sqrt(self._t_s_series["M"][0,self.get_T_0():]))
            else:
                rho_prime_k = None

        return rho_prime_k


    def calc_rho_prime2_k(self):

        gamma_1_f = self._parameters.gamma_1_s.get_value()[0]
        gamma_2_f = self._parameters.gamma_2_s.get_value()[0]

        T = self.get_T()
        psi = self._parameters.psi.get_value()

        if psi==0:
            rho_prime2_k = np.zeros(T)
        else:
            inv_psi_2 = psi**(-2)
            if self._parameters.rho_type_s.get_value()[0]=="linear":
                rho_prime2_k =  np.zeros(T)
            elif self._parameters.rho_type_s.get_value()[0]=="cubic":
                rho_prime2_k =  (inv_psi_2*6*gamma_1_f)*self._t_s_series["M"][0,self.get_T_0():]
            elif self._parameters.rho_type_s.get_value()[0]=="tanh":
                rho_prime2_k = -(inv_psi_2*2*(gamma_1_f**2))*self._t_s_series["rho"][0,self.get_T_0():]*(1 - (self._t_s_series["rho"][0,self.get_T_0():]**2))
            elif self._parameters.rho_type_s.get_value()[0]=="wtar":
                rho_prime2_k = (-inv_psi_2*((gamma_1_f/gamma_2_f)**2)/8) * (self._t_s_series["M"][0,self.get_T_0():]**(-3/2)) * (1 + 4*np.sqrt(self._t_s_series["M"][0,self.get_T_0():])*self._t_s_series["rho"][0,self.get_T_0():]) * (1 - (self._t_s_series["rho"][0,self.get_T_0():]**2)) 


        return rho_prime2_k



    
 #--#===========================
    #--------------------------
    # Optimization
    #--------------------------
    #===========================

    def calc_p_F_0_lin(self, extend_back=None):

        if extend_back is None:
            extend_back = 0

        ts = (1-extend_back) + np.arange(extend_back+self.get_T())
    

        return self._parameters.p_0_F.get_value() + self._parameters.psi.get_value()*self._parameters.mu_F.get_value()*ts
    

    def calc_p_F_0_gauss(self, sigma=None):

        if sigma is None:
            sigma = self.get_gauss_sigma()

        p_F_0_bd = self._parameters.p_0_F.get_value() + np.arange(self.get_T()+2) * self._parameters.psi.get_value() * self._parameters.mu_F.get_value()
        p_F_0_bd[1:-1] = self.get_t_series("p")

        return scipy.ndimage.gaussian_filter(p_F_0_bd, sigma=sigma)[1:-1]
    


    def copy_log_L_dict_res(self):

        log_L_dict = {field: dict() for field in ["spec", "x", "opt", "end", "res"]}

        for field in log_L_dict:
            for sf in self._log_L_dict[field]:
                log_L_dict[field][sf] = np.copy(self._log_L_dict[field][sf])

        return log_L_dict


    def reset_log_L_dict(self, N_k=None, reset=None, clear_track=None, N_track=None, keep_opt_last=None):


        if reset is None:
            reset = False
        
        if keep_opt_last is None:
            if reset:
                keep_opt_last = False
            else:
                keep_opt_last = True

        if reset:
            self._log_L_dict = {field: dict() for field in ["spec", "x", "k", "opt", "end", "res", "temp"]}

            if N_k is None:
                N_k = 50   # N_k mod


            
            

        if reset or N_k is not None:



            self._log_L_dict["spec"]["N_k"] = N_k
            self._log_L_dict["spec"]["N_k_trunc"] = self.get_N_k_trunc()
            self._log_L_dict["spec"]["gauss_sigma"] = self.get_gauss_sigma()
            self._log_L_dict["spec"]["epsilon"] = self.get_epsilon()
            self._log_L_dict["spec"]["step_norm_cutoff"] = self.get_step_norm_cutoff()
            self._log_L_dict["spec"]["check_opt_last"] = self.get_check_opt_last()


            self._log_L_dict["x"]["p_F_0"] = np.full(self.get_T(), np.nan)
            self._log_L_dict["x"]["p_F_0_set"] = np.full(self.get_T(), np.nan)
            self._log_L_dict["x"]["p_F_opt"] = np.full(self.get_T(), np.nan)
            self._log_L_dict["x"]["p_F_end"] = np.full(self.get_T(), np.nan)

            if not keep_opt_last:
                self._log_L_dict["x"]["p_F_0_opt_last_0"] =  np.full(self.get_T(), np.nan)
                self._log_L_dict["x"]["p_F_0_opt_last_1"] =  np.full(self.get_T(), np.nan)


            self._log_L_dict["k"]["log_L_k"] =  np.full(N_k, np.nan)
            self._log_L_dict["k"]["step_norm_k"] =  np.full(N_k, np.nan)
            self._log_L_dict["k"]["p_F_k"] =  np.full((N_k, self.get_T()), np.nan)
           
           
           
            self._log_L_dict["opt"]["k_opt"] =  0
            self._log_L_dict["opt"]["log_L_opt"] =  np.nan
            self._log_L_dict["opt"]["step_norm_opt"] =  np.nan

            self._log_L_dict["end"]["k_end"] =  N_k
            self._log_L_dict["end"]["log_L_end"] =  np.nan
            self._log_L_dict["end"]["step_norm_end"] =  np.nan



            self._log_L_dict["temp"]["p_F_k0"] =  np.full(self.get_T(), np.nan)
            self._log_L_dict["temp"]["p_F_k1"] =  np.full(self.get_T(), np.nan)

            self._log_L_dict["temp"]["A"] =  np.full((self.get_T(), self.get_T()), np.nan)
            self._log_L_dict["temp"]["B"] =  np.full(self.get_T(), np.nan)

            self._log_L_dict["res"]["use_opt_last"] =  False
            self._log_L_dict["res"]["succeed"] =  True


            self.reset_log_L_dict_track(N_track=N_track, reset=True)



        else:


            self._log_L_dict["spec"]["N_k_trunc"] = self.get_N_k_trunc()
            self._log_L_dict["spec"]["gauss_sigma"] = self.get_gauss_sigma()
            self._log_L_dict["spec"]["epsilon"] = self.get_epsilon()
            self._log_L_dict["spec"]["step_norm_cutoff"] = self.get_step_norm_cutoff()
            self._log_L_dict["spec"]["check_opt_last"] = self.get_check_opt_last()


            self._log_L_dict["x"]["p_F_0"][:] = np.nan
            self._log_L_dict["x"]["p_F_0_set"][:] = np.nan
            self._log_L_dict["x"]["p_F_opt"][:] = np.nan
            self._log_L_dict["x"]["p_F_end"][:] = np.nan

            if not keep_opt_last:
                self._log_L_dict["x"]["p_F_0_opt_last_0"][:] =  np.nan
                self._log_L_dict["x"]["p_F_0_opt_last_1"][:] =  np.nan


            self._log_L_dict["k"]["log_L_k"][:] = np.nan
            self._log_L_dict["k"]["step_norm_k"][:] = np.nan
            self._log_L_dict["k"]["p_F_k"][:,:] = np.nan
            

            self._log_L_dict["opt"]["k_opt"] =  0
            self._log_L_dict["opt"]["log_L_opt"] =  np.nan
            self._log_L_dict["opt"]["step_norm_opt"] =  np.nan


            self._log_L_dict["end"]["k_end"] =  self._log_L_dict["spec"]["N_k"]
            self._log_L_dict["end"]["log_L_end"] = np.nan
            self._log_L_dict["end"]["step_norm_end"] = np.nan
            

            self._log_L_dict["temp"]["p_F_k0"][:] = np.nan
            self._log_L_dict["temp"]["p_F_k1"][:] = np.nan

            self._log_L_dict["temp"]["A"][:,:] = np.nan
            self._log_L_dict["temp"]["B"][:] = np.nan

            self._log_L_dict["res"]["use_opt_last"] =  False
            self._log_L_dict["res"]["succeed"] =  True

            if clear_track is None:
                clear_track = False

            if clear_track:

                self.reset_log_L_dict_track(N_track=N_track, reset=False)
    


    def reset_log_L_dict_track(self, N_track=None, reset=None):

        if reset is None:
            reset = False
        if reset or N_track is not None:
            self._log_L_dict_track = dict()

            if N_track is None:
                N_track = 15000
            self._log_L_dict_track["N_track"] = N_track
            self._log_L_dict_track["track_i"] = 0

            self._log_L_dict_track["track_k_opt"] = np.full(self._log_L_dict_track["N_track"], -1, dtype=int)
            self._log_L_dict_track["track_k_end"] = np.full(self._log_L_dict_track["N_track"], -1, dtype=int)
            self._log_L_dict_track["track_log_L_k"] = np.full((self._log_L_dict_track["N_track"], self._log_L_dict["spec"]["N_k"]), np.nan)
            self._log_L_dict_track["track_step_norm_k"] = np.full((self._log_L_dict_track["N_track"], self._log_L_dict["spec"]["N_k"]), np.nan)

            self._log_L_dict_track["track_succeed"] = np.full(self._log_L_dict_track["N_track"], -1, dtype=int)
            self._log_L_dict_track["track_use_opt_last"] = np.full(self._log_L_dict_track["N_track"], -1, dtype=int)

        else:
            self._log_L_dict_track["track_i"] = 0
            self._log_L_dict_track["track_k_opt"][:] = -1
            self._log_L_dict_track["track_k_end"][:] = -1
            self._log_L_dict_track["track_log_L_k"][:,:] = np.nan
            self._log_L_dict_track["track_step_norm_k"][:,:] = np.nan

            self._log_L_dict_track["track_succeed"][:] = -1
            self._log_L_dict_track["track_use_opt_last"][:] = -1

        






    def set_p_F_nb(self, new_p_F, rho_type_int_s=None, A_G=None, Q_q=None, Hess_F_D_N=None):

        if Hess_F_D_N is None:
            Hess_F_D_N = False

        if rho_type_int_s is None:
            rho_type_int_s = self._parameters.calc_rho_type_int_s()

        if A_G is None:
            A_G = self._parameters.calc_A_G()
        if Q_q is None:
            #Q_q = self._Hcomp["Q_q"]
            Q_q = self._parameters.calc_Q_q()
        

        (
                self._Hess, self._Jac, 
                self._log_L["LA"], self._log_L["FDH"],
                self._t_series["p"], self._t_series["r"], 
                self._t_series["p_F"], self._t_series["r_F"],
                self._t_series["varsigma"],
                self._t_s_series["D"], self._t_s_series["a"], self._t_s_series["n"], self._t_s_series["S"], self._t_s_series["M"], self._t_s_series["rho"], self._t_s_series["g"], self._t_s_series["w"], self._t_s_series["nu"], self._t_s_series["nepsilon"],
                self._Hcomp["mu_t_D"], self._Hcomp["sigma_t_D_2"], self._Hcomp["rho_prime_k"], self._Hcomp["rho_prime2_k"], self._Hcomp["F_mu"], self._Hcomp["D_mu"], self._Hcomp["dr_mu"], self._Hcomp["ns2_m_1"], self._Hcomp["nu_w"], self._Hcomp["r_1"],
            ) = ssv_abm_nb.set_p_F(new_p_F=new_p_F,
                                p_F=self._t_series["p_F"], r_F=self._t_series["r_F"], p_0_F=self._parameters.p_0_F.get_value(), 
                                T=self.get_T(), T_1=self.get_T_tot_buff(), T_buff=self.get_T_buff(),
                                Hess=self._Hess, Jac=self._Jac, 
                                log_L_LA=self._log_L["LA"], log_L_FDH=self._log_L["FDH"],
                                    p=self._t_series["p"], r=self._t_series["r"],
                                    varsigma_t=self._t_series["varsigma"],
                                    N_s_a_bool=self._parameters.N_s_a_bool.get_value(), N_s_e_bool=self._parameters.N_s_e_bool.get_value(), rho_type_int_s=rho_type_int_s,
                                    psi=self._parameters.psi.get_value(), mu_F=self._parameters.mu_F.get_value(), sigma_F=self._parameters.sigma_F.get_value(), delta_s=self._parameters.delta_s.get_value(), gamma_1_s=self._parameters.gamma_1_s.get_value(), gamma_2_s=self._parameters.gamma_2_s.get_value(), sigma_s=self._parameters.sigma_s.get_value(), eta_s=self._parameters.eta_s.get_value(), eta_varsigma=self._parameters.eta_varsigma.get_value(), varsigma_0=self._parameters.varsigma_0.get_value(),
                                    beta=self._parameters.beta.get_value(), eta_w=self._parameters.eta_w.get_value(), alpha_0_s=self._parameters.alpha_0_s.get_value(), alpha_n=self._parameters.alpha_n.get_value(), alpha_w=self._parameters.alpha_w.get_value(), lambda_w_s=self._parameters.lambda_w_s.get_value(), 
                                    G_sigma=np.float64(self._parameters.G_sigma.get_value()), A_G=A_G, Q_q=Q_q, 
                                    D_t_s=self._t_s_series["D"], a_t_s=self._t_s_series["a"], n_t_s=self._t_s_series["n"], S_t_s=self._t_s_series["S"], M_t_s=self._t_s_series["M"], rho_t_s=self._t_s_series["rho"], g_t_s=self._t_s_series["g"], w_t_s=self._t_s_series["w"], nu_t_s=self._t_s_series["nu"], nepsilon_t_s=self._t_s_series["nepsilon"], 
                                    mu_t_D=self._Hcomp["mu_t_D"], sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], rho_prime2_k=self._Hcomp["rho_prime2_k"], F_mu=self._Hcomp["F_mu"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                T_0=self.get_T_0(), run_p_t=True, calc_log_L=True, Hess_F_D_N=Hess_F_D_N)


        
        
        
    def calc_p_F_opt_nb_solve(self, p_F_0=None, N_k=None, N_k_trunc=None, gauss_sigma=None, epsilon=None, step_norm_cutoff=None, return_dict=None, check_opt_last=None, overwrite_a=True, overwrite_b=True, break_nan=None, Hess_F_D_N=None, track=None):


        if Hess_F_D_N is None:
            Hess_F_D_N = False


        if track is None:
            track = True



        if break_nan is None:
            break_nan = False
        

        
        if return_dict is None:
            return_dict=True


        self.set_N_k_trunc(N_k_trunc=N_k_trunc, reset=False)
        self.set_gauss_sigma(gauss_sigma=gauss_sigma, reset=False)
        self.set_epsilon(epsilon=epsilon, reset=False)
        self.set_step_norm_cutoff(step_norm_cutoff=step_norm_cutoff, reset=False)
        self.set_check_opt_last(check_opt_last=check_opt_last, reset=False)
        

        self.reset_log_L_dict(N_k=N_k, reset=False, keep_opt_last=True)


        
        A_G = self._parameters.calc_A_G()
        Q_q = self._parameters.calc_Q_q()
        rho_type_int_s = self._parameters.calc_rho_type_int_s()

        T = self.get_T()
        T_sqrt = np.sqrt(T)

        if p_F_0 is None:
            self._log_L_dict["x"]["p_F_0"] = self.calc_p_F_0_gauss()
        else:
            self._log_L_dict["x"]["p_F_0"] = p_F_0

        if np.any(np.isnan(self._log_L_dict["x"]["p_F_0_opt_last_1"])):
            self._log_L_dict["x"]["p_F_0_opt_last_1"] = np.copy(self._log_L_dict["x"]["p_F_0"])
            check_opt_last = False
        else:
            check_opt_last = self._log_L_dict["spec"]["check_opt_last"]

        self._log_L_dict["x"]["p_F_0_opt_last_0"] = np.copy(self._log_L_dict["x"]["p_F_0_opt_last_1"])

        #self.set_p_F_np(p_F=self._log_L_dict["x"]["p_F_0"], run_p_t=True, set_Hess_adj=False, set_log_L=True)
        self.set_p_F_nb(new_p_F=self._log_L_dict["x"]["p_F_0"], rho_type_int_s=rho_type_int_s, A_G=A_G, Hess_F_D_N=Hess_F_D_N)
        self._log_L_dict["opt"]["k_opt"] = 0
        self._log_L_dict["x"]["p_F_opt"] = np.copy(self._log_L_dict["x"]["p_F_0"])
        self._log_L_dict["opt"]["log_L_opt"] = self._log_L["LA"]

        if check_opt_last:
            #self.set_p_F_np(p_F=self._log_L_dict["x"]["p_F_0_opt_last_1"], run_p_t=True, set_Hess_adj=False, set_log_L=True)
            self.set_p_F_nb(new_p_F=self._log_L_dict["x"]["p_F_0_opt_last_1"], rho_type_int_s=rho_type_int_s, A_G=A_G, Hess_F_D_N=Hess_F_D_N)
            if not np.isnan(self._log_L["LA"]):
                self._log_L_dict["res"]["use_opt_last"] = self._log_L["LA"] > self._log_L_dict["opt"]["log_L_opt"]
                if self._log_L_dict["res"]["use_opt_last"]:
                    self._log_L_dict["x"]["p_F_opt"] = np.copy(self._log_L_dict["x"]["p_F_0_opt_last_1"])
                    self._log_L_dict["opt"]["log_L_opt"] = self._log_L["LA"]
                else:
                    #self.set_p_F_np(p_F=self._log_L_dict["x"]["p_F_0"], run_p_t=True, set_Hess_adj=False, set_log_L=True)
                    self.set_p_F_nb(new_p_F=self._log_L_dict["x"]["p_F_0"], rho_type_int_s=rho_type_int_s, A_G=A_G, Hess_F_D_N=Hess_F_D_N)

        self._log_L_dict["opt"]["p_F_0_set"] = np.copy(self._log_L_dict["x"]["p_F_opt"])


        self._log_L_dict["k"]["p_F_k"][0,:] = np.copy(self._log_L_dict["x"]["p_F_opt"])
        self._log_L_dict["k"]["log_L_k"][0] = self._log_L["LA"]

        self._log_L_dict["temp"]["p_F_k0"] =  np.copy(self._log_L_dict["x"]["p_F_opt"])
        self._log_L_dict["temp"]["p_F_k1"] = np.copy(self._log_L_dict["x"]["p_F_opt"])

        for k in range(1,self._log_L_dict["spec"]["N_k"]):
            self._log_L_dict["temp"]["p_F_k0"] = np.copy(self._log_L_dict["temp"]["p_F_k1"])

            #self._log_L_dict["temp"]["A"] = -self._Hess_adj
            self._log_L_dict["temp"]["A"] = -self.calc_Hess_adj_copy()
            self._log_L_dict["temp"]["B"] = (self._log_L_dict["temp"]["A"] @ self._log_L_dict["temp"]["p_F_k0"]) + self._Jac
            self._log_L_dict["temp"]["p_F_k1"] = scipy.linalg.solve(self._log_L_dict["temp"]["A"], self._log_L_dict["temp"]["B"], assume_a="sym", overwrite_a=overwrite_a, overwrite_b=overwrite_b)

            self._log_L_dict["k"]["step_norm_k"][k] = np.sqrt(np.sum((self._log_L_dict["temp"]["p_F_k1"] - self._log_L_dict["temp"]["p_F_k0"])**2))/T_sqrt

            #self.set_p_F_np(p_F=self._log_L_dict["temp"]["p_F_k1"], run_p_t=True, set_Hess_adj=False, set_log_L=True)
            self.set_p_F_nb(new_p_F=self._log_L_dict["temp"]["p_F_k1"], rho_type_int_s=rho_type_int_s, A_G=A_G, Hess_F_D_N=Hess_F_D_N)
            self._log_L_dict["k"]["log_L_k"][k] = self._log_L["LA"]

            self._log_L_dict["k"]["p_F_k"][k,:] = self._log_L_dict["temp"]["p_F_k1"]

            if self._log_L_dict["k"]["log_L_k"][k] > self._log_L_dict["opt"]["log_L_opt"]:
                self._log_L_dict["opt"]["k_opt"] = k
                self._log_L_dict["opt"]["log_L_opt"] = self._log_L_dict["k"]["log_L_k"][self._log_L_dict["opt"]["k_opt"]]
                self._log_L_dict["opt"]["step_norm_opt"] = self._log_L_dict["k"]["step_norm_k"][self._log_L_dict["opt"]["k_opt"]]
                self._log_L_dict["x"]["p_F_opt"] = np.copy(self._log_L_dict["temp"]["p_F_k1"])

            if self._log_L_dict["spec"]["step_norm_cutoff"]>0:
                if self._log_L_dict["k"]["step_norm_k"][k]<self._log_L_dict["spec"]["step_norm_cutoff"]:
                    break

            if self._log_L_dict["spec"]["N_k_trunc"]>0:
                if k - self._log_L_dict["opt"]["k_opt"] >= self._log_L_dict["spec"]["N_k_trunc"]:
                    break

            if np.any(np.isnan(self._Jac)) or np.any(np.isnan(self._Hess)):
                self._log_L_dict["res"]["succeed"] = False
                if break_nan:
                    self._log_L_dict["opt"]["log_L_opt"] = np.nan
                    break
        

        self._log_L_dict["end"]["k_end"] = k
        self._log_L_dict["opt"]["log_L_end"] = self._log_L_dict["k"]["log_L_k"][self._log_L_dict["end"]["k_end"]]
        self._log_L_dict["opt"]["step_norm_end"] = self._log_L_dict["k"]["step_norm_k"][self._log_L_dict["end"]["k_end"]]
        self._log_L_dict["x"]["p_F_end"] = np.copy(self._log_L_dict["temp"]["p_F_k1"])
        

        #x_iK = np.copy(x_i1)
        if self._log_L_dict["res"]["succeed"] or not break_nan:
            self._log_L_dict["x"]["p_F_0_opt_last_1"] = np.copy(self._log_L_dict["x"]["p_F_opt"])
            if k!=self._log_L_dict["opt"]["k_opt"]:
                #self.set_p_F_np(p_F=self._log_L_dict["x"]["p_F_opt"], run_p_t=True, set_log_L=True)
                self.set_p_F_nb(new_p_F=self._log_L_dict["x"]["p_F_opt"], rho_type_int_s=rho_type_int_s, A_G=A_G, Hess_F_D_N=Hess_F_D_N)

        if track:
            if self._log_L_dict_track["track_i"] < self._log_L_dict_track["N_track"]:
                self._log_L_dict_track["track_k_opt"][self._log_L_dict_track["track_i"]] = np.copy(self._log_L_dict["opt"]["k_opt"])
                self._log_L_dict_track["track_k_end"][self._log_L_dict_track["track_i"]] = np.copy(self._log_L_dict["end"]["k_end"])
                self._log_L_dict_track["track_log_L_k"][self._log_L_dict_track["track_i"],:] = np.copy(self._log_L_dict["k"]["log_L_k"])
                self._log_L_dict_track["track_step_norm_k"][self._log_L_dict_track["track_i"],:] = np.copy(self._log_L_dict["k"]["step_norm_k"])

                self._log_L_dict_track["track_succeed"][self._log_L_dict_track["track_i"]] = int(self._log_L_dict["res"]["succeed"])
                self._log_L_dict_track["track_use_opt_last"][self._log_L_dict_track["track_i"]] = int(self._log_L_dict["res"]["use_opt_last"])
                

                self._log_L_dict_track["track_i"] += 1


        return self._log_L_dict["opt"]["log_L_opt"]
    

    


    def calc_p_F_opt_nb(self, p_F_0=None, N_k=None, N_k_trunc=None, gauss_sigma=None, epsilon=None, step_norm_cutoff=None, return_dict=None, check_opt_last=None, overwrite_a=True, overwrite_b=True, break_nan=None, Hess_F_D_N=None, track=None):

        if Hess_F_D_N is None:
            Hess_F_D_N = False

        if break_nan is None:
            break_nan = False
        
        if return_dict is None:
            return_dict=True

        if track is None:
            track = True

        self.set_N_k_trunc(N_k_trunc=N_k_trunc, reset=False)
        self.set_gauss_sigma(gauss_sigma=gauss_sigma, reset=False)
        self.set_epsilon(epsilon=epsilon, reset=False)
        self.set_step_norm_cutoff(step_norm_cutoff=step_norm_cutoff, reset=False)
        self.set_check_opt_last(check_opt_last=check_opt_last, reset=False)

        self.reset_log_L_dict(N_k=N_k, reset=False, keep_opt_last=True)



        Q_q = self._parameters.calc_Q_q()
        A_G = self._parameters.calc_A_G()
        rho_type_int_s = self._parameters.calc_rho_type_int_s()

        if p_F_0 is None:
            self._log_L_dict["x"]["p_F_0"] = self.calc_p_F_0_gauss()
        else:
            self._log_L_dict["x"]["p_F_0"] = p_F_0

        if np.any(np.isnan(self._log_L_dict["x"]["p_F_0_opt_last_1"])):
            self._log_L_dict["x"]["p_F_0_opt_last_1"] = np.copy(self._log_L_dict["x"]["p_F_0"])
            check_opt_last = False
        else:
            check_opt_last = self._log_L_dict["spec"]["check_opt_last"]

        self._log_L_dict["x"]["p_F_0_opt_last_0"] = np.copy(self._log_L_dict["x"]["p_F_0_opt_last_1"])



        (
            self._log_L_dict["x"]["p_F_0"], self._log_L_dict["x"]["p_F_0_opt_last_1"], self._log_L_dict["x"]["p_F_0_set"],
            self._log_L_dict["x"]["p_F_opt"], self._log_L_dict["x"]["p_F_end"], 
            self._log_L_dict["k"]["log_L_k"], self._log_L_dict["k"]["step_norm_k"], self._log_L_dict["k"]["p_F_k"],
            self._log_L_dict["opt"]["k_opt"], self._log_L_dict["opt"]["log_L_opt"], self._log_L_dict["opt"]["step_norm_opt"],
            self._log_L_dict["end"]["k_end"], self._log_L_dict["end"]["log_L_end"], self._log_L_dict["end"]["step_norm_end"],
            self._log_L_dict["temp"]["p_F_k0"], self._log_L_dict["temp"]["p_F_k1"], self._log_L_dict["temp"]["A"], self._log_L_dict["temp"]["B"],
            self._log_L_dict["res"]["use_opt_last"], self._log_L_dict["res"]["succeed"],
            #self._log_L_dict["x"]["p_F_0"], self._log_L_dict["spec"]["N_k"], self._log_L_dict["spec"]["epsilon"], self._log_L_dict["spec"]["step_norm_cutoff"], self._log_L_dict["x"]["p_F_0_opt_last_1"],
            #self._log_L_dict["opt"]["k_opt"], self._log_L_dict["k"]["p_F_k"], self._log_L_dict["temp"]["p_F_k0"], self._log_L_dict["temp"]["p_F_k1"], self._log_L_dict["x"]["p_F_opt"], self._log_L_dict["k"]["log_L_k"], self._log_L_dict["opt"]["log_L_opt"], self._log_L_dict["k"]["step_norm_k"], self._log_L_dict["temp"]["A"], self._log_L_dict["temp"]["B"], self._log_L_dict["res"]["succeed"], self._log_L_dict["res"]["use_opt_last"],
            #self._log_L_dict["N_track"], self._log_L_dict["track_i"], self._log_L_dict["track_k_opt"], self._log_L_dict["track_log_L_k"], self._log_L_dict["track_step_norm_k"], 
            self._Hess, self._Jac, 
            self._log_L["LA"], self._log_L["FDH"],
            self._t_series["p"], self._t_series["r"],
            self._t_series["p_F"], self._t_series["r_F"],
            self._t_series["varsigma"],
            self._t_s_series["D"], self._t_s_series["a"], self._t_s_series["n"], self._t_s_series["S"], self._t_s_series["M"], self._t_s_series["rho"], self._t_s_series["g"], self._t_s_series["w"], self._t_s_series["nu"], self._t_s_series["nepsilon"],
            self._Hcomp["mu_t_D"], self._Hcomp["sigma_t_D_2"], self._Hcomp["rho_prime_k"], self._Hcomp["rho_prime2_k"], self._Hcomp["F_mu"], self._Hcomp["D_mu"], self._Hcomp["dr_mu"], self._Hcomp["ns2_m_1"], self._Hcomp["nu_w"], self._Hcomp["r_1"],
            ) = ssv_abm_nb.calc_p_F_opt(
                                T=self.get_T(), 
                                p_F_0=self._log_L_dict["x"]["p_F_0"], p_F_0_opt_last_1=self._log_L_dict["x"]["p_F_0_opt_last_1"], p_F_0_set=self._log_L_dict["x"]["p_F_0_set"],
                                N_k=self._log_L_dict["spec"]["N_k"], N_k_trunc=self._log_L_dict["spec"]["N_k_trunc"], epsilon=self._log_L_dict["spec"]["epsilon"], step_norm_cutoff=self._log_L_dict["spec"]["step_norm_cutoff"], #check_opt_last=
                                p_F_opt=self._log_L_dict["x"]["p_F_opt"], p_F_end=self._log_L_dict["x"]["p_F_end"], 
                                log_L_k=self._log_L_dict["k"]["log_L_k"], step_norm_k=self._log_L_dict["k"]["step_norm_k"], p_F_k=self._log_L_dict["k"]["p_F_k"],
                                k_opt=self._log_L_dict["opt"]["k_opt"], log_L_opt=self._log_L_dict["opt"]["log_L_opt"], step_norm_opt=self._log_L_dict["opt"]["step_norm_opt"],
                                k_end=self._log_L_dict["end"]["k_end"], log_L_end=self._log_L_dict["end"]["log_L_end"], step_norm_end=self._log_L_dict["end"]["step_norm_end"],
                                p_F_k0=self._log_L_dict["temp"]["p_F_k0"], p_F_k1=self._log_L_dict["temp"]["p_F_k1"], A=self._log_L_dict["temp"]["A"], B=self._log_L_dict["temp"]["B"],
                                use_opt_last=self._log_L_dict["res"]["use_opt_last"], succeed=self._log_L_dict["res"]["succeed"],
                                #p_F_0=self._log_L_dict["x"]["p_F_0"], N_k=self._log_L_dict["spec"]["N_k"], epsilon=self._log_L_dict["spec"]["epsilon"], step_norm_cutoff=self._log_L_dict["spec"]["step_norm_cutoff"], p_F_0_opt_last_1=self._log_L_dict["x"]["p_F_0_opt_last_1"],
                                #k_opt=self._log_L_dict["opt"]["k_opt"], p_F_k=self._log_L_dict["k"]["p_F_k"], p_F_k0=self._log_L_dict["temp"]["p_F_k0"], p_F_k1=self._log_L_dict["temp"]["p_F_k1"], p_F_opt=self._log_L_dict["x"]["p_F_opt"], log_L_k=self._log_L_dict["k"]["log_L_k"], log_L_opt=self._log_L_dict["opt"]["log_L_opt"], step_norm_k=self._log_L_dict["k"]["step_norm_k"], A=self._log_L_dict["temp"]["A"], B=self._log_L_dict["temp"]["B"], succeed=self._log_L_dict["res"]["succeed"], use_opt_last=self._log_L_dict["res"]["use_opt_last"],
                                #N_track=self._log_L_dict["N_track"], track_i=self._log_L_dict["track_i"], track_k_opt=self._log_L_dict["track_k_opt"], track_log_L_k=self._log_L_dict["track_log_L_k"], track_step_norm_k=self._log_L_dict["track_step_norm_k"], 
                                Hess=self._Hess, Jac=self._Jac, 
                                log_L_LA=self._log_L["LA"], log_L_FDH=self._log_L["FDH"],
                                T_1=self.get_T_tot_buff(), T_0=self.get_T_0(), T_buff=self.get_T_buff(),
                                p=self._t_series["p"], r=self._t_series["r"],
                                p_F=self._t_series["p_F"], r_F=self._t_series["r_F"],
                                varsigma_t=self._t_series["varsigma"],
                                p_0_F=self._parameters.p_0_F.get_value(),
                                N_s_a_bool=self._parameters.N_s_a_bool.get_value(), N_s_e_bool=self._parameters.N_s_e_bool.get_value(), rho_type_int_s=rho_type_int_s,
                                psi=self._parameters.psi.get_value(), mu_F=self._parameters.mu_F.get_value(), sigma_F=self._parameters.sigma_F.get_value(), delta_s=self._parameters.delta_s.get_value(), gamma_1_s=self._parameters.gamma_1_s.get_value(), gamma_2_s=self._parameters.gamma_2_s.get_value(), sigma_s=self._parameters.sigma_s.get_value(), eta_s=self._parameters.eta_s.get_value(), eta_varsigma=self._parameters.eta_varsigma.get_value(), varsigma_0=self._parameters.varsigma_0.get_value(),
                                beta=self._parameters.beta.get_value(), eta_w=self._parameters.eta_w.get_value(), alpha_0_s=self._parameters.alpha_0_s.get_value(), alpha_n=self._parameters.alpha_n.get_value(), alpha_w=self._parameters.alpha_w.get_value(), lambda_w_s=self._parameters.lambda_w_s.get_value(), 
                                G_sigma=np.float64(self._parameters.G_sigma.get_value()), A_G=A_G, Q_q=Q_q, 
                                D_t_s=self._t_s_series["D"], a_t_s=self._t_s_series["a"], n_t_s=self._t_s_series["n"], S_t_s=self._t_s_series["S"], M_t_s=self._t_s_series["M"], rho_t_s=self._t_s_series["rho"], g_t_s=self._t_s_series["g"], w_t_s=self._t_s_series["w"], nu_t_s=self._t_s_series["nu"], nepsilon_t_s=self._t_s_series["nepsilon"], 
                                mu_t_D=self._Hcomp["mu_t_D"], sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], rho_prime2_k=self._Hcomp["rho_prime2_k"], F_mu=self._Hcomp["F_mu"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                check_opt_last=check_opt_last, overwrite_a=overwrite_a, overwrite_b=overwrite_b, break_nan=break_nan, Hess_F_D_N=Hess_F_D_N)
        
        if track:
            if self._log_L_dict_track["track_i"] < self._log_L_dict_track["N_track"]:
                self._log_L_dict_track["track_k_opt"][self._log_L_dict_track["track_i"]] = np.copy(self._log_L_dict["opt"]["k_opt"])
                self._log_L_dict_track["track_k_end"][self._log_L_dict_track["track_i"]] = np.copy(self._log_L_dict["end"]["k_end"])
                self._log_L_dict_track["track_log_L_k"][self._log_L_dict_track["track_i"],:] = np.copy(self._log_L_dict["k"]["log_L_k"])
                self._log_L_dict_track["track_step_norm_k"][self._log_L_dict_track["track_i"],:] = np.copy(self._log_L_dict["k"]["step_norm_k"])

                self._log_L_dict_track["track_succeed"][self._log_L_dict_track["track_i"]] = int(self._log_L_dict["res"]["succeed"])
                self._log_L_dict_track["track_use_opt_last"][self._log_L_dict_track["track_i"]] = int(self._log_L_dict["res"]["use_opt_last"])

                self._log_L_dict_track["track_i"] += 1
        
        return self._log_L_dict["opt"]["log_L_opt"]



    def calc_p_F_opt_no_term_nb(self, p_F_0=None, N_k=None, N_k_trunc=None, gauss_sigma=None, epsilon=None, step_norm_cutoff=None, return_dict=None, check_opt_last=None, overwrite_a=True, overwrite_b=True, break_nan=None, Hess_F_D_N=None, track=None):

        if Hess_F_D_N is None:
            Hess_F_D_N = False

        if break_nan is None:
            break_nan = False
        
        if return_dict is None:
            return_dict=True

        if track is None:
            track = True

       
        self.set_N_k_trunc(N_k_trunc=N_k_trunc, reset=False)
        self.set_gauss_sigma(gauss_sigma=gauss_sigma, reset=False)
        self.set_epsilon(epsilon=epsilon, reset=False)
        self.set_step_norm_cutoff(step_norm_cutoff=step_norm_cutoff, reset=False)
        self.set_check_opt_last(check_opt_last=check_opt_last, reset=False)

        self.reset_log_L_dict(N_k=N_k, reset=False, keep_opt_last=True)



        Q_q = self._parameters.calc_Q_q()
        A_G = self._parameters.calc_A_G()
        rho_type_int_s = self._parameters.calc_rho_type_int_s()

        if p_F_0 is None:
            self._log_L_dict["x"]["p_F_0"] = self.calc_p_F_0_gauss()
        else:
            self._log_L_dict["x"]["p_F_0"] = p_F_0

        if np.any(np.isnan(self._log_L_dict["x"]["p_F_0_opt_last_1"])):
            self._log_L_dict["x"]["p_F_0_opt_last_1"] = np.copy(self._log_L_dict["x"]["p_F_0"])
            check_opt_last = False
        else:
            check_opt_last = self._log_L_dict["spec"]["check_opt_last"]

        self._log_L_dict["x"]["p_F_0_opt_last_0"] = np.copy(self._log_L_dict["x"]["p_F_0_opt_last_1"])



        (
            self._log_L_dict["x"]["p_F_0"], self._log_L_dict["x"]["p_F_0_opt_last_1"], self._log_L_dict["x"]["p_F_0_set"],
            self._log_L_dict["x"]["p_F_opt"], self._log_L_dict["x"]["p_F_end"], 
            self._log_L_dict["k"]["log_L_k"], self._log_L_dict["k"]["step_norm_k"], self._log_L_dict["k"]["p_F_k"],
            self._log_L_dict["opt"]["k_opt"], self._log_L_dict["opt"]["log_L_opt"], self._log_L_dict["opt"]["step_norm_opt"],
            self._log_L_dict["end"]["k_end"], self._log_L_dict["end"]["log_L_end"], self._log_L_dict["end"]["step_norm_end"],
            self._log_L_dict["temp"]["p_F_k0"], self._log_L_dict["temp"]["p_F_k1"], self._log_L_dict["temp"]["A"], self._log_L_dict["temp"]["B"],
            self._log_L_dict["res"]["use_opt_last"], self._log_L_dict["res"]["succeed"],
            #self._log_L_dict["x"]["p_F_0"], self._log_L_dict["spec"]["N_k"], self._log_L_dict["spec"]["epsilon"], self._log_L_dict["spec"]["step_norm_cutoff"], self._log_L_dict["x"]["p_F_0_opt_last_1"],
            #self._log_L_dict["opt"]["k_opt"], self._log_L_dict["k"]["p_F_k"], self._log_L_dict["temp"]["p_F_k0"], self._log_L_dict["temp"]["p_F_k1"], self._log_L_dict["x"]["p_F_opt"], self._log_L_dict["k"]["log_L_k"], self._log_L_dict["opt"]["log_L_opt"], self._log_L_dict["k"]["step_norm_k"], self._log_L_dict["temp"]["A"], self._log_L_dict["temp"]["B"], self._log_L_dict["res"]["succeed"], self._log_L_dict["res"]["use_opt_last"],
            #self._log_L_dict["N_track"], self._log_L_dict["track_i"], self._log_L_dict["track_k_opt"], self._log_L_dict["track_log_L_k"], self._log_L_dict["track_step_norm_k"], 
            self._Hess, self._Jac, 
            self._log_L["LA"], self._log_L["FDH"],
            self._t_series["p"], self._t_series["r"],
            self._t_series["p_F"], self._t_series["r_F"],
            self._t_series["varsigma"],
            self._t_s_series["D"], self._t_s_series["a"], self._t_s_series["n"], self._t_s_series["S"], self._t_s_series["M"], self._t_s_series["rho"], self._t_s_series["g"], self._t_s_series["w"], self._t_s_series["nu"], self._t_s_series["nepsilon"],
            self._Hcomp["mu_t_D"], self._Hcomp["sigma_t_D_2"], self._Hcomp["rho_prime_k"], self._Hcomp["rho_prime2_k"], self._Hcomp["F_mu"], self._Hcomp["D_mu"], self._Hcomp["dr_mu"], self._Hcomp["ns2_m_1"], self._Hcomp["nu_w"], self._Hcomp["r_1"],
            ) = ssv_abm_nb.calc_p_F_opt_no_term(
                                T=self.get_T(), 
                                p_F_0=self._log_L_dict["x"]["p_F_0"], p_F_0_opt_last_1=self._log_L_dict["x"]["p_F_0_opt_last_1"], p_F_0_set=self._log_L_dict["x"]["p_F_0_set"],
                                N_k=self._log_L_dict["spec"]["N_k"], N_k_trunc=self._log_L_dict["spec"]["N_k_trunc"], epsilon=self._log_L_dict["spec"]["epsilon"], step_norm_cutoff=self._log_L_dict["spec"]["step_norm_cutoff"], #check_opt_last=
                                p_F_opt=self._log_L_dict["x"]["p_F_opt"], p_F_end=self._log_L_dict["x"]["p_F_end"], 
                                log_L_k=self._log_L_dict["k"]["log_L_k"], step_norm_k=self._log_L_dict["k"]["step_norm_k"], p_F_k=self._log_L_dict["k"]["p_F_k"],
                                k_opt=self._log_L_dict["opt"]["k_opt"], log_L_opt=self._log_L_dict["opt"]["log_L_opt"], step_norm_opt=self._log_L_dict["opt"]["step_norm_opt"],
                                k_end=self._log_L_dict["end"]["k_end"], log_L_end=self._log_L_dict["end"]["log_L_end"], step_norm_end=self._log_L_dict["end"]["step_norm_end"],
                                p_F_k0=self._log_L_dict["temp"]["p_F_k0"], p_F_k1=self._log_L_dict["temp"]["p_F_k1"], A=self._log_L_dict["temp"]["A"], B=self._log_L_dict["temp"]["B"],
                                use_opt_last=self._log_L_dict["res"]["use_opt_last"], succeed=self._log_L_dict["res"]["succeed"],
                                #p_F_0=self._log_L_dict["x"]["p_F_0"], N_k=self._log_L_dict["spec"]["N_k"], epsilon=self._log_L_dict["spec"]["epsilon"], step_norm_cutoff=self._log_L_dict["spec"]["step_norm_cutoff"], p_F_0_opt_last_1=self._log_L_dict["x"]["p_F_0_opt_last_1"],
                                #k_opt=self._log_L_dict["opt"]["k_opt"], p_F_k=self._log_L_dict["k"]["p_F_k"], p_F_k0=self._log_L_dict["temp"]["p_F_k0"], p_F_k1=self._log_L_dict["temp"]["p_F_k1"], p_F_opt=self._log_L_dict["x"]["p_F_opt"], log_L_k=self._log_L_dict["k"]["log_L_k"], log_L_opt=self._log_L_dict["opt"]["log_L_opt"], step_norm_k=self._log_L_dict["k"]["step_norm_k"], A=self._log_L_dict["temp"]["A"], B=self._log_L_dict["temp"]["B"], succeed=self._log_L_dict["res"]["succeed"], use_opt_last=self._log_L_dict["res"]["use_opt_last"],
                                #N_track=self._log_L_dict["N_track"], track_i=self._log_L_dict["track_i"], track_k_opt=self._log_L_dict["track_k_opt"], track_log_L_k=self._log_L_dict["track_log_L_k"], track_step_norm_k=self._log_L_dict["track_step_norm_k"], 
                                Hess=self._Hess, Jac=self._Jac, 
                                log_L_LA=self._log_L["LA"], log_L_FDH=self._log_L["FDH"],
                                T_1=self.get_T_tot_buff(), T_0=self.get_T_0(), T_buff=self.get_T_buff(),
                                p=self._t_series["p"], r=self._t_series["r"],
                                p_F=self._t_series["p_F"], r_F=self._t_series["r_F"],
                                varsigma_t=self._t_series["varsigma"],
                                p_0_F=self._parameters.p_0_F.get_value(),
                                N_s_a_bool=self._parameters.N_s_a_bool.get_value(), N_s_e_bool=self._parameters.N_s_e_bool.get_value(), rho_type_int_s=rho_type_int_s,
                                psi=self._parameters.psi.get_value(), mu_F=self._parameters.mu_F.get_value(), sigma_F=self._parameters.sigma_F.get_value(), delta_s=self._parameters.delta_s.get_value(), gamma_1_s=self._parameters.gamma_1_s.get_value(), gamma_2_s=self._parameters.gamma_2_s.get_value(), sigma_s=self._parameters.sigma_s.get_value(), eta_s=self._parameters.eta_s.get_value(), eta_varsigma=self._parameters.eta_varsigma.get_value(), varsigma_0=self._parameters.varsigma_0.get_value(),
                                beta=self._parameters.beta.get_value(), eta_w=self._parameters.eta_w.get_value(), alpha_0_s=self._parameters.alpha_0_s.get_value(), alpha_n=self._parameters.alpha_n.get_value(), alpha_w=self._parameters.alpha_w.get_value(), lambda_w_s=self._parameters.lambda_w_s.get_value(), 
                                G_sigma=np.float64(self._parameters.G_sigma.get_value()), A_G=A_G, Q_q=Q_q, 
                                D_t_s=self._t_s_series["D"], a_t_s=self._t_s_series["a"], n_t_s=self._t_s_series["n"], S_t_s=self._t_s_series["S"], M_t_s=self._t_s_series["M"], rho_t_s=self._t_s_series["rho"], g_t_s=self._t_s_series["g"], w_t_s=self._t_s_series["w"], nu_t_s=self._t_s_series["nu"], nepsilon_t_s=self._t_s_series["nepsilon"], 
                                mu_t_D=self._Hcomp["mu_t_D"], sigma_t_D_2=self._Hcomp["sigma_t_D_2"], rho_prime_k=self._Hcomp["rho_prime_k"], rho_prime2_k=self._Hcomp["rho_prime2_k"], F_mu=self._Hcomp["F_mu"], D_mu=self._Hcomp["D_mu"], dr_mu=self._Hcomp["dr_mu"], ns2_m_1=self._Hcomp["ns2_m_1"], nu_w=self._Hcomp["nu_w"], r_1=self._Hcomp["r_1"],
                                check_opt_last=check_opt_last, overwrite_a=overwrite_a, overwrite_b=overwrite_b, break_nan=break_nan, Hess_F_D_N=Hess_F_D_N)
        
        if track:
            if self._log_L_dict_track["track_i"] < self._log_L_dict_track["N_track"]:
                self._log_L_dict_track["track_k_opt"][self._log_L_dict_track["track_i"]] = np.copy(self._log_L_dict["opt"]["k_opt"])
                self._log_L_dict_track["track_k_end"][self._log_L_dict_track["track_i"]] = np.copy(self._log_L_dict["end"]["k_end"])
                self._log_L_dict_track["track_log_L_k"][self._log_L_dict_track["track_i"],:] = np.copy(self._log_L_dict["k"]["log_L_k"])
                self._log_L_dict_track["track_step_norm_k"][self._log_L_dict_track["track_i"],:] = np.copy(self._log_L_dict["k"]["step_norm_k"])

                self._log_L_dict_track["track_succeed"][self._log_L_dict_track["track_i"]] = int(self._log_L_dict["res"]["succeed"])
                self._log_L_dict_track["track_use_opt_last"][self._log_L_dict_track["track_i"]] = int(self._log_L_dict["res"]["use_opt_last"])

                self._log_L_dict_track["track_i"] += 1
        
        return self._log_L_dict["opt"]["log_L_opt"]



    def calc_p_F_opt_np(self, p_F_0=None, N_k=None, N_k_trunc=None, gauss_sigma=None, epsilon=None, step_norm_cutoff=None, return_dict=None, check_opt_last=None, overwrite_a=True, overwrite_b=True, break_nan=None, track=None):



        if break_nan is None:
            break_nan = False


        
        if return_dict is None:
            return_dict=True

        if track is None:
            track = True


        self.set_N_k_trunc(N_k_trunc=N_k_trunc, reset=False)
        self.set_gauss_sigma(gauss_sigma=gauss_sigma, reset=False)
        self.set_epsilon(epsilon=epsilon, reset=False)
        self.set_step_norm_cutoff(step_norm_cutoff=step_norm_cutoff, reset=False)
        self.set_check_opt_last(check_opt_last=check_opt_last, reset=False)

        self.reset_log_L_dict(N_k=N_k, reset=False, keep_opt_last=True)

        

        T = self.get_T()
        T_sqrt = np.sqrt(T)

        if p_F_0 is None:
            self._log_L_dict["x"]["p_F_0"] = self.calc_p_F_0_gauss()
        else:
            self._log_L_dict["x"]["p_F_0"] = p_F_0

        if np.any(np.isnan(self._log_L_dict["x"]["p_F_0_opt_last_1"])):
            self._log_L_dict["x"]["p_F_0_opt_last_1"] = np.copy(self._log_L_dict["x"]["p_F_0"])
            check_opt_last = False
        else:
            check_opt_last = self._log_L_dict["spec"]["check_opt_last"]

        self._log_L_dict["x"]["p_F_0_opt_last_0"] = np.copy(self._log_L_dict["x"]["p_F_0_opt_last_1"])


        self.set_p_F_np(p_F=self._log_L_dict["x"]["p_F_0"], run_p_t=True, set_Hess_adj=False, set_log_L=True)
        self._log_L_dict["opt"]["k_opt"] = 0
        self._log_L_dict["x"]["p_F_opt"] = np.copy(self._log_L_dict["x"]["p_F_0"])
        self._log_L_dict["opt"]["log_L_opt"] = self._log_L["LA"]
        

        if check_opt_last:
            self.set_p_F_np(p_F=self._log_L_dict["x"]["p_F_0_opt_last_1"], run_p_t=True, set_Hess_adj=False, set_log_L=True)
            if not np.isnan(self._log_L["LA"]):
                self._log_L_dict["res"]["use_opt_last"] = self._log_L["LA"] > self._log_L_dict["opt"]["log_L_opt"]
                if self._log_L_dict["res"]["use_opt_last"]:
                    self._log_L_dict["x"]["p_F_opt"] = np.copy(self._log_L_dict["x"]["p_F_0_opt_last_1"])
                    self._log_L_dict["opt"]["log_L_opt"] = self._log_L["LA"]
                else:
                    self.set_p_F_np(p_F=self._log_L_dict["x"]["p_F_0"], run_p_t=True, set_Hess_adj=False, set_log_L=True)

        self._log_L_dict["opt"]["p_F_0_set"] = np.copy(self._log_L_dict["x"]["p_F_opt"])


        self._log_L_dict["k"]["p_F_k"][0,:] = np.copy(self._log_L_dict["x"]["p_F_opt"])
        self._log_L_dict["k"]["log_L_k"][0] = self._log_L["LA"]

        self._log_L_dict["temp"]["p_F_k0"] =  np.copy(self._log_L_dict["x"]["p_F_opt"])
        self._log_L_dict["temp"]["p_F_k1"] = np.copy(self._log_L_dict["x"]["p_F_opt"])

        for k in range(1,self._log_L_dict["spec"]["N_k"]):
            self._log_L_dict["temp"]["p_F_k0"] = np.copy(self._log_L_dict["temp"]["p_F_k1"])

            #self._log_L_dict["temp"]["A"] = -self._Hess_adj
            self._log_L_dict["temp"]["A"] = -self.calc_Hess_adj_copy()
            self._log_L_dict["temp"]["B"] = (self._log_L_dict["temp"]["A"] @ self._log_L_dict["temp"]["p_F_k0"]) + self._Jac
            self._log_L_dict["temp"]["p_F_k1"] = scipy.linalg.solve(self._log_L_dict["temp"]["A"], self._log_L_dict["temp"]["B"], assume_a="sym", overwrite_a=overwrite_a, overwrite_b=overwrite_b)

            self._log_L_dict["k"]["step_norm_k"][k] = np.sqrt(np.sum((self._log_L_dict["temp"]["p_F_k1"] - self._log_L_dict["temp"]["p_F_k0"])**2))/T_sqrt

            self.set_p_F_np(p_F=self._log_L_dict["temp"]["p_F_k1"], run_p_t=True, set_Hess_adj=False, set_log_L=True)
            self._log_L_dict["k"]["log_L_k"][k] = self._log_L["LA"]

            self._log_L_dict["k"]["p_F_k"][k,:] = self._log_L_dict["temp"]["p_F_k1"]

            if self._log_L_dict["k"]["log_L_k"][k] > self._log_L_dict["opt"]["log_L_opt"]:
                self._log_L_dict["opt"]["k_opt"] = k
                self._log_L_dict["opt"]["log_L_opt"] = self._log_L_dict["k"]["log_L_k"][self._log_L_dict["opt"]["k_opt"]]
                self._log_L_dict["x"]["p_F_opt"] = np.copy(self._log_L_dict["temp"]["p_F_k1"])

            if self._log_L_dict["spec"]["step_norm_cutoff"]>0:
                if self._log_L_dict["k"]["step_norm_k"][k]<self._log_L_dict["spec"]["step_norm_cutoff"]:
                    break

            if self._log_L_dict["spec"]["N_k_trunc"]>0:
                if k - self._log_L_dict["opt"]["k_opt"] >= self._log_L_dict["spec"]["N_k_trunc"]:
                    break

            if np.any(np.isnan(self._Jac)) or np.any(np.isnan(self._Hess)):
                self._log_L_dict["res"]["succeed"] = False
                if break_nan:
                    self._log_L_dict["opt"]["log_L_opt"] = np.nan
                    break
                

        #x_iK = np.copy(x_i1)
        if self._log_L_dict["res"]["succeed"] or not break_nan:
            self._log_L_dict["x"]["p_F_0_opt_last_1"] = np.copy(self._log_L_dict["x"]["p_F_opt"])
            if k!=self._log_L_dict["opt"]["k_opt"]:
                self.set_p_F_np(p_F=self._log_L_dict["x"]["p_F_opt"], run_p_t=True, set_log_L=True)



        self._log_L_dict["end"]["k_end"] = k
        self._log_L_dict["opt"]["log_L_end"] = self._log_L_dict["k"]["log_L_k"][self._log_L_dict["end"]["k_end"]]
        self._log_L_dict["opt"]["step_norm_end"] = self._log_L_dict["k"]["step_norm_k"][self._log_L_dict["end"]["k_end"]]
        self._log_L_dict["x"]["p_F_end"] = np.copy(self._log_L_dict["temp"]["p_F_k1"])


        if track:
            if self._log_L_dict_track["track_i"] < self._log_L_dict_track["N_track"]:
                self._log_L_dict_track["track_k_opt"][self._log_L_dict_track["track_i"]] = np.copy(self._log_L_dict["opt"]["k_opt"])
                self._log_L_dict_track["track_k_end"][self._log_L_dict_track["track_i"]] = np.copy(self._log_L_dict["end"]["k_end"])
                self._log_L_dict_track["track_log_L_k"][self._log_L_dict_track["track_i"],:] = np.copy(self._log_L_dict["k"]["log_L_k"])
                self._log_L_dict_track["track_step_norm_k"][self._log_L_dict_track["track_i"],:] = np.copy(self._log_L_dict["k"]["step_norm_k"])

                self._log_L_dict_track["track_succeed"][self._log_L_dict_track["track_i"]] = int(self._log_L_dict["res"]["succeed"])
                self._log_L_dict_track["track_use_opt_last"][self._log_L_dict_track["track_i"]] = int(self._log_L_dict["res"]["use_opt_last"])

                self._log_L_dict_track["track_i"] += 1


        return self._log_L_dict["opt"]["log_L_opt"]


    #def set_log_L_dict(self, p_F_0=None, N_i=None, epsilon=None, step_norm_cutoff=None, overwrite_a=True, overwrite_b=True, break_nan=None):
    #    self._log_L_dict = self.calc_log_L_np(p_F_0=p_F_0, N_i=N_i, epsilon=epsilon, step_norm_cutoff=step_norm_cutoff, return_dict=True, overwrite_a=overwrite_a, overwrite_b=overwrite_b, break_nan=break_nan)



    def set_linreg_p(self):

        self._linreg_p = dict()

        T = self.get_T()

        H = np.transpose(np.array([1+np.arange(T), np.ones(T)]))
        z = self.get_t_series("p")

        self._linreg_p["cov"] = np.linalg.inv(H.T @ H)
        self._linreg_p["theta"] = self._linreg_p["cov"] @ H.T @ z
        self._linreg_p["theta_se"] = np.sqrt(np.diag(self._linreg_p["cov"]))

        # linreg_1 = np.linalg.lstsq(a=np.transpose(np.array([1+ts, np.ones(self.get_T())])), b=self.get_t_series("p"))



    
    def calc_theta_p_F_endpoints(self, p_0=None, est_p_0=None):

        if np.any(self._t_series["p"]==np.nan):
            mu_F = None
            p_0_F = None
            p_0 = None
            sigma_F = None
        
        else:

            if est_p_0 is None:
                if p_0 is None:
                    est_p_0 = True
                else:
                    est_p_0 = False

            if est_p_0:
                mu_F = (self._t_series["p"][self.get_T_tot_buff()-1] - self._t_series["p"][self.get_T_0()]) / (self._parameters.psi.get_value() * (self.get_T()-1))
                p_0 = self._t_series["p"][self.get_T_0()] - (self._parameters.psi.get_value() * mu_F)
            else:

                if p_0 is None:
                    p_0 = self._parameters.p_0.get_value()
                mu_F = (self._t_series["p"][self.get_T_tot_buff()-1] - self._t_series["p"][self.get_T_0()-1]) / (self._parameters.psi.get_value() * (self.get_T()))

            p_0_F = p_0
            sigma_F = np.std(self.get_t_series("r")/self._parameters.psi.get_value())/np.sqrt(2)
        
        
        return mu_F, p_0_F, p_0, sigma_F




    def calc_theta_p_F_ls(self):


        self.set_linreg_p()
        
        psi = self._parameters.psi.get_value()

        #self._parameters.p_0_F.set_default(self._linreg_p["theta"][1])
        #self._parameters.mu_F.set_default(self._linreg_p["theta"][0])

        if np.any(np.isnan(self._linreg_p["theta"])):
            mu_F = None
            p_0_F = None
            p_0 = None
            sigma_F = None
        else:

            mu_F = self._linreg_p["theta"][0]/psi
            p_0_F = self._linreg_p["theta"][1]
            p_0 = self.get_t_series("p")[0] - psi*mu_F

            sigma_F = np.std(self.get_t_series("r")/psi)/np.sqrt(2)

        return mu_F, p_0_F, p_0, sigma_F


    def push_theta_p_F_estimate(self, set_mu_F=None, set_p_0_F=None, set_p_0=None, set_sigma_F=None):


        if set_mu_F is None:
            set_mu_F = True
        if set_p_0_F is None:
            set_p_0_F = True
        if set_p_0 is None:
            set_p_0 = True
        if set_sigma_F is None:
            set_sigma_F = True

        mu_F, p_0_F, p_0, sigma_F = self.calc_theta_p_F_ls()


        if set_mu_F:
            if mu_F is not None:
                self._parameters.mu_F.set_value(mu_F)
        if set_p_0_F:
            if p_0_F is not None:
                self._parameters.p_0_F.set_value(p_0_F)
        if set_p_0:
            if p_0 is not None:
                self._parameters.p_0.set_value(p_0)
        if set_sigma_F:
            if sigma_F is not None:
                self._parameters.sigma_F.set_value(sigma_F)
        
        




    def set_parameters_defaults(self, set_opt_Bounds=True, factor_se=None):

        if factor_se is None:
            factor_se = 2

        psi = self._parameters.psi.get_value()

        self.set_linreg_p()
        if np.any(np.isnan(self._linreg_p["theta"])):
            mu_F = None
            p_0_F = None
            p_0 = None
            sigma_F = None

        else:
            mu_F, p_0_F, p_0, sigma_F = self.calc_theta_p_F_endpoints()



        self._parameters.set_mode_defaults(p_0_F=p_0_F, p_0=p_0,
                                            mu_F=mu_F, sigma_F=sigma_F, sigma_D=sigma_F)


        if False:
            linreg_param = ["mu_F", "p_0_F"]
            for k in range(len(linreg_param)):
                self._parameters.__dict__[linreg_param[k]]._opt_range.set_range(incl_low=self._linreg_p["theta"][k] - factor_se*self._linreg_p["theta_se"][k],
                                                                                incl_high=self._linreg_p["theta"][k] + factor_se*self._linreg_p["theta_se"][k])
        if np.any(np.isnan(self._linreg_p["theta"])) or np.any(np.isnan(self._linreg_p["theta_se"])):
            self._parameters.mu_F._opt_range.set_range(incl_low=-np.infty, incl_high=np.infty)
            self._parameters.p_0_F._opt_range.set_range(incl_low=-np.infty, incl_high=np.infty)
        else:
            self._parameters.mu_F._opt_range.set_range(incl_low=(self._linreg_p["theta"][0] - factor_se*self._linreg_p["theta_se"][0])/psi,
                                                                                incl_high=(self._linreg_p["theta"][0] + factor_se*self._linreg_p["theta_se"][0])/psi)
            self._parameters.p_0_F._opt_range.set_range(incl_low=(self._linreg_p["theta"][1] - factor_se*self._linreg_p["theta_se"][1]),
                                                                                incl_high=(self._linreg_p["theta"][1] + factor_se*self._linreg_p["theta_se"][1]))

        





        self._parameters.p_0._opt_range.set_range(incl_low=self._parameters.p_0.get_default() - 4*self._parameters.sigma_F.get_default(), incl_high=self._parameters.p_0.get_default() + 4*self._parameters.sigma_F.get_default())

        sigma_s_default = np.max(self._parameters.sigma_s.get_default())
        gamma_2_s_default = np.max(self._parameters.gamma_2_s.get_default())


        self._parameters.sigma_F._opt_range.set_range(incl_low=self._parameters.sigma_F.get_default()*1E-2, incl_high=self._parameters.sigma_F.get_default()*1E+2)
        self._parameters.sigma_s._opt_range.set_range(incl_low=sigma_s_default*1E-2, incl_high=sigma_s_default*1E+2)

        self._parameters.gamma_2_s._opt_range.set_incl_high(2)
        #self._parameters.gamma_2_s._opt_range.set_incl_low(2)
        self._parameters.gamma_2_s._opt_range.set_incl_low(np.sqrt(2/np.pi))


        

        #self._parameters.delta_s._opt_range.set_range(incl_high=1)
        #self._parameters.gamma_1_s._opt_range.set_range(incl_high=1)
        self._parameters.eta_s._opt_range.set_incl_low(1E-2)


        self._parameters.set_opt_Bounds()


        #self._parameters._opt_Bounds[np.array(self._parameters._flat_index["dict"]["range"]["delta_s"])[0],0] = self._parameters.delta_s.get_default()[0]*1E-1





        self._parameters._opt_Bounds[np.array(self._parameters._flat_index["dict"]["range"]["delta_s"])[self._parameters.rho_type_s.get_value()=="linear"],1] = 1/np.sqrt(np.sum(self._parameters.N_s_a_bool.get_value()))
        self._parameters._opt_Bounds[np.array(self._parameters._flat_index["dict"]["range"]["gamma_1_s"])[self._parameters.rho_type_s.get_value()=="cubic"],1] = 2.0


        rho_tanh_where = np.where(np.sum(self._parameters._rmode_array_bool[3:,:], axis=0))[0]
        self._parameters._opt_Bounds[np.array(self._parameters._flat_index["dict"]["range"]["gamma_1_s"])[rho_tanh_where],0] = self._parameters.gamma_1_s.get_default()[rho_tanh_where]*1E-2
        self._parameters._opt_Bounds[np.array(self._parameters._flat_index["dict"]["range"]["gamma_1_s"])[rho_tanh_where],1] = self._parameters.gamma_1_s.get_default()[rho_tanh_where]*1E+2
        

        

        if set_opt_Bounds:
            dfrac = 0.1


    def set_flat_parameters(self, flat_parameters, subordinate=None, push=None):

        if push is None:
            push = True

        self._parameters.set_flat_parameters(flat_parameters=flat_parameters, subordinate=subordinate, push=push)
        #if push:
        #    self.set_Hcomp_param()


    def set_flat_parameters_free(self, flat_parameters_free, flat_free):


        self._parameters.set_flat_parameters_free(flat_parameters_free=flat_parameters_free, flat_free=flat_free)
        #self.set_Hcomp_param()





    def set_parameters_estimate(self, flat_parameters_0=None, flat_fix=None, rbe_str=None, rmode=None, bmode=None, emode=None, insert=None, print_res_time=True, opt_method=None, factor_se=None, epsilon=None, step_norm_cutoff=None, check_opt_last=None, Hess_F_D_N=None, track=None, set_se=None, N_k=None, N_k_trunc=None, gauss_sigma=None):

        


        res_timer = misc_fns.timer(print_timer=print_res_time, print_tab="\t", print_head="res")



        if set_se is None:
            set_se = True

        if check_opt_last is None:
            if self._parameters.rho_type_s.get_value()[0]=="linear":
                check_opt_last = False
                #check_opt_last = True
            else:
                #check_opt_last = True
                check_opt_last = False

        if Hess_F_D_N is None:
            Hess_F_D_N = False

        if track is None:
            track = True


        self._parameters._train_t = self._train_t

        self._parameters.set_mode(rbe_str=rbe_str, rmode=rmode, bmode=bmode, emode=emode)
        self._parameters.set_opt_method(opt_method)


        self.set_N_k_trunc(N_k_trunc=N_k_trunc, reset=True)
        self.set_gauss_sigma(gauss_sigma=gauss_sigma, reset=True)
        self.set_epsilon(epsilon=epsilon, reset=True)
        self.set_step_norm_cutoff(step_norm_cutoff=step_norm_cutoff, reset=True)
        self.set_check_opt_last(check_opt_last=check_opt_last, reset=True)

        self._parameters.res_time['opt']['method'] = self._parameters.get_opt_method()

        # phi_type
        
        self._parameters.pull_flat_parameters()
        self.set_parameters_defaults(factor_se=factor_se)
        

        self._parameters.set_flat_fix(fix=flat_fix)


        flat_defaults = self._parameters.flatten_defaults()
        # Set initial parameter values
        if flat_parameters_0 is None:
            self._parameters.res_fp0 = np.copy(flat_defaults)
        else:
            self._parameters.res_fp0 = np.copy(flat_parameters_0)

        


        flat_free = np.copy(self._parameters._flat_free)

        self.set_flat_parameters(self._parameters.res_fp0)

        self._parameters.res_fp1 = self._parameters.flatten()


        N_s_a_bool = self._parameters.N_s_a_bool.get_value()

        sigma_s_free = self._parameters._flat_free[self._parameters._flat_index["dict"]["range"]["sigma_s"]]
        sigma_s_subordinate = self._parameters._flat_subordinate[self._parameters._flat_index["dict"]["range"]["sigma_s"]] - self._parameters._flat_index["dict"]["start"]["sigma_s"]
        sigma_s_flat_start = np.sum(self._parameters._flat_free[:self._parameters._flat_index["dict"]["start"]["sigma_s"]])
        sigma_s_flat_end = sigma_s_flat_start + np.sum(sigma_s_free)

        self._hsigma_s = self._parameters.sigma_s.get_value()
        self._sigma_s = self._hsigma_s[sigma_s_subordinate[:,1]]

        def sigma_s_constraint(flat_parameters_free):

            self._hsigma_s[sigma_s_free] = flat_parameters_free[sigma_s_flat_start:sigma_s_flat_end]
            self._sigma_s = self._hsigma_s[sigma_s_subordinate[:,1]]


            return np.sum(self._sigma_s[N_s_a_bool]) - 1E-3

        lin_con =   {
                            "type": "ineq",
                            "fun": sigma_s_constraint,
                        }


        #cons = np.concatenate([[sigma_s_con]])

        self.reset_log_L_dict(N_k=N_k, reset=True, keep_opt_last=False)

        # Function to minimise (i.e. minus likelihood with parameters as vector)
        def log_L_min(flat_parameters_free, flat_free):
            self.set_flat_parameters_free(flat_parameters_free, flat_free)
            return -self.calc_p_F_opt_nb(overwrite_a=False, overwrite_b=False, Hess_F_D_N=Hess_F_D_N, track=track)


        self._log_L_dict["x"]["p_F_0_opt_last_1"] = self.calc_p_F_0_gauss()

        
        # Optimisation
        opt_timer = misc_fns.timer(print_timer=print_res_time, print_tab="\t\t", print_head=self._parameters.get_opt_method())
        self._parameters.res = scipy.optimize.minimize(log_L_min, x0=self._parameters.res_fp1[flat_free], args=(flat_free),
                                        method=self._parameters.get_opt_method(), options={'disp': print_res_time, 'ftol': self._parameters.ftol},
                                        bounds=scipy.optimize.Bounds(lb=self._parameters._opt_Bounds[flat_free,0], ub=self._parameters._opt_Bounds[flat_free,1]),
                                        )
                                        #onstraints=lin_con)
        opt_timer.stop()
        self._parameters.res_time['opt'] = opt_timer.get_timer_dict()



        self._parameters.res_fp2 = np.copy(self._parameters.res_fp1)
        self._parameters.res_fp2[flat_free] = self._parameters.res.x
        self._parameters.res_fp2[np.logical_and(flat_free, np.abs(self._parameters.res_fp2) < 1E-8)] = 0
        self._parameters.res_fp2 = self._parameters.res_fp2[self._parameters._flat_subordinate[:,1]]
        self.set_flat_parameters(flat_parameters=self._parameters.res_fp2)

        self._parameters.res_fp2_p_F = np.copy(self.get_t_series("p_F"))
        self._parameters.res_log_L_dict = self.copy_log_L_dict_res()


        if set_se:

            Hess_step = 1E-6
            zero_eff = 1E-6

            # Unconstrained non-zero parameters
            if False:

                unconstrained_nz = np.logical_and(self._parameters._flat_free, np.abs(self._parameters.res_fp2) >= zero_eff)

                eta_max = np.concatenate([np.array(self._parameters._flat_index["dict"]["range"][f"eta_{sf}"])[np.where(self._parameters.res_fp2[self._parameters._flat_index["dict"]["range"][f"eta_{sf}"]] == 1)[0]] for sf in ["w", "s"]])
                eta_nmax = np.concatenate([np.array(self._parameters._flat_index["dict"]["range"][f"eta_{sf}"])[np.where(self._parameters.res_fp2[self._parameters._flat_index["dict"]["range"][f"eta_{sf}"]] >= 1*(1-Hess_step))[0]] for sf in ["w", "s"]])


                unconstrained_nz[eta_max] = False

                steps = self._parameters.res_fp2*Hess_step
                steps[eta_nmax] = 1 - self._parameters.res_fp2[eta_nmax]

            else:

                unconstrained_nz = np.copy(self._parameters._flat_free)
                steps = self._parameters.res_fp2*Hess_step

                fp2_zero = np.abs(self._parameters.res_fp2) <= zero_eff

                if False:
                    unconstrained_nz[fp2_zero] = False
                else:
                    steps[fp2_zero] = Hess_step*zero_eff

                

                fp2_minmax = self._parameters.res_fp2 == np.transpose(self._parameters._opt_Bounds)

                fp2_nearminmax = np.array([
                                                self._parameters.res_fp2-steps < self._parameters._opt_Bounds[:,0],
                                                self._parameters.res_fp2+steps > self._parameters._opt_Bounds[:,1],
                                                ])
                
                unconstrained_nz[np.sum(fp2_minmax, axis=0, dtype=bool)] = False
                

                steps[fp2_nearminmax[0,:]] = self._parameters.res_fp2[fp2_nearminmax[0,:]]
                steps[fp2_nearminmax[1,:]] = self._parameters._opt_Bounds[fp2_nearminmax[1,:],1] - self._parameters.res_fp2[fp2_nearminmax[1,:]]

            # Function to minimise exclusive to unconstrained non-zero parameters (i.e. minus likelihood with parameters as vector)

            #track = False
            def log_L_min_unconstrained(flat_parameters_unconstrained):
                return log_L_min(flat_parameters_unconstrained, unconstrained_nz)

            # Standard error
            se = [None for i in range(self._parameters.res_fp0.size)]
            # Ad-hoc check for fitting across u range
            #if self._parameters.res.fun < 500
            self._parameters.fitted_success = True
            hess = numdifftools.Hessian(log_L_min_unconstrained, full_output=True, method="central", step=steps[unconstrained_nz])(self._parameters.res_fp2[unconstrained_nz])[0]
            try:
                np.linalg.inv(hess)
            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    se_unconstrained = np.full(np.sum(unconstrained_nz), np.nan)
                    self._parameters.fitted_success = False
            
            if self._parameters.fitted_success:
                se_unconstrained = np.sqrt(np.abs(np.diag(np.linalg.inv(hess))))     
            for i in range(se_unconstrained.size):
                se[np.where(unconstrained_nz)[0][i]] = se_unconstrained[i]
            for i in range(len(se)):
                se[i] = se[self._parameters._flat_subordinate[i,1]]
            #else:
                #self._parameters.fitted_success = False
        else:
            se = None
    

        # Set optimised parameters 
        self.set_flat_parameters(flat_parameters=self._parameters.res_fp2)
        self.set_p_F_nb(new_p_F=self._parameters.res_fp2_p_F, Hess_F_D_N=Hess_F_D_N)


        self._parameters.set_fit(se=se, fp1=self._parameters.res_fp1)



        # Run model at optimised parameters
        #self.set_t_series()       

        # Information criteria
        self._N_params = self._parameters._N_params
        self._parameters._log_L = -self._parameters.res.fun

        self._parameters._fit_scores = misc_fns.IC_dict(log_L=self._parameters._log_L, N_params=self._parameters._N_params, N_data=self.get_T())
        
        res_timer.stop()
        self._parameters.res_time["res"] = res_timer.get_timer_dict()

        # Save copy of estimate
        self._parameters_estimate = copy.deepcopy(self._parameters)

        return self._parameters.res



    def calc_n_fev_dict(self, fdict=None):

        if fdict is None:
            fdict = "est"

        n_fev_dict = dict()

        n_fev = self._parameters.res.nfev
        n_fev_se = self._log_L_dict_track["track_i"] - n_fev


        if fdict=="est":
            n_fev_dict["n_fev"] = n_fev
            if n_fev > self._log_L_dict_track["N_track"]:
                n_fev_dict["n_p_F_opt"] = self.get_opt_spec_dict
            else:
                n_fev_dict["n_p_F_opt"] = np.sum(self._log_L_dict_track["track_k_end"][:n_fev])
        elif fdict=="se":
            n_fev_dict["n_fev_se"] = n_fev_se
            if n_fev > self._log_L_dict_track["N_track"]:
                n_fev_dict["n_p_F_opt_se"] = 0
            elif n_fev+n_fev_se > self._log_L_dict_track["N_track"]:
                n_fev_dict["n_p_F_opt_se"] = 0
            else:
                n_fev_dict["n_p_F_opt_se"] = np.sum(self._log_L_dict_track["track_k_end"][n_fev:n_fev+n_fev_se])

        return n_fev_dict
    


    def calc_fit_df(self, fdict=None, lead=None, tail=None):

        if fdict is None:
            fdict=="est"

        if lead is None:
            lead = dict()
        if tail is None:
            tail = dict()

        lead_2 = {**lead, **self.get_opt_spec_dict()}


        if fdict=="est":
            tail_2 = {
                        **self.calc_n_fev_dict(fdict="est"),
                        **self.calc_n_fev_dict(fdict="se"),
                        **tail,
                    }

            return self._parameters.calc_fit_est_df(lead=lead_2, tail=tail_2)
        elif fdict=="se":
            tail_2 = {
                        **self.calc_n_fev_dict(fdict="se"),
                        **tail,
                    }
            return self._parameters.calc_fit_se_df(lead=lead_2, tail=tail_2)
        elif fdict=="fp1":
            return self._parameters.calc_fit_fp1_df(lead=lead_2, tail=tail)
        elif fdict=="eval_train":
            return self.calc_eval_df(lead=lead, tail=tail, train_only=True)
        elif fdict=="eval":
            return self.calc_eval_df(lead=lead, tail=tail, train_only=False)
        elif fdict=="meta":
            return self.calc_meta_df(lead=lead, tail=tail)
        


    def calc_eval_df(self, lead=None, tail=None, train_only=None):
        if lead is None:
            lead = dict()

        lead_2 = {
                        **lead,
                        **self.get_opt_spec_dict(),
                        **self._parameters.res_log_L_dict["spec"],
                        **self._parameters.calc_mode_dict(),
                        **self._parameters.calc_fixed_param_dict(),
                    }
        
        return self.calc_pabm_eval(lead=lead_2, tail=tail, normalize=False, lags=5, boxpierce=True, pvalue=True, train_only=train_only)


    def calc_meta_df(self, lead=None, tail=None):
        if lead is None:
            lead = dict()
        if tail is None:
            tail = dict()

        return {
                        **lead,
                        **self.get_opt_spec_dict(),
                        **self._parameters.res_log_L_dict["spec"],
                        **self._modes,
                        **self._parameters.calc_fixed_param_dict(),
                        **self._all_fit_time,
                        **tail,
                    }
        
        

 #--#===========================
    #--------------------------
    # Fit
    #--------------------------
    #===========================




    def calc_fit_fp_dict_old(self):

        return {fp: {rbe_str: None for rbe_str in self.calc_modes_rbe_str_list_old()} for fp in ["fp0_0", "fp0", "fp1", "fp2"]}

    def set_fit_fp_dict_old(self):

        self._fit_fp_dict = self.calc_fit_fp_dict()


    def calc_fit_fp_dict(self):

        return {fp: {rbe_str: None for rbe_str in self.calc_modes_rbe_str_0_dict()} for fp in ["fp0_0", "fp0", "fp1", "fp2"]}

    def set_fit_fp_dict(self):

        self._fit_fp_dict = self.calc_fit_fp_dict()


    def set_use_fp0(self, use_fp0=None, reset=None):

        if reset is None:
            reset = False

        if reset:
            self._use_fp0 = True

        if use_fp0 is not None:
            self._use_fp0 = use_fp0

    def get_use_fp0(self):

        return self._use_fp0


    def get_opt_spec_dict(self):

        return {
                "opt_method": self._parameters._opt_method,
                "use_fp0": self.get_use_fp0(),
                }

    def all_loop_fit_old(self, rmode_s0s=None, rmode_s1s=None, rmode_s2s=None, bmodes=None, emodes=None, opt_method=None, use_fp0=None, check_opt_last=None, export_fit_df=None, export_figs=None, fig_suppress=None, reset_fit_df=None, reset_datetime_w_str=None, print_fit_time=None):

    
        self._parameters.set_opt_method(opt_method=opt_method)
        
        self.set_use_fp0(use_fp0=use_fp0, reset=False)


        self.set_datetime_w_str(set_fit_dir_w=False, reset=reset_datetime_w_str)



        if reset_fit_df is None:
            reset_fit_df = True

        if reset_fit_df:
            self.reset_fit_df_dict()
        

        self.set_modes(rmode_s0=rmode_s0s, rmode_s1=rmode_s1s, rmode_s2=rmode_s2s, bmode=bmodes, emode=emodes)
        self.set_fit_fp_dict()


        all_fit_timer = misc_fns.timer(print_timer=print_fit_time, print_tab=None, print_head="all_fit")

        for s0 in range(len(self._modes["rmode_s0"])):
            rmode_s0 = self._modes["rmode_s0"][s0]
            #rmode_s0_0 = self._modes["rmode_s0"][np.max([0, s0-1])]
            rmode_s0_0 = rmode_s0
            for s1 in range(len(self._modes["rmode_s1"])):
                if s1>0:
                    rmode_s0_0 = rmode_s0
                rmode_s1 = self._modes["rmode_s1"][s1]
                #rmode_s1_0 = self._modes["rmode_s1"][np.max([0, s1-1])]
                rmode_s1_0 = rmode_s1
                for s2 in range(len(self._modes["rmode_s2"])):
                    if s2>0:
                        rmode_s0_0 = rmode_s0
                        rmode_s1_0 = rmode_s1
                    rmode_s2 = self._modes["rmode_s2"][s2]
                    rmode = f"{rmode_s0}{rmode_s1}{rmode_s2}"
                    #rmode_s2_0 = self._modes["rmode_s2"][np.max([0, s2-1])]
                    rmode_s2_0 = rmode_s2
                    for b in range(len(self._modes["bmode"])):
                        if b>0:
                            rmode_s0_0 = rmode_s0
                            rmode_s1_0 = rmode_s1
                            rmode_s2_0 = rmode_s2
                        

                        bmode = self._modes["bmode"][b]
                        if bmode > 1:
                            bmode_0 = bmode - 1
                        else:
                            bmode_0 = bmode

                        for e in range(len(self._modes["emode"])):
                            
                            emode = self._modes["emode"][e]
                            if bmode_0 < bmode:
                                emode_0 = emode
                            else:
                                emode_0 = np.max([0, emode-1])


                            rmode_0 = f"{rmode_s0_0}{rmode_s1_0}{rmode_s2_0}"

                            self.set_mode_reset(rmode=rmode, bmode=bmode, emode=emode)

                            rbe_str = self._parameters.calc_rbe_str()
                            rbe_str_0 = pack_rbe_str(rmode=rmode_0, bmode=bmode_0, emode=emode_0)

                            

                            

                            N_s_e_bool = self._parameters.N_s_e_bool.get_value()
                            N_s_e = np.sum(N_s_e_bool)
                            N_s_a = self._parameters.N_s_a_bool.get_value()
                            N_s_a_e = np.copy(N_s_a)
                            N_s_a_e[0] = False


                            
                            if self._fit_fp_dict["fp2"][rbe_str_0] is not None:
                                self._fit_fp_dict["fp0_0"][rbe_str] = np.copy(self._fit_fp_dict["fp2"][rbe_str_0])
                                if emode!=emode_0:
                                    for param in ["eta_s"]:
                                        self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][param]] = self._parameters.__dict__[param].get_default()
                                        #self._fit_fp_dict["fp0_0"][rbe_str][np.array(self._parameters._flat_index["dict"]["range"][param])[N_s_a_e]] = 0.1
                                if bmode!=bmode_0:
                                    if bmode==2:
                                        for param in ["lambda_w_s"]:
                                            self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][param]] = self._parameters.__dict__[param].get_default()
                                    elif bmode==1:
                                        for param in ["beta", "alpha_n", "alpha_w", "eta_w"]:
                                            self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][param]] = self._parameters.__dict__[param].get_default()
                                        for param in ["alpha_0_s"]:
                                            self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][param]] = self._parameters.__dict__[param].get_default()
                                            
                                        #for param in ["delta_s"]:
                                            #self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][param]] *= np.sqrt(N_s_e) # A_n mod

                                        #self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][f"sigma_s"][:2]] = self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][f"sigma_s"][-1]] * np.sqrt(N_s_e) # A_n mod
                                        #self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][f"sigma_s"][-1]] = 0

                                            
                                if rmode_s2!=rmode_s2_0:
                                    for param in ["delta_s", "gamma_1_s", "gamma_2_s", "sigma_s"]:  # A_n mod
                                        self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][param][2]] = self._parameters.__dict__[param].get_default()[2]
                                if rmode_s1!=rmode_s1_0:
                                    for param in ["delta_s", "gamma_1_s", "gamma_2_s", "sigma_s"]:
                                        self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][param][1]] = self._parameters.__dict__[param].get_default()[1]

                                if rmode_s0!=rmode_s0_0:
                                    for param in ["delta_s", "gamma_1_s", "gamma_2_s", "sigma_s"]:
                                        self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][param][0]] = self._parameters.__dict__[param].get_default()[0]


                            self.fit_rbe_mode(rmode=rmode, bmode=bmode, emode=emode, opt_method=None, use_fp0=None, check_opt_last=check_opt_last, export_fit_df=export_fit_df, export_figs=export_figs, fig_suppress=fig_suppress)



        all_fit_timer.stop()
        self._all_fit_time = all_fit_timer.get_timer_dict()

        self.append_fit_w(export_fit_df=export_fit_df, export_fit_df_meta_only=True)

    

    def all_loop_fit(self, rmode_s0s=None, rmode_s1s=None, rmode_s2s=None, bmodes=None, emodes=None, opt_method=None, use_fp0=None, check_opt_last=None, 
                        N_k=None, N_k_trunc=None, gauss_sigma=None,
                        export_fit_df=None, export_figs=None, fig_suppress=None, reset_fit_df=None, reset_datetime_w_str=None, print_fit_time=None):

    
        self._parameters.set_opt_method(opt_method=opt_method)
        
        self.set_use_fp0(use_fp0=use_fp0, reset=False)


        self.set_datetime_w_str(set_fit_dir_w=False, reset=reset_datetime_w_str)


        if reset_fit_df is None:
            reset_fit_df = True

        if reset_fit_df:
            self.reset_fit_df_dict()
        

        self.set_modes(rmode_s0=rmode_s0s, rmode_s1=rmode_s1s, rmode_s2=rmode_s2s, bmode=bmodes, emode=emodes)
        
        self.set_fit_fp_dict_old()


        all_fit_timer = misc_fns.timer(print_timer=print_fit_time, print_tab=None, print_head="all_fit")

        
        for rbe_str in self._modes_rbe_str_0_dict:
                            
            rbe_str_0 = self._modes_rbe_str_0_dict[rbe_str]

            rmode, bmode, emode = unpack_rbe_str(rbe_str=rbe_str)
            rmode_0, bmode_0, emode_0 = unpack_rbe_str(rbe_str=rbe_str_0)

            #rmode_s = unpack_rmode(rmode=rmode)


            self.set_mode_reset(rmode=rmode, bmode=bmode, emode=emode, print_rbe_str=True)


            N_s_e_bool = self._parameters.N_s_e_bool.get_value()
            N_s_e = np.sum(N_s_e_bool)
            N_s_a = self._parameters.N_s_a_bool.get_value()
            N_s_a_e = np.copy(N_s_a)
            N_s_a_e[0] = False

            
            if self._fit_fp_dict["fp2"][rbe_str_0] is not None:
                self._fit_fp_dict["fp0_0"][rbe_str] = np.copy(self._fit_fp_dict["fp2"][rbe_str_0])
                if emode!=emode_0:
                    for param in ["eta_s"]:
                        self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][param]] = self._parameters.__dict__[param].get_default()
                        #self._fit_fp_dict["fp0_0"][rbe_str][np.array(self._parameters._flat_index["dict"]["range"][param])[N_s_a_e]] = 0.1
                if bmode!=bmode_0:
                    if bmode==2:
                        for param in ["lambda_w_s"]:
                            self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][param]] = self._parameters.__dict__[param].get_default()
                    elif bmode==1:
                        for param in ["beta", "alpha_n", "alpha_w", "eta_w"]:
                            self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][param]] = self._parameters.__dict__[param].get_default()
                        for param in ["alpha_0_s"]:
                            self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][param]] = self._parameters.__dict__[param].get_default()
                            
                        #for param in ["delta_s"]:
                        #    self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][param]] *= np.sqrt(N_s_e) # A_n mod

                        #self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][f"sigma_s"][:2]] = self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][f"sigma_s"][-1]] * np.sqrt(N_s_e)  # A_n mod
                        #self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][f"sigma_s"][-1]] = 0

                for k in range(len(rmode)):
                    if rmode[k]!=rmode_0[k]:
                        for param in ["delta_s", "gamma_1_s", "gamma_2_s", "sigma_s"]: # A_n mod
                            self._fit_fp_dict["fp0_0"][rbe_str][self._parameters._flat_index["dict"]["range"][param][k]] = self._parameters.__dict__[param].get_default()[k]



            self.fit_rbe_mode(rmode=rmode, bmode=bmode, emode=emode, opt_method=None, use_fp0=None, check_opt_last=check_opt_last, 
                                N_k=N_k, N_k_trunc=N_k_trunc, gauss_sigma=gauss_sigma,
                                export_fit_df=export_fit_df, export_figs=export_figs, fig_suppress=fig_suppress)


        all_fit_timer.stop()
        self._all_fit_time = all_fit_timer.get_timer_dict()

        self.append_fit_w(export_fit_df=export_fit_df, export_fit_df_meta_only=True)


    def append_fit_w(self, export_fit_df=None, export_fit_df_meta_only=None):

        if export_fit_df is None:
            export_fit_df = True
        if export_fit_df is None:
            export_fit_df_meta_only = True

        

        self._rw["fit"]["w"]["pabm"]["meta"] = self._rw["fit"]["w"]["pabm"]["meta"].append(self.calc_fit_df(lead=self._rw["meta"]["series"], fdict="meta"), ignore_index=True) 
        if export_fit_df:
            self._rw["fit"]["w"]["pabm"]["meta"].to_csv(path_or_buf=f"{self._fit_dir_w}fit_pabm_meta.csv", sep=",", index=False)
        for fdict in self._fit_df_dict:
            self._rw["fit"]["w"]["pabm"][fdict] = self._rw["fit"]["w"]["pabm"][fdict].append(self._fit_df_dict[fdict], ignore_index=True)
            if export_fit_df and not export_fit_df_meta_only:
                self._rw["fit"]["w"]["pabm"][fdict].to_csv(path_or_buf=f"{self._fit_dir_w}fit_pabm_{fdict}.csv", sep=",", index=False)






    def calc_modes_rbe_str_list_old(self):
        return [pack_rbe_str(rmode=f"{rs0}{rs1}{rs2}", bmode=b, emode=e) for rs0 in self._modes["rmode_s0"] for rs1 in self._modes["rmode_s1"] for rs2 in self._modes["rmode_s2"] for b in self._modes["bmode"] for e in self._modes["emode"]]


    def calc_modes_rbe_str_0_dict(self):

        modes_rbe_str_0_dict = dict()

        for s0 in range(len(self._modes["rmode_s0"])):
            rmode_s0 = self._modes["rmode_s0"][s0]
            rmode_s0_0 = rmode_s0
            for s1 in range(len(self._modes["rmode_s1"])):
                if s1>0:
                    rmode_s0_0 = rmode_s0
                rmode_s1 = self._modes["rmode_s1"][s1]
                rmode_s1_0 = rmode_s1
                for s2 in range(len(self._modes["rmode_s2"])):
                    if s2>0:
                        rmode_s0_0 = rmode_s0
                        rmode_s1_0 = rmode_s1
                    rmode_s2 = self._modes["rmode_s2"][s2]
                    rmode = f"{rmode_s0}{rmode_s1}{rmode_s2}"
                    rmode_s2_0 = rmode_s2

                    z_bool = np.array([rmode[s] == "z" for s in range(len(rmode))])

                    if not np.all(z_bool):
                        for b in range(len(self._modes["bmode"])):
                            if (b==0) or ((np.size(z_bool) - np.sum(z_bool)) > 1):

                                if b>0:
                                    rmode_s0_0 = rmode_s0
                                    rmode_s1_0 = rmode_s1
                                    rmode_s2_0 = rmode_s2
                                

                                bmode = self._modes["bmode"][b]
                                if bmode > 1:
                                    bmode_0 = bmode - 1
                                else:
                                    bmode_0 = bmode

                                for e in range(len(self._modes["emode"])):
                                    
                                    emode = self._modes["emode"][e]
                                    if bmode_0 < bmode:
                                        emode_0 = emode
                                    else:
                                        emode_0 = np.max([0, emode-1])


                                    rmode_0 = f"{rmode_s0_0}{rmode_s1_0}{rmode_s2_0}"

                                    rbe_str = pack_rbe_str(rmode=rmode, bmode=bmode, emode=emode)
                                    rbe_str_0 = pack_rbe_str(rmode=rmode_0, bmode=bmode_0, emode=emode_0)


                                    modes_rbe_str_0_dict[rbe_str] = rbe_str_0

        return modes_rbe_str_0_dict
    
    def set_modes_rbe_str_0_dict(self):

        self._modes_rbe_str_0_dict = self.calc_modes_rbe_str_0_dict()


    def reset_modes(self, init=None):

        if init is None:
            init = False
        
        mode_fields = ["rmode_s0", "rmode_s1", "rmode_s2", "bmode", "emode"]

        if init:
            self._modes = {field: None for field in mode_fields}

        for field in mode_fields:
            self.set_modes_field(field=field, value=None, reset=True)
            

    def set_modes_field(self, field=None, value=None, reset=None):

        if field is None:
            field=="emode"


        if reset is None:
            reset = False
        if reset:
            if field=="rmode_s0":
                self._modes[field] = misc_fns.make_iterable_array("l")
            elif field=="rmode_s1":
                self._modes[field] = misc_fns.make_iterable_array(["z", "l"])
            elif field=="rmode_s2":
                self._modes[field] = misc_fns.make_iterable_array(["z", "l"])
            elif field=="bmode":
                self._modes[field] = misc_fns.make_iterable_array(np.arange(3))
            elif field=="emode":
                #self._modes[field] = misc_fns.make_iterable_array(np.arange(2))
                self._modes[field] = misc_fns.make_iterable_array(np.arange(1))

        if value is not None:
            self._modes[field] = misc_fns.make_iterable_array(value)


    def set_modes_field_alt(self, field=None, value=None, reset=None):

        if field is None:
            field=="emode"


        if reset is None:
            reset = False


        if field=="rmode_s0":
            if reset:
                self._modes[field] = misc_fns.make_iterable_array("l")
            if value is not None:
                self._modes[field] = misc_fns.make_iterable_array(value)

        if field=="rmode_s1":
            if reset:
                self._modes[field] = misc_fns.make_iterable_array(["z", "l"])
            if value is not None:
                self._modes[field] = misc_fns.make_iterable_array(value)

        if field=="rmode_s2":
            if reset:
                self._modes[field] = misc_fns.make_iterable_array(["z", "l"])
            if value is not None:
                self._modes[field] = misc_fns.make_iterable_array(value)

        if field=="bmode":
            if reset:
                self._modes[field] = misc_fns.make_iterable_array(np.arange(3))
            if value is not None:
                self._modes[field] = misc_fns.make_iterable_array(value)

        if field=="emode":
            if reset:
                #self._modes[field] = misc_fns.make_iterable_array(np.arange(2))
                self._modes[field] = misc_fns.make_iterable_array(np.arange(1))
            if value is not None:
                self._modes[field] = misc_fns.make_iterable_array(value)



    def set_modes_alt(self, rmode_s0=None, rmode_s1=None, rmode_s2=None, bmode=None, emode=None, reset=None, set_rbe_0=None, set_defaults=None):

        if reset is None:
            reset = False

        if reset:
            for field in self._modes:
                self.set_modes_field(field=field, value=None, reset=True)

        
        self.set_modes_field(field="rmode_s0", value=rmode_s0, reset=False)
        self.set_modes_field(field="rmode_s1", value=rmode_s1, reset=False)
        self.set_modes_field(field="rmode_s2", value=rmode_s2, reset=False)
        self.set_modes_field(field="bmode", value=bmode, reset=False)
        self.set_modes_field(field="emode", value=emode, reset=False)

        if set_rbe_0 is None:
            set_rbe_0 = True

        if set_rbe_0:
            rmode = f"{self._modes['rmode_s0'][0]}{self._modes['rmode_s1'][0]}{self._modes['rmode_s2'][0]}"
            bmode = self._modes['bmode'][0]
            emode = self._modes['emode'][0]

            self.set_mode_reset(rmode=rmode, bmode=bmode, emode=emode, set_defaults=set_defaults)


    def set_modes(self, rmode_s0=None, rmode_s1=None, rmode_s2=None, bmode=None, emode=None, reset=None, set_rbe_0=None, set_defaults=None):

        if reset is None:
            reset = False

        
        self.set_modes_field(field="rmode_s0", value=rmode_s0, reset=reset)
        self.set_modes_field(field="rmode_s1", value=rmode_s1, reset=reset)
        self.set_modes_field(field="rmode_s2", value=rmode_s2, reset=reset)
        self.set_modes_field(field="bmode", value=bmode, reset=reset)
        self.set_modes_field(field="emode", value=emode, reset=reset)

        self.set_modes_rbe_str_0_dict()

        if set_rbe_0 is None:
            set_rbe_0 = True

        if set_rbe_0:
            rmode = f"{self._modes['rmode_s0'][0]}{self._modes['rmode_s1'][0]}{self._modes['rmode_s2'][0]}"
            bmode = self._modes['bmode'][0]
            emode = self._modes['emode'][0]

            self.set_mode_reset(rmode=rmode, bmode=bmode, emode=emode, set_defaults=set_defaults)




    def set_mode_reset(self, rbe_str=None, rmode=None, bmode=None, emode=None, set_defaults=None, print_rbe_str=None):

        if set_defaults is None:
            set_defaults = True
        if print_rbe_str is None:
            print_rbe_str = False


        self._parameters.set_mode(rbe_str=rbe_str, rmode=rmode, bmode=bmode, emode=emode, reset=False)

        if print_rbe_str:
            print(self._parameters.calc_rbe_str())

        if set_defaults:
            self.set_parameters_defaults()
            self._parameters.pull_flat_parameters()
            self._parameters.set_flat_parameters(self._parameters.flatten_defaults())

        self.sim_p_t_reset(reset_t_series=False)




    def reset_fit_df_dict(self):

        self._fit_df_dict = empty_fit_df_dict()





    def write_fit_dict(self, export_fit_df=None, reset=None):

        if export_fit_df is None:
            export_fit_df = True

        if reset is None:
            reset = False
        
        if reset:
            self.reset_fit_df_dict()


        for fdict in self._fit_df_dict:
            self._fit_df_dict[fdict] = self._fit_df_dict[fdict].append(self.calc_fit_df(lead=self._rw["meta"]["series"], fdict=fdict), ignore_index=True) 
            if export_fit_df:
                self._fit_df_dict[fdict].to_csv(path_or_buf=f"{self._fit_dir_w}fit_pabm_{fdict}.csv", sep=",", index=False)


    def fit_rbe_mode(self, rbe_str=None, rmode=None, bmode=None, emode=None, opt_method=None, use_fp0=None, check_opt_last=None, 
                     N_k=None, N_k_trunc=None, gauss_sigma=None,
                     export_fit_df=None, export_figs=None, fig_suppress=None):

        self._parameters.set_opt_method(opt_method=opt_method)
        self.set_use_fp0(use_fp0=use_fp0, reset=False)

        if export_fit_df is None:
            export_fit_df = True
        if export_figs is None:
            export_figs = True

        set_fit_dir_w = bool(export_fit_df*export_figs)


        if fig_suppress is None:
            fig_suppress = True


        #self.sim_p_t_reset()
        #self.load_p_t_r_t(p_t=p_t, r_t=r_t, p_0=p_0, run=False)



        self.set_mode_reset(rbe_str=rbe_str, rmode=rmode, bmode=bmode, emode=emode, set_defaults=True)
        self.sim_p_t_nb(sim=False)

        rbe_str = self._parameters.calc_rbe_str()
        if self.get_use_fp0():
            fp0 = self._fit_fp_dict["fp0_0"][rbe_str]
        else:
            fp0 = None
            

        self.set_parameters_estimate(set_se=True, check_opt_last=check_opt_last, flat_parameters_0=fp0, N_k=N_k, N_k_trunc=N_k_trunc, gauss_sigma=gauss_sigma)

        self._fit_fp_dict["fp0"][rbe_str] = np.copy(self._parameters.res_fp0)
        self._fit_fp_dict["fp1"][rbe_str] = np.copy(self._parameters.res_fp1)
        self._fit_fp_dict["fp2"][rbe_str] = np.copy(self._parameters.res_fp2)
        

        
        if set_fit_dir_w:
            self.set_fit_dir_w()


        self.write_fit_dict(export_fit_df=export_fit_df, reset=False)
        
        


        if export_figs:
            self.calc_fig_process(export_fig=True, force_dir=self._fit_dir_w, fig_suppress=fig_suppress)
            self.set_fit_plot_track(se=True)
            for suff in ["est", "se"]:
                self.calc_fig_track_i(export_fig=True, set_fit_plot_track=False, suff=suff, force_dir=self._fit_dir_w, fig_suppress=fig_suppress)
                self.calc_fig_track_k(export_fig=True, set_fit_plot_track=False, suff=suff, force_dir=self._fit_dir_w, fig_suppress=fig_suppress)
                self.calc_fig_track_k_opt_last(export_fig=True, set_fit_plot_track=False, suff=suff, force_dir=self._fit_dir_w, fig_suppress=fig_suppress)
                self.calc_fig_track_opt(export_fig=True, set_fit_plot_track=False, suff=suff, force_dir=self._fit_dir_w, fig_suppress=fig_suppress)
                self.calc_fig_track_opt_3d(export_fig=True, set_fit_plot_track=False, suff=suff, force_dir=self._fit_dir_w, fig_suppress=fig_suppress)













 #--#===========================
    #--------------------------
    # Figures
    #--------------------------
    #===========================


    def calc_fig_process(self, export_fig=None, force_dir=None, fig_suppress=None, cols=None):

        if export_fig is None:
            export_fig = False

        if fig_suppress is None:
            fig_suppress = False

        if cols is None:
            cols = plot_fns.calc_cols()

        ts = np.arange(self.get_T())




        fig_pruna = plot_fns.rc_plot_setup(name=f"pabm_{self._parameters.calc_rbe_str()}_full", nrows=6, allborderlabels=False)

        s_cols = [cols[3], cols[2], cols[9], cols[7]]
        s_super = ["f", "c", "v", "n"]

        N_s_a_bool = self._parameters.N_s_a_bool.get_value()
        N_s_a = np.sum(N_s_a_bool)

        N_s_a_show_bool = np.copy(N_s_a_bool)
        #N_s_a_show_bool[3] = False # A_n mod
        N_s_a_show = np.sum(N_s_a_show_bool)
        plotted = np.where(N_s_a_show_bool)[0]


        sim = False
        N_p = 3 + int(sim)

        fig_pruna["ax"][0][0].plot(self.get_t_series("p"), lw=0.5, color='black')
        fig_pruna["ax"][0][0].plot(self.get_t_series("p_F"), lw=0.5, color=cols[self._log_L_dict["opt"]["k_opt"]%len(cols)])
        fig_pruna["ax"][0][0].plot(self.calc_p_F_0_lin(), lw=0.5, color="magenta", ls=(0, (8, 8)))
        if sim:
            fig_pruna["ax"][0][0].plot(self.get_t_series("p_F_sim"), lw=0.5, color=cols[3])

        for k in range(20):
            fig_pruna["ax"][0][0].plot(self._log_L_dict["k"]["p_F_k"] [k,:], lw=0.25)


        p_leg = np.array([f"$p_{{t}}$", f"$p_{{t}}^{{F,K={self._log_L_dict['opt']['k_opt']}}}$", f"$p_{{t}}^{{F,\\mathrm{{lin}}}}$", f"$p_{{t}}^{{F,S}}$"])

        fig_pruna["ax"][0][0].legend([fig_pruna["ax"][0][0].get_children()[k] for k in range(N_p)],
                                            p_leg[:N_p], 
                                            fontsize=6.5, loc="upper left", ncol=N_p)



        fig_pruna["ax"][0][0].set_ylabel(f"$p_{{t}}$")



        a_u = 0.025
        u_q = np.quantile(self.get_t_series("r"), [a_u, 1-a_u])
        fig_pruna["ax"][1][0].plot(np.repeat(ts,4), np.concatenate(misc_fns.stack(self.get_t_series("r"))), lw=0.3, color=cols[7])
        fig_pruna["ax"][1][0].plot(ts, self._parameters.psi.get_value()*self.calc_mu_t_D(), color=cols[1], lw=0.5)

        for k in range(2):
            fig_pruna["ax"][1][0].plot(ts, ((-1)**k)*self._parameters.psi.get_value()*self.calc_sigma_t_D(), color=cols[0], lw=0.5)
            fig_pruna["ax"][1][0].axhline(u_q[k], color='black', lw=1, ls=":")

        fig_pruna["ax"][1][0].legend([fig_pruna["ax"][1][0].get_children()[k] for k in [0,1,2,3]],
                                        [f"$r_{{t}}$", f"$\\psi \\mu_{{t}}^{{D}}$", f"$\\pm \\psi \\sigma_{{t}}^{{D}}$", f"$u^{{\\leftrightharpoons}}{{\\left(a_{{u}}={a_u}\\right)}}$"], 
                                        fontsize=6.5, loc="upper left", ncol=4)
        fig_pruna["ax"][1][0].set_ylabel(f"$r_{{t}}$")



        for k in plotted:
            fig_pruna["ax"][2][0].plot(self.get_t_s_series("n")[k,:] * self._parameters.delta_s.get_value()[k] * self.get_t_s_series("rho")[k,:], lw=0.3, color=s_cols[k])


        fig_pruna["ax"][2][0].set_ylabel(f"$\\mathbf{{\\mu}}_{{t}} = \\mathbf{{n}}_{{t}} \\circ \\mathbf{{\\delta}} \circ \\mathbf{{\\rho}}_{{t}}$")

        fig_pruna["ax"][2][0].legend([fig_pruna["ax"][2][0].get_children()[k] for k in range(N_s_a_show)],
                                    [f"$\\mu_{{t}}^{{\\mathrm{{{s_super[s]}}}}}$" for s in plotted], 
                                    fontsize=6.5, loc="upper left", ncol=N_s_a_show)

        for k in plotted:
            fig_pruna["ax"][3][0].plot(self.get_t_s_series("rho")[k,:], lw=0.3, color=s_cols[k])

        fig_pruna["ax"][3][0].set_ylabel(f"$\\mathbf{{\\rho}}_{{t}}$")

        fig_pruna["ax"][3][0].legend([fig_pruna["ax"][3][0].get_children()[k] for k in range(N_s_a_show)],
                                    [f"$\\rho_{{t}}^{{\\mathrm{{{s_super[s]}}}}}$" for s in plotted], 
                                    fontsize=6.5, loc="upper left", ncol=N_s_a_show)


        for k in plotted:
            fig_pruna["ax"][4][0].plot(self.get_t_s_series("n")[k,:], lw=0.3, color=s_cols[k])

        fig_pruna["ax"][4][0].set_ylabel(f"$\\mathbf{{n}}_{{t}}$")

        fig_pruna["ax"][4][0].legend([fig_pruna["ax"][4][0].get_children()[k] for k in range(N_s_a_show)],
                                    [f"$n_{{t}}^{{\\mathrm{{{s_super[s]}}}}}$" for s in plotted], 
                                    fontsize=6.5, loc="upper left", ncol=N_s_a_show)

        #fig_pruna["ax"][4][0].set_ylim(np.array([0,1]) + 0.01*np.array([-1,1])) # A_n mod
        fig_pruna["ax"][4][0].set_ylim(np.array([0,self._parameters.calc_A_n()]) + 0.01*np.array([-1,1])) # A_n mod


        for k in plotted:
            fig_pruna["ax"][5][0].plot(self.get_t_s_series("w")[k,:], lw=0.3, color=s_cols[k])

        fig_pruna["ax"][5][0].set_ylabel(f"$\\mathbf{{w}}_{{t}}$")

        fig_pruna["ax"][5][0].legend([fig_pruna["ax"][4][0].get_children()[k] for k in range(N_s_a_show)],
                                    [f"$w_{{t}}^{{\\mathrm{{{s_super[s]}}}}}$" for s in plotted], 
                                    fontsize=6.5, loc="upper left", ncol=N_s_a_show)











        for i in range(len(fig_pruna["ax"])):
            fig_pruna["ax"][i][0].set_xlim([0,self.get_T()])



        #fig_pruna["ax"][4][0].set_ylabel(f"$r_{{t}}/{{\\left(\\psi \\sigma_{{t}}^{{D}}\\right)}}$")


        fig_pruna["ax"][-1][0].set_xlabel(f"$t$")

        fig_pruna["fig"].set_size_inches(7,8.5)
        fig_pruna["fig"].tight_layout()

        if export_fig:
            plot_fns.fig_export(fig_pruna, force_dir=force_dir, height=8.5, formats=['pdf'])

        if fig_suppress:
            fig_pruna["fig"].clear()




    #def calc_fit_plot_vals(self, se=None):

    def calc_fit_plot_track(self, se=None):


        if se is None:
            se = False


        N_fevs = dict()
        N_fevs["est"] = self._parameters.res["nfev"]
        if se:
            N_fevs["se"] = self._log_L_dict_track["track_i"]

        fit_plot_track = {suf: dict() for suf in N_fevs}
        for suff in fit_plot_track:
            N_fev = N_fevs[suff]
            fit_plot_track[suff]["N_fev"] = N_fev

            fit_plot_track[suff]["log_L_opt"] = np.array([self._log_L_dict_track["track_log_L_k"][i,self._log_L_dict_track["track_k_opt"][i]] for i in range(N_fev)])
            fit_plot_track[suff]["step_norm_opt"] = np.array([self._log_L_dict_track["track_step_norm_k"][i,self._log_L_dict_track["track_k_opt"][i]] for i in range(N_fev)])

            fit_plot_track[suff]["log_L_end"] = np.array([self._log_L_dict_track["track_log_L_k"][i,self._log_L_dict_track["track_k_end"][i]] for i in range(N_fev)])
            fit_plot_track[suff]["step_norm_end"] = np.array([self._log_L_dict_track["track_step_norm_k"][i,self._log_L_dict_track["track_k_end"][i]] for i in range(N_fev)])

            fit_plot_track[suff]["where_opt_last"] = np.where(self._log_L_dict_track["track_use_opt_last"][:N_fev])[0]
            fit_plot_track[suff]["where_not_opt_last"] = np.where(1-self._log_L_dict_track["track_use_opt_last"][:N_fev])[0]
            fit_plot_track[suff]["where_fail"] = np.where(1 - self._log_L_dict_track["track_succeed"][:N_fev])[0]
            fit_plot_track[suff]["where_not_fail"] = np.where(self._log_L_dict_track["track_succeed"][:N_fev])[0]

        
        return fit_plot_track

    def set_fit_plot_track(self, se=None):

        self._fit_plot_track = self.calc_fit_plot_track(se=se)


    def calc_fig_track_i(self, export_fig=None, set_fit_plot_track=None, suff=None, force_dir=None, fig_suppress=None, cols=None):


        if export_fig is None:
            export_fig = False


        if set_fit_plot_track is None:
            set_fit_plot_track = True
        if suff is None:
            suff = "est"
        if fig_suppress is None:
            fig_suppress = False

        if cols is None:
            cols = plot_fns.calc_cols()

        if set_fit_plot_track:
            self.set_fit_plot_track(se=True)


        N_fev = self._fit_plot_track[suff]["N_fev"]
        log_L_opt = self._fit_plot_track[suff]["log_L_opt"]
        log_L_end = self._fit_plot_track[suff]["log_L_end"]

        step_norm_opt = self._fit_plot_track[suff]["step_norm_opt"]
        step_norm_end = self._fit_plot_track[suff]["step_norm_end"]

        where_opt_last = self._fit_plot_track[suff]["where_opt_last"]
        where_not_opt_last = self._fit_plot_track[suff]["where_not_opt_last"]

        where_fail = self._fit_plot_track[suff]["where_fail"]
        where_not_fail = self._fit_plot_track[suff]["where_not_fail"]
        
        

        fig_track = plot_fns.fig_column_setup(name=f"pabm_fit_{self._parameters.calc_rbe_str()}_track_i_{suff}",  hratio=np.array([3,3,3,3,1,1]), labeltopbottom=True)


        fig_track["ax"][0][0].plot(np.cumsum(self._log_L_dict_track["track_k_opt"][:N_fev]), lw=0.5, color=cols[0])
        fig_track["ax"][0][0].plot(np.cumsum(self._log_L_dict_track["track_k_end"][:N_fev]), lw=0.5, color=cols[1])
        fig_track["ax"][0][0].set_ylabel("$\\int{{\\hat{{k}}}}$")

        fig_track["ax"][0][0].legend([fig_track["ax"][0][0].get_children()[k] for k in range(2)],
                                            [f"$\\hat{{k}}_{{\\mathrm{{opt}}}}$", f"$\\hat{{k}}_{{\\mathrm{{end}}}}$"],
                                            loc="upper left", ncol=2)

        fig_track["ax"][1][0].plot(self._log_L_dict_track["track_k_opt"][:N_fev], lw=0.5, color=cols[0])
        fig_track["ax"][1][0].plot(self._log_L_dict_track["track_k_end"][:N_fev], lw=0.5, color=cols[1])
        fig_track["ax"][1][0].set_ylabel("$\\hat{{k}}$")



        fig_track["ax"][2][0].plot(log_L_opt, lw=0.5, color=cols[0])
        fig_track["ax"][2][0].plot(log_L_end, lw=0.5, color=cols[1])
        fig_track["ax"][2][0].set_ylabel("$\\ell{{\\left(\\mathbf{{\\theta}}\\right)}}_{{\\hat{{k}}}}$")


        fig_track["ax"][3][0].axhline(self._log_L_dict["spec"]["step_norm_cutoff"] , color="black", lw=0.3, ls="--")

        fig_track["ax"][3][0].plot(step_norm_opt, lw=0.5, color=cols[0])
        fig_track["ax"][3][0].plot(step_norm_end, lw=0.5, color=cols[1])
        fig_track["ax"][3][0].set_ylabel("$\\left|\\Delta{{p^{{F}}}}\\right|_{{\\hat{{k}}}}/\\sqrt{{T}}$")
        fig_track["ax"][3][0].set_yscale('log')

        fig_track["ax"][3][0].legend([fig_track["ax"][3][0].get_children()[0]],
                                    [f"$\\left|\\Delta{{p^{{F}}}}\\right|_{{\mathrm{{cut}}}}/\\sqrt{{T}}$"],
                                    #loc="upper right"
                                    )


        plot_opt_last = where_opt_last[where_opt_last<N_fev]
        plot_not_opt_last = where_not_opt_last[where_not_opt_last<N_fev]

        fig_track["ax"][4][0].scatter(plot_opt_last, np.repeat(1, plot_opt_last.size), color=cols[1], s=5, lw=0.2, marker="|")
        fig_track["ax"][4][0].scatter(plot_not_opt_last, np.repeat(0, plot_not_opt_last.size), color=cols[0], s=5, lw=0.2, marker="|")
        fig_track["ax"][4][0].set_ylim([-0.5,1.5])
        fig_track["ax"][4][0].set_ylabel("optlast")

        fig_track["ax"][4][0].tick_params(
                                            labelleft=False,
                                            left=False,
                                            right=False
                                        )


        plot_fail = where_fail[where_fail<N_fev]
        plot_not_fail = where_not_fail[where_not_fail<N_fev]

        fig_track["ax"][5][0].scatter(plot_fail, np.repeat(1, plot_fail.size), color=cols[1], s=5, lw=0.2, marker="|")
        fig_track["ax"][5][0].scatter(plot_not_fail, np.repeat(0, plot_not_fail.size), color=cols[0], s=5, lw=0.2, marker="|")
        fig_track["ax"][5][0].set_ylim([-0.5,1.5])
        fig_track["ax"][5][0].set_ylabel("fail")

        fig_track["ax"][5][0].tick_params(
                                            labelleft=False,
                                            left=False,
                                            right=False
                                        )


        for i in range(len(fig_track["ax"])):
            if suff=="se":
                fig_track["ax"][i][0].axvline(self._fit_plot_track["est"]["N_fev"], color="black", lw=0.3)
            fig_track["ax"][i][0].set_xlim([0,N_fev])
            




        fig_track["ax"][0][0].tick_params(
                                            labeltop=False,
                                        )



        fig_track["ax"][-1][0].set_xlabel(f"$i$")

        fig_track["fig"].set_size_inches(7,5)
        fig_track["fig"].tight_layout()


        if export_fig:
            plot_fns.fig_export(fig_track, height=5, force_dir=force_dir, formats=['pdf'])

        if fig_suppress:
            fig_track["fig"].clear()

        
    def calc_fig_track_k(self, export_fig=None, set_fit_plot_track=None, suff=None, force_dir=None, fig_suppress=None):


        if export_fig is None:
            export_fig = False


        if set_fit_plot_track is None:
            set_fit_plot_track = True
        if suff is None:
            suff = "est"
        if fig_suppress is None:
            fig_suppress = False

        if set_fit_plot_track:
            self.set_fit_plot_track(se=True)


        N_fev = self._fit_plot_track[suff]["N_fev"]
        log_L_opt = self._fit_plot_track[suff]["log_L_opt"]

        step_norm_opt = self._fit_plot_track[suff]["step_norm_opt"]

        
        grads = np.linspace(0,1,N_fev)
        cmap = matplotlib.cm.get_cmap('viridis')
        grad_cols = cmap(grads)

        fig_track = plot_fns.rc_plot_setup(name=f"pabm_fit_{self._parameters.calc_rbe_str()}_track_k_{suff}", nrows=2, allborderlabels=False)

        grads = np.linspace(0,1,N_fev)
        cmap = matplotlib.cm.get_cmap('viridis')
        grad_cols = cmap(grads)


        fig_track["ax"][1][0].axhline(self._log_L_dict["spec"]["step_norm_cutoff"] , color="black", lw=0.3, ls="--")

        fig_track["ax"][0][0].scatter(self._log_L_dict_track["track_k_opt"][:N_fev], log_L_opt[:N_fev], s=5, marker="D", lw=0, c=grad_cols)
        fig_track["ax"][1][0].scatter(self._log_L_dict_track["track_k_opt"][:N_fev], step_norm_opt[:N_fev], s=5, marker="D", lw=0, c=grad_cols)


        for i in range(N_fev):

            fig_track["ax"][0][0].plot(self._log_L_dict_track["track_log_L_k"][i,:], lw=0.2, color=grad_cols[i])
            fig_track["ax"][1][0].plot(self._log_L_dict_track["track_step_norm_k"][i,:], lw=0.2, color=grad_cols[i])



        fig_track["ax"][1][0].legend([fig_track["ax"][1][0].get_children()[0]],
                                    [f"$\\left|\\Delta{{p^{{F}}}}\\right|_{{\mathrm{{cut}}}}/\\sqrt{{T}}$"],
                                    loc="upper right")



        fig_track["ax"][0][0].set_ylabel("$\\ell{{\\left(\\mathbf{{\\theta}}\\right)}}_{{k}}$")

        fig_track["ax"][1][0].set_ylabel("$\\left|\\Delta{{p^{{F}}}}\\right|_{{k}}/\\sqrt{{T}}$")


        fig_track["ax"][1][0].set_yscale('log')



        for i in range(len(fig_track["ax"])):
            fig_track["ax"][i][0].set_xlim([0,self._log_L_dict["spec"]["N_k"]])
            fig_track["ax"][i][0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))


        fig_track["ax"][-1][0].set_xlabel(f"$k$")

        fig_track["fig"].set_size_inches(7,3)
        fig_track["fig"].tight_layout()


        if export_fig:
            plot_fns.fig_export(fig_track, height=3, force_dir=force_dir, formats=['pdf'])

        if fig_suppress:
            fig_track["fig"].clear()


    def calc_fig_track_k_opt_last(self, export_fig=None, set_fit_plot_track=None, suff=None, force_dir=None, fig_suppress=None):


        if export_fig is None:
            export_fig = False


        if set_fit_plot_track is None:
            set_fit_plot_track = True
        if suff is None:
            suff = "est"
        if fig_suppress is None:
            fig_suppress = False

        if set_fit_plot_track:
            self.set_fit_plot_track(se=True)
            

        N_fev = self._fit_plot_track[suff]["N_fev"]
        log_L_opt = self._fit_plot_track[suff]["log_L_opt"]

        step_norm_opt = self._fit_plot_track[suff]["step_norm_opt"]

        where_opt_last = self._fit_plot_track[suff]["where_opt_last"]
        where_not_opt_last = self._fit_plot_track[suff]["where_not_opt_last"]


        
        grads = np.linspace(0,1,N_fev)
        cmap = matplotlib.cm.get_cmap('viridis')
        grad_cols = cmap(grads)


        fig_track = plot_fns.rc_plot_setup(name=f"pabm_fit_{self._parameters.calc_rbe_str()}_track_k_opt_last_{suff}", nrows=6, allborderlabels=False)

        

        plot_opt_last = where_opt_last[where_opt_last<N_fev]
        plot_not_opt_last = where_not_opt_last[where_not_opt_last<N_fev]

        cuts = [np.arange(N_fev), plot_opt_last, plot_not_opt_last]
        cut_labels = ["all", "opt last", "lin"]


        N_cut = len(cuts)

        for j in range(N_cut):



            fig_track["ax"][1*N_cut+j][0].axhline(self._log_L_dict["spec"]["step_norm_cutoff"] , color="black", lw=0.3, ls="--")

            fig_track["ax"][0*N_cut+j][0].scatter(self._log_L_dict_track["track_k_opt"][cuts[j]], log_L_opt[cuts[j]], s=5, marker="D", lw=0, c=grad_cols[cuts[j]])
            fig_track["ax"][1*N_cut+j][0].scatter(self._log_L_dict_track["track_k_opt"][cuts[j]], step_norm_opt[cuts[j]], s=5, marker="D", lw=0, c=grad_cols[cuts[j]])


            for i in cuts[j]:

                fig_track["ax"][0*N_cut+j][0].plot(self._log_L_dict_track["track_log_L_k"][i,:], lw=0.2, color=grad_cols[i])
                fig_track["ax"][1*N_cut+j][0].plot(self._log_L_dict_track["track_step_norm_k"][i,:], lw=0.2, color=grad_cols[i])



            fig_track["ax"][1*N_cut+j][0].legend([fig_track["ax"][1*N_cut+j][0].get_children()[0]],
                                        [f"$\\left|\\Delta{{p^{{F}}}}\\right|_{{\mathrm{{cut}}}}/\\sqrt{{T}}$"],
                                        loc="upper right")



            fig_track["ax"][0*N_cut+j][0].set_ylabel(f"$\\ell{{\\left(\\mathbf{{\\theta}}\\right)}}_{{k}}^{{\\mathrm{{{cut_labels[j]}}}}}$")

            fig_track["ax"][1*N_cut+j][0].set_ylabel(f"$\\left|\\Delta{{p^{{F}}}}\\right|_{{k}}^{{\\mathrm{{{cut_labels[j]}}}}}/\\sqrt{{T}}$")


            fig_track["ax"][1*N_cut+j][0].set_yscale('log')



        for i in range(len(fig_track["ax"])):
            fig_track["ax"][i][0].set_xlim([0,self._log_L_dict["spec"]["N_k"]])
            fig_track["ax"][i][0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))


        fig_track["ax"][-1][0].set_xlabel(f"$k$")

        fig_track["fig"].set_size_inches(7,8.5)
        fig_track["fig"].tight_layout()

        if export_fig:
            plot_fns.fig_export(fig_track, height=8.5, force_dir=force_dir, formats=['pdf'])

        if fig_suppress:
            fig_track["fig"].clear()


    def calc_fig_track_opt(self, export_fig=None, set_fit_plot_track=None, suff=None, force_dir=None, fig_suppress=None):

        if export_fig is None:
            export_fig = False


        if set_fit_plot_track is None:
            set_fit_plot_track = True
        if suff is None:
            suff = "est"
        if fig_suppress is None:
            fig_suppress = False

        if set_fit_plot_track:
            self.set_fit_plot_track(se=True)


        N_fev = self._fit_plot_track[suff]["N_fev"]
        log_L_opt = self._fit_plot_track[suff]["log_L_opt"]

        step_norm_opt = self._fit_plot_track[suff]["step_norm_opt"]


        grads = np.linspace(0,1,N_fev)
        cmap = matplotlib.cm.get_cmap('viridis')
        grad_cols = cmap(grads)


        fig_track_opt = plot_fns.rc_plot_setup(name=f"pabm_fit_{self._parameters.calc_rbe_str()}_track_opt_{suff}", allborderlabels=False)


        fig_track_opt["ax"][0][0].scatter(log_L_opt[:N_fev], step_norm_opt[:N_fev], s=5, marker="D", lw=0, c=grad_cols[:N_fev])


        fig_track_opt["ax"][0][0].set_xlabel("$\\ell{{\\left(\\mathbf{{\\theta}}\\right)}}_{{k}}$")
        fig_track_opt["ax"][0][0].set_ylabel("$\\left|\\Delta{{p^{{F}}}}\\right|_{{k}}/\\sqrt{{T}}$")

        fig_track_opt["ax"][0][0].set_yscale('log')



        fig_track_opt["fig"].set_size_inches(3.4,3)
        fig_track_opt["fig"].tight_layout()

        if export_fig:
            plot_fns.fig_export(fig_track_opt, height=3, width=3.4, force_dir=force_dir, formats=['pdf'])

        if fig_suppress:
            fig_track_opt["fig"].clear()


    def calc_fig_track_opt_3d(self, export_fig=None, set_fit_plot_track=None, suff=None, force_dir=None, fig_suppress=None):

        if export_fig is None:
            export_fig = False


        

        if set_fit_plot_track is None:
            set_fit_plot_track = True
        if suff is None:
            suff = "est"
        if fig_suppress is None:
            fig_suppress = False

        if set_fit_plot_track:
            self.set_fit_plot_track(se=True)


        N_fev = self._fit_plot_track[suff]["N_fev"]
        log_L_opt = self._fit_plot_track[suff]["log_L_opt"]

        step_norm_opt = self._fit_plot_track[suff]["step_norm_opt"]




        grads = np.linspace(0,1,N_fev)
        cmap = matplotlib.cm.get_cmap('viridis')
        grad_cols = cmap(grads)

        fig_track_opt_3d = {
                                "name": f"pabm_fit_{self._parameters.calc_rbe_str()}_track_opt_3d_{suff}",
                                "fig": None,
                                "ax": [[None]],
                            }


        fig_track_opt_3d["fig"], fig_track_opt_3d["ax"][0][0] = plt.subplots(subplot_kw={"projection": "3d"})
        fig_track_opt_3d["ax"][0][0].scatter(log_L_opt[:N_fev], step_norm_opt[:N_fev], np.arange(N_fev), s=5, marker="D", lw=0, c=grad_cols[:N_fev])

        fig_track_opt_3d["ax"][0][0].set_xlabel("$\\ell{{\\left(\\mathbf{{\\theta}}\\right)}}_{{k}}$")
        fig_track_opt_3d["ax"][0][0].set_ylabel("$\\left|\\Delta{{p^{{F}}}}\\right|_{{k}}/\\sqrt{{T}}$")
        fig_track_opt_3d["ax"][0][0].set_zlabel("k")

        fig_track_opt_3d["ax"][0][0].set_yscale('log')


        fig_track_opt_3d["fig"].set_size_inches(3.4,3)
        fig_track_opt_3d["fig"].tight_layout()

        if export_fig:
            plot_fns.fig_export(fig_track_opt_3d, height=3, width=3.4, force_dir=force_dir, formats=['pdf'])

        if fig_suppress:
            fig_track_opt_3d["fig"].clear()
        

 #--#===========================
    #--------------------------
    # Residual analysis
    #--------------------------
    #===========================

    def calc_pabm_eval(self, lead=None, tail=None, normalize=None, lags=None, boxpierce=None, head=None, pvalue=None, train_only=None):



        if lead is None:
            lead = dict()
        if tail is None:
            tail = dict()

        

        eval_atf = self.calc_pabm_eval_atf(normalize=normalize, lags=lags, boxpierce=boxpierce, head=head, pvalue=pvalue, train_only=train_only)

        columns = [     
                        [k for k in lead],
                        ["period"],
                        [k for k in eval_atf["train"].keys()],
                        [k for k in tail],
                ]
        df_dict = {c: [] for c in np.concatenate(columns)}
        
        for period in eval_atf:
            df_dict["period"].append(period)
            for k in columns[2]:
                df_dict[k].append(eval_atf[period][k])
        for k in columns[0]:
            df_dict[k] = np.repeat(lead[k], len(df_dict["period"]))
        for k in columns[3]:
            df_dict[k] = np.repeat(tail[k], len(df_dict["period"]))

        return pd.DataFrame(df_dict)

    def calc_pabm_eval_atf(self, normalize=None, lags=None, boxpierce=None, head=None, pvalue=None, train_only=None):


        eval = get_atf(train_only=train_only)


        eval_log_L = self.calc_pabm_eval_atf_log_L(train_only=train_only)
        for period in eval:
            eval[period] = {
                                    **eval_log_L[period],
                                    **self.calc_pabm_eval_resid(period=period, normalize=normalize, lags=lags, boxpierce=boxpierce, head=head, pvalue=pvalue),
                                }


        return eval
    

    def calc_log_L_all_dict_period(self, period=None):

        if period is None:
            period = "all"

        return self.calc_log_L_all_dict()



    def calc_pabm_eval_atf_log_L(self, train_only=None):


        eval = get_atf(train_only=train_only)

        eval_atf_T_N_M = self.calc_pabm_eval_atf_T(train_only=train_only)

        for period in eval:
            
            eval[period] = {
                                    **eval_atf_T_N_M[period],
                                    **self.calc_pabm_eval_atf_log_L_suff(period=period, flatten=True),
                                }

        return eval





    def calc_pabm_eval_atf_log_L_suff(self, period=None, flatten=None):

        if flatten is None:
            flatten=False


        log_L_all_dict = self.calc_log_L_all_dict_period(period=period)
        log_L_suff = {suff: {dev: None for dev in ["N_data", "N_params", "log_L", "AIC", "BIC"]} for suff in log_L_all_dict}

        for suff in log_L_suff:
            log_L_suff[suff]["log_L"] = log_L_all_dict[suff]
            log_L_suff[suff]["N_data"] = self.calc_N_data(period=period, component=suff)
            log_L_suff[suff]["N_params"] = self._parameters.calc_N_params(component=suff)
            log_L_suff[suff]["AIC"] = misc_fns.calc_AIC(N_params=log_L_suff[suff]["N_params"], log_L=log_L_suff[suff]["log_L"])
            log_L_suff[suff]["BIC"] = misc_fns.calc_BIC(N_data=log_L_suff[suff]["N_data"], N_params=log_L_suff[suff]["N_params"], log_L=log_L_suff[suff]["log_L"])


        if flatten:
            return {f"{val}_{suff}": log_L_suff[suff][val] for suff in log_L_suff for val in log_L_suff[suff]}
        else:
            return log_L_suff


    def calc_pabm_eval_resid_atf(self, node=None, normalize=None, lags=None, boxpierce=None, head=None, pvalue=None, train_only=None):

        eval = get_atf(train_only=train_only)
        for period in eval:
            eval[period] = self.calc_pabm_eval_resid(node=node, period=period, normalize=normalize, lags=lags, boxpierce=boxpierce, head=head, pvalue=pvalue)




    def calc_pabm_eval_resid(self, node=None, period=None, normalize=None, lags=None, boxpierce=None, head=None, pvalue=None):

        if lags is None:
            lags = 5


        eval =  {
                    **self.calc_resid_KS_LB(val="D", period=period, normalize=normalize, lags=lags, boxpierce=boxpierce, head=head, pvalue=pvalue, rehead=True, val_head=True),
                    **self.calc_resid_KS_LB(val="F", period=period, normalize=normalize, lags=lags, boxpierce=boxpierce, head=head, pvalue=pvalue, rehead=True, val_head=True),
                }

        return eval





    def calc_resid_KS_LB(self, val=None, period=None, normalize=None, theory_cdf=None, lags=None, boxpierce=None, head=None, pvalue=None, rehead=None, val_head=None):
        
        KS_dict = self.calc_resid_KS(val=val, period=period, normalize=normalize, theory_cdf=theory_cdf, val_head=val_head)
        LB_dict = self.calc_resid_LB(val=val, period=period, normalize=normalize, theory_cdf=theory_cdf, lags=lags, boxpierce=boxpierce, flatten=True, head=head, pvalue=pvalue, rehead=rehead, val_head=val_head)

        return {**KS_dict, **LB_dict}


    def calc_resid_LB(self, val=None, period=None, normalize=None, theory_cdf=None, lags=None, boxpierce=None, flatten=None, head=None, pvalue=None, rehead=None, val_head=None):

        if normalize is None:
            normalize = False


        if theory_cdf is None:
            theory_cdf = self.calc_theory_cdf(val=val)

        if flatten is None:
            flatten=False
        if rehead is None:
            rehead = True

        if pvalue is None:
            pvalue = True
        if pvalue:
            suff = ["stat", "pvalue"]
            suff_rehead = ["stat", "p"]
        else:
            suff = ["stat"]
            suff_rehead = ["stat"]
        if head is None:
            if boxpierce is None:
                boxpierce = False
            if boxpierce:
                head = ["lb", "bp"]
            else:
                head = ["lb"]
            
        
        head = misc_fns.make_iterable_array(head)
        head = [head[k].lower() for k in range(len(head))]
        if np.isin("bp", head):
            boxpierce = True
        
        head_df = [f"{h}_{s}" for h in head for s in suff]

        if rehead:
            head_rehead = [head[k].upper() for k in range(len(head))]
            head_df_rehead = [f"{h}_{s}" for h in head_rehead for s in suff_rehead]

            columns_dict = {head_df[k]: head_df_rehead[k] for k in range(len(head_df))}
            head_out = head_df_rehead
        else:
            head_out = head_df



        resid = self.calc_resid(val=val, period=period, normalize=normalize, theory_cdf=theory_cdf)

        if lags is None:
            lags = 5
        T = np.size(resid)
        lag_eff = np.min([T-1, lags])
        dict_val = {f"{h}": [np.nan for k in range(lags)] for h in head_out}
        df = pd.DataFrame(dict_val, index=1+np.array(range(lags)))
        if lag_eff>0:
            df_lb =  acorr_ljungbox(x=resid,
                                                    lags=lag_eff,
                                                    #auto_lag=True,
                                                    boxpierce=boxpierce,
                                                    return_df=True)
            if rehead:
                df_lb = df_lb.rename(columns=columns_dict)

            df.iloc[:lag_eff] = df_lb[head_out].iloc[:lag_eff]
            
        if not flatten:
            return df
        else:
            val_head = calc_val_head(val_head=val_head, val=val, normalize=normalize)
            return {f"{val_head}{h}_{1+k}": df[h].values[k] for k in range(df.shape[0]) for h in head_out}



    

    
    def calc_resid_KS(self, val=None, period=None, normalize=None, theory_cdf=None, val_head=None):

        if normalize is None:
            normalize = False
        #if val=="F_M":
        #    normalize=True

        if theory_cdf is None:
            theory_cdf = self.calc_theory_cdf(val=val)
        if normalize:
            KS_cdf = scipy.stats.norm().cdf
        else:
            KS_cdf = theory_cdf

        resid = self.calc_resid(val=val, period=period, normalize=normalize, theory_cdf=theory_cdf)
        if np.size(resid)==0:
            KS_result = np.full(2, np.nan)
        else:
            KS_result = scipy.stats.kstest(resid, KS_cdf)

        val_head = calc_val_head(val_head=val_head, val=val, normalize=normalize)
        return {f"{val_head}KS_stat": KS_result[0], f"{val_head}KS_p": KS_result[1]}



    def calc_resid(self, val=None, period=None, normalize=None, theory_cdf=None):

        if val is None:
            val = "D"
        
        if period is None:
            period = "all"


        if val=="D":
            resid = self._Hcomp["D_mu"]/np.sqrt(self._Hcomp["sigma_t_D_2"])
        elif val=="F":
            resid = self._Hcomp["F_mu"]/self._parameters.sigma_F.get_value()



        if normalize is None:
            normalize = False
        if normalize:
            if theory_cdf is None:
                theory_cdf = self.calc_theory_cdf(val=val)
                
            resid = scipy.stats.norm().ppf(theory_cdf(resid))

        return resid


            
        

    def calc_theory_cdf(self, val=None):
        if val=="D":
            theory_cdf = scipy.stats.norm().cdf
        elif val=="F":
            theory_cdf = scipy.stats.norm().cdf

        return theory_cdf
        

    def calc_pabm_eval_atf_T(self, train_only=None):

        eval = get_atf(train_only=train_only)
        for period in eval:

            eval[period] = {
                                    "T": self.calc_T(period=period),
                            }

        return eval


    def calc_T(self, period=None):

        if period is None:
            period = "all"

        if period=="train":
            return self.get_T()
        elif period=="forecast":
            return 0
        else:
            return self.get_T()
        

        
    def calc_N_data(self, period=None, component=None):

        if component is None:
            component = "LA"

        T = self.calc_T(period=period)

        
        factor = 1

        return factor*T

   

    

 #--#===========================
    #--------------------------
    # base paramaeter functions
    #--------------------------
    #===========================


    def get_N_s(self):
        """Get function for N_s
                -- the number of agent strategies
        """
        return self._N_s


 #--#========================
    #-----------------------
    # data functions
    #-----------------------
    #=======================


    if False:
        def set_train_t(self, start=None, end=None, reset=False):

            if reset:
                #self._train_t = np.array([0, self.get_t_size()])

                self.get_T_buff()
            if start is not None:
                if type(start)==int:
                    self._train_t[0] = start
                else:
                    self._train_t[0] = np.sum(self._index < start)
            if end is not None:
                if type(start)==int:
                    self._train_t[1] = end
                else:
                    self._train_t[1] = np.sum(self._index < end)
    else:
        def set_train_t(self, start=None, end=None, reset=False):

            if reset:
                self._train_t = np.array([self.get_T_0(), self.get_T_tot_buff()])
            if start is not None:
                if type(start)==int:
                    self._train_t[0] = start
                #else:
                #    self._train_t[0] = np.sum(self._index < start)
            if end is not None:
                if type(start)==int:
                    self._train_t[1] = end
                #else:
                #    self._train_t[1] = np.sum(self._index < end)


#--#========================
    #-----------------------
    # process functions
    #-----------------------
    #=======================

    def reset_process(self):

        self.reset_t_t_s_series()

    def reset_t_t_s_series(self):
        """
        """
        self.reset_t_series()
        self.reset_t_s_series()


    def reset_t_series(self):
        """
        """
        self._t_series = self.new_t_series()


    def reset_t_s_series(self):
        """
        """
        self._t_s_series = self.new_t_s_series()


    def new_t_series(self):

        base_float = np.full(self.get_T_tot_buff(), np.nan)

        p_t_series =    {
                            "p": np.copy(base_float),
                            "p_F": np.copy(base_float),
                            "r": np.copy(base_float),
                            "r_F": np.copy(base_float),
                            "p_F_sim": np.copy(base_float),
                            "r_F_sim": np.copy(base_float),
                            "varsigma": np.copy(base_float),
                        }
        
        return p_t_series



    def new_t_s_series(self):
        """
        """
        base_float_s = np.full((self.get_N_s(), self.get_T_tot_buff()), np.nan)

        t_series =    {
                            "D": np.copy(base_float_s),
                            "rho": np.copy(base_float_s),
                            "M": np.copy(base_float_s),
                            "S": np.copy(base_float_s),
                            "nepsilon": np.copy(base_float_s),

                            "g": np.copy(base_float_s),
                            "w": np.copy(base_float_s),
                            "nu": np.copy(base_float_s),

                            "n": np.copy(base_float_s),
                            "a": np.copy(base_float_s),
                        }

        return t_series








#============
#============
#
# rbe-pabm
#
#===========
#===========


class rbe_pabm():
    """Class for Pruna abm group
    """
    def __init__(self, data, exchanges=None, train_t_end=None, meta_series=None, opt_method=None):
        """Constructor tt_pot class
        """
        self._pabm_fit_time = {t: None for t in ["start", "end", "delta"]}
        self.reset_rw(meta_series=meta_series)
        self.set_pabm(data=data, exchanges=exchanges, train_t_end=train_t_end, opt_method=opt_method, reset=True)


    

    def reset_rw(self, meta_series=None):
        self._rw = dict()
        self._rw['meta'] = dict()
        self.set_meta_series(meta_series=meta_series)

        for cat in ['fit']:
            self._rw[cat] = {rw: dict() for rw in ['r', 'w']}

    
        self._rw['fit']['w']['pabm'] = empty_fit_df_dict(incl_meta=True)

    def set_meta_series(self, meta_series=None):
        if meta_series is not None:
            self._rw['meta']['series'] = meta_series
        


    def set_pabm(self, data, exchanges=None,train_t_end=None, opt_method=None, reset=False):


        self._exchanges = exchanges



        #self.tt = create_models_2T_POT_nt(data=data, a_u=self._a_u, exchanges=self._exchanges, dt_dt_fine=self._dt_dt_fine, train_t_end=train_t_end, fix_a_lambda=fix_a_lambda, insert=insert, opt_method=opt_method)

        self.pabm = create_models_r12r3be_pabm(data=data, exchanges=self._exchanges, train_t_end=train_t_end, opt_method=opt_method)

    def set_pabm_fit(self, print_pabm_fit_time=None, print_res_time=None, append_w=None, fix_a_lambda=None, insert=None, opt_method=None, fp0=None):
        #self._tt_fit = fit_models_2T_POT_nt(self.tt, print_tt_fit_time=print_tt_fit_time, print_res_time=print_res_time, fix_a_lambda=fix_a_lambda, insert=insert, opt_method=opt_method, fp0=fp0)

        self._pabm_fit = fit_models_r12r3be_pabm(self.pabm, print_pabm_fit_time=print_pabm_fit_time, print_res_time=print_res_time, opt_method=opt_method, fp0=fp0)


        self._pabm_fit_time = self._pabm_fit["pabm_fit"]

        if append_w is None:
            append_w = True
        if append_w:
            w = self.calc_opt_w()
            for fn in w:
                self._rw['fit']['w']['pabm'][fn] = self._rw['fit']['w']['pabm'][fn].append(w[fn])






    


 #--#========================
    #-----------------------
    # rbe_pabm model set functions
    #-----------------------
    #=======================


def create_dict_be(element=None):
    return {
                bmode: {emode: element for emode in _emode} for bmode in _bmode
            } 

def create_dict_r12r3(element=None):
    return {
                r12mode: {r3mode: element for r3mode in _r3mode} for r12mode in _r12mode
            } 

def create_dict_r12r3be(element=None):
    return create_dict_r12r3(element=create_dict_be(element=element))  



def create_models_r12r3be_pabm(data, exchanges=None, train_t_start=None, train_t_end=None, fopt_method=None):

    models = create_dict_r12r3be(element=None)


    if False:
        base = hawkes_process(data=data_2, dt_dt_fine=dt_dt_fine, exchanges=exchanges, fix_a_lambda=fix_a_lambda, insert=insert, opt_method=opt_method, 
                                                                mode="bi", mark="mark", a_u=a_u, u_dir=["left", "right"],
                                                                train_t_start=train_t_start, train_t_end=train_t_end)
    else:
        base = 1

    for r12 in models:
        for r3 in models[r12]:
            r = f"{r12}{r3}" 
            for b in models[r12][r3]:
                for e in models[r12][r3][b]:
                    models[r12][r3][b][e] = copy.deepcopy(base)
                    if False:
                        models[r12][r3][b][e]._parameters.set_rmode(r)
                        models[r12][r3][b][e]._parameters.set_bmode(int(b))
                        models[r12][r3][b][e]._parameters.set_emode(int(e))

    return models


def fit_models_r12r3be_pabm(self, print_pabm_fit_time=None, print_res_time=None, opt_method=None, fp0=None):

    if print_pabm_fit_time is None:
        print_pabm_fit_time = True
    if print_res_time is None:
        print_res_time = True
    
    pabm_fit_timer = misc_fns.timer(print_timer=print_pabm_fit_time, print_head="fit_models_pabm")


    if fp0 is None:
        set_fp0 = True
        fp0 = {
                    r12:   {
                                r3: {
                                        b:  {
                                                e: None for e in self[r12][r3][b]
                                            } for b in self[r12][r3]
                                    } for r3 in self[r12]
                            } for r12 in self
                }
        for r12 in self:
            for r3 in self[r12]:
                for b in self[r12][r3]:
                    for e in self[r12][r3][b]:
                        self[r12][r3][b][e].set_parameters_defaults()
                        fp0[r12][r3][b][e] = self[r12][r3][b][e]._parameters.flatten_defaults()
    else:
        set_fp0 = False

    

    fp0_0 = copy.deepcopy(fp0)
    fp0_1 = copy.deepcopy(fp0)


    if False:
        param_2T_bool = {tail: np.full(fp0["bi_d"]["unmark"].size, False) for tail in ["left", "right"]}
        for i in range(len(self["ci_s"]["unmark"]._parameters._flat_index["list"]["range"])):
            param_2T_bool["left"][self["ci_s"]["unmark"]._parameters._flat_index["list"]["range"][i][0]] = True
            param_2T_bool["right"][self["ci_s"]["unmark"]._parameters._flat_index["list"]["range"][i][-1]] = True


################################################################################

    def fit_r12r3be(r12, r3, b, e):
        if print_res_time:
            print(r12, r3, b, e)
        self[r12][r3][b][e].set_parameters_estimate(flat_parameters_0=fp0[r12][r3][b][e], print_res_time=print_res_time, opt_method=opt_method)

##############################################################################

    r3="z"
    b="0"
    e="0"
    fit_r12r3be(r12="ll", r3=r3, b=b, e=e)
    if set_fp0:
        for r12 in self:
            if bool(1 - (r12=="ll")):
                fp0[r12][r3][b][e] = self["ll"][r3][b][e]._parameters.flatten()

    fit_r12r3be(r12="lt", r3=r3, b=b, e=e)
    fit_r12r3be(r12="cl", r3=r3, b=b, e=e)
    fit_r12r3be(r12="ct", r3=r3, b=b, e=e)

    if set_fp0:
        for r12 in self:
            for r3 in self[r12]:
                for b in self[r12][r3]:
                    for e in self[r12][r3][b]:
                        if bool(1 - (r3=="z")*(b=="0")*(e=="0")):
                            fp0[r12][r3][b][e] = self[r12]["z"]["0"]["0"]._parameters.flatten()


    r3="w"
    b="0"
    e="0"
    fit_r12r3be(r12="ll", r3=r3, b=b, e=e)
    if set_fp0:
        for r12 in self:
            if bool(1 - (r12=="ll")):
                fp0[r12][r3][b][e] = self["ll"][r3][b][e]._parameters.flatten()

    fit_r12r3be(r12="lt", r3=r3, b=b, e=e)
    fit_r12r3be(r12="cl", r3=r3, b=b, e=e)
    fit_r12r3be(r12="ct", r3=r3, b=b, e=e)


    if set_fp0:
        for r12 in self:
            for r3 in ["w"]:
                for b in self[r12][r3]:
                    for e in self[r12][r3][b]:
                        if bool(1 - (b=="0")*(e=="0")):
                            fp0[r12][r3][b][e] = self[r12]["w"]["0"]["0"]._parameters.flatten()

    

    r3="z"
    b="1"
    e="0"
    fit_r12r3be(r12="ll", r3=r3, b=b, e=e)
    if set_fp0:
        for r12 in self:
            if bool(1 - (r12=="ll")):
                fp0[r12][r3][b][e] = self["ll"][r3][b][e]._parameters.flatten()

    fit_r12r3be(r12="lt", r3=r3, b=b, e=e)
    fit_r12r3be(r12="cl", r3=r3, b=b, e=e)
    fit_r12r3be(r12="ct", r3=r3, b=b, e=e)

    if set_fp0:
        for r12 in self:
            for r3 in self[r12]:
                for b in ["1"]:
                    for e in self[r12][r3][b]:
                        if bool(1 - (r3=="z")*(e=="0")):
                            fp0[r12][r3][b][e] = self[r12]["z"]["1"]["0"]._parameters.flatten()


    r3="w"
    b="1"
    e="0"
    fit_r12r3be(r12="ll", r3=r3, b=b, e=e)
    if set_fp0:
        for r12 in self:
            if bool(1 - (r12=="ll")):
                fp0[r12][r3][b][e] = self["ll"][r3][b][e]._parameters.flatten()

    fit_r12r3be(r12="lt", r3=r3, b=b, e=e)
    fit_r12r3be(r12="cl", r3=r3, b=b, e=e)
    fit_r12r3be(r12="ct", r3=r3, b=b, e=e)


    if set_fp0:
        for r12 in self:
            for r3 in ["w"]:
                for b in ["1"]:
                    for e in self[r12][r3][b]:
                        if bool(1 - (e=="0")):
                            fp0[r12][r3][b][e] = self[r12]["w"]["1"]["0"]._parameters.flatten()


        
    r3="z"
    b="0"
    e="1"
    fit_r12r3be(r12="ll", r3=r3, b=b, e=e)
    if set_fp0:
        for r12 in self:
            if bool(1 - (r12=="ll")):
                fp0[r12][r3][b][e] = self["ll"][r3][b][e]._parameters.flatten()

    fit_r12r3be(r12="lt", r3=r3, b=b, e=e)
    fit_r12r3be(r12="cl", r3=r3, b=b, e=e)
    fit_r12r3be(r12="ct", r3=r3, b=b, e=e)

    if set_fp0:
        for r12 in self:
            for r3 in self[r12]:
                for b in self[r12][r3]:
                    for e in ["1"]:
                        if bool(1 - (r3=="z")*(b=="0")):
                            fp0[r12][r3][b][e] = self[r12]["z"]["0"]["1"]._parameters.flatten()


    r3="w"
    b="0"
    e="1"
    fit_r12r3be(r12="ll", r3=r3, b=b, e=e)
    if set_fp0:
        for r12 in self:
            if bool(1 - (r12=="ll")):
                fp0[r12][r3][b][e] = self["ll"][r3][b][e]._parameters.flatten()

    fit_r12r3be(r12="lt", r3=r3, b=b, e=e)
    fit_r12r3be(r12="cl", r3=r3, b=b, e=e)
    fit_r12r3be(r12="ct", r3=r3, b=b, e=e)


    if set_fp0:
        for r12 in self:
            for r3 in ["w"]:
                for b in self[r12][r3]:
                    for e in ["1"]:
                        if bool(1 - (b=="0")):
                            fp0[r12][r3][b][e] = self[r12]["w"]["0"]["1"]._parameters.flatten()

    

    r3="z"
    b="1"
    e="1"
    fit_r12r3be(r12="ll", r3=r3, b=b, e=e)
    if set_fp0:
        for r12 in self:
            if bool(1 - (r12=="ll")):
                fp0[r12][r3][b][e] = self["ll"][r3][b][e]._parameters.flatten()

    fit_r12r3be(r12="lt", r3=r3, b=b, e=e)
    fit_r12r3be(r12="cl", r3=r3, b=b, e=e)
    fit_r12r3be(r12="ct", r3=r3, b=b, e=e)

    if set_fp0:
        for r12 in self:
            for r3 in self[r12]:
                for b in ["1"]:
                    for e in ["1"]:
                        if bool(1 - (r3=="z")):
                            fp0[r12][r3][b][e] = self[r12]["z"]["1"]["1"]._parameters.flatten()


    r3="w"
    b="1"
    e="1"
    fit_r12r3be(r12="ll", r3=r3, b=b, e=e)
    if set_fp0:
        for r12 in self:
            if bool(1 - (r12=="ll")):
                fp0[r12][r3][b][e] = self["ll"][r3][b][e]._parameters.flatten()

    fit_r12r3be(r12="lt", r3=r3, b=b, e=e)
    fit_r12r3be(r12="cl", r3=r3, b=b, e=e)
    fit_r12r3be(r12="ct", r3=r3, b=b, e=e)






    for r12 in self:
        for r3 in self[r12]:
            for b in self[r12][r3]:
                for e in self[r12][r3][b]:
                    fp0_1[r12][r3][b][e] = self[r12][r3][b][e]._parameters.flatten_defaults()


    pabm_fit_timer.stop()

    fit_dict =  {
                    "pabm_fit": pabm_fit_timer.get_timer_dict(),
                    "fp0": fp0,
                    "fp0_0": fp0_0,
                    "fp0_1": fp0_1,
                }
    return fit_dict



 

def get_atf(train_only=None):

    if train_only is None:
        train_only=False

    if train_only:
        periods = ["train"]
    else:
        periods = ["all", "train", "forecast"]

    return {period: None for period in periods}





def calc_val_head(val_head=None, val=None, normalize=None):

    if val_head is None:
        val_head = False

    if type(val_head)!=bool:
        return val_head
    elif val_head==False or val is None:
        return f""
    else:
        
        if normalize is None:
            normalize = False

        if normalize:
            normalize_str = f"Norm_"
        else:
            normalize_str = f""

        return f"{normalize_str}{val}_"





def rho(x=None, type=None, gamma_1=None, gamma_2=None):

    if x is None:
        x = 0
    if type is None:
        type = "zero"
    if gamma_1 is None:
        gamma_1 = 0
        
    if type=="zero":
        return np.zeros_like(x)
    elif type=="linear":
        return x
    elif type=="cubic":
        return x + gamma_1*(x**3)
    elif type=="tanh":
        return np.tanh(gamma_1*x)
    elif type=="wtar":
        if gamma_2 is None:
            gamma_2 = 1
        return np.tanh(gamma_1*(1 - np.sqrt(x)/gamma_2))


def flatten_parameters(mu_F, sigma_F, p_0_F, p_0, beta, alpha_0_s, alpha_n, alpha_w, eta_w, lambda_w_s, psi, delta_s, gamma_1_s, gamma_2_s, eta_s, sigma_s, eta_varsigma, varsigma_0):
    return np.hstack([mu_F, sigma_F, p_0_F, p_0, beta, np.hstack(alpha_0_s), alpha_n, alpha_w, eta_w, np.hstack(lambda_w_s), psi, np.hstack(delta_s), np.hstack(gamma_1_s), np.hstack(gamma_2_s), np.hstack(eta_s), np.hstack(sigma_s), eta_varsigma, varsigma_0])


def unflatten_parameters(D_nodes, flat_parameters):


    stypes = ["scalar", "scalar", "scalar", "scalar", "scalar", "vector", "scalar", "scalar", "scalar", "vector", "scalar", "vector", "vector", "vector", "vector", "vector", "scalar", "scalar"]
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
        if True:
            indicies.append(range(index_sizes_cumsum[i], index_sizes_cumsum[i+1]))
            parameters.append(np.resize(flat_parameters[indicies[i]], stypes_shape[stypes[i]]))
        else:
            if stypes[i]=="scalar":
                parameters.append(flat_parameters[index_sizes_cumsum[i]])
            else:
                index = range(index_sizes_cumsum[i], index_sizes_cumsum[i+1])
                parameters.append(np.resize(flat_parameters[index], stypes_shape[stypes[i]]))

    return parameters

def append_0_M(X):

    X_out = np.zeros((X.shape[0]+1, X.shape[1]+1))
    X_out[:-1,:-1] = X
    return X_out


def where_non_symmetyric(Hess):

    return np.where(np.triu(Hess) != np.triu(Hess.T))

def non_symmetric_vectors(Hess, where_ns=None):

    if where_ns is None:
        where_ns = where_non_symmetyric(Hess)

    return np.array([Hess[where_ns[0], where_ns[1]],
                    Hess.T[where_ns[0], where_ns[1]]])


def non_symmetric_matrix(Hess):

    outHess = np.zeros_like(Hess)

    where_ns = where_non_symmetyric(Hess=Hess)

    ns_vectors = non_symmetric_vectors(Hess=Hess, where_ns=None)

    outHess[where_ns[0], where_ns[1]] = ns_vectors[0,:]
    outHess[where_ns[1], where_ns[0]] = ns_vectors[1,:]

    return outHess


def non_symmetric_stats(Hess):

    ns_res = dict()

    ns_res["nsv"] = non_symmetric_vectors(Hess)
    ns_res["diff_nsv"] = np.diff(ns_res["nsv"], axis=0)
    ns_res["N_ns"] = ns_res["diff_nsv"].size

    
    if ns_res["N_ns"]==0:
        ns_res["max"] = 0.0
        ns_res["mean"] = 0.0
        ns_res["std"] = np.nan
        ns_res["median"] = 0.0

    else:
        ns_res["max"] = np.max(ns_res["diff_nsv"])
        ns_res["mean"] = np.mean(ns_res["diff_nsv"])
        ns_res["std"] = np.std(ns_res["diff_nsv"])
        ns_res["median"] = np.quantile(ns_res["diff_nsv"], 0.5)

    ns_res["log_nsv"] = np.log(np.abs(ns_res["nsv"]))
    ns_res["diff_log_nsv"] = np.diff(ns_res["log_nsv"], axis=0)

    if ns_res["N_ns"]==0:
        ns_res["max_log"] = 0.0
        ns_res["mean_log"] = 0.0
        ns_res["std_log"] = np.nan
        ns_res["median_log"] = 0.0

    else:
        ns_res["max_log"] = np.max(ns_res["diff_log_nsv"])
        ns_res["mean_log"] = np.mean(ns_res["diff_log_nsv"])
        ns_res["std_log"] = np.std(ns_res["diff_log_nsv"])
        ns_res["median_log"] = np.quantile(ns_res["diff_log_nsv"], 0.5)


    return ns_res


def print_non_symmetric_stats(Hess, return_ns_res=None):

    if return_ns_res is None:
        return_ns_res = True

    ns_res = non_symmetric_stats(Hess=Hess)


    #for key in ["N_ns", "max", "mean", "std", "median"]:
    for key in ns_res:
        print(f"{key} = {ns_res[key]}")

    if return_ns_res:
        return ns_res



def pack_rbe_str(rmode=None, bmode=None, emode=None):

    if rmode is None:
        rmode = "llz"
    if bmode is None:
        bmode = 0
    if emode is None:
        emode = 0

    return f"R{rmode}B{int(bmode)}E{int(emode)}"


def unpack_rbe_str(rbe_str):

    rmode = rbe_str[rbe_str.find("R")+1:rbe_str.find("B")]
    bmode = int(rbe_str[rbe_str.find("B")+1:rbe_str.find("E")])
    emode = int(rbe_str[rbe_str.find("E")+1:])

    return rmode, bmode, emode

def unpack_rmode(rmode):

    return np.array([rmode[i] for i in range(len(rmode))])


def empty_fit_df_dict(incl_meta=None):

    if incl_meta is None:
        incl_meta = False

    if incl_meta:
        fields = ["meta", "est", "se", "fp1", "eval_train"]
    else:
        fields = ["est", "se", "fp1", "eval_train"]

    return {fdict: pd.DataFrame() for fdict in fields}


def calc_T_T_burn(p_t=None, r_t=None, p_0=None, T=None, T_burn=None):

    if p_t is not None or r_t is not None:
        T_burn = 0
        if p_t is not None:
            if p_0 is None:
                T = np.size(p_t) - 1
            else:
                T = np.size(p_t)
        else:
            T = np.size(r_t)

    return T, T_burn       


_rho_type_int_list = ["zero", "linear", "cubic", "tanh", "wtar"]

_rho_type_int = {_rho_type_int_list[i]: i for i in range(len(_rho_type_int_list))}





_r1mode = ["l", "c"]
_r2mode = ["l", "t"]
_r3mode = ["z", "w"]

_r12mode = [f"{r1}{r2}" for r1 in _r1mode for r2 in _r2mode]

_bmode = ["0", "1", "2"]

_emode = ["0", "1"]


_rmode_dict =   {
                    "z": "zero",
                    "l": "linear",
                    "c": "cubic",
                    "t": "tanh",
                    "w": "wtar",
                }

_rmode_letters = [l for l in _rmode_dict]
_rmode_names = [_rmode_dict[l] for l in _rmode_letters]




