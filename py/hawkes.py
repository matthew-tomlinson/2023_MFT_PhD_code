import numpy as np
from numpy.lib.function_base import diff
import scipy
import scipy.stats as stats
from scipy.stats import genpareto
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import pandas as pd
import copy
import functools
import time
import numba as nb
from numba import jit, njit
import numdifftools
import os
# Local files
import financial_data_handling as fin_data_handl
import parameters
import cvol as cvl
import bulk_gp
import threshold

import misc_functions as misc_fns
import plot_functions as plot_fns
import dist_functions as dist_fns
import hawkes_functions as hwks_fns





class hawkes_threshold_opt(threshold.thresholds):
    """Class for hawkes_threshold_opt
    """
    def __init__(self, D_nodes,
                 opt_method=None, fix_a_lambda=None,
                 a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None):
        
        if False:
        
            self._D_nodes = parameters.clean_D_nodes(D_nodes=D_nodes)
            

            self.reset_params_threshold()

            self._set_params_threshold(a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, reset=False)
        else:
            super().__init__(D_nodes=D_nodes, 
                                a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive,
                                #X=X, train_t=train_t, calc_u=calc_u, calc_a_u=calc_a_u
                                )
            self._set_opt_settings(opt_method=opt_method, fix_a_lambda=fix_a_lambda, reset=True)



    def _set_opt_settings(self, opt_method=None, fix_a_lambda=None, reset=None, reset_opt_method=None, reset_fix_a_lambda=None):

        resets = hwks_fns.calc_resets_opt_setting(reset=reset, opt_method=reset_opt_method, fix_a_lambda=reset_fix_a_lambda)

        self._set_fix_a_lambda(fix_a_lambda=fix_a_lambda, reset=resets['fix_a_lambda'])
        self._set_opt_method(opt_method=opt_method, reset=resets['opt_method'])



    def _set_opt_method(self, opt_method=None, reset=None):

        if opt_method is not None:
            self._opt_method = opt_method
        elif reset:
                self._opt_method = 'SLSQP'

    def get_opt_method(self):
        return self._opt_method


    def _set_fix_a_lambda(self, fix_a_lambda=None, reset=None, free=None):


        if reset is None:
            reset = False
        if free is None:
            free = True

        if fix_a_lambda is not None:
            self._fix_a_lambda = fix_a_lambda
        elif reset:
                self._fix_a_lambda = True



    def get_fix_a_lambda(self):
        return self._fix_a_lambda


    def calc_opt_settings_dict(self):

        return hwks_fns.calc_opt_settings_dict(opt_method=self.get_opt_method(), fix_a_lambda=self.get_fix_a_lambda())
    




class hawkes_parameters(hawkes_threshold_opt):
    """Class for all Hawkes parameters
    """
    def __init__(self, D_nodes, 
                    a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None,
                    opt_method=None, fix_a_lambda=None, 
                    mode=None, mark=None, 
                    mu=None, gamma=None, 
                    phi_type=None, beta=None, nu=None, omega=None,
                    xi=None, varsigma=None, eta=None,
                    kappa_type=None, alpha=None):
        """Constructor for hawkes_parameters class
        """

        if False:

            self._D_nodes = parameters.clean_D_nodes(D_nodes=D_nodes)
            self.set_params_threshold(a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, reset=False)
            self.set_opt_settings(opt_method=opt_method, fix_a_lambda=fix_a_lambda, reset=True)
            


        else:
            super().__init__(D_nodes=D_nodes, 
                                a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive,
                                #X=X, train_t=train_t, calc_u=calc_u, calc_a_u=calc_a_u,
                                opt_method=opt_method, fix_a_lambda=fix_a_lambda, 
                                )
            

        self.reset_fp()
        self.set_flat_fix(reset=True, free=False)
        self.set_all(mu=mu, gamma=gamma, 
                    phi_type=phi_type, beta=beta, nu=nu, omega=omega,
                    xi=xi, varsigma=varsigma, eta=eta,
                    kappa_type=kappa_type, alpha=alpha,
                    fix=False)

        self.set_mode(mode)
        self.set_mark(mark)

        self.set_flat_fix_tmm()

        


        # Parameter estimation
        self.res = None
        self.res_time = {step: dict() for step in ["res", 'opt']}
        self.res_time['res']['fix_a_lambda'] = None
        self.res_time['opt']['method'] = None
        for step in self.res_time:
            for t in ["start", "end", "delta"]:
                self.res_time[step][t] = None

        self.res_fp0 = None
        self.res_fp1 = None
        self.res_fp2 = None

        self.ftol = 2.220446049250313e-09

        self._fitted = {param_name: None for param_name in self._fp_names}
        self._fitted_se = self._fitted.copy()
        self._fitted_scores = misc_fns.IC_dict()


        self._N_params = None
        self._log_L = None


        self._train_t = None



    def reset_fp(self):

        self.reset_fp_params()
        self.reset_flat()


    def set_params_threshold(self, a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None, 
                           reset=None, reset_a_u=None, reset_u_dir=None, reset_u=None, reset_u_zero=None):
        

        self._set_params_threshold(a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, 
                           reset=reset, reset_a_u=reset_a_u, reset_u_dir=reset_u_dir, reset_u=reset_u, reset_u_zero=reset_u_zero)
        


    def reset_fp_params(self):

        #self.reset_params_threshold()

        #----------------------------
        # Intensity parameters
        #----------------------------
        # Expected intensity
        self.a_lambda = parameters.p_vector(D_nodes=self._D_nodes, name="a_lambda", etype=[int, float], family="intensity", default=0.05, incl_low=0)

        # Background intentity
        self.mu = parameters.p_vector(D_nodes=self._D_nodes, name="mu", etype=[int, float], family="intensity", default=1E-8, excl_low=0, opt_incl_high=2, opt_incl_low=1E-3, unit=f"{hwks_fns.get_unit_d_tr()}^{{-1}}")

        # Excitement coefficients
        self.gamma = parameters.p_matrix(D_nodes=self._D_nodes, name="gamma", etype=[int, float], family="intensity", default=0, incl_low=0, opt_incl_high=2)


        #----------------------------
        # Time kernel parameters
        #----------------------------
        # Kernel type
        self.phi_type = parameters.p_scalar(D_nodes=self._D_nodes, name="phi_type", etype=[str], family="phi", default="exp", permitted=["exp", "pow"])

        # Exponential kernel parameters
        self.beta = parameters.p_vector(D_nodes=self._D_nodes, name="beta", etype=[int, float,], family="phi", default=1, incl_low=0, opt_incl_high=2, opt_incl_low=1E-3, unit=f"{hwks_fns.get_unit_d_tr()}^{{-1}}")

        # Power-law kernel parameters
        self.nu = parameters.p_vector(D_nodes=self._D_nodes, name="nu", etype=[int, float], family="phi", default=0, incl_low=0, opt_incl_high=2, unit=f"{hwks_fns.get_unit_d_tr()}^{{-1}}")
        self.omega = parameters.p_vector(D_nodes=self._D_nodes, name="omega", etype=[int, float], family="phi", default=0, incl_low=0, opt_incl_high=5)


        #----------------------------
        # Magnitude parameters
        #----------------------------

        # Magnitude GPD exponent
        self.xi = parameters.p_vector(D_nodes=self._D_nodes, name="xi", etype=[int, float], family="magnitude", default=0, incl_low=-1, opt_incl_high=0.5)

        # Unconditional scale parameter
        self.varsigma = parameters.p_vector(D_nodes=self._D_nodes, name="varsigma", etype=[int, float], family="magnitude", default=1, excl_low=0, opt_incl_high=2, opt_incl_low=1E-3)

        # Conditional scale parameter
        self.sigma = parameters.p_vector(D_nodes=self._D_nodes, name="sigma", etype=[int, float], family="magnitude", default=self.varsigma.get_value(), excl_low=0)

        # Intensity dependence coefficient
        self.eta = parameters.p_vector(D_nodes=self._D_nodes, name="eta", etype=[int, float], family="magnitude", default=0, incl_low=0, opt_incl_high=2)


        #----------------------------
        # Impact function parameters
        #----------------------------
        # Impact function type (0 = 'gpd', 1 = 'exp', 2 = 'pow')
        self.kappa_type = parameters.p_vector(D_nodes=self._D_nodes, name="kappa_type", etype=[str], family="kappa", default="gpd", permitted=["one", "exp", "pow", "gpd"])

        # Impact parameter
        self.alpha = parameters.p_vector(D_nodes=self._D_nodes, name="alpha", etype=[int, float], family="kappa", default=0, incl_low=0, opt_incl_high=1000)


        self.reset_fp_names()



    def get_fp_name(self, name=None, name_rm=None):
        if name is None:
            name = self._fp_name
        return misc_fns.A_rm(A=name, A_rm=name_rm)
    
    def calc_fp_names(self, name=None, name_rm=None):

        return self.calc_names(name=self.get_fp_name(name=name, name_rm=name_rm))

    def get_fp_names(self, names=None, names_rm=None):
        if names is None:
            names = self._fp_names
        return misc_fns.A_rm(A=names, A_rm=names_rm)


    def reset_fp_names(self):

        self._fp_name = ["a_lambda", "gamma", "beta", "nu", "omega", "xi", "varsigma", "eta", "alpha"]
        self._fp_names = self.calc_names(name=self._fp_name)


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


    def set_mode(self, mode=None):
        # Modes 
        # generic
        # nme = not mutually exclusive
        # me = mutually exclusive
        # dc = decoupled
        # "2T-POT"
        # bi = bivariate [equiv. to nme]
        # bi_d = bivariate (decoupled) [equiv. to dc]
        # ci = common intensity
        # ci_s = common_intensity (symmetric)
        if mode is None:
            mode="nme"

        if mode=="bi":
            mode="nme"
        elif mode=="bi_d":
            mode="dc"
        elif mode=="bi_l":
            self.gamma._value[1:,:] = 0
            self.pull_flat_parameters()
        elif mode=="bi_r":
            self.gamma._value[:-1,:] = 0
            self.pull_flat_parameters()

        self._mode = mode

        self.set_flat_subordinate(free=False)
        self.set_flat_fix(reset=True, free=False)
        self.set_flat_fix_type(free=False)
        self.set_flat_fix_mode(free=False)
        self.set_flat_free()

        self.set_M()

        self.set_defaults_mode()
             

    def get_mode(self):
        return self._mode

    def set_M(self):

        if self._mode=="ci" or self._mode=="ci_s":
            D_ci=2
        else:
            D_ci=1
        self._M_sum = misc_fns.M_sum(D_nodes=self.get_D_nodes(), D_ci=D_ci)
        self._M_weight = self._M_sum/np.transpose(np.array([np.sum(self._M_sum, axis=1)]))

    def get_M_sum(self):
        return self._M_sum

    def get_M_weight(self):
        return self._M_weight

    def set_mark(self, mark=None):

        if type(mark)==bool:
            self._mark=mark
        elif mark=="unmark":
            self._mark=False
        else:
            self._mark=True
        if self._mark==False:
            self.eta.set_value(0)
            self.alpha.set_value(0)
            self.pull_flat_parameters()

        self.set_flat_fix_mark()
        self.set_flat_free()
        self.set_defaults_mark()


    def get_mark(self):
        return self._mark
    
    def calc_mm_dict(self):
        mm_dict =   {
                        "mode": hwks_fns.mode_translate(self._mode),
                        "mark": hwks_fns.mark_translate(self._mark),
                    }
        return mm_dict
    


    def set_opt_settings(self, opt_method=None, fix_a_lambda=None, 
                         reset=None, reset_opt_method=None, reset_fix_a_lambda=None, 
                         free=None):

        resets = hwks_fns.calc_resets_opt_setting(reset=reset, opt_method=reset_opt_method, fix_a_lambda=reset_fix_a_lambda)

        self.set_opt_method(opt_method=opt_method, reset=resets['opt_method'])
        self.set_fix_a_lambda(fix_a_lambda=fix_a_lambda, reset=resets['fix_a_lambda'], free=free)



    def set_opt_method(self, opt_method=None, reset=None):

        self._set_opt_method(opt_method=opt_method, reset=reset)
    

    def set_fix_a_lambda(self, fix_a_lambda=None, reset=None, free=None):

        self._set_fix_a_lambda(fix_a_lambda=fix_a_lambda, reset=reset)

        if free is None:
            free = True

        if self.get_fix_a_lambda():
            self.set_flat_fix(fix=np.array(self._flat_index["dict"]["range"]["a_lambda"]))
        else:
            self.set_flat_fix(unfix=np.array(self._flat_index["dict"]["range"]["a_lambda"]))
            self.set_flat_fix_mode()

        if free:
            self.set_flat_free()

        



        



    def set_flat_subordinate(self, free=True):

        self._flat_subordinate = np.reshape(np.repeat(range(self._flat_size),2), (self._flat_size,2))
        if self._mode=="ci":
            param_ci =  {
                            param: np.resize(self._flat_subordinate[self._flat_index["dict"]["range"][param], 0], self.__dict__[param]._shape) for param in ["a_lambda", "gamma"]
                        }
            for param in param_ci.keys():
                for i in range(2//2):
                    param_ci[param][1+i] = param_ci[param][i]   
                self._flat_subordinate[self._flat_index["dict"]["range"][param], 1] = np.hstack(param_ci[param])            
        elif self._mode=="ci_s":
            for param in self._flat_index["dict"]["range"]:
                self._flat_subordinate[self._flat_index["dict"]["range"][param], 1] = self._flat_index["dict"]["start"][param]

        if free:
            self.set_flat_free()


    def get_flat_subordinate(self):

        return self._flat_fix


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


    def set_flat_fix_tmm(self, free=True):
        self.set_flat_fix_type(free=False)
        self.set_flat_fix_mode(free=False)
        self.set_flat_fix_mark(free=False)
        self._set_fix_a_lambda(free=False)
        if free:
            self.set_flat_free()



    def set_flat_fix_type(self, free=True):
        # Fix unused phi parameters
        if self.phi_type.get_value() == "exp":
            phi_fix = np.concatenate([self._flat_index["dict"]["range"]["nu"], self._flat_index["dict"]["range"]["omega"]])
        elif self.phi_type.get_value() == "pow":
            phi_fix = self._flat_index["dict"]["range"]["beta"]
        else:
            raise ValueError("{} {} {}".format(self.phi_type._fp_name, 
                                                self.phi_type._p_range._messages["permitted"], 
                                                self.phi_type._p_range._values["permitted"]))
        self._flat_fix[phi_fix] = True

        # Fix alpha for one type impact
        alpha_one = self.kappa_type.get_value() == "one"
        if np.any(alpha_one):  
            alpha_one_fix = self._flat_index["dict"]["start"]["alpha"] + np.where(alpha_one)[0]
            self._flat_fix[alpha_one_fix] = True
        if free:
            self.set_flat_free()


    def set_flat_fix_mode(self, free=True):
        # Decoupled mode
        if self._mode=="dc":
            self._flat_fix[self._flat_index["dict"]["start"]["gamma"] + np.where(np.hstack(~np.eye(self.get_D_nodes(),dtype=bool)))[0]] = True
        elif self._mode=="bi_l":
            G = np.ones((self.get_D_nodes(),self.get_D_nodes()),dtype=bool)
            G[0,:] = False
            self._flat_fix[self._flat_index["dict"]["start"]["gamma"] + np.where(np.hstack(G))[0]] = True
        elif self._mode=="bi_r":
            G = np.ones((self.get_D_nodes(),self.get_D_nodes()),dtype=bool)
            G[-1,:] = False
            self._flat_fix[self._flat_index["dict"]["start"]["gamma"] + np.where(np.hstack(G))[0]] = True
        if free:
            self.set_flat_free()

    def set_flat_fix_mark(self, free=True):
        mark_index = np.concatenate([self._flat_index["dict"]["range"]["eta"], self._flat_index["dict"]["range"]["alpha"]])
        # Fix eta and alpha for unmark process
        if not self._mark:
            self._flat_fix[mark_index] = True
        else:
            self._flat_fix[mark_index] = False
        if free:
            self.set_flat_free()


    def set_defaults_mm(self):
        self.set_defaults_mode()
        self.set_defaults_mark()


    def set_defaults_mode(self):
        if self._mode=="dc":
            gamma = self.gamma.get_default()
            gamma[~np.eye(gamma.shape[0],dtype=bool)] = 0
            self.gamma.set_default(gamma)
        elif self._mode=="bi_l":
            gamma = np.zeros(self.gamma.get_default().shape)
            gamma[0,:] = self.gamma.get_default()[0,:]
            self.gamma.set_default(gamma)
        elif self._mode=="bi_r":
            gamma = np.zeros(self.gamma.get_default().shape)
            gamma[1,:] = self.gamma.get_default()[1,:]
            self.gamma.set_default(gamma)
        
    
    def set_defaults_mark(self):
        if not self._mark:
            self.eta.set_default(default=0)
            self.alpha.set_default(default=0)
     

    def get_flat_fix(self):

        return self._flat_fix

    def set_flat_free(self):
        
        self._flat_free = np.logical_and(~self._flat_fix, self._flat_subordinate[:,0]==self._flat_subordinate[:,1])
        self._N_params = np.sum(self._flat_free)

    def get_flat_free(self):

        return self._flat_free

    def flatten(self):
        return hwks_fns.flatten_parameters(self.a_lambda._value, self.gamma._value,
                                        self.beta._value, self.nu._value, self.omega._value,
                                        self.xi._value, self.varsigma._value, self.eta._value,
                                        self.alpha._value)

    def flatten_defaults(self):
        return hwks_fns.flatten_parameters(self.a_lambda._default, self.gamma._default,
                                        self.beta._default, self.nu._default, self.omega._default,
                                        self.xi._default, self.varsigma._default, self.eta._default,
                                        self.alpha._default)


    def unflatten_parameters(self, flat_parameters):
        
        return [np.resize(flat_parameters[self._flat_index["dict"]["range"][name]], self.__dict__[name]._shape) for name in self._fp_name]


    def set_flat_parameters(self, flat_parameters, subordinate=True, push=True):

        if subordinate:
            self._flat_parameters[self._flat_subordinate[~self._flat_fix,0]] = flat_parameters[self._flat_subordinate[~self._flat_fix,1]]
        if push:
            self.push_flat_parameters()

    def set_flat_parameters_free(self, flat_parameters_free, flat_free):

        self._flat_parameters[flat_free] = flat_parameters_free
        self._flat_parameters[self._flat_subordinate[~self._flat_fix,0]] = self._flat_parameters[self._flat_subordinate[~self._flat_fix,1]]
        self.push_flat_parameters()


    def get_flat_parameters(self, pull=True):

        if pull:
            self.pull_flat_parameters()
            
        return self._flat_parameters

    


    def push_flat_parameters(self):

        unflat_parameters = self.unflatten_parameters(self._flat_parameters)

        self.a_lambda._value = unflat_parameters[0]
        self.gamma._value = unflat_parameters[1]
        self.update_mu()

        self.beta._value = unflat_parameters[2]
        self.nu._value = unflat_parameters[3]
        self.omega._value = unflat_parameters[4]

        self.xi._value = unflat_parameters[5]
        self.varsigma._value = unflat_parameters[6]
        self.eta._value = unflat_parameters[7]

        self.alpha._value = unflat_parameters[8]

    def pull_flat_parameters(self):

        self._flat_parameters = self.flatten()


        



    def set_all(self, mu=None, gamma=None, 
                    phi_type=None, beta=None, nu=None, omega=None,
                    xi=None, varsigma=None, eta=None,
                    kappa_type=None, alpha=None,
                    pull=True, fix=True):
        """
        """
        self.set_intensity_parameters(mu=mu, gamma=gamma, pull=False)
        if phi_type is None:
            phi_type = self.phi_type.get_value()
        self.set_phi(phi_type=phi_type, beta=beta, nu=nu, omega=omega, pull=False, fix=False)
        self.set_magnitude_parameters(xi=xi, varsigma=varsigma, eta=eta, pull=False)
        self.set_kappa_parameters(kappa_type=kappa_type, alpha=alpha, pull=False, fix=False)
        if fix:
            self.set_flat_fix_type()
        if pull:
            self.pull_flat_parameters()


    def set_intensity_parameters(self, mu=None, gamma=None, pull=True):
        """
        """
        if mu is not None:
            self.mu.set_value(value=mu)
        if gamma is not None:
            self.gamma.set_value(value=gamma)   
        if pull:
            self.pull_flat_parameters()       
            

    def update_mu(self):
        self.mu._value = (np.eye(self.get_D_nodes()) - self.gamma.get_value()) @ self.a_lambda.get_value()

    def update_a_lambda(self):
        self.a_lambda._value = np.linalg.inv(np.eye(self.get_D_nodes()) - self.gamma.get_value()) @ self.mu.get_value()


    def set_phi(self, phi_type, beta=None, nu=None, omega=None, pull=True, fix=True):
        self.phi_type.set_value(value=phi_type)
        if self.phi_type.get_value() == "exp":
            self.set_phi_exp(beta=beta, pull=False, fix=False)
        elif self.phi_type.get_value() == "pow":
            self.set_phi_pow(nu=nu, omega=omega, pull=False, fix=False)
        else:
            raise ValueError("{} {} {}".format(self.phi_type._name, 
                                                self.phi_type._p_range._messages["permitted"], 
                                                self.phi_type._p_range._values["permitted"]))
        if fix:
            self.set_flat_fix_type()
        if pull:
            self.pull_flat_parameters() 
            
    def set_phi_exp(self, beta, pull=True, fix=True):
        self.phi_type.set_value(value="exp")

        self.beta.set_value(value=beta)
        self.nu.set_value(value=0)
        self.omega.set_value(value=0)

        if fix:
            self.set_flat_fix_type()
        if pull:
            self.pull_flat_parameters() 

    def set_phi_pow(self, nu, omega, fix=True, pull=True):
        self.phi_type.set_value(value="pow")

        self.beta.set_value(value=0)
        self.nu.set_value(value=nu)
        self.omega.set_value(value=omega)
        if fix:
            self.set_flat_fix_type()
        if pull:
            self.pull_flat_parameters()

    def phi(self, lag, node=0):

        if np.any(lag < 0):
            raise ValueError("all elements of lag must be greater than or equal to 0")
        
        if self.phi_type._value == "exp":
            return phi_exp(lag, self.beta.get_value()[node])
        elif self.phi_type._value == "pow":
            return phi_pow(lag, self.nu.get_value(), self.omega.get_value())
        else:
            raise TypeError("phi_type is not set as either 'exp' or 'pow'")

    def int_phi(self, lag, node=0):

        if np.any(lag < 0):
            raise ValueError("all elements of lag must be greater than or equal to 0")
        
        if self.phi_type._value == "exp":
            return int_phi_exp(lag, self.beta.get_value()[node])
        elif self.phi_type._value == "pow":
            return int_phi_pow(lag, self.nu.get_value(), self.omega.get_value())
        else:
            raise TypeError("phi_type is not set as either 'exp' or 'pow'")


    def set_magnitude_parameters(self, xi=None, varsigma=None, eta=None, pull=True):
        """Set function for all parameters of the magnitude distribution(s)
        """
        if type(xi) != type(None):
            self.xi.set_value(value=xi)
        if type(varsigma) != type(None):
            self.varsigma.set_value(value=varsigma)
        if type(eta) != type(None):
            self.eta.set_value(value=eta)

        if pull:
            self.pull_flat_parameters()



    # lambda_tilde_t
    def calc_sigma_old(self, lambda_t=None, nodes=None):

        if type(lambda_t) == type(None):
            return self.varsigma.get_value(nodes=nodes)

        nodes = self.clean_nodes(nodes=nodes)
        if np.size(lambda_t) != nodes.size:
            raise ValueError("lambda_t must be of size nodes.size = {}".format(nodes.size))

        sigma = self.varsigma.get_value(nodes) + self.eta.get_value(nodes)*(lambda_t - self.mu.get_value(nodes))

        return sigma
    
    def calc_sigma(self, lambda_t=None, nodes=None):

        if type(lambda_t) == type(None):
            return self.varsigma.get_value(nodes=nodes)

        nodes = self.clean_nodes(nodes=nodes)
        if len(np.shape(lambda_t))==2:
            T = np.shape(lambda_t)[1]
            T_ones = np.ones(T)

            varsigma = np.outer(self.varsigma.get_value(nodes), T_ones)
            eta = np.outer(self.eta.get_value(nodes), T_ones)
            mu = np.outer(self.mu.get_value(nodes), T_ones)

            sigma = varsigma + eta * (lambda_t[nodes,:] - mu)
        
        else:
            if np.size(lambda_t) != nodes.size:
                raise ValueError("lambda_t must be of size nodes.size = {}".format(nodes.size))

            sigma = self.varsigma.get_value(nodes) + self.eta.get_value(nodes)*(lambda_t - self.mu.get_value(nodes))

        return sigma

    def set_sigma(self, lambda_t=None, nodes=None):
        sigma = self.calc_sigma(lambda_t=lambda_t, nodes=nodes)
        self.sigma.set_value(value=sigma, nodes=nodes)


    def set_kappa_parameters(self, kappa_type=None, alpha=None, pull=True, fix=True):
        """
        """
        if kappa_type is not None:
            self.kappa_type.set_value(value=kappa_type)
        if alpha is not None:
            self.alpha.set_value(value=alpha)
        if fix:
            self.set_flat_fix_type()
        if pull:
            self.pull_flat_parameters()


    def gpd_cdf(self, nodes, z):

        xi = self.xi.get_value(nodes)

        is_zero = xi == 0
        if np.any(is_zero):
            F = 1 - np.exp(-z)
            if np.all(is_zero) == False:
                not_zero = np.logical_not(is_zero)
                F[not_zero] = 1 - (1 + xi[not_zero] * z[not_zero])**(-(1/xi[not_zero]))
        else:
            F = 1 - (1 + xi * z)**(-(1/xi))

        #print(xi, u, sigma, G)

        return F

    def gpd_ppf(self, nodes, F):

        xi = self.xi.get_value(nodes)

        is_zero = xi == 0
        if np.any(is_zero):
            z = -np.log(1 - F)
            if np.all(is_zero) == False:
                not_zero = np.logical_not(is_zero)
                z[not_zero] = ((1 - F[not_zero])**(-xi) - 1)/xi
        else:
            z = ((1 - F)**(-xi) - 1)/xi

        return z


    def kappa(self, nodes=0, z=None, unconditional=False):

        z = np.squeeze(z)
        nodes = np.squeeze(nodes)
        if nodes.ndim == 0 and z.ndim > 0:
            nodes = np.full_like(z, nodes, dtype=int) 

        [z, nodes] = parameters.clean_vector(D_nodes=self._D_nodes, value=z, nodes=nodes, name="z", unique=False)

        if np.any(z < 0):
            raise ValueError("z must be greater than or equal to threshold 0")

        kappa = np.zeros(z.size, dtype=float)
        nodes = misc_fns.make_iterable_array(nodes)

        for index in range(z.size):
            if self.kappa_type.get_value(nodes[index]) == "one":
                kappa[index] = 1
            elif self.kappa_type.get_value(nodes[index]) == "exp":
                kappa[index] = np.exp(self.alpha.get_value(nodes[index]) * z[index])
            elif self.kappa_type.get_value(nodes[index]) == "pow":
                kappa[index] = np.power(1 + (z[index]*self.sigma.get_value(nodes[index])/self.u.get_value(nodes[index])), self.alpha.get_value(nodes[index]))
            elif self.kappa_type.get_value(nodes[index]) == "gpd":
                kappa[index] = (1 - self.alpha.get_value(nodes[index]) * np.log(1 - self.gpd_cdf(nodes=nodes[index], z=z[index]))) / (1 + self.alpha.get_value(nodes[index]))
            else:
                raise ValueError("{} {} {}".format(self.kappa_type._name, 
                                                self.kappa_type._p_range._messages["permitted"], 
                                                self.kappa_type._p_range._values["permitted"]))

        return kappa


    def print(self):
        """
        """
        print("")
        print("==================")
        print("HAWKES PARAMETERS")
        print("==================")
        self.print_params_threshold()
        print("")
        print("Intensity parameters:")
        print("a_lambda =", self.a_lambda.get_value())
        print("mu =", self.mu.get_value())
        print("gamma =")
        print(self.gamma.get_value())
        print("")
        print("Time kernel parameters:")
        print("phi_type =", self.phi_type.get_value())
        print("beta =", self.beta.get_value())
        print("nu =", self.nu.get_value())
        print("omega =", self.omega.get_value())
        print("")
        print("Event magnitude parameters:")
        print("xi =", self.xi.get_value())
        print("varsigma =", self.varsigma.get_value())
        print("eta =", self.eta.get_value())
        print("")
        print("Impact function parameters:")
        print("kappa_type =", self.kappa_type.get_value())
        print("alpha =", self.alpha.get_value())
        print("")

    def print_fitted(self):

        for param in self._fitted:
            print(f"{param} = {self._fitted[param]} +/- {self._fitted_se[param]}")



    def set_Bounds(self, delta_excl=None):

        self._Bounds = np.zeros((np.sum(self._flat_index["list"]["size"]), 2))
        for name in self._fp_name:
            self._Bounds[self._flat_index["dict"]["range"][name],:] = self.__dict__[name]._p_range.get_Bounds(delta_excl=delta_excl)


    def set_opt_Bounds(self, delta_excl=None):

        self._opt_Bounds = np.zeros((np.sum(self._flat_index["list"]["size"]), 2))
        for name in self._fp_name:
            self._opt_Bounds[self._flat_index["dict"]["range"][name],:] = self.__dict__[name]._opt_range.get_Bounds(delta_excl=delta_excl)





    






    def set_fitted(self, se=None):

        vals = self.flatten()
        N_params = np.size(vals)
        if se is None:
            se = [None for i in range(N_params)]
        for i in range(N_params):
            name = self._fp_names[i]
            if vals[i] != 0:
                self._fitted[name] = vals[i]
                self._fitted_se[name] = se[i]
            else:
                self._fitted[name] = None
                self._fitted_se[name] = None

        for i in range(self.get_D_nodes()):
            self._fitted[self.mu._names[i]] = self.mu.get_value()[i]
            self._fitted_se[self.mu._names[i]] = None

    if False:
        def set_n(self):
            if self.phi_type.get_value() == "exp":
                self._n = (self.gamma.get_value()/self.beta.get_value())
            elif self.phi_type.get_value() == "pow":
                self._n = (self.gamma.get_value()/(self.nu.get_value()*self.omega.get_value()))
            self.set_n_sd()
            return self._n

        def set_n_sd(self):

            gamma_sd = np.array([[self._fitted_se[self.gamma._names[i][j]] for j in range(self._D_nodes)] for i in range(self._D_nodes)])

            if False:
                alpha_sd = np.zeros(self._D_nodes)
                for i in range(self._D_nodes):
                    if self._fitted_se[self.alpha._names[i]] is not None:
                        alpha_sd[i] = self._fitted_se[self.alpha._names[i]]

            if self.phi_type.get_value() == "exp":
                beta_sd = self._fitted_se["beta"]
                phi_sd = (beta_sd/self.beta.get_value())**2
            elif self.phi_type.get_value() == "pow":
                nu_sd = self._fitted_se["nu"]
                omega_sd = self._fitted_se["omega"]
                phi_sd = (nu_sd/self.nu.get_value())**2 + (omega_sd/self.omega.get_value())**2

            self._n_sd = self._n * np.sqrt(phi_sd + ((gamma_sd/self.gamma.get_value())**2)) #+ ((alpha_sd/(1+self.alpha.get_value()))**2))

            return self._n_sd



    def calc_N_params(self, node=None, tail=None, component=None):

        if component is None:
            component = 'u'

        if tail is not None:
            node = hwks_fns.get_tail_node()[tail]


        flat_free = np.copy(self._flat_free)

        if node is None or self._mode=='ci_s':
            node_names = []
        else:
            if self._mode=='dc':
                node_names = np.copy(self._fp_name)
            else:
                if self._mark:
                    M_names = []
                    arrival_names = []
                else:
                    M_names = ['xi', 'varsigma']
                    if self._mode!='ci' and self._mode!='ci_s':
                        arrival_names = ['a_lambda', 'gamma']
                    else:
                        arrival_names = []

                node_names = np.concatenate([arrival_names, M_names])

        for param in node_names:
            rng = np.array(self._flat_index['dict']['range'][param])
            rng_select = np.full(rng.size, True)
            if param =='gamma':
                #if self._parameters._mode=='dc':
                #    L=1
                #else:
                L = self.get_D_nodes()
            else:
                L=1
            rng_select[node*L:(node+1)*L] = False
            flat_free[rng[rng_select]] = False


        if not self._mark:
            if component=='lambda':
                all_names_extra =  ['xi', 'varsigma']
            elif component=='M':
                all_names_extra = ['a_lambda', 'gamma', 'beta']
            else:
                all_names_extra = []
            all_names = np.concatenate([all_names_extra, ['nu', 'alpha']])
        else:
            all_names = [] 
            

        for param in all_names:
            rng = np.array(self._flat_index['dict']['range'][param])
            flat_free[rng] = False
                
        return np.sum(flat_free)




    



class hawkes_process(parameters.D_nodes):
    """Class for tracking events and time series of Hawkes process
    """

    def __init__(self, data=None, D_nodes=None, t_size=None, dt_dt_fine=None, 
                    exchanges=None, start_date=None,
                    N_max_node=None, N_max_total=None,
                    a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None,
                    mode=None, mark=None, opt_method=None, fix_a_lambda=None,
                    train_t_start=None, train_t_end=None):
        """
        """
        self._D_nodes=None
        if dt_dt_fine is None:
            dt_dt_fine = 1

        if data is None:
            self._df = None
            self._index = None
            self._data = None

            self._exchanges = None
            self._start_date = None

            if D_nodes is None:
                D_nodes = 1
            if dt_dt_fine is None:
                dt_dt_fine = 1
            if t_size is None:
                t_size = 1
        else:
            if type(data) == pd.core.frame.DataFrame or type(data) == pd.core.series.Series:
                self._df = data
                self._index = self._df.index
                self._start_date = self._index[0]

                if np.ndim(self._df) < 2:
                    self._data = np.resize(self._df.values, (1, np.size(self._df)))
                else:
                    self._data = np.transpose(self._df.values)
            else:
                self._data = data
                self._index = None
                self._exchanges = None
                self._start_date = None

         

            shape = np.shape(self._data)
            D_nodes = shape[0]
            t_size = shape[1]


        self._D_nodes = parameters.clean_D_nodes(D_nodes=D_nodes)
        self._parameters = hawkes_parameters(D_nodes=self.get_D_nodes(), a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, opt_method=opt_method, fix_a_lambda=fix_a_lambda, mode=mode, mark=mark)


        self.set_t_size(t_size)
        self.set_dt_dt_fine(dt_dt_fine)
        self.set_t()


        if self._start_date is None:
            if start_date is None:
                self._start_date = pd.Timestamp("1970-01-01")
            else:
                self._start_date = pd.Timestamp(start_date)


        if exchanges is not None:
            if np.shape(exchanges)==() and np.size(exchanges)==1:
                exchanges = [exchanges for i in range(self.get_D_nodes())]
            if len(exchanges) != self.get_D_nodes():
                raise ValueError("exchanges must be of length D_nodes = {}".format(self.get_D_nodes()))
            self._exchanges = exchanges

            self._calenders = []
            self._schedules = []
            indicies = []
            for i in range(self.get_D_nodes()):
                [calender, schedule] = fin_data_handl.get_calender_schedule(self._exchanges[i], start_date=self._start_date)
                self._calenders.append(copy.deepcopy(calender))
                self._schedules.append(copy.deepcopy(schedule))
                indicies.append(self._schedules[i].index)


            if self._index is None:
                self._index = functools.reduce(np.intersect1d, (indicies))[:self.get_t_size()]

            self._schedules_index = [self._schedules[i].loc[self._index] for i in range(self.get_D_nodes())]
        else:
            self._exchanges = exchanges = [None for i in range(self.get_D_nodes())]
            self._schedules_index = None

        

        self.set_TS()
        self.set_train_t(start=train_t_start, end=train_t_end, reset=True, set_train_events_bool=False)

        # Set process parameters
        self.set_N_max_node(N_max_node)
        self.set_N_max_total(N_max_total)

        self.reset_process()

        self.set_u()
        self.detect_events()


        self._fitted_scores = misc_fns.IC_dict()

        self._hb = None


 #--#===========================
    #--------------------------
    # base paramaeter functions
    #--------------------------
    #===========================

    
    def set_new_data(self, data, reset_u=True):

        if type(data) == pd.core.frame.DataFrame or type(data) == pd.core.series.Series:
            self._df = data
            self._index = self._df.index
            self._start_date = self._index[0]

            if np.ndim(self._df) < 2:
                self._data = np.resize(self._df.values, (1, np.size(self._df)))
            else:
                self._data = np.transpose(self._df.values)
        else:
            self._data = data

        if reset_u:
            self.set_u()
        self.detect_events()




    def set_t_size(self, t_size):
        """Set function for t_size
                -- Number of elements in event-resolution discrete time series
        """
        if np.isin(type(t_size), [int]) == False:
            raise TypeError("t_size must be of type: int")
        if t_size < 1:
            raise ValueError("t_size must be greater than or equal to 1")
        self._t_size = t_size

    def get_t_size(self):
        """Get function for t_size
                -- the end-time of the point processes (the start-time is defined to be 0)
        """
        return self._t_size


    def set_dt_dt_fine(self, dt_dt_fine):
        """Set function for dt
                -- the discrete time-step of the point processes
        """
        if np.isin(type(dt_dt_fine), [int]) == False:
            raise TypeError("dt_dt_fine must be of type: int")
        if dt_dt_fine < 1:
            raise ValueError("dt must be greater than or equal to 1")
        self._dt_dt_fine = dt_dt_fine
        self._dt_fine = 1/self._dt_dt_fine

    def get_dt_dt_fine(self):
        """Get function for dt
                -- the discrete time-step of the point processes
        """
        return self._dt_dt_fine

    def get_dt_fine(self):
        """Get function for dt
                -- the discrete time-step of the point processes
        """
        return self._dt_fine

    def get_t_fine_size(self):
        """Get function for T
                -- the end-time of the point processes (the start-time is defined to be 0)
        """
        return self._t_fine_size
        

    def set_t(self):
        """Set function for t_vec
                -- the vector of all discrete times within the considered time window

        Also implicitly sets T_index
                -- the index of the final element of t_vec, corresponing to time T
        """

        self._t = np.array(range(self.get_t_size()))





    def get_t(self, t_index=None):
        """Get function for t_vec
                -- the vector of all discrete times within the considered time window
        """
        if type(t_index) == type(None):
            return self._t
        else:
            return self._t[t_index]





    def set_TS(self, set_t_fine=False):

        self._TS =  {
                        "t": None,
                        "t_fine": None,
                    }

        if self._schedules_index is not None:
            self._TS["t"] = np.array([[self._schedules_index[i].loc[self._index[k],"market_close"] for k in range(self._index.size)] for i in range(self.get_D_nodes())])
            fail = np.logical_not(self._TS["t"] >= pd.Timestamp("1678", tz='utc'))
            if np.any(fail):
                wherefail = np.where(fail)
                for i in range(self.get_D_nodes()):
                    N_fail = np.sum(wherefail[0]==i)
                    deltas = np.array([self._TS["t"][0, misc_fns.dropint(wherefail[1][wherefail[0]==i])][k] - pd.Timestamp(self._TS["t"][0, misc_fns.dropint(wherefail[1][wherefail[0]==i])][k].strftime(format="%Y-%m-%d"), tz='utc') for k in range(N_fail)])
                    self._TS["t"][i, wherefail[1][wherefail[0]==i]] = np.array([pd.Timestamp(self._df.index[wherefail[1][wherefail[0]==i]][k], tz='utc') for k in range(N_fail)]) + deltas
        else:
            self._TS["t"] = np.array([self._index for i in range(self.get_D_nodes())])

                
        if set_t_fine:
            self.set_t_fine()


    def set_t_fine(self):
            
        TS_dt_fine = np.full(self.get_dt_dt_fine(), np.timedelta64(int(np.timedelta64(1, 'D')/np.timedelta64(self.get_dt_dt_fine(), 'ms')), 'ms'))
        cumsum_dt = -np.flip(np.insert(np.cumsum(TS_dt_fine), 0, TS_dt_fine[0] - TS_dt_fine[0]))
        self._TS["t_fine"] = np.array([[self._TS["t"][i,k] + pd.to_timedelta(cumsum_dt) for k in range(self._index.size)] for i in range(self.get_D_nodes())], dtype=object)


    def set_train_t(self, start=None, end=None, reset=False, set_train_events_bool=True):

        if reset:
            self._train_t = np.array([0, self.get_t_size()])
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
        if set_train_events_bool:
            self.set_train_events_bool()

    
    def set_train_events_bool(self):
        self._train_events_bool = np.logical_and(self._events["t"]>=self._train_t[0], self._events["t"]<self._train_t[1])


    def set_N_max_node(self, N_max_node=None):
        """Set function for N_max_node
                -- the maximum number of events in a process for simulations
        """
        if N_max_node == None:
            N_max_node = self.get_t_size()
        else:
            if type(N_max_node) != int: 
                raise ValueError("N_max_node must be of type int or None")
            if N_max_node != None and N_max_node < 1:
                raise ValueError("N_max_node must be greater than or equal to 1")
        self._N_max_node = N_max_node

    def get_N_max_node(self):
        return self._N_max_node


    def set_N_max_total(self, N_max_total=None):
        """Set function for N_max_total
                -- the maximum number of total events for simulations
        """
        if N_max_total == None:
            N_max_total = self.get_N_max_node() * self.get_D_nodes()
        else:
            if type(N_max_total) != int: 
                raise ValueError("N_max_node must be of type int or None")
            if N_max_total < 1:
                raise ValueError("N_max_node must be greater than or equal to 1")
        self._N_max_total = N_max_total

    def get_N_max_total(self):
        return self._N_max_total



 #--#========================
    #-----------------------
    # data functions
    #-----------------------
    #=======================

    def exceedance_transform(self, node, data=None):

        if data is None:
            data = self._data[node,:]
        return misc_fns.exceedance_transform(data, u_dir=self._parameters.u_dir.get_value()[node],
                                            u=self._parameters.u.get_value()[node], 
                                            u_zero=self._parameters.u_zero.get_value()[node])

    def exceedance_transform_inv(self, node, data=None):

        if data is None:
            data = self._events['M'][node,:self._events["N"][node]]

        return misc_fns.exceedance_transform_inv(data, u_dir=self._parameters.u_dir.get_value()[node],
                                                u=self._parameters.u.get_value()[node], 
                                                u_zero=self._parameters.u_zero.get_value()[node])


    def calc_u_quantiles(self, quantiles=None, quantiles_methods=None, nodes=None):
        """
        """

        [quantiles, nodes_quantiles] = parameters.clean_vector(D_nodes=self.get_D_nodes(), value=quantiles, nodes=nodes, unique=False, name="quantiles")
        [quantiles_methods, nodes_methods] = parameters.clean_vector(D_nodes=self.get_D_nodes(), value=quantiles_methods, nodes=nodes, unique=False, name="quantiles_methods")

        if np.array_equal(nodes_quantiles, nodes_methods) == False:
            raise ValueError("quantiles and quantiles_methods do not match")

        u_quantile = np.full(nodes_quantiles.size, np.nan)

        for i in range(nodes_quantiles.size):
            data_node = self._data[nodes_quantiles[i],:]

            if quantiles_methods[i] == "q":
                u_quantile[i] = np.quantile(a=data_node, q=quantiles[i])
            elif quantiles_methods[i] == "q_abs":
                u_quantile[i] = np.quantile(a=np.abs(data_node), q=quantiles[i])
            elif quantiles_methods[i] == "q_loss":
                u_quantile[i] =  np.quantile(a=data_node[data_node<0], q=1-quantiles)
            elif quantiles_methods == "q_gain":
                u_quantile[i] = np.quantile(a=data_node[data_node>0], q=quantiles)
            else:
                raise ValueError("Invalid quantiles_methods: '{}'".format(quantiles_methods[i]))

            return u_quantile





    def set_u(self, set_zero=True):

        if self._data is not None:

            q = np.outer(np.ones(2), self._parameters.a_u.get_value())
            q_abs = q/2
            q[1,:] = 1-q[0,:]
            q_abs[1,:] = 1-q_abs[0,:]

            u = np.full((2, self.get_D_nodes()), np.nan)
            u_abs = np.copy(u)

            u_final = np.full(self.get_D_nodes(), np.nan)
            zero_final = np.zeros(self.get_D_nodes())

            for i in range(self.get_D_nodes()):
                if self._parameters.u_dir.get_value()[i] == "abs":
                    u_abs[:,i] = np.quantile(self._data[i,self._train_t[0]:self._train_t[1]], q_abs[:,i])
                    u_final[i] = np.diff(u_abs[:,i])[0]/2
                    zero_final[i] = np.sum(u_abs[:,i])/2     
                else:
                    u[:,i] = np.quantile(self._data[i,self._train_t[0]:self._train_t[1]], q[:,i])
                    if self._parameters.u_dir.get_value()[i] == "left":
                        u_final[i] = u[0,i]
                    elif self._parameters.u_dir.get_value()[i] == "right":
                        u_final[i] = u[1,i]
                

            self._parameters.u.set_value(u_final)
            if set_zero:
                self._parameters.u_zero.set_value(zero_final)






    def detect_events(self):

        if self._data is not None:

            self.set_N_max_node()    
            self.set_N_max_total()
            self.reset_events()

            if False:
                for i in range(self.get_D_nodes()):
                    
                    data = self.exceedance_transform(node=i)
                    events_t = np.where(data > 0)[0]
                    N = events_t.size
                    self._events["N"][i] = N
                    self._events["t"][i, 0:N] = events_t
                    self._events["TS"][i, 0:N] = self._TS["t"][i, events_t]
                    self._events['M'][i, 0:N] = data[events_t]
            else:
                u_dir_sign = np.array([hwks_fns.get_u_dir_sign(u_dir) for u_dir in self._parameters.u_dir.get_value()])
                events = dist_fns.calc_events(x=self._df[0].values, 
                                            a_u=self._parameters.a_u.get_value(), u=self._parameters.u.get_value(), u_dir=u_dir_sign, u_zero=self._parameters.u_zero.get_value(), 
                                            as_array=True, first_u=True, override_u2=True)
                
                for e in events:
                    self._events[e] = events[e]
                    for i in range(self.get_D_nodes()):
                        self._events["TS"][i, :self._events["N"][i]] = self._TS["t"][i, self._events["t"][i, :self._events["N"][i]]]
                        
            self.shrink_events()
            self.set_events_N_t()
            self.set_t_unique()
            self.set_train_events_bool()
            self.set_tail_fit()


    def set_tail_fit(self):

        u_dir_sign = np.array([hwks_fns.get_u_dir_sign(u_dir) for u_dir in self._parameters.u_dir.get_value()])

        self._tail_fit = {
                            'fix_a_lambda':     {
                                                False:   dist_fns.calc_tail_fit_atf(x=self._df[0].values, train_t=self._train_t, 
                                                            a_u=np.transpose([self._parameters.a_u.get_value()]), u=np.transpose([self._parameters.u.get_value()]), 
                                                            u_dir=np.transpose([u_dir_sign]), u_zero=np.transpose([self._parameters.u_zero.get_value()]),
                                                            incl_M_stat=True, incl_gp_fit=True, fa_lambda=None, fscale=None, fc=None, fNparams_free=None),
                                                True:   dist_fns.calc_tail_fit_atf(x=self._df[0].values, train_t=self._train_t, 
                                                            a_u=np.transpose([self._parameters.a_u.get_value()]), u=np.transpose([self._parameters.u.get_value()]), 
                                                            u_dir=np.transpose([u_dir_sign]), u_zero=np.transpose([self._parameters.u_zero.get_value()]),
                                                            incl_M_stat=True, incl_gp_fit=True, fa_lambda=np.transpose([self._parameters.a_u.get_value()]), fscale=None, fc=None, fNparams_free=None),
                                                }
                        }



 #--#========================
    #-----------------------
    # process functions
    #-----------------------
    #=======================


    def reset_process(self):

        self.reset_events()
        self.reset_t_series()
    

    def reset_events(self):

        base_float = np.full((self.get_D_nodes(), self.get_N_max_node()), np.nan, dtype=float)
        base_int = np.full((self.get_D_nodes(), self.get_N_max_node()), -(self.get_t_size()+1), dtype=int)

        # dist_fns.calc_events -> dict_keys(['t', 'tilde_t_a_u', 'tilde_t_a_lambda', 'delta_t', 'M', 'N', 'a_lambda', 'N_t'])

        # lambda_tilde_t
        self._events =  {
                            "t": np.copy(base_int),                 # dist_fns.calc_events
                            "TS": np.full((self.get_D_nodes(), self.get_N_max_node()), pd.Timestamp("2199-12-31", tz="UTC")),
                            'M': np.copy(base_float), # dist_fns.calc_events
                            'Y_M': np.copy(base_float),
                            'tilde_M': np.copy(base_float),
                            'F_M': np.copy(base_float),
                            "kappa": np.copy(base_float),
                            "tilde_t": np.copy(base_float),
                            "tilde_t_eff": np.copy(base_float),
                            "tilde_t_all": None,
                            "sigma": np.copy(base_float),
                            "lambda": np.copy(base_float),
                            #"tilde_t_t": np.copy(base_float),
                            "N": np.full(self.get_D_nodes(), 0, dtype=int), # dist_fns.calc_events
                            "N_t": np.full((self.get_D_nodes(), self.get_t_size()), 0, dtype=int), # dist_fns.calc_events
                            "list": [],
                            "t_unique": None,
                            "tilde_t_t_unique": None,
                        }


    def shrink_events(self, N_max_node=None):
        # Shrink (i.e. remove NaNs) from unfilled events arrays
        if N_max_node == None:
            N_max_node = max(self._events["N"])

        for i in self._events:
            #if i!="tilde_t_all" and i!="N" and i!="N_t" and i!="list" and i!="t_unique" and i!="tilde_t_t_unique" and i!='tilde_t_a_u' and i!='tilde_t_a_lambda' and i!='a_lambda':
            if not np.isin(i, ["tilde_t_all", "N", "N_t", "list", "t_unique", "tilde_t_t_unique", "tilde_t_a_u", "tilde_t_a_lambda", "a_lambda"]):
                self._events[i] = self._events[i][:, 0:N_max_node]

    def set_t_unique(self):
        self._events["t_unique"] = np.unique(self._events["t"][self._events["t"] >= 0])


    def get_events_2d(self, name, nodes=None, k=None):

        if type(name) != str:
            raise TypeError("name must be of type str")

        if type(nodes) == type(None) and type(k) == type(None):
            return self._events[name]
        elif type(nodes) == type(None) and type(k) != type(None):
            return self._events[name][:,k]
        elif type(nodes) != type(None) and type(k) == type(None):
            if np.size(nodes) == 1 and self.get_events_N() != 0:
                return self._events[name][nodes,0:self.get_events_N()]
            else:
                return self._events[name][nodes,:]
        else:
            return self._events[name][nodes,k]

    def get_events(self):
        return self._events

    def get_events_t(self, nodes=None, k=None):
        return self.get_events_2d("t", nodes=nodes, k=k)

    def get_events_m(self, nodes=None, k=None):
        return self.get_events_2d('M', nodes=nodes, k=k)

    def get_events_kappa(self, nodes=None, k=None):
        return self.get_events_2d("kappa", nodes=nodes, k=k)

    def get_events_tilde_t(self, nodes=None, k=None):
        return self.get_events_2d("tilde_t", nodes=nodes, k=k)

    def get_events_N(self, nodes=None):
        if type(nodes) == type(None):
            return self._events["N"]
        else:
            nodes = self.clean_nodes(nodes=nodes)
            return self._events["N"][nodes]

    def get_events_N_t(self, nodes=None, t_index=None):
        if type(nodes) == type(None) and type(t_index) == type(None):
            return self._events["N_t"]
        elif type(nodes) == type(None) and type(t_index) != type(None):
            return self._events["N_t"][:,t_index]
        elif type(nodes) != type(None) and type(t_index) == type(None):
            return self._events["N_t"][nodes,:]
        else:
            return self._events["N_t"][nodes,t_index]

    def set_events_N_t(self):
        events_N_diff = np.full((self.get_D_nodes(), self.get_t_size()), 0, dtype=int)
        for i in range(self.get_D_nodes()):
            events_N_diff[i, self._events["t"][i,0:self._events["N"][i]]] = 1
        self._events["N_t"] = np.cumsum(events_N_diff, axis=1)


    def get_events_list(self, k=None):
        if type(k) == type(None):
            return self._events["list"]
        else:
            return self._events["list"][k]  


    def reset_t_series(self):
        """
        """
        self._t_series = self.new_t_series()


    def new_t_series(self):
        """
        """

        base_float = np.full((self.get_D_nodes(), self.get_t_size()), np.nan)

        t_series =    {
                            "lambda": np.copy(base_float),
                            "sigma": np.copy(base_float),
                            "tilde_t": np.copy(base_float),
                            "delta_tilde_t": np.copy(base_float),
                            "p_event": np.copy(base_float),
                            "lambda_fine": None,
                            "chi_fine": None,
                        }

        return t_series


    def get_t_series(self):
        return self._t_series

    def get_t_series_2d(self, name, nodes=None, t=None):

        if type(name) != str:
            raise TypeError("name must be of type str")

        if type(nodes) == type(None) and type(t) == type(None):
            return self._t_series[name]
        elif type(nodes) == type(None) and type(t) != type(None):
            return self._t_series[name][:,t]
        elif type(nodes) != type(None) and type(t) == type(None):
            return self._t_series[name][nodes,:]
        else:
            return self._t_series[name][nodes,t]

    #def get_t_series_lambda(self, nodes=None, t=None):
    #    return self.get_t_series_2d("lambda", nodes=nodes, t=t)

    def get_t_series_p_event(self, nodes=None, t=None):
        return self.get_t_series_2d("p_event", nodes=nodes, t=t)


 #--#========================
    #-----------------------
    # forecast functions
    #-----------------------
    #=======================

    def lambda_forecast_disc(self, t, N=None, t_present=None, disc=None):

        #num_cores = multiprocessing.cpu_count()

        if disc is None:
            disc = True

        if N is not None:
            if N.size != self.get_D_nodes():
                raise ValueError("N must be of size D_nodes")
        else:
            if t_present is None:
                t_present = np.min(t)
            
            N = np.zeros(self.get_D_nodes(), dtype=int)
            for i in range(self.get_D_nodes()):
                past_events_index = np.logical_not(np.isnan(self._events["t"][i,:]))
                past_events_index[past_events_index] = self._events["t"][i,past_events_index] < t_present
                N[i] = np.sum(past_events_index)

        t = np.array(t)
        if t.ndim > 1:
            raise ValueError("t must be of ndim 0 or 1")

        is_phi_exp = self._parameters.phi_type.get_value() == "exp"

        if disc:
            return lambda_forecast_fast_disc(D_nodes=self.get_D_nodes(), 
                                        mu=self._parameters.mu.get_value(), 
                                        gamma=self._parameters.gamma.get_value(), 
                                        events_t=self._events["t"], 
                                        events_kappa=self._events["kappa"], 
                                        t=t, 
                                        N=N,
                                        is_phi_exp=is_phi_exp, 
                                        beta=self._parameters.beta.get_value(),
                                        nu=self._parameters.nu.get_value(),
                                        omega=self._parameters.omega.get_value(),
                                        )
        
        else:
            return lambda_forecast_fast_cont(D_nodes=self.get_D_nodes(), 
                                        mu=self._parameters.mu.get_value(), 
                                        gamma=self._parameters.gamma.get_value(), 
                                        events_t=self._events["t"], 
                                        events_kappa=self._events["kappa"], 
                                        t=t, 
                                        N=N,
                                        is_phi_exp=is_phi_exp, 
                                        beta=self._parameters.beta.get_value(),
                                        nu=self._parameters.nu.get_value(),
                                        omega=self._parameters.omega.get_value(),
                                        )

    

    def tilde_t_forecast(self, t, N=None, t_present=None):

        #num_cores = multiprocessing.cpu_count()

        if N is not None:
            if N.size != self.get_D_nodes():
                raise ValueError("N must be of size D_nodes")
        else:
            if t_present is None:
                t_present = np.min(t)
            
            N = np.zeros(self.get_D_nodes(), dtype=int)
            for i in range(self.get_D_nodes()):
                past_events_index = np.logical_not(np.isnan(self._events["t"][i,:]))
                past_events_index[past_events_index] = self._events["t"][i,past_events_index] < t_present
                N[i] = np.sum(past_events_index)

        t = np.array(t)
        if t.ndim > 1:
            raise ValueError("t must be of ndim 0 or 1")

        is_phi_exp = self._parameters.phi_type.get_value() == "exp"

        return tilde_t_forecast_fast(D_nodes=self.get_D_nodes(), 
                                    mu=self._parameters.mu.get_value(), 
                                    gamma=self._parameters.gamma.get_value(), 
                                    events_t=self._events["t"], 
                                    events_kappa=self._events["kappa"], 
                                    t=t, 
                                    N=N,
                                    is_phi_exp=is_phi_exp, 
                                    beta=self._parameters.beta.get_value(),
                                    nu=self._parameters.nu.get_value(),
                                    omega=self._parameters.omega.get_value(),
                                    )
    #def calc_chi_node(self, node, t, N):
    #    lags = np.outer(t, np.ones(N[node])) - self._events["t"][node, 0:N[node]]
    #    chis_lags_events = self._parameters.phi(lags) * self._events["kappa"][node, 0:N[node]]
    #    return np.sum(chis_lags_events, axis=1)

    
#--#========================
    #-----------------------
    # estimation functions
    #-----------------------
    #=======================

    def get_events_bool(self, t_start=None, t_end=None, train_t=False, post_train_t=False, node=None):

        if train_t:
            t_start=self._train_t[0]
            t_end=self._train_t[1]
            #events_bool = self._train_events_bool
        elif post_train_t:
                t_start=self._train_t[1]
                t_end=self.get_t_size()
        else:
            if t_start is None:
                t_start=0
            if t_end is None:
                t_end=self.get_t_size()
        events_bool = np.logical_and(self._events["t"]>=t_start, self._events["t"]<t_end)

        if node is not None:
            for i in range(self.get_D_nodes()):
                if i!=node:
                    events_bool[i,:] = False

        return t_start, t_end, events_bool

    
    def set_tilde_t_2(self, is_phi_exp=True, t_start=None, t_end=None, train_t=False):

        t_start, t_end, events_bool = self.get_events_bool(t_start=t_start, t_end=t_end, train_t=train_t)
        events_N = np.sum(events_bool, axis=1)

        N_t = np.sum(np.logical_and(self._events["t"]>=0, self._events["t"]<t_start), axis=1)
        t_unique_0 = self._events["t_unique"][np.logical_and(self._events["t_unique"]>=t_start, self._events["t_unique"]<t_end)]

        for k in range(t_unique_0.size):
            triggered = np.array([np.isin(t_unique_0[k], self._events["t"][i,events_bool[i,:]]) for i in range(self.get_D_nodes())])
            # lambda_tilde_t
            lambda_t = self.lambda_forecast_disc(t=np.array([t_unique_0[k]]), N=N_t)[:,0]
            sigma_t = self._parameters.calc_sigma(lambda_t = lambda_t)
            #tilde_t_t = np.diff(self.tilde_t_forecast(t=np.array([t_unique_0[k]-1, t_unique_0[k]]), N=N_t), axis=1)
            #sigma_t = self._parameters.calc_sigma(tilde_t_t = tilde_t_t)

            for i in np.where(triggered)[0]:
                #print(N_t)
                #import pdb
                #pdb.set_trace()
                # lambda_tilde_t
                self._events["lambda"][i, N_t[i]] = lambda_t[i]
                #self._events["tilde_t_t"][i, N_t[i]] = tilde_t_t[i]

                self._events["sigma"][i, N_t[i]] = sigma_t[i]
                #self._parameters.sigma._value[i] = self._events["sigma"][i, N_t[i]]
                self._events['Y_M'][i, N_t[i]] = self._events['M'][i, N_t[i]]/self._events["sigma"][i, N_t[i]]
                self._events["kappa"][i, N_t[i]] = self._parameters.kappa(nodes=int(i), z=self._events['Y_M'][i, N_t[i]])
            
            N_t[triggered] +=1

        t_unique = np.unique(np.concatenate(([t_start-1], t_unique_0, [t_end])))
        cumulative_chi = np.zeros(shape=(self.get_D_nodes(), t_unique.size-1, np.max(events_N)), dtype=float)
        present = [np.where(np.isin(t_unique[1:], self._events["t"][i,events_bool[i,:]]))[0] for i in range(self.get_D_nodes())]

        for i in range(self.get_D_nodes()):
            for k in range(events_N[i]):
                cumulative_chi[i, 1+present[i][k]:, k] = np.diff(self._parameters.int_phi(t_unique[1+present[i][k]:] - self._events["t"][i,k], node=i)) * self._events["kappa"][i,k]

        self._events["tilde_t_t_unique"] = np.cumsum(np.outer(self._parameters.mu.get_value(), np.diff(t_unique)) + self._parameters.gamma.get_value() @ np.sum(cumulative_chi, axis=2), axis=1)
        tilde_t_all_t_unique = np.sum(self._events["tilde_t_t_unique"], axis=0)

        for i in range(self.get_D_nodes()):
            self._events["tilde_t"][i,events_bool[i,:]] = self._events["tilde_t_t_unique"][i, present[i]]
            self._events['F_M'][i,:] = self._parameters.gpd_cdf(nodes=i, z=self._events['Y_M'][i,:])

        self._events["tilde_t_all"] = np.sort(np.concatenate([tilde_t_all_t_unique[present[i]] for i in range(self.get_D_nodes())]))

        self._log_likelihood = self.calc_log_L_2(t_start=t_start, t_end=t_end, train_t=train_t)





    def set_triggers(self, is_phi_exp=True, t_start=None, t_end=None, train_t=False):

        t_start, t_end, events_bool = self.get_events_bool(t_start=t_start, t_end=t_end, train_t=train_t)
        events_N = np.sum(events_bool, axis=1)

        N_t = np.sum(np.logical_and(self._events["t"]>=0, self._events["t"]<t_start), axis=1)
        t_unique_0 = self._events["t_unique"][np.logical_and(self._events["t_unique"]>=t_start, self._events["t_unique"]<t_end)]

        for k in range(t_unique_0.size):
            triggered = np.array([np.isin(t_unique_0[k], self._events["t"][i,events_bool[i,:]]) for i in range(self.get_D_nodes())])
            # lambda_tilde_t
            lambda_t = self.lambda_forecast_disc(t=np.array([t_unique_0[k]]), N=N_t)[:,0]
            sigma_t = self._parameters.calc_sigma(lambda_t = lambda_t)
            #tilde_t_t = np.diff(self.tilde_t_forecast(t=np.array([t_unique_0[k]-1, t_unique_0[k]]), N=N_t), axis=1)[:,0]
            #sigma_t = self._parameters.calc_sigma(tilde_t_t = tilde_t_t)


            for i in np.where(triggered)[0]:

                # lambda_tilde_t
                self._events["lambda"][i, N_t[i]] = lambda_t[i]
                #self._events["tilde_t_t"][i, N_t[i]] = tilde_t_t[i]

                self._events["sigma"][i, N_t[i]] = sigma_t[i]
                #self._parameters.sigma._value[i] = self._events["sigma"][i, N_t[i]]
                self._events['Y_M'][i, N_t[i]] = self._events['M'][i, N_t[i]]/self._events["sigma"][i, N_t[i]]
                self._events["kappa"][i, N_t[i]] = self._parameters.kappa(nodes=int(i), z=self._events['Y_M'][i, N_t[i]])
            
            N_t[triggered] +=1

    

    def set_tilde_t(self, is_phi_exp=True, t_start=None, t_end=None, train_t=False):

        t_start, t_end, events_bool = self.get_events_bool(t_start=t_start, t_end=t_end, train_t=train_t)
        events_N = np.sum(events_bool, axis=1)

        t_unique_0 = self._events["t_unique"][np.logical_and(self._events["t_unique"]>=t_start, self._events["t_unique"]<t_end)]

        t_unique = np.unique(np.concatenate(([t_start-1], t_unique_0, [t_end])))
        cumulative_chi = np.zeros(shape=(self.get_D_nodes(), t_unique.size-1, np.max(events_N)), dtype=float)
        present = [np.where(np.isin(t_unique[1:], self._events["t"][i,events_bool[i,:]]))[0] for i in range(self.get_D_nodes())]

        for i in range(self.get_D_nodes()):
            for k in range(events_N[i]):
                cumulative_chi[i, 1+present[i][k]:, k] = np.diff(self._parameters.int_phi(t_unique[1+present[i][k]:] - self._events["t"][i,k], node=i)) * self._events["kappa"][i,k]

        self._events["tilde_t_t_unique"] = np.cumsum(np.outer(self._parameters.mu.get_value(), np.diff(t_unique)) + self._parameters.gamma.get_value() @ np.sum(cumulative_chi, axis=2), axis=1)
        tilde_t_all_t_unique = np.sum(self._events["tilde_t_t_unique"], axis=0)

        for i in range(self.get_D_nodes()):
            self._events["tilde_t"][i,events_bool[i,:]] = self._events["tilde_t_t_unique"][i, present[i]]
            self._events['F_M'][i,:] = self._parameters.gpd_cdf(nodes=i, z=self._events['Y_M'][i,:])
            self._events['tilde_M'][i,:] = scipy.stats.expon().ppf(self._events['F_M'][i,:])

        self._events["tilde_t_all"] = np.sort(np.concatenate([tilde_t_all_t_unique[present[i]] for i in range(self.get_D_nodes())]))





    def set_t_series(self):

        self.reset_t_series()
        self._log_L = self.calc_log_L()
        self.set_tilde_t()
        self.set_residual_KS()
        self.fill_t_series()
        self._fitted_scores = misc_fns.IC_dict(log_L=self._log_L, N_params=self._parameters._N_params, N_data=2*np.sum(self._events["N"]))
        #self.set_tt_eval_blr_atf()

    def set_t_series_2(self):

        self.reset_t_series()
        self.set_tilde_t_2()
        self.set_residual_KS()
        self.fill_t_series()


    def fill_t_series(self):

        t_unique = np.unique(np.concatenate((self._events["t_unique"], [self.get_t()[-1]])))

        t_initial_event = np.min(t_unique)
        T = self._t_series["lambda"].shape[1]

        if False:
            N_t = np.zeros(shape=self.get_D_nodes(), dtype=int)
            # lambda_tilde_t
            for t in range(self._t_series["lambda"].shape[1]):
                triggered = np.any(self._events["t"] == t, axis=1)
                self._t_series["lambda"][:,t] = self.lambda_forecast_disc(t=np.array([t]), N=N_t)[:,0]
                self._t_series["sigma"][:,t] = self._parameters.calc_sigma(lambda_t = self._t_series["lambda"][:,t])
                N_t[triggered] +=1
        else:
            N_t = np.zeros(shape=self.get_D_nodes(), dtype=int)
            # lambda_tilde_t
            t_0 = 0
            for t in t_unique:
                t_rng = np.arange(t_0, t+1)
                triggered = np.any(self._events["t"] == t, axis=1)
                self._t_series["lambda"][:,t_0:t+1] = self.lambda_forecast_disc(t=t_rng, N=N_t)
                self._t_series["sigma"][:,t_0:t+1] = self._parameters.calc_sigma(lambda_t = self._t_series["lambda"][:,t_0:t+1])
                N_t[triggered] +=1
                t_0 = t+1
            if t_0<T:
                t=T
                t_rng = np.arange(t_0, t+1)
                self._t_series["lambda"][:,t_0:t+1] = self.lambda_forecast_disc(t=t_rng, N=N_t)
                self._t_series["sigma"][:,t_0:t+1] = self._parameters.calc_sigma(lambda_t = self._t_series["lambda"][:,t_0:t+1])




        if False:

            cumulative_chi = np.zeros(shape=(self.get_D_nodes(), self.get_t_size(), np.max(self._events["N"])), dtype=float)
            for i in range(self.get_D_nodes()):
                for k in range(self._events["N"][i]):
                    cumulative_chi[i, 1+self._events["t"][i,k]:, k] = (1 + self._parameters.int_phi(np.array(range(1, self.get_t_size()-self._events["t"][i,k])), node=i)) * self._events["kappa"][i,k]
 
        
            self._t_series["tilde_t"] = np.cumsum(np.outer(self._parameters.mu.get_value(), np.ones(self.get_t_size())), axis=1) + self._parameters.gamma.get_value() @ np.sum(cumulative_chi, axis=2)
            self._t_series["delta_tilde_t"][:,0] = self._t_series["tilde_t"][:,0]
            self._t_series["delta_tilde_t"][:,1:] = np.diff(self._t_series["tilde_t"], axis=1)
            self._t_series["p_event"] = np.transpose(self._parameters.get_M_weight()) @ (1 - np.exp(-self._parameters.get_M_sum() @ self._t_series["delta_tilde_t"]))
            self._t_series["tilde_t_eff"] = np.cumsum(-np.log(1-self._t_series["p_event"]), axis=1)
        else:
            self._t_series["tilde_t"] = np.cumsum(self._t_series["lambda"], axis=1)
            self._t_series["p_event"] = np.transpose(self._parameters.get_M_weight()) @ (1 - np.exp(-self._parameters.get_M_sum() @ self._t_series["lambda"]))
            self._t_series["tilde_t_eff"] = np.cumsum(-np.log(1-self._t_series["p_event"]), axis=1)

        # lambda_tilde_t
        #self._t_series["sigma"][:,k] = self._parameters.calc_sigma(tilde_t_t = self._t_series["delta_tilde_t"][:,k])
    
    def fill_t_series_fine(self):

        self._t_series["chi_fine"] = np.zeros(shape=(self.get_D_nodes(), self.get_t_size(), 1+self.get_dt_dt_fine()), dtype=float)
        t_grid = np.outer(np.array(range(self._t_series["chi_fine"].shape[1])), np.ones(self._t_series["chi_fine"].shape[2])) + np.outer(np.ones(self._t_series["chi_fine"].shape[1]), np.linspace(0,1,self._t_series["chi_fine"].shape[2]))
        phi_grid = np.array([self._parameters.phi(lag=t_grid, node=i) for i in range(self.get_D_nodes())])
        for i in range(self.get_D_nodes()):
            for k in range(self._events["N"][i]):
                if self._events["t"][i,k] != self.get_t_size():
                    self._t_series["chi_fine"][i,1+self._events["t"][i,k]:,:] += self._events["kappa"][i,k] * phi_grid[i,:self.get_t_size()-self._events["t"][i,k]-1,:]
        self._t_series["lambda_fine"] = np.reshape(np.outer(self._parameters.mu.get_value(), np.ones(self._t_series["chi_fine"].shape[1]*self._t_series["chi_fine"].shape[2])) + self._parameters.gamma.get_value() @ np.reshape(self._t_series["chi_fine"], (self._t_series["chi_fine"].shape[0], self._t_series["chi_fine"].shape[1]*self._t_series["chi_fine"].shape[2])), (self._t_series["chi_fine"].shape[0], self._t_series["chi_fine"].shape[1], self._t_series["chi_fine"].shape[2]))


    def calc_tilde_t_interval(self, t_start=None, t_end=None, train_t=False):

        t_start, t_end, events_bool = self.get_events_bool(t_start=t_start, t_end=t_end, train_t=train_t)

        life = (t_end-1) - self._events["t"]
        cumulative_chi_event = np.full(self._events["t"].shape, np.nan)
        for i in range(2):
            cumulative_chi_event[i,events_bool[i]] = (self._parameters.int_phi(life[i,events_bool[i]], node=i) - self._parameters.int_phi(0, node=i))*self._events["kappa"][i,events_bool[i]]
        cumulative_chi_node = np.array([np.sum(cumulative_chi_event[i,events_bool[i]]) for i in range(self.get_D_nodes())])

        return (t_end-t_start)*self._parameters.mu.get_value() + self._parameters.gamma.get_value() @ cumulative_chi_node
    



    def calc_log_L(self, t_start=None, t_end=None, train_t=False, set_triggers=True):

        t_start, t_end, events_bool = self.get_events_bool(t_start=t_start, t_end=t_end, train_t=train_t)
        if set_triggers:
            self.set_triggers(t_start=t_start, t_end=t_end, train_t=train_t)

        terms = np.zeros(3, dtype=float)

        # lambda_tilde_t
        terms[0] = np.sum(np.log(self._events["lambda"][events_bool]))
        #terms[0] = np.sum(np.log(self._events["tilde_t_t"][events_bool]))
        terms[1] = -np.sum(self.calc_tilde_t_interval(t_start=t_start, t_end=t_end, train_t=train_t))
        terms[2] = np.sum([-np.sum(np.log(self._events["sigma"][i,events_bool[i,:]]))
                                +np.sum(np.log(scipy.stats.genpareto(c=self._parameters.xi.get_value()[i]).pdf(self._events['Y_M'][i,events_bool[i,:]]))) for i in range(self.get_D_nodes())])

        return np.sum(terms)


    def calc_log_L_terms(self, t_start=None, t_end=None, train_t=False, set_triggers=True):

        t_start, t_end, events_bool = self.get_events_bool(t_start=t_start, t_end=t_end, train_t=train_t)
        if set_triggers:
            self.set_triggers(t_start=t_start, t_end=t_end, train_t=train_t)

        terms = np.zeros((self.get_D_nodes(),3), dtype=float)


        terms[:,1] = -self.calc_tilde_t_interval(t_start=t_start, t_end=t_end, train_t=train_t)
        for i in range(self.get_D_nodes()):
            # lambda_tilde_t
            terms[i,0] = np.sum(np.log(self._events["lambda"][i, events_bool[i,:]]))
            #terms[i,0] = np.sum(np.log(self._events["tilde_t_t"][i, events_bool[i,:]]))
            
            terms[i,2] = -np.sum(np.log(self._events["sigma"][i,events_bool[i,:]]))+np.sum(np.log(scipy.stats.genpareto(c=self._parameters.xi.get_value()[i]).pdf(self._events['Y_M'][i,events_bool[i,:]])))

        return terms

    
    def calc_log_L_2(self, t_start=None, t_end=None, train_t=False):

        t_start, t_end, events_bool = self.get_events_bool(t_start=t_start, t_end=t_end, train_t=train_t)
        terms = np.zeros(3, dtype=float)

        # lambda_tilde_t
        terms[0] = np.sum(np.log(self._events["lambda"][events_bool]))
        #terms[0] = np.sum(np.log(self._events["tilde_t_t"][events_bool]))
        terms[1] = -np.sum(self._events["tilde_t_t_unique"][:,-1])
        terms[2] = np.sum([-np.sum(np.log(self._events["sigma"][i,events_bool[i,:]]))
                                +np.sum(np.log(scipy.stats.genpareto(c=self._parameters.xi.get_value()[i]).pdf(self._events['Y_M'][i,events_bool[i,:]]))) for i in range(self.get_D_nodes())])

        return np.sum(terms)



    def set_parameters_defaults(self, gamma=None, branch=None, set_opt_Bounds=True, a_u=None):


        if branch is None:
            branch=0.8

        if self._parameters._mode=="ci_s":
            m = [None for i in range(self.get_D_nodes())]
            for i in range(np.shape(self._parameters.get_M_sum())[0]):
                J_nodes = np.where(self._parameters.get_M_sum()[i])[0]
                m_nodes = self._events['M'][J_nodes,:][self._train_events_bool[J_nodes]]
                for j in J_nodes:
                    m[j] = m_nodes[~np.isnan(m_nodes)]
        else:
            m = [self._events['M'][i,self._train_events_bool[i]] for i in range(self.get_D_nodes())]

        if gamma is None:
            gamma = branch * np.transpose(self._parameters.get_M_weight()) @ self._parameters.get_M_sum()
        
        self._parameters.beta.set_default(default=1)
        self._parameters.nu.set_default(default=1)
        self._parameters.omega.set_default(default=1)


        gpd_fit = np.array([scipy.stats.genpareto.fit(m[i], floc=0) for i in range(self.get_D_nodes())])
        #for i in range(self.get_D_nodes()):
        #    if gpd_fit[i,0] < 0:
        #        gpd_fit[i,:] = scipy.stats.genpareto.fit(m[i], f0=0, floc=0)
        self._parameters.xi.set_default(default=gpd_fit[:,0])
        self._parameters.varsigma.set_default(default=gpd_fit[:,2])

        self._parameters.eta.set_default(default=0)
        self._parameters.alpha.set_default(default=0)


        self._parameters.gamma.set_default(default=gamma)

        self._parameters.set_defaults_mm()

        if a_u is None:
            a_u = np.transpose(self._parameters.get_M_weight()) @ self._parameters.get_M_sum() @ self._events["N"]/self.get_t_size()
        self._parameters.a_lambda.set_default(default=a_u)

        mu_0 = (np.eye(self.get_D_nodes()) - self._parameters.gamma.get_default()) @ self._parameters.a_lambda.get_default()
        self._parameters.mu.set_default(default=mu_0)


        if set_opt_Bounds:
            dfrac = 0.1

            self._parameters.a_lambda._opt_range.set_incl_high((1+dfrac)*np.max([m[i].size/self.get_t_size() for i in range(self.get_D_nodes())]))
            self._parameters.varsigma._opt_range.set_incl_high((1+dfrac)*np.max(self._parameters.varsigma.get_default()))



    



        



    def set_parameters_estimate(self, phi_type=None, kappa_type=None, flat_parameters_0=None, flat_fix=None, mode=None, mark=None, branch=None, fix_a_lambda=None, print_res_time=True, opt_method=None):

        


        res_timer = misc_fns.timer(print_timer=print_res_time, print_tab="\t", print_head="res")

        if mode is None:
            mode = self._parameters.get_mode()
        if mark is None:
            mark = self._parameters.get_mark()

        self._parameters._train_t = self._train_t
        self._parameters._train_events_bool = self._train_events_bool

        self._parameters.set_mode(mode)
        self._parameters.set_mark(mark)
        self._parameters.set_opt_settings(opt_method=opt_method, fix_a_lambda=fix_a_lambda)

        self._parameters.res_time['opt']['method'] = self._parameters.get_opt_method()
        self._parameters.res_time["res"]["fix_a_lambda"] = self._parameters.get_fix_a_lambda()
        


        if phi_type is not None:
            self._parameters.phi_type.set_value(value=phi_type)
        if kappa_type is not None:
            self._parameters.kappa_type.set_value(value=kappa_type)
        
        self._parameters.pull_flat_parameters()
        if self._parameters.get_fix_a_lambda():
            a_u = self._parameters.a_u.get_value()
        else:
            a_u = None
        self.set_parameters_defaults(branch=branch, a_u=a_u)
        self._parameters.set_flat_fix_type()
        
        self._parameters.set_flat_fix(fix=flat_fix)

        flat_defaults = self._parameters.flatten_defaults()
        # Set initial parameter values
        if flat_parameters_0 is None:
            self._parameters.res_fp0 = np.copy(flat_defaults)
        else:
            self._parameters.res_fp0 = np.copy(flat_parameters_0)

        self._parameters.set_opt_Bounds()

        flat_free = np.copy(self._parameters._flat_free)
        if not self._parameters._mark:
            gpd_flat = np.concatenate([self._parameters._flat_index["dict"]["range"]["xi"], self._parameters._flat_index["dict"]["range"]["varsigma"]])
            self._parameters.res_fp0[gpd_flat] = flat_defaults[gpd_flat]
            flat_free[gpd_flat] = False

        self._parameters.set_flat_parameters(self._parameters.res_fp0)
        #if a_u is not None:
        #    self._parameters._flat_parameters[self._parameters._flat_index['dict']['range']['a_lambda']] = a_u
        self._parameters.res_fp1 = self._parameters.flatten()

        a_lambda_free = self._parameters._flat_free[self._parameters._flat_index["dict"]["range"]["a_lambda"]]
        a_lambda_subordinate = self._parameters._flat_subordinate[self._parameters._flat_index["dict"]["range"]["a_lambda"]] - self._parameters._flat_index["dict"]["start"]["a_lambda"]
        a_lambda_flat_start = np.sum(self._parameters._flat_free[:self._parameters._flat_index["dict"]["start"]["a_lambda"]])
        a_lambda_flat_end = a_lambda_flat_start + np.sum(a_lambda_free)

        gamma_free = self._parameters._flat_free[self._parameters._flat_index["dict"]["range"]["gamma"]]
        gamma_subordinate = self._parameters._flat_subordinate[self._parameters._flat_index["dict"]["range"]["gamma"]] - self._parameters._flat_index["dict"]["start"]["gamma"]
        gamma_flat_start = np.sum(self._parameters._flat_free[:self._parameters._flat_index["dict"]["start"]["gamma"]])
        gamma_flat_end = gamma_flat_start + np.sum(gamma_free)

        ha_lambda = self._parameters.a_lambda.get_value()
        hgamma = np.hstack(self._parameters.gamma.get_value())

        def mu_constraint(flat_parameters_free):

            ha_lambda[a_lambda_free] = flat_parameters_free[a_lambda_flat_start:a_lambda_flat_end]
            a_lambda = ha_lambda[a_lambda_subordinate[:,1]]

            hgamma[gamma_free] = flat_parameters_free[gamma_flat_start:gamma_flat_end]
            gamma = np.reshape(hgamma[gamma_subordinate[:,1]], (self.get_D_nodes(),self.get_D_nodes()))

            return self._parameters.get_M_sum() @ (np.eye(self.get_D_nodes()) - gamma) @ a_lambda

        lin_con =   {
                        "type": "ineq",
                        "fun": mu_constraint,
                    }

        # Function to minimise (i.e. minus likelihood with parameters as vector)
        def log_L_min(flat_parameters_free, flat_free):
            self._parameters.set_flat_parameters_free(flat_parameters_free, flat_free)
            return -self.calc_log_L(train_t=True, set_triggers=True)


        
        # Optimisation
        opt_timer = misc_fns.timer(print_timer=print_res_time, print_tab="\t\t", print_head=self._parameters.get_opt_method())
        self._parameters.res = scipy.optimize.minimize(log_L_min, x0=self._parameters.res_fp1[flat_free], args=(flat_free),
                                        method=self._parameters.get_opt_method(), options={'disp': print_res_time, 'ftol': self._parameters.ftol},
                                        bounds=scipy.optimize.Bounds(lb=self._parameters._opt_Bounds[flat_free,0], ub=self._parameters._opt_Bounds[flat_free,1]),
                                        constraints=lin_con)
        opt_timer.stop()
        self._parameters.res_time['opt'] = opt_timer.get_timer_dict()


        self._parameters.res_fp2 = np.copy(self._parameters.res_fp1)
        self._parameters.res_fp2[flat_free] = self._parameters.res.x
        self._parameters.res_fp2[np.logical_and(flat_free, np.abs(self._parameters.res_fp2) < 1E-8)] = 0
        self._parameters.res_fp2 = self._parameters.res_fp2[self._parameters._flat_subordinate[:,1]]
        self._parameters.set_flat_parameters(flat_parameters=self._parameters.res_fp2)

        # Unconstrained non-zero parameters
        unconstrained_nz = np.logical_and(self._parameters._flat_free, np.abs(self._parameters.res_fp2) >= 1E-8)
        unconstrained_nz[self._parameters._flat_index["dict"]["start"]["beta"] + np.where(np.sum(self._parameters.gamma.get_value(), axis=0) == 0)[0]] = False

        # Function to minimise exclusive to unconstrained non-zero parameters (i.e. minus likelihood with parameters as vector)
        def log_L_min_unconstrained(flat_parameters_unconstrained):
            return log_L_min(flat_parameters_unconstrained, unconstrained_nz)

        # Standard error
        se = [None for i in range(self._parameters.res_fp0.size)]
        # Ad-hoc check for fitting across u range
        #if self._parameters.res.fun < 500
        self._parameters.fitted_success = True
        hess = numdifftools.Hessian(log_L_min_unconstrained, full_output=True, method="central", step=self._parameters.res_fp2[unconstrained_nz]/1E6)(self._parameters.res_fp2[unconstrained_nz])[0]
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

        # Set optimised parameters 
        self._parameters.set_flat_parameters(flat_parameters=self._parameters.res_fp2)
        self._parameters.set_phi(phi_type=self._parameters.phi_type.get_value(),
                                            beta=self._parameters.beta.get_value(),
                                            nu=self._parameters.nu.get_value(),
                                            omega=self._parameters.omega.get_value())
        self._parameters.set_fitted(se=se)

        # Run model at optimised parameters
        self.set_t_series()       

        # Information criteria
        self._N_params = self._parameters._N_params
        self._parameters._log_L = -self._parameters.res.fun

        self._parameters._fitted_scores = misc_fns.IC_dict(log_L=self._parameters._log_L, N_params=self._parameters._N_params, N_data=2*np.sum(self._train_events_bool))
        
        res_timer.stop()
        self._parameters.res_time["res"] = res_timer.get_timer_dict()

        # Save copy of estimate
        self._parameters_estimate = copy.deepcopy(self._parameters)

        return self._parameters.res






    def calc_tt_eval(self, lead=None, normalize=None, lags=None, boxpierce=None, head=None, pvalue=None, static=None, fix_a_lambda=None):
        if lead is None:
            lead = dict()

        eval_blr_atf = self.calc_tt_eval_blr_atf(normalize=normalize, lags=lags, boxpierce=boxpierce, head=head, pvalue=pvalue, static=static, fix_a_lambda=fix_a_lambda)

        columns = [     
                        [k for k in lead],
                        ["tail", "period"],
                        [k for k in eval_blr_atf['both']['all'].keys()],
                ]
        df_dict = {c: [] for c in np.concatenate(columns)}
        for tail in eval_blr_atf:
            for period in eval_blr_atf[tail]:
                df_dict["tail"].append(tail)
                df_dict["period"].append(period)
                for k in columns[-1]:
                    df_dict[k].append(eval_blr_atf[tail][period][k])
        for k in columns[0]:
            df_dict[k] = np.repeat(lead[k], len(df_dict["tail"]))

        return pd.DataFrame(df_dict)





    #def set_tt_eval_blr_atf(self):
    #    self.eval_blr_atf = self.calc_tt_eval_blr_atf()




    def calc_tt_eval_blr_atf(self, normalize=None, lags=None, boxpierce=None, head=None, pvalue=None, static=None, fix_a_lambda=None):

        if static is None:
            static = False

        eval = misc_fns.get_blr_atf()
        eval_log_L = self.calc_tt_eval_blr_atf_log_L(static=static, fix_a_lambda=fix_a_lambda)
        for tail in eval:
            for period in eval[tail]:
                eval[tail][period] = {
                                        **eval_log_L[tail][period],
                                        **self.calc_tt_eval_resid(tail=tail, period=period, normalize=normalize, lags=lags, boxpierce=boxpierce, head=head, pvalue=pvalue, static=static, fix_a_lambda=fix_a_lambda),
                                    }


        return eval



    

    def calc_tt_eval_blr_atf_log_L(self, static=None, fix_a_lambda=None):

        if static is None:
            static = False


        eval = misc_fns.get_blr_atf()
        log_L_terms = {period: None for period in eval['both']}


        if static:
            if fix_a_lambda is None:
                fix_a_lambda = self._parameters.get_fix_a_lambda()
            for period in log_L_terms:
                T = self._tail_fit['fix_a_lambda'][fix_a_lambda][period]['lambda']['T'][:,0]
                N = self._tail_fit['fix_a_lambda'][fix_a_lambda][period]['lambda']['N'][:,0]
                a_lambda = self._tail_fit['fix_a_lambda'][fix_a_lambda][period]['lambda']['a_lambda'][:,0]
                log_L_terms[period] = np.array([[-T[i]*a_lambda[i], N[i]*np.log(a_lambda[i]), self._tail_fit['fix_a_lambda'][fix_a_lambda][period]['gp_fit']['eval']['log_L'][i,0]] for i in range(2)])

            
        else:
            log_L_terms['all'] = self.calc_log_L_terms(train_t=False)
            log_L_terms['train'] = self.calc_log_L_terms(train_t=True)
            log_L_terms['forecast'] = log_L_terms['all'] - log_L_terms['train']


        eval_blr_atf_T_N_M = self.calc_tt_eval_blr_atf_T_N_M()



        for tail in eval:
            for period in eval[tail]:
                if tail=='left':
                    log_L_term = np.copy(log_L_terms[period][0,:])
                elif tail=='right':
                    log_L_term = np.copy(log_L_terms[period][1,:])
                else:
                    log_L_term = np.sum(log_L_terms[period], axis=0)
                
                
                eval[tail][period] = {
                                        **eval_blr_atf_T_N_M[tail][period],
                                        **self.calc_tt_eval_blr_atf_log_L_suff(log_L_term, tail=tail, period=period, flatten=True, static=static, fix_a_lambda=fix_a_lambda),
                                    }

        return eval





    def calc_tt_eval_blr_atf_log_L_suff(self, log_L_term, node=None, tail=None, period=None, flatten=None, static=None, fix_a_lambda=None):

        if static is None:
            static = True

        if flatten is None:
            flatten=False

        log_L_suff = {suff: {dev: None for dev in ['N_data', 'N_params', 'log_L', 'AIC', 'BIC']} for suff in ['u', 'lambda', 'M']}
        log_L_suff['u']['log_L'] = np.sum(log_L_term)
        log_L_suff['lambda']['log_L'] = np.sum(log_L_term[:2])
        log_L_suff['M']['log_L'] = np.sum(log_L_term[2])

        if fix_a_lambda is None:
            fix_a_lambda = False



        for suff in log_L_suff:
            log_L_suff[suff]['N_data'] = self.calc_N_M_data(node=node, tail=tail, period=period, component=suff)
            if static:
                if tail=='both':
                    N_params_lambda = 1 - int(fix_a_lambda)
                    N_params_M = 4
                else:
                    N_params_lambda = 1 - int(fix_a_lambda)
                    N_params_M = 2

                if suff=='lambda':
                    log_L_suff[suff]['N_params'] = N_params_lambda
                elif suff=='M':
                    log_L_suff[suff]['N_params'] = N_params_M
                else:
                    log_L_suff[suff]['N_params'] = N_params_lambda+N_params_M
            else:
                log_L_suff[suff]['N_params'] = self._parameters.calc_N_params(node=node, tail=tail, component=suff)
            log_L_suff[suff]['AIC'] = misc_fns.calc_AIC(N_params=log_L_suff[suff]['N_params'], log_L=log_L_suff[suff]['log_L'])
            log_L_suff[suff]['BIC'] = misc_fns.calc_BIC(N_data=log_L_suff[suff]['N_data'], N_params=log_L_suff[suff]['N_params'], log_L=log_L_suff[suff]['log_L'])


        if flatten:
            return {f"{val}_{suff}": log_L_suff[suff][val] for suff in log_L_suff for val in log_L_suff[suff]}
        else:
            return log_L_suff


    def calc_tt_eval_resid_atf(self, node=None, normalize=None, lags=None, boxpierce=None, head=None, pvalue=None, static=None, fix_a_lambda=None):

        eval = misc_fns.get_blr_atf()
        for tail in eval:
            for period in eval[tail]:
                eval[tail][period] = self.calc_tt_eval_resid(node=node, tail=tail, period=period, normalize=normalize, lags=lags, boxpierce=boxpierce, head=head, pvalue=pvalue, static=static, fix_a_lambda=fix_a_lambda)




    def calc_tt_eval_resid(self, node=None, tail=None, period=None, normalize=None, lags=None, boxpierce=None, head=None, pvalue=None, static=None, fix_a_lambda=None):

        if lags is None:
            lags = 5
        if static is None:
            static = False

        if static:
            eval =  {
                        **self.calc_N_KS(val='t', node=node, tail=tail, period=period, val_head=True, fix_a_lambda=fix_a_lambda),
                        **self.calc_events_k_KS_LB(val='t', node=node, tail=tail, period=period, set_zero=True, delta=True, normalize=normalize, lags=lags, boxpierce=boxpierce, head=head, pvalue=pvalue, rehead=True, val_head=True, fix_a_lambda=fix_a_lambda),
                        **self.calc_events_k_KS_LB(val='M', node=node, tail=tail, period=period, set_zero=False, delta=False, normalize=normalize, lags=lags, boxpierce=boxpierce, head=head, pvalue=pvalue, rehead=True, val_head=True),
                    }
        else:
            eval =  {
                        **self.calc_N_KS(val='tilde_t', node=node, tail=tail, period=period, val_head=True),
                        **self.calc_events_k_KS_LB(val='tilde_t', node=node, tail=tail, period=period, set_zero=True, delta=True, normalize=normalize, lags=lags, boxpierce=boxpierce, head=head, pvalue=pvalue, rehead=True, val_head=True),
                        **self.calc_N_KS(val='tilde_t_eff', node=node, tail=tail, period=period, val_head=True),
                        **self.calc_events_k_KS_LB(val='tilde_t_eff', node=node, tail=tail, period=period, set_zero=True, delta=True, normalize=normalize, lags=lags, boxpierce=boxpierce, head=head, pvalue=pvalue, rehead=True, val_head=True),
                        **self.calc_events_k_KS_LB(val='tilde_M', node=node, tail=tail, period=period, set_zero=False, delta=False, normalize=normalize, lags=lags, boxpierce=boxpierce, head=head, pvalue=pvalue, rehead=True, val_head=True),
                    }

        return eval





    
    def calc_N_KS(self, val=None, node=None, tail=None, period=None, val_head=None, fix_a_lambda=None):

        #tilde_t = self.calc_tilde_t_k_zero(node=node, tail=tail, period=period)
        if val is None:
            val = 'tilde_t_eff'
        t_tilde_t = self.calc_events_k(val=val, node=node, tail=tail, period=period, set_zero=True, delta=False, normalize=False)
        if val=='t':
            if False:
                if node is None:
                    a_lambda = np.sum(self._parameters.a_u.get_value())
                else:
                    a_lambda = self._parameters.a_u.get_value()[node]
            else:
                if fix_a_lambda is None:
                    fix_a_lambda = self._parameters.get_fix_a_lambda()
                a_lambda = self._tail_fit['fix_a_lambda'][fix_a_lambda]['train']['lambda']['a_lambda'][:,0]
                if node is None:
                    a_lambda_eff = np.sum(a_lambda)
                else:
                    a_lambda_eff = a_lambda[node]

            t_tilde_t = t_tilde_t * a_lambda_eff

        if False:
            N = t_tilde_t.size
            if N>0:
                dist = (t_tilde_t - (1+np.array(range(N))))/N
                sup  = np.max(np.abs(dist))
                stat = sup*np.sqrt(N)
                p = scipy.special.kolmogorov(stat)
            else:
                stat= np.nan
                p = np.nan
        else:
            KS_res = misc_fns.KS_test_Poisson_scipy(tilde_t=t_tilde_t)
            stat = KS_res[0]
            p = KS_res[1]



        val_head = misc_fns.calc_val_head(val_head=val_head, val=f"N_{val}", delta=False, normalize=False)
        return {f'{val_head}KS_stat': stat, f'{val_head}KS_p': p}



    def calc_events_k_KS_LB(self, val=None, node=None, tail=None, period=None, set_zero=None, delta=None, normalize=None, theory_cdf=None, lags=None, boxpierce=None, head=None, pvalue=None, rehead=None, val_head=None, fix_a_lambda=None):
        
        KS_dict = self.calc_events_k_KS(val=val, node=node, tail=tail, period=period, set_zero=set_zero, delta=delta, normalize=normalize, theory_cdf=theory_cdf, val_head=val_head, fix_a_lambda=fix_a_lambda)
        LB_dict = self.calc_events_k_LB(val=val, node=node, tail=tail, period=period, set_zero=set_zero, delta=delta, normalize=normalize, theory_cdf=theory_cdf, lags=lags, boxpierce=boxpierce, flatten=True, head=head, pvalue=pvalue, rehead=rehead, val_head=val_head, fix_a_lambda=fix_a_lambda)

        return {**KS_dict, **LB_dict}


    def calc_events_k_LB(self, val=None, node=None, tail=None, period=None, set_zero=None, delta=None, normalize=None, theory_cdf=None, lags=None, boxpierce=None, flatten=None, head=None, pvalue=None, rehead=None, val_head=None, fix_a_lambda=None):

        if normalize is None:
            normalize = False
        #if val=='F_M':
        #    normalize=True

        if theory_cdf is None:
            theory_cdf = self.calc_theory_cdf(val=val, node=node, delta=delta, fix_a_lambda=fix_a_lambda)

        if flatten is None:
            flatten=False
        if rehead is None:
            rehead = True

        #if rehead or flatten:
        if pvalue is None:
            pvalue = True
        if pvalue:
            suff = ['stat', 'pvalue']
            suff_rehead = ['stat', 'p']
        else:
            suff = ['stat']
            suff_rehead = ['stat']
        if head is None:
            if boxpierce is None:
                boxpierce = False
            if boxpierce:
                head = ['lb', 'bp']
            else:
                head = ['lb']
            
        
        head = misc_fns.make_iterable_array(head)
        head = [head[k].lower() for k in range(len(head))]
        if np.isin('bp', head):
            boxpierce = True
        
        head_df = [f"{h}_{s}" for h in head for s in suff]

        if rehead:
            head_rehead = [head[k].upper() for k in range(len(head))]
            head_df_rehead = [f"{h}_{s}" for h in head_rehead for s in suff_rehead]

            columns_dict = {head_df[k]: head_df_rehead[k] for k in range(len(head_df))}
            head_out = head_df_rehead
        else:
            head_out = head_df



        events = self.calc_events_k(val=val, node=node, tail=tail, period=period, set_zero=set_zero, delta=delta, normalize=normalize, theory_cdf=theory_cdf)

        if lags is None:
            lags = 5
        N_events = np.size(events)
        lag_eff = np.min([N_events-1, lags])
        dict_val = {f"{h}": [np.nan for k in range(lags)] for h in head_out}
        df = pd.DataFrame(dict_val, index=1+np.array(range(lags)))
        if lag_eff>0:
            df_lb =  acorr_ljungbox(x=events,
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
            val_head = misc_fns.calc_val_head(val_head=val_head, val=val, delta=delta, normalize=normalize)
            return {f"{val_head}{h}_{1+k}": df[h].values[k] for k in range(df.shape[0]) for h in head_out}



    

    
    def calc_events_k_KS(self, val=None, node=None, tail=None, period=None, set_zero=None, delta=None, normalize=None, theory_cdf=None, val_head=None, fix_a_lambda=None):

        if normalize is None:
            normalize = False
        #if val=='F_M':
        #    normalize=True

        if theory_cdf is None:
            theory_cdf = self.calc_theory_cdf(val=val, node=node, delta=delta, fix_a_lambda=fix_a_lambda)
        if normalize:
            KS_cdf = scipy.stats.norm().cdf
        else:
            KS_cdf = theory_cdf

        events = self.calc_events_k(val=val, node=node, tail=tail, period=period, set_zero=set_zero, delta=delta, normalize=normalize, theory_cdf=theory_cdf)
        if np.size(events)==0:
            KS_result = np.full(2, np.nan)
        else:
            KS_result = scipy.stats.kstest(events, KS_cdf)

        val_head = misc_fns.calc_val_head(val_head=val_head, val=val, delta=delta, normalize=normalize)
        return {f'{val_head}KS_stat': KS_result[0], f'{val_head}KS_p': KS_result[1]}


    
        



    def calc_tilde_t_k_zero(self, node=None, tail=None, period=None):
        
        if tail is not None:
            node = hwks_fns.get_tail_node()[tail]
        if period is None:
            period = 'all'


        if False:
            if ci is None:
                if self._parameters._mode[:2]=='ci':
                    ci=True
                else:
                    ci=False
            tilde_t_k = np.full((self.get_D_nodes(), np.max(N_T), self.get_D_nodes()), np.nan)
            for i in range(self.get_D_nodes()):
                tilde_t_k[i,:N_T[i],:] = np.transpose(self._t_series['tilde_t'][:,self._events['t'][i,events_bool[i]]])


        tilde_t_0 = 0
        
        if node is None:
            

            if period=='forecast':
                N_start = self.calc_N_M(period='train')
            if N_start > 0:
                tilde_t_0 = self._events['tilde_t_all'][N_start-1]
            return self._events['tilde_t_all'][N_start:N_start+self.calc_N_M(period=period)] - tilde_t_0
        else:

            if period=='train':
                t_start, t_end, events_bool = self.get_events_bool(train_t=True)
            elif period=='forecast':
                t_start, t_end, events_bool = self.get_events_bool(train_t=False, post_train_t=True)
                N_train = np.sum(self._train_events_bool, axis=1)[node]
                if N_train > 0:
                    tilde_t_0 = self._events['tilde_t'][node,N_train-1]
            else:
                t_start, t_end, events_bool = self.get_events_bool(train_t=False)

            return self._events['tilde_t'][node,events_bool[node]] - tilde_t_0



    
    def calc_delta_tilde_t_k_zero(self, node=None, tail=None, period=None):

        return misc_fns.diff_zero(self.calc_tilde_t_k_zero(node=node, tail=tail, period=period))

    def calc_delta_tilde_t_AR_l(self, lags=None, node=None, tail=None, period=None):

        if lags is None:
            lags = 1
        
        delta_tilde_t = self.calc_delta_tilde_t_k_zero(node=node, tail=tail, period=period)
        
        return plot_fns.correlogram(data=delta_tilde_t, lags=lags)




    

    def calc_events_k(self, val=None, node=None, tail=None, period=None, set_zero=None, delta=False, normalize=None, theory_cdf=None):

        if val is None:
            val = 't'


        delta = misc_fns.calc_eval_default(measure_name='delta', measure_inp=delta, val=val)
        set_zero = misc_fns.calc_eval_default(measure_name='set_zero', measure_inp=set_zero, val=val)

        
        if tail is not None:
            node = hwks_fns.get_tail_node()[tail]
        if period is None:
            period = 'all'


        if False:
            if ci is None:
                if self._parameters._mode[:2]=='ci':
                    ci=True
                else:
                    ci=False
            tilde_t_k = np.full((self.get_D_nodes(), np.max(N_T), self.get_D_nodes()), np.nan)
            for i in range(self.get_D_nodes()):
                tilde_t_k[i,:N_T[i],:] = np.transpose(self._t_series['tilde_t'][:,self._events['t'][i,events_bool[i]]])


        t_start_train, t_end_train, events_bool_train = self.get_events_bool(train_t=True, node=node)
        t_start_all, t_end_all, events_bool_all = self.get_events_bool(train_t=False, node=node)
        t_start_forecast, t_end_forecast, events_bool_forecast = self.get_events_bool(train_t=False, post_train_t=True, node=node)

        if val=='tilde_M':
            tilde_M = True
            val_eff='Y_M'
        else:
            val_eff=val


        N_start=0
        if period=='train':
            t_start=t_start_train
            t_end=t_end_train
            events_bool=events_bool_train
        elif period=='forecast':
            t_start=t_start_forecast
            t_end=t_end_forecast
            events_bool=events_bool_forecast
            N_start = self.calc_N_M(period='train', node=node)
        else:
            t_start=t_start_all
            t_end=t_end_all
            events_bool=events_bool_all

        if val_eff=='tilde_t' and node is None:
            events_array = np.copy(self._events['tilde_t_all'])
            events_bool_all = np.full(events_array.shape, True)
            events_bool = np.full(events_array.shape, False)
            events_bool[t_start:t_end] = True
        else:
            events_array = np.copy(self._events[val_eff])

        if val=='tilde_M':
            for i in range(self.get_D_nodes()):
                events_array[i,:] = scipy.stats.expon().ppf(scipy.stats.genpareto(c=self._parameters.xi.get_value()[i], loc=0, scale=1).cdf(events_array[i,:]))

        if not np.any(events_bool):
            return np.array([])
        else:
            zero = 0
            if set_zero:  
                if N_start > 0:
                    zero = events_array[events_bool_all][N_start-1]
            events =  events_array[events_bool] - zero


            if delta:
                events = misc_fns.diff_zero(events)

            if normalize is None:
                normalize = False
            if normalize:
                if theory_cdf is None:
                    theory_cdf = self.calc_theory_cdf(val=val, node=node, delta=delta)
                    


                events = scipy.stats.norm().ppf(theory_cdf(events))

            return events


            
        

    def calc_theory_cdf(self, val=None, node=None, delta=None, fix_a_lambda=None):
        if val is None:
            val = ''
        delta = misc_fns.calc_eval_default(measure_name='delta', measure_inp=delta, val=val)
        if np.isin(val, ['tilde_t', 'tilde_t_eff', 'tilde_M']) and delta:
            theory_cdf = scipy.stats.expon().cdf
        elif np.isin(val, ['t']) and delta:
            if fix_a_lambda is None:
                fix_a_lambda = self._parameters.get_fix_a_lambda()
            a_lambda = self._tail_fit['fix_a_lambda'][fix_a_lambda]['train']['lambda']['a_lambda'][:,0]
            if node is None:
                a_lambda_eff = np.sum(a_lambda)
            else:
                a_lambda_eff = a_lambda[node]
            theory_cdf = scipy.stats.expon(scale=1/a_lambda_eff).cdf

        elif np.isin(val, ['F_M']):
            theory_cdf = lambda x : x
        elif  np.isin(val, ['Y_M']):
            if node is not None:
                xi = self._parameters.xi.get_value()[node]
            else:
                xi = np.mean(self._parameters.xi.get_value())
            theory_cdf = scipy.stats.genpareto(loc=0, scale=1, c=xi).cdf
        
        elif  np.isin(val, ['M']):
            theory_cdf = scipy.stats.genpareto(*scipy.stats.genpareto.fit(self.calc_events_k(val='M', node=node, period='train', normalize=False), floc=0)).cdf
        else:
            theory_cdf = scipy.stats.norm.cdf

        return theory_cdf
        



    

    def calc_tt_eval_blr_atf_T_N_M(self):

        eval = misc_fns.get_blr_atf()
        for tail in eval:
            for period in eval[tail]:

                eval[tail][period] = {
                                        'T': self.calc_T(period=period),
                                        'N_M': self.calc_N_M(tail=tail, period=period),
                                    }

        return eval


    def calc_T(self, period=None):

        if period is None:
            period = 'all'

        if period=='train':
            return np.diff(self._train_t)[0]
        elif period=='forecast':
            return self.get_t_size() - self._train_t[-1]
        else:
            return self.get_t_size()



    def calc_N_M(self, node=None, tail=None, period=None):

        if tail is not None:
            node = hwks_fns.get_tail_node()[tail]
        if period is None:
            period = 'all'

        N_all = np.copy(self._events['N'])
        N_train = np.sum(self._train_events_bool, axis=1)

        if period=='train':
            N_i = N_train
        elif period=='forecast':
            N_i = N_all - N_train
        else:
            N_i = N_all

        if node is None:
            return np.sum(N_i)
        else:
            return N_i[node]


    def calc_N_M_data(self, node=None, tail=None, period=None, component=None):

        if component is None:
            component = 'u'

        N_m = self.calc_N_M(node=node, tail=tail, period=period)

        
        factor = 2
        if component=='lambda' or component=='M':
            if not self._parameters._mark:
                factor = 1

        return factor*N_m








    


    
    



 #--#========================
    #-----------------------
    # forecast functions
    #-----------------------
    #=======================

    def calc_forecast(self, delta_forecast=5, p_threshold=0.5, periods=None):

        if periods is None:
            periods = [np.array(self._index[0], self._index[-1])]
        elif type(periods) != list:
            periods = [periods]


        forecast = {str(delta): dict() for delta in delta_forecast}

        for delta in forecast:

            forecast[delta]["delta_forecast"] = int(delta)
            forecast_size = self.get_t_size() - (forecast[delta]["delta_forecast"]  - 1)

            forecast[delta]["probability"] = np.full((self.get_D_nodes(), forecast_size), np.nan)
            initial_events = self._events["t"][:,0]
            t_initial_event = np.min(initial_events[initial_events >= 0])
            forecast[delta]["probability"][:, 0:t_initial_event+1] = np.outer((1 - np.exp(-forecast[delta]["delta_forecast"] * self._parameters.mu.get_value())), np.ones(t_initial_event+1))
            for k in range(t_initial_event+1, forecast_size):
                forecast[delta]["probability"][:,k] = self.calc_event_probability_window(t=k, window_sizes=forecast[delta]["delta_forecast"], N=self._events["N_t"][:, k-1])[:,0] 

            
            forecast[delta]["indicator"] = np.full((self.get_D_nodes(), forecast_size), False)
            for i in range(self.get_D_nodes()):
                index = np.unique(np.outer(self._events["t"][i,:], np.ones(forecast[delta]["delta_forecast"], dtype=int)) - np.array(range(forecast[delta]["delta_forecast"])))
                forecast[delta]["indicator"][i, index[np.logical_and(index >= 0, index < forecast_size)]] = True

            SPE = (forecast[delta]["probability"] - forecast[delta]["indicator"])**2

            forecast[delta]["binary"] = {str(p): dict() for p in p_threshold}
            for p in forecast[delta]["binary"]:

                forecast[delta]["binary"][p]["p_threshold"] = float(p)
                forecast[delta]["binary"][p]["prediction"] = (forecast[delta]["probability"] >= forecast[delta]["binary"][p]["p_threshold"])[:, :forecast_size]

                forecast[delta]["binary"][p]["outcome"] =   {
                                                                "all": np.full((2, 2, self.get_D_nodes(), forecast_size), False),
                                                                "Pos": {"True": None, "False": None},
                                                                "Neg": {"True": None, "False": None},
                                                            }

                forecast[delta]["binary"][p]["outcome"]["all"][0,0,:,:] = forecast[delta]["binary"][p]["outcome"]["Pos"]["True"] = np.logical_and(forecast[delta]["binary"][p]["prediction"], forecast[delta]["indicator"])
                forecast[delta]["binary"][p]["outcome"]["all"][0,1,:,:] = forecast[delta]["binary"][p]["outcome"]["Pos"]["False"] = np.logical_and(forecast[delta]["binary"][p]["prediction"], np.logical_not(forecast[delta]["indicator"]))
                forecast[delta]["binary"][p]["outcome"]["all"][1,0,:,:] = forecast[delta]["binary"][p]["outcome"]["Neg"]["True"]  = np.logical_and(np.logical_not(forecast[delta]["binary"][p]["prediction"]), forecast[delta]["indicator"])
                forecast[delta]["binary"][p]["outcome"]["all"][1,1,:,:] = forecast[delta]["binary"][p]["outcome"]["Neg"]["False"]  = np.logical_and(np.logical_not(forecast[delta]["binary"][p]["prediction"]), np.logical_not(forecast[delta]["indicator"]))
                
            forecast[delta]["periods"] = [self.make_score_period(TS_bounds=periods[k], probability=forecast[delta]["probability"], indicator=forecast[delta]["indicator"], SPE=SPE, binary=forecast[delta]["binary"], delta_forecast=forecast[delta]["delta_forecast"]) for k in range(len(periods))]

                
        return forecast




    def make_score_period(self, TS_bounds, probability, indicator, SPE, binary, delta_forecast):

        period = dict()
        period["t_bounds"] = np.array(  [
                                            np.sum(self._index < TS_bounds[0]), 
                                            np.min( [
                                                        np.sum(self._index < TS_bounds[1]) - 1,
                                                        indicator[0].size - 1
                                                    ])
                                        ])
        period["t_range"] = range(period["t_bounds"][0], period["t_bounds"][1]+1)
        period["size"] = np.size(period["t_range"])

        period["TS_bounds"] = self._index[period["t_bounds"]]
        period["TS_bounds_str"] = "[" + period["TS_bounds"][0].strftime('%Y-%m-%d') + " : " + period["TS_bounds"][1].strftime('%Y-%m-%d') + "]"


        period["prob_score"] =  {
                                    "all_nodes": dict(),
                                    "node": dict(),
                                    "export": None,
                                }
        period["binary_score"] = {p: copy.deepcopy(period["prob_score"]) for p in binary}

        period["prob_score"]["node"]["QPS"] = 2 * np.mean(SPE[:, period["t_range"]], axis=1)
        period["prob_score"]["node"]["LPS"] = -1 * np.mean(np.logical_not(indicator[:, period["t_range"]])*np.log(1-probability[:, period["t_range"]]) + indicator[:, period["t_range"]]*np.log(probability[:, period["t_range"]]), axis=1)
        period["prob_score"]["all_nodes"]["QPS"] = np.mean(period["prob_score"]["node"]["QPS"])
        period["prob_score"]["all_nodes"]["LPS"] = np.mean(period["prob_score"]["node"]["LPS"])

        period["prob_score"]["export"] =    {
                                                "delta_forecast": delta_forecast,
                                                
                                                "forecast_period_0": period["TS_bounds"][0],
                                                "forecast_period_1": period["TS_bounds"][1],
                                                **period["prob_score"]["all_nodes"],
                                            }


        for p in period["binary_score"]:

            period["binary_score"][p]["node"]["confusion_matrix"] = np.sum(binary[p]["outcome"]["all"][:,:,:, period["t_range"]], axis=3)
            period["binary_score"][p]["node"]["positive_rates"] = period["binary_score"][p]["node"]["confusion_matrix"][0] / np.sum(period["binary_score"][p]["node"]["confusion_matrix"], axis=0)

            period["binary_score"][p]["all_nodes"]["confusion_matrix"] = np.sum(period["binary_score"][p]["node"]["confusion_matrix"], axis=2)
            period["binary_score"][p]["all_nodes"]["positive_rates"] = np.mean(period["binary_score"][p]["node"]["positive_rates"], axis=1)

            for name in ["all_nodes", "node"]:  
                period["binary_score"][p][name]["KSS"] = misc_fns.calc_KSS(c_mat=period["binary_score"][p][name]["confusion_matrix"])
                period["binary_score"][p][name]["ETS"] = misc_fns.calc_ETS(c_mat=period["binary_score"][p][name]["confusion_matrix"])
                period["binary_score"][p][name]["FBS"] = misc_fns.calc_FBS(c_mat=period["binary_score"][p][name]["confusion_matrix"])

            period["binary_score"][p]["export"] =   {
                                                        "delta_forecast": delta_forecast,
                                                        "p_threshold": binary[p]["p_threshold"],

                                                        "forecast_period_0": period["TS_bounds"][0],
                                                        "forecast_period_1": period["TS_bounds"][1],

                                                        "True positives": period["binary_score"][p]["all_nodes"]["confusion_matrix"][0,0],
                                                        "False positives": period["binary_score"][p]["all_nodes"]["confusion_matrix"][0,1],
                                                        "False negatives": period["binary_score"][p]["all_nodes"]["confusion_matrix"][1,0],
                                                        "True negatives": period["binary_score"][p]["all_nodes"]["confusion_matrix"][1,1],

                                                        "True positive rate": period["binary_score"][p]["all_nodes"]["positive_rates"][0],
                                                        "False positive rate": period["binary_score"][p]["all_nodes"]["positive_rates"][1],

                                                        "KSS": period["binary_score"][p]["all_nodes"]["KSS"],
                                                        "ETS": period["binary_score"][p]["all_nodes"]["ETS"],
                                                        "FBS": period["binary_score"][p]["all_nodes"]["FBS"]
                                                    }
        

        return period




    def calc_int_lambda_forcast(self, t, window_sizes, N=None):

        if N is None:
            N = np.array([np.sum(self._events["t"][i,:] < t) for i in range(self.get_D_nodes())])

        t_forecast = t + window_sizes

        chi = np.zeros(shape=(self.get_D_nodes(), np.size(window_sizes), np.max(N)), dtype=float)

        for i in range(self.get_D_nodes()):
            for k in range(N[i]):
                chi[i, :, k] = (self._parameters.int_phi(t_forecast - self._events["t"][i,k]) - self._parameters.int_phi(t - self._events["t"][i,k], node=i)) * self._events["kappa"][i,k]

        return np.outer(self._parameters.mu.get_value(), window_sizes) + self._parameters.gamma.get_value() @ np.sum(chi, axis=2)

    
    def calc_event_probability_window(self, t, window_sizes, N=None):

        return 1 - np.exp(-self.calc_int_lambda_forcast(t, window_sizes, N))





 #--#========================
    #-----------------------
    # analysis functions
    #-----------------------
    #=======================



    def set_residual_KS(self, print_results=False, nodes=False):
        """
        """
        self._residual_KS = {
                                "all_nodes": misc_fns.calc_residual_KS(tilde_t=self._events["tilde_t_all"]),
                                "node": [misc_fns.calc_residual_KS(tilde_t=self._events["tilde_t"][node,0:self._events["N"][node]]) for node in range(self._D_nodes)],
                            }
        if print_results:
            self.print_residual_KS(nodes=nodes)

    def print_residual_KS(self, nodes=False):

        print("")
        print("======================")
        print("KS residual analysis")
        print("======================")
        print("")
        print("All nodes:")
        print(self._residual_KS["all_nodes"])
        print("")

        if type(nodes) == bool:
            if nodes == False:
                return
            else:
                nodes=None

        nodes = self.clean_nodes(nodes=nodes)

        print("Individual nodes")
        for node in nodes:
            print("Node {}:".format(node))
            print(self._residual_KS["node"][node])
        print("")



#---#========================
    #-----------------------
    # plotting functions
    #-----------------------
    #=======================

    def get_t_fine_plot(self, market_time=None):
        if market_time is None:
            market_time=False
        if market_time:
            t_fine_plot = np.full((self.get_D_nodes(), self.get_t_size(), 4+self.get_dt_dt_fine()), np.nan)
            t_fine_plot[:,:,1:-2] = np.outer(np.array(range(self.get_t_size())), np.ones(1+self.get_dt_dt_fine())) + np.outer(np.ones(self.get_t_size()), np.linspace(0,1,1+self.get_dt_dt_fine()))
        else:
            t_fine_plot = np.full((self.get_D_nodes(), self.get_t_size(), 4+self.get_dt_dt_fine()), None)
            t_fine_plot[:,:,1:-2] = self._TS["t_fine"]
        t_fine_plot[:,:,0] = t_fine_plot[:,:,1]
        t_fine_plot[:,:,-1] = t_fine_plot[:,:,-3]
        t_fine_plot[:,:,-2] = t_fine_plot[:,:,-3]

        return t_fine_plot


    def plot_lambda_ax(self, nodes=None, plot_lambda=None, plot_chi=None, markers=True, marker_size=None, marker_magnitude=False, market_time=None, colours=None, ax=None):

        if ax is None:
            ax = plt.gca()

        if colours is None:
            colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

        if plot_lambda is None:
            plot_lambda = True
        if plot_chi is None:
            plot_chi = False

        nodes_set = self.clean_nodes(nodes)
        #[t_events, m_events, kappa_events, N_events, t_range_index] = self.extract_events(nodes=nodes_set, t_start=t_start, t_end=t_end)

        if self._TS["t_fine"] is None:
            self.set_t_fine()
        t_fine_plot = self.get_t_fine_plot(market_time=market_time)

        if self._t_series["lambda_fine"] is None:
            self.fill_t_series_fine()
        lambda_fine = np.full_like(t_fine_plot, np.nan)
        chi_fine = np.full_like(t_fine_plot, np.nan)
        lambda_fine[:,:,1:-2] = self._t_series["lambda_fine"]
        lambda_fine[:,:-1,-2] = lambda_fine[:,1:,1]

        chi_fine[:,:,1:-2] = self._t_series["chi_fine"]
        chi_fine[:,:-1,-2] = chi_fine[:,1:,1]

        

        if markers:
            if type(marker_size) != type(None):
                marker_size = np.squeeze(marker_size)
                if marker_size.ndim != 0:
                    raise ValueError("marker_size must be of ndim 0")

            if marker_magnitude:
                m_mean = np.mean(np.concatenate(m_events))
                
                marker_scale = marker_size
                if type(marker_scale) == type(None):
                    marker_scale=20
                marker_size = [marker_scale*m_events[i]/m_mean for i in range(nodes_set.size)]
            else:
                if type(marker_size) != type(None):
                    marker_size = marker_size * np.ones(nodes_set.size)

        if markers:
            for i in range(nodes_set.size):
                colour = colours[np.mod(nodes_set[i],len(colours))]

                x_markers = t_fine_plot[nodes_set[i], t_events[i], -2]
                Y_Markers = lambda_fine[nodes_set[i], t_events[i], -2]
                if type(marker_size) != type(None):
                    ax.scatter(x_markers, Y_Markers, s=marker_size[i], color=colour)
                else:
                    ax.scatter(x_markers, Y_Markers, color=colour)


        if plot_lambda:
            for i in range(nodes_set.size):
                colour = colours[np.mod(nodes_set[i],len(colours))]
                ax.plot(np.concatenate(t_fine_plot[nodes_set[i]]), np.concatenate(lambda_fine[nodes_set[i]]), color=colour, linewidth=1)  
        if plot_chi:
            for i in range(nodes_set.size):
                colour = colours[np.mod(nodes_set[i],len(colours))]
                ax.plot(np.concatenate(t_fine_plot[nodes_set[i]]), np.concatenate(chi_fine[nodes_set[i]]), color=colour, linewidth=0.5)  

        ax.set_xlim(left=np.min(t_fine_plot[:,0,0]), right=np.max(t_fine_plot[:,-1,-1]))

        ax.set_ylim(bottom=0)

        ax.set_ylabel(f"$\\lambda_{{t}} \\: / \\: {hwks_fns.get_unit_d_tr()}^{{-1}}$")

        return ax

    
    def plot_events_ax(self, nodes=None, show_x=None, show_sub=None, market_time=None, colours=None, ax=None):

        if ax is None:
            ax = plt.gca()
        if colours is None:
            colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if show_x is None:
            show_x = True
        if show_sub is None:
            show_sub = False

        if self._TS["t_fine"] is None:
            self.set_t_fine()
        t_fine_plot = self.get_t_fine_plot(market_time=market_time)


        nodes_set = self.clean_nodes(nodes)

        for j in range(nodes_set.size):
            i = nodes_set[j]
            exceed_bool = np.full(self.get_t_size(), False)
            exceed_bool[self._events["t"][i,:self._events["N"][i]]] = True
            excess_x = np.repeat(t_fine_plot[i,exceed_bool,-2], 4)
            excess_y = misc_fns.stack(self._events['M'][i,:self._events["N"][i]])

            if show_x:
                excess_y[:,1] = self._parameters.u.get_value()[i]
                excess_y[:,2] = self.exceedance_transform_inv(node=i, data=excess_y[:,2])
                
            

            if show_sub:
                excess_y_zero = misc_fns.stack(np.repeat(self._parameters.u.get_value()[i], self._events["N"][i]),0)
                zero_bool = self.exceedance_transform(node=i) > self.exceedance_transform(node=i, data=self._parameters.u_zero.get_value()[i])
                sub_bool = np.logical_and(zero_bool, ~exceed_bool)
                sub_x = np.repeat(t_fine_plot[i,sub_bool,-2], 4)
                sub_y = misc_fns.stack(self._data[i][sub_bool], self._parameters.u_zero.get_value()[i])

                ax.plot(sub_x, np.concatenate(sub_y), linewidth=0.3, color=plot_fns.colour_transform_alpha(colours[i], alpha=0.25))
                ax.plot(excess_x, np.concatenate(excess_y_zero), linewidth=0.3, color=plot_fns.colour_transform_alpha(colours[i], alpha=0.75))

            ax.plot(excess_x, np.concatenate(excess_y), linewidth=0.3, color=plot_fns.colour_transform_alpha(colours[j], alpha=1))

            ax.set_xlim(left=np.min(t_fine_plot[:,0,2]), right=np.max(t_fine_plot[:,-1,-2]))

            if show_x:
                ax.set_ylabel("$r_{{t}}$")
            else:
                ax.set_ylabel("$m$")




    
    def plot_lambda_fig(self, nodes=None, plot_lambda=None, plot_chi=None, markers=None, marker_size=None, marker_magnitude=None, market_time=False, colours=None, fig=None):

        fig, ax = plt.subplots(nrows=1)
        self.plot_lambda_ax(nodes=nodes, plot_lambda=plot_lambda, plot_chi=plot_chi, markers=markers, marker_size=marker_size, marker_magnitude=marker_magnitude, market_time=market_time, colours=colours, ax=ax)

        return fig

    def plot_process_fig(self, nodes=None, plot_lambda=None, plot_chi=None, markers=None, marker_size=None, marker_magnitude=None, show_x=None, show_sub=None, market_time=None, colours=None, fig=None):

        if colours is None:
            colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

        nodes_set = misc_fns.make_iterable_array(values=self.clean_nodes(nodes=nodes))
        fig, ax = plt.subplots(nrows=1+nodes_set.size)
        self.plot_lambda_ax(nodes=nodes, plot_lambda=plot_lambda, plot_chi=plot_chi, markers=markers, marker_size=marker_size, marker_magnitude=marker_magnitude, market_time=market_time, colours=colours, ax=ax[0])
        for i in range(nodes_set.size):
            self.plot_events_ax(nodes=nodes_set[i],  show_x=show_x, show_sub=show_sub, market_time=market_time, colours=[colours[i]], ax=ax[1+i])

        return fig


#---#========================
    #-----------------------
    # bulk functions
    #-----------------------
    #=======================

    def set_hb(self, dist=None, bound=None, set_fit=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):

        self._hb = hawkes_bulk_dist(hawkes=self, dist=dist, bound=bound, set_fit=set_fit, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

    def set_vrs(self, a_q, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        for dist in self._hb.hb_d:
            self._hb.hb_d[dist].set_vrs(hawkes=self, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)


    def set_hb_mp(self):

        self.set_hb(dist=self._hb_mp_params['dist'], bound=self._hb_mp_params['bound'], set_fit=self._hb_mp_params['set_fit'], a_q=self._hb_mp_params['a_q'], set_eval=self._hb_mp_params['set_eval'], 
                    CT_Q=self._hb_mp_params['CT_Q'], S_Q=self._hb_mp_params['S_Q'], S_E=self._hb_mp_params['S_E'], save_t=self._hb_mp_params['save_t'])

        return self._hb

    def set_vrs_mp(self):

        self.set_vrs(a_q=self._hb_mp_params['a_q'], set_eval=self._hb_mp_params['set_eval'], 
                    CT_Q=self._hb_mp_params['CT_Q'], S_Q=self._hb_mp_params['S_Q'], S_E=self._hb_mp_params['S_E'], save_t=self._hb_mp_params['save_t'])

        return self._hb


    def set_hb_mp_params(self, dist=None, bound=None, set_fit=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):

        self._hb_mp_params =   {
                                    'dist': dist,
                                    'bound': bound,
                                    'set_fit': set_fit,
                                    'a_q': a_q,
                                    'set_eval': set_eval,
                                    'CT_Q': CT_Q,
                                    'S_Q': S_Q,
                                    'S_E': S_E,
                                    'save_t': save_t,
                                }

    def set_vrs_eval(self, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        for dist in self._hb.hb_d:
            self._hb.hb_d[dist].set_vrs_eval(hawkes=self, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

    def calc_vrs_eval(self, lead=None, atf=None):
        return self._hb.calc_vrs_eval(lead=lead, atf=atf)

    def calc_fit_tt_hb_w(self, lead=None, page=None):
        return self._hb.calc_fit_tt_hb_w(lead=lead, page=page)
    

    def set_t_bound(self, t_start=None, t_end=None, train_t=False, full_output=False):

        t_bound = np.array([0, self.get_t_size()], dtype=int)
        if full_output:
            return t_bound
        elif train_t:
            t_bound = self._train_t
        else:
            if t_start is not None:
                t_bound[0] = t_start
            if t_end is not None:
                t_bound[1] = t_end
        return t_bound


    def bulk_parameters(self, p=None, dist="norm", df=1E+18, nc=0, bound=None, t_start=None, t_end=None, train_t=False, full_output=False):

        if p is None:
            t_bound = self.set_t_bound(t_start=t_start, t_end=t_end, train_t=train_t, full_output=full_output)
            p = np.copy(self._t_series["p_event"][:, t_bound[0]:t_bound[1]])
        if bound=="left":
            p[1,:] = 1 - p[0,:]
        elif bound=="right":
            p[0,:] = p[1,:]
            p[1,:] = 1 - p[1,:]
        else:
            p[1,:] = 1 - p[1,:]

        if dist=="norm":
            Q = scipy.stats.norm().ppf(p)
            df = np.nan
            nc = np.nan
        elif dist=="t":
            Q = scipy.stats.t(df=df).ppf(p)
            nc = np.nan
        elif dist=="nct":
            Q = scipy.stats.nct(df=df, nc=nc).ppf(p)

        scale = np.diff(self._parameters.u.get_value())[0]/np.diff(Q, axis=0)[0]
        loc = (np.sum(self._parameters.u.get_value()) - scale*np.sum(Q, axis=0))/2

        return loc, scale, df, nc


    def bulk_norm_dev(self, bulk_bool, bulk, bound=None, t_start=None, t_end=None, train_t=True, full_output=False):
        loc, scale, df, nc = self.bulk_parameters(dist="norm", bound=bound, t_start=t_start, t_end=t_end, train_t=train_t, full_output=full_output)
        dist = scipy.stats.norm(loc=loc[bulk_bool],
                                scale=scale[bulk_bool])
        return -2*np.sum(np.log(dist.pdf(bulk)))

    def bulk_t_dev(self, x, bulk_bool, bulk, bound=None, t_start=None, t_end=None, train_t=True, full_output=False):
        loc, scale, df, nc = self.bulk_parameters(dist="t", df=x[0], bound=bound, t_start=t_start, t_end=t_end, train_t=train_t, full_output=full_output)
        dist = scipy.stats.t(df=df,
                                loc=loc[bulk_bool],
                                scale=scale[bulk_bool])

        return -2*np.sum(np.log(dist.pdf(bulk)))

    def bulk_nct_dev(self, x, bulk_bool, bulk, bound=None, t_start=None, t_end=None, train_t=True, full_output=False):
        loc, scale, df, nc = self.bulk_parameters(dist="nct", df=x[0], nc=x[1], bound=bound, t_start=t_start, t_end=t_end, train_t=train_t, full_output=full_output)
        dist = scipy.stats.nct(df=df, nc=nc,
                                loc=loc[bulk_bool],
                                scale=scale[bulk_bool])

        return -2*np.sum(np.log(dist.pdf(bulk)))

    
    def bulk_fit(self, p=None, dist=None, df_0=None, nc_0=None, bound=None, opt_method=None, t_start=None, t_end=None, train_t=True, full_output=True):

        if dist is None:
            dist = 'norm'
        if bound is None:
            bound = 'both'
        if df_0 is None:
            df_0 = 6
        if nc_0 is None:
            nc_0 = 0
        if opt_method is None:
            opt_method = 'SLSQP'

        t_bound = self.set_t_bound(t_start=t_start, t_end=t_end, train_t=train_t, full_output=False)
        y = self._df[0].values[t_bound[0]:t_bound[1]]

        bulk_bool = ~np.logical_or(y < self._parameters.u.get_value()[0], y > self._parameters.u.get_value()[1])
        bulk = y[bulk_bool]


        nct_bounds = np.array([
                                [2.1, 100],
                                [-3, 3]
                            ])

        se = np.full(2, np.nan)
        res = None
       

        if dist=="norm":
            def dev_fun(x=None):
                return self.bulk_norm_dev(bulk_bool, bulk, bound, t_start, t_end, train_t, False)
            x0 = np.array([])
        elif dist=="t":
            def dev_fun(x):
                return self.bulk_t_dev(x, bulk_bool, bulk, bound, t_start, t_end, train_t, False)
            x0 = np.array([df_0])
        elif dist=='nct': 
            def dev_fun(x):
                return self.bulk_nct_dev(x, bulk_bool, bulk, bound, t_start, t_end, train_t, False)
            x0 = np.array([df_0, nc_0])
        
        if opt_method == 'load':
            loc, scale, df, nc = self.bulk_parameters(dist=dist, df=df_0, nc=nc_0, bound=bound, t_start=t_start, t_end=t_end, train_t=train_t, full_output=full_output)
            log_L = -dev_fun(x=x0)/2
        
        else:
            if dist=="norm":
                fitted_success=None
                opt_timer = misc_fns.timer(reset_start=True)
                

                loc, scale, df, nc = self.bulk_parameters(dist=dist, bound=bound, t_start=t_start, t_end=t_end, train_t=train_t, full_output=full_output)
                log_L = -self.bulk_norm_dev(bulk_bool, bulk, bound, t_start, t_end, train_t, False)/2
            else:
                fitted_success=True
                

                opt_timer = misc_fns.timer(print_timer=False)
                res = scipy.optimize.minimize(dev_fun, x0=x0, method=opt_method,
                                                bounds=scipy.optimize.Bounds(lb=nct_bounds[0,0], ub=nct_bounds[0,1]))
                opt_timer.stop()

                if dist=="t":
                    loc, scale, df, nc = self.bulk_parameters(dist=dist, df=res.x[0], bound=bound, t_start=t_start, t_end=t_end, train_t=train_t, full_output=full_output)
                elif dist=='nct':
                    loc, scale, df, nc = self.bulk_parameters(dist=dist, df=res.x[0], nc=res.x[1], bound=bound, t_start=t_start, t_end=t_end, train_t=train_t, full_output=full_output)

                hess = numdifftools.Hessian(dev_fun, full_output=True, method="central", step=res.x/1E6)(res.x)[0]
                try:
                    np.linalg.inv(hess)
                except np.linalg.LinAlgError as err:
                    if 'Singular matrix' in str(err):
                        se = np.full(2, np.nan)
                        fitted_success=False
                if fitted_success:
                    se[:res.x.size] = np.sqrt(np.abs(np.diag(np.linalg.inv(hess))))

                log_L = -res.fun/2
       



        ret_dict =  {    
                        'hb':   {
                                    'a_u': {f'a_u_{1+i}': self._parameters.a_lambda.get_value()[i] for i in range(2)},
                                    'opt': self._parameters.calc_opt_settings_dict(),
                                    'mm': self._parameters.calc_mm_dict(),
                                    'db': {'dist': dist, 'bound': bound},
                                },
                        'ls': {'est': {'loc': loc, 'scale': scale}},
                        'nct':  {
                                    'est': {'df': df, 'nc': nc},
                                    'se': {'df': se[0], 'nc': se[1]},
                                },
                        'res':  {
                                    'log_L': log_L,
                                    'opt_time': opt_timer.get_timer_dict(),
                                    'res': res,
                                },
                    }


        return ret_dict






#---#========================
    #-----------------------
    # tail analysis functions
    #-----------------------
    #=======================

        
    def calc_VaR(self, a_q, bdist=None):


        VaR = -scipy.stats.genpareto(c=self._parameters.xi.get_value()[0],
                            loc=np.abs(self._parameters.u.get_value()[0]),
                            scale=self._t_series["sigma"][0,:]).ppf(1-(a_q/self._t_series["p_event"][0,:]))

        if bdist is not None:
            sub_u = np.isnan(VaR)
            VaR[sub_u] = bdist.ppf(a_q)[sub_u]
            
        return VaR

    def calc_ES(self, a_q, bdist=None, VaR=None):

        if VaR is None:
            VaR = self.calc_VaR(a_q=a_q, bdist=bdist)
            sub_u = np.isnan(VaR)
        else:
            sub_u = VaR > self._parameters.u.get_value()[0]

        ES = np.full_like(VaR, np.nan)
        mean_excess = -(self._t_series["sigma"][0,~sub_u] + self._parameters.xi.get_value()[0] * VaR[~sub_u]-self._parameters.u.get_value()[0])/(1 - self._parameters.xi.get_value()[0])
        ES[~sub_u] = VaR[~sub_u] + mean_excess

        if bdist is not None:
            loc = bdist.kwds["loc"]
            scale = bdist.kwds["scale"]
            if bdist.dist.name=="t" or bdist.dist.name=="nct":
                df = bdist.kwds["df"]
            else:
                df = None
            if bdist.dist.name=="nct":
                nc = bdist.kwds["nc"]
            else:
                nc = None
            
            ES_recess = misc_fns.bound_xint_dist(dist=bdist.dist.name, lb=self._parameters.u.get_value()[0], ub=VaR[sub_u], loc=loc[sub_u], scale=scale[sub_u], df=df, nc=nc)
            ES_excess = (self._parameters.u.get_value()[0] - self._t_series["sigma"][0,sub_u]/(1 - self._parameters.xi.get_value()[0]))*self._t_series["p_event"][0,sub_u]
        
            ES[sub_u] = (ES_excess + ES_recess) /  a_q

        return ES




#---#========================
    #-----------------------
    # simulation functions
    #-----------------------
    #=======================

    def simulate(self, run=False, parameters=None):

        if parameters is None:
            parameters = copy.deepcopy(self._parameters)

        sim = hawkes_simulation(D_nodes=self.get_D_nodes(), t_size=self.get_t_size(), dt_dt_fine=self.get_dt_dt_fine(), start_date=self._start_date, exchanges=self._exchanges,
                                            parameters=parameters)
        if run:
            sim.simulate()
        return sim

    def simulate_estimate(self, run=False, multivariate_mi=False):

        return self.simulate(run=True, parameters=copy.deepcopy(self._parameters_estimate))

    


    


class hawkes_simulation(hawkes_process):
    """Class for a multidimensional Hawkes process
    """


    def __init__(self, D_nodes=None, t_size=None, dt_dt_fine=None, exchanges=None, start_date=None, N_max_node=None, N_max_total=None,
                    parameters=None,
                    mode=None, mark=None, opt_method=None, fix_a_lambda=None,
                    a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None,
                    mu=None, gamma=None, 
                    phi_type=None, beta=None, nu=None, omega=None,
                    xi=None, varsigma=None, eta=None,
                    kappa_type=None, alpha=None):
        """Constructs a Hawkes process

        D_nodes -- the number of point processes

        T -- the end-time of the point processes (the start-time is defined to be 0)
        dt -- the discrete time-step of the point processes

        mu -- the background intensity of the point processes (scalar or vector)
        gamma -- the coefficients for self- and mutual-excitation of intensity (scalar or matrix)


        u -- threshold event magnitude of each point process (scalar or vector)
        xi -- parameterises the event magnitude GP distribution(s) (scalar or vector) -> exponent of GPD
        varsigma -- parameterises the event magnitude GP distribution(s) (scalar or vector) -> unconditional scale parameter
        eta -- parameterises the event magnitude GP distribution(s) (scalar or vector) -> coefficient of excitation in conditional scale parameteer (sigma)


        phi_type -- specifies the parametric form of the time kernel
            Exponential time kernel (phi_type = 'exp')

            Power-law time kernel (phi_type = 'pow')

        beta -- parameterises the exponential (phi_type = 'exp') time kernel -> coefficient of exponent
        nu -- parameterises the power-law (phi_type = 'pow') time kernel -> coefficient of time
        omega -- parameterises the power-law (phi_type = 'pow') time kernel -> exponent


        kappa_type -- specifies the parametric form of each processes intensity impact function (scalar or vector)
            Derived from event magnitude GPD (kappa_type = 0)

            Exponential impact function (kappa_type = 1)

            Power-law impact function (kappa_type = 1)

        alpha -- parameterises each processes impact function (scalar or vector)


        N_max_total -- the maximum number of total events for simulations
        M_max_node -- the maximum number of events in a process for simulations
        seed -- the seed for random number generation
        """

        if parameters is not None:
            if D_nodes is None:
                D_nodes = parameters._D_nodes
            else:
                if D_nodes != parameters._D_nodes:
                    raise ValueError("parameters must be of D_nodes = {}".format(D_nodes))

        super().__init__(D_nodes=D_nodes, t_size=t_size, dt_dt_fine=dt_dt_fine, exchanges=exchanges, start_date=start_date, N_max_node=N_max_node, N_max_total=N_max_total)

        if parameters is None:
            self._parameters = hawkes_parameters(D_nodes=self.get_D_nodes(), 
                                                    a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive,
                                                    opt_method=opt_method, fix_a_lambda=fix_a_lambda,
                                                    mode=mode, mark=mark, 
                                                    mu=mu, gamma=gamma, 
                                                    phi_type=phi_type, beta=beta, nu=nu, omega=omega,
                                                    xi=xi, varsigma=varsigma, eta=eta,
                                                    kappa_type=kappa_type, alpha=alpha)
        else:
            self._parameters = parameters


    

 #--#========================
    #-----------------------
    # simulation functions
    #-----------------------
    #=======================

    def simulate(self, N_max_node=None, N_max_total=None, ret=False):
        """

        """

        #--------------------------------
        # Set-up
        #--------------------------------
        # Reset process
        if N_max_node != None:
            self.set_N_max_node(N_max_node)
            self.set_N_max_total(N_max_total)
        self.reset_process()

        #--------------------------------
        # Purely exogenous phase
        #--------------------------------
        # Simulate time to the first event from exponential distribution
        [initial_node, initial_t_event] = self.find_initial_event()

        self._t_series["tilde_t"] = np.cumsum(np.outer(self._parameters.mu.get_value(), np.ones(self.get_t_size())), axis=1)

        # Fill-in simulation vectors up to initial event
        self._t_series["delta_tilde_t"][:,0:1+initial_t_event] = np.outer(self._parameters.mu.get_value(), np.ones(1+initial_t_event))
        self._t_series["p_event"][:,0:1+initial_t_event] = np.transpose(self._parameters.get_M_weight()) @ (1 - np.exp(-self._parameters.get_M_sum() @ self._t_series["delta_tilde_t"][:,0:1+initial_t_event]))
        
        # Trigger initial event
        self._events["list"].append([self.trigger_event(initial_node, t=initial_t_event, N_t=self._events["N"])])

        #--------------------------------
        # Endogenous and exogenous phase
        #--------------------------------
        for k in range(1+initial_t_event, self.get_t_size()):

            self._t_series["delta_tilde_t"][:, k] = np.diff(self._t_series["tilde_t"][:, k-1:k+1], axis=1)[:,0]
            self._t_series["p_event"][:,k] = np.transpose(self._parameters.get_M_weight()) @ (1 - np.exp(-self._parameters.get_M_sum() @ self._t_series["delta_tilde_t"][:,k]))
            # Stochastic event triggering
            if self._parameters._mode=="ci" or self._parameters._mode=="ci_s" or self._parameters._mode=="me":
                # Probability of event in next time step
                triggered = np.sum(np.cumsum(self._t_series["p_event"][:,k]) <= np.random.uniform())
                if triggered < 2:
                    self._events["list"].append(self.trigger_event(triggered, t=k, N_t=self._events["N"]))
            else:
                triggered = np.random.uniform(size=self.get_D_nodes()) < self._t_series["p_event"][:,k]
                if np.any(triggered != 0):
                    N_t = np.copy(self._events["N"])
                    for i in range(self.get_D_nodes()):
                        if triggered[i]:
                            self._events["list"].append(self.trigger_event(i, t=k, N_t=N_t))
            if np.any(self._events["N"] == self.get_N_max_node()):
                break
            if sum(self._events["N"]) == self.get_N_max_total():
                break

        self.shrink_events() 
        events = copy.deepcopy(self._events)
        t_series = copy.deepcopy(self._t_series)

        #--------------------------------
        # Post-simulation processing
        #--------------------------------
        
        self.set_events_N_t()
        self.set_t_unique()
        self.set_t_series()

        if ret:
            return events, t_series

        


    def find_initial_event(self):

        mu = np.transpose(self._parameters.get_M_weight()) @ self._parameters.get_M_sum() @ self._parameters.mu.get_value()
        initial_t_candidates = np.random.exponential(scale = 1/mu)
        initial_node = int(np.argmin(initial_t_candidates))
        initial_t_event = int(initial_t_candidates[initial_node])

        return initial_node, initial_t_event


    def trigger_event(self, node, t, N_t=None):

        if type(t) != int:
            raise TypeError("t must be of type: int")

        if N_t is None:
            N_t = np.sum(np.logical_and(self._events["t"] < t, self._events["t"] >= 0), axis=1)


        self._events["t"][node, N_t[node]] = t
        self._events["TS"][node, N_t[node]] = self._TS["t"][node, t]


        # lambda_tilde_t
        self._events["lambda"][node, N_t[node]] = self.lambda_forecast_disc(t=[t], N=N_t)[node, 0]
        self._events["sigma"][node, N_t[node]] = self._parameters.calc_sigma(nodes=node, lambda_t=self._events["lambda"][node, N_t[node]])
        #self._events["tilde_t_t"][node, N_t[node]] = np.diff(self.tilde_t_forecast(t=[t-1, t], N=N_t), axis=1)[node]
        #self._events["sigma"][node, N_t[node]] = self._parameters.calc_sigma(nodes=node, tilde_t_t=self._events["tilde_t_t"][node, N_t[node]])



        self._events['F_M'][node, N_t[node]] = scipy.stats.uniform().rvs()
        self._events['Y_M'][node, N_t[node]] = self._parameters.gpd_ppf(nodes=node, F=self._events['F_M'][node, N_t[node]])
        self._events["kappa"][node, N_t[node]] = self._parameters.kappa(nodes=node, z=self._events['Y_M'][node, N_t[node]])
        self._events['M'][node, N_t[node]] = self._events['Y_M'][node, N_t[node]]*self._events["sigma"][node, N_t[node]]

        self._t_series["tilde_t"][:, 1+t:] = self._t_series["tilde_t"][:, 1+t:] + np.outer(self._parameters.gamma.get_value()[:,node], ((1 + self._parameters.int_phi(np.array(range(1, self.get_t_size()-t)), node=node)))*self._events["kappa"][node, N_t[node]])


    
        # lambda_tilde_t
        event = [
                    node, 
                    self._events["t"][node, N_t[node]],
                    self._events["TS"][node, N_t[node]], 
                    self._events['M'][node, N_t[node]],
                    self._events['Y_M'][node, N_t[node]], 
                    self._events['F_M'][node, N_t[node]], 
                    self._events["kappa"][node, N_t[node]],
                    self._events["sigma"][node, N_t[node]],
                    self._events["lambda"][node, N_t[node]],
                    #self._t_series["tilde_t_t"][node, N_t[node]],
                ]

        self._events["N"][node] += 1

        return event


class hawkes_bulk_dist():
    """Class for all two-tail Hawkes bulk distribution distributions
    """
    def __init__(self, hawkes, dist=None, bound=None, set_fit=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        """Constructor for hawkes_bulk_dist class
        """
        self.set_hb_d(hawkes=hawkes, dist=dist, bound=bound, set_fit=set_fit, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t, reset=True)


    def set_hb_d(self, hawkes, dist=None, bound=None, set_fit=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, reset=False):
        if dist is not None:
            self._dist = misc_fns.make_iterable_array(dist)
        elif reset:
           self._dist = misc_fns.make_iterable_array("norm")

        self.hb_d = calc_hb_d(hawkes, dist=self._dist, bound=bound, set_fit=set_fit, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)


    def set_vrs(self, hawkes, a_q, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        for dist in self.hb_d:
            self.hb_d[dist].set_vrs(hawkes=hawkes, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)



    def set_vrs_eval(self, hawkes, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        for dist in self.hb_d:
            self.hb_d[dist].set_vrs_eval(hawkes=hawkes, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

    def calc_vrs_eval(self, lead=None, atf=None):
        df = pd.DataFrame()
        for d in self.hb_d:
            df = df.append(self.hb_d[d].calc_vrs_eval(lead=lead, atf=atf))
        return df

    def calc_fit_tt_hb_w(self, lead=None, page=None):
        df = pd.DataFrame()
        for d in self.hb_d:
            df = df.append(self.hb_d[d].calc_fit_tt_hb_w(lead=lead, page=page))
        return df


class hawkes_bulk_bound():
    """Class for all two-tail Hawkes bulk distribution boundary conditions
    """
    def __init__(self, hawkes, dist=None, bound=None, set_fit=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        """Constructor for hawkes_bulk_bound class
        """
        self.set_hb_b(hawkes=hawkes, dist=dist, bound=bound, set_fit=set_fit, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t, reset=True)


    def set_hb_b(self, hawkes, dist=None, bound=None, set_fit=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, reset=False):
        if bound is not None:
            self._bound = misc_fns.make_iterable_array(bound)
        elif reset:
           self._bound = misc_fns.make_iterable_array("both")

        self.hb_b = calc_hb_b(hawkes=hawkes, dist=dist, bound=self._bound, set_fit=set_fit, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)



    def set_vrs(self, hawkes, a_q, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        for bound in self.hb_b:
            self.hb_b[bound].set_vrs(hawkes=hawkes, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

    def set_vrs_eval(self, hawkes, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        for bound in self.hb_b:
            self.hb_b[bound].set_vrs_eval(hawkes=hawkes, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

    def calc_vrs_eval(self, lead=None, atf=None):
        df = pd.DataFrame()
        for b in self.hb_b:
            df = df.append(self.hb_b[b].calc_vrs_eval(lead=lead, atf=atf))
        return df

    def calc_fit_tt_hb_w(self, lead=None, page=None):
        df = pd.DataFrame()
        for b in self.hb_b:
            df = df.append(self.hb_b[b].calc_fit_tt_hb_w(lead=lead, page=page), ignore_index=True)
        return df




class hawkes_bulk():
    """Class for all two-tail Hawkes bulk distribution
    """
    def __init__(self, hawkes, dist=None, bound=None, set_fit=None, a_q=None,  set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):
        """Constructor for hawkes_bulk class
        """

        if dist is not None:
            self._dist = dist
        else:
            self._dist = "norm"
        if bound is not None:
            self._bound = bound
        else:
            self._bound = "both"
        self._train_t_end = hawkes._train_t[1]
        self._train_t = hawkes._train_t

    
        self.set_hawkes_meta(hawkes=hawkes)

        if set_fit is None:
            set_fit = True
        if set_fit:
            self.set_bdist(hawkes=hawkes)
            #self.set_fit_2(hawkes=hawkes)

        if set_fit and a_q is not None:
            self.set_vrs(hawkes=hawkes, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)


    def set_hawkes_meta(self, hawkes):

        self._hawkes_T_M_N = hawkes.calc_tt_eval_blr_atf_T_N_M()
        self._hawkes_N_data = {suff: {tail: {period: hawkes.calc_N_M_data(tail=tail, period=period, component=suff) for period in ['all', 'train', 'forecast']} for tail in ['both', 'left', 'right']} for suff in ['u', 'lambda', 'M']}
        self._hawkes_N_params = {suff: {tail: hawkes._parameters.calc_N_params(tail=tail, component=suff) for tail in ['both', 'left', 'right']} for suff in ['u', 'lambda', 'M']}
        self._hawkes_events_t = np.copy(hawkes._events['t'])
        self._hawkes_events_N = np.copy(hawkes._events['N'])
        self._hawkes_N_params_a_u = np.unique(hawkes._parameters.a_u.get_value()).size




    def set_bdist(self, hawkes, df_0=None, nc_0=None, opt_method=None):

        self._bdist = bulk_gp.bulk_gp(data=hawkes._df[0].values, 
                                    bulk_dist=self._dist, fit_method='fit_bulk', train_t=self._train_t, bulk_bound=self._bound, opt_method=opt_method,
                                    bulk_df=df_0, bulk_nc=nc_0,
                                    gp_c=hawkes._parameters.xi.get_value(),
                                    gp_loc=hawkes._parameters.u.get_value(), 
                                    gp_scale=hawkes._t_series['sigma'],
                                    gp_weight=hawkes._t_series['p_event'])


        self._bdist._bulk_fit['hb'] =   {
                                            'a_u': {f'a_u_{1+i}': hawkes._parameters.a_u.get_value()[i] for i in range(2)},
                                            'opt': hawkes._parameters.calc_opt_settings_dict(),
                                            'mm': hawkes._parameters.calc_mm_dict(),
                                            'db': {'dist': self._dist, 'bound': self._bound},
                                        }

        self.set_F_X(hawkes=hawkes)



    def set_F_X(self, hawkes):

        self.F_X = self._bdist.cdf(hawkes._df[0].values)
        self.f_X = self._bdist.pdf(hawkes._df[0].values)
        self.log_L = cvl.calc_log_L(f_X=self.f_X, train_t_end=hawkes._train_t[1])

        
    def set_fit_2(self, hawkes, df_0=None, nc_0=None, opt_method=None):
        self._fit_2 = hawkes.bulk_fit(dist=self._dist, bound=self._bound, df_0=df_0, nc_0=nc_0, opt_method=opt_method)
        self.set_bdist_2()
        self.set_F_X_2(hawkes=hawkes)

    def load_bulk(self, hawkes, df_0=None, nc_0=None):
        self.set_bdist(hawkes=hawkes, df_0=df_0, nc_0=nc_0, opt_method='load')

    def load_bulk_2(self, hawkes, df_0=None, nc_0=None):
        self.set_fit_2(hawkes=hawkes, df_0=df_0, nc_0=nc_0, opt_method='load')

    def set_bdist_2(self):
        if self._dist=="norm":
            self._bdist_2 = scipy.stats.norm(loc=self._fit_2['ls']['est']['loc'], scale=self._fit_2['ls']['est']['scale'])
        elif self._dist=="t":
            self._bdist_2 = scipy.stats.t(loc=self._fit_2['ls']['est']['loc'], scale=self._fit_2['ls']['est']['scale'], df=self._fit_2['nct']['est']['df'])
        elif self._dist=="nct":
            self._bdist_2 = scipy.stats.nct(loc=self._fit_2['ls']['est']['loc'], scale=self._fit_2['ls']['est']['scale'], df=self._fit_2['nct']['est']['df'], nc=self._fit_2['nct']['est']['nc'])
        self.set_tdist_2()

    def set_tdist_2(self):
        if self._dist=="norm":
            self._tdist_2 = scipy.stats.norm()
        elif self._dist=="t":
            self._tdist_2 = scipy.stats.t(df=self._bdist_2.kwds['df'])
        elif self._dist=="nct":
            self._tdist_2 = scipy.stats.nct(df=self._bdist_2.kwds['df'], nc=self._bdist_2.kwds['nc'])


    def calc_emmdb_dict(self, hawkes=None):
        if hawkes is None:
            return self.calc_db_dict()
        else:
            return {
                        **{f"a_u_{i}": hawkes._parameters.a_u.get_value()[i] for i in range(2)}
                        **hawkes._parameters.calc_opt_settings_dict(), 
                        **hawkes._parameters.calc_mm_dict(),  
                        **self.calc_db_dict()
                    }

    def calc_db_dict(self):
        return {p: self.__dict__[f"_{p}"] for p in ["dist", "bound"]}

    def calc_fit_emmdb_dict(self):

        return {
                    **self._bdist._bulk_fit['hb']['a_u'],
                    **self._bdist._bulk_fit['hb']['mm'],
                    **self._bdist._bulk_fit['hb']['db'],
                }

    def calc_nct_est_dict(self):
        hb_dict =   {
                        **self.calc_fit_emmdb_dict(),
                        **self._bdist._bulk_fit['nct']['est'],
                        **self.calc_opt_dict(),
                    }
        return hb_dict

    def calc_nct_se_dict(self):
        hb_dict =   {
                        **self.calc_fit_emmdb_dict(),
                        **self._bdist._bulk_fit['nct']['se'],
                    }
        return hb_dict




    def calc_log_L_X(self, part=None, period=None):

        pp_bool = self.calc_part_period_bool(part=part, period=period)

        return np.sum(np.log(self.f_X[pp_bool]))




    def calc_tt_hb_eval(self, lead=None):
        if lead is None:
            lead = dict()

        eval_abblr_atf = self.calc_tt_hb_eval_abblr_atf()

        columns = [     
                        [k for k in lead],
                        ["part", "period"],
                        [k for k in eval_abblr_atf['all']['all'].keys()],
                ]
        df_dict = {c: [] for c in np.concatenate(columns)}
        for part in eval_abblr_atf:
            for period in eval_abblr_atf[part]:
                df_dict['part'].append(part)
                df_dict['period'].append(period)
                for k in columns[-1]:
                    df_dict[k].append(eval_abblr_atf[part][period][k])
        for k in columns[0]:
            df_dict[k] = np.repeat(lead[k], len(df_dict['part']))

        return pd.DataFrame(df_dict)


    def calc_tt_hb_eval_abblr_atf(self):


        eval = misc_fns.get_abblr_atf()

        eval_abblr_atf_log_L = self.calc_tt_hb_eval_abblr_atf_log_L()

        for part in eval:
            for period in eval[part]:
                
                eval[part][period] = {
                                        **eval_abblr_atf_log_L[part][period],
                                    }

        return eval


    def calc_tt_hb_eval_abblr_atf_log_L(self):


        eval = misc_fns.get_abblr_atf()

        eval_abblr_atf_T_N_M = self.calc_tt_hb_eval_abblr_atf_T_N_M()

        for part in eval:
            for period in eval[part]:
                
                
                eval[part][period] = {
                                        **eval_abblr_atf_T_N_M[part][period],
                                        **self.calc_tt_hb_eval_abblr_atf_log_L_suff(part=part, period=period, flatten=True),
                                    }

        return eval


    def calc_tt_hb_eval_abblr_atf_T_N_M(self):

        eval = misc_fns.get_abblr_atf()
        for part in eval:
            for period in eval[part]:

                eval[part][period] = {
                                        'T': self.calc_T(period=period),
                                        'N_M': self.calc_N_M(part=part, period=period),
                                    }

        return eval

    def calc_T(self, period=None):

        if period is None:
            period = 'all'
        return np.copy(self._hawkes_T_M_N['both'][period]['T'])

        t_size = self._hawkes_t_size
        if period=='train':
            return np.diff(self._train_t)[0]
        elif period=='forecast':
            return t_size - self._train_t[-1]
        else:
            return t_size

    def calc_N_M(self, part=None, period=None):

        return np.sum(self.calc_part_period_bool(part=part, period=period))

    
    def calc_N_params(self, part=None):


        N_param_B = {
                        'norm': 0,
                        't': 1,
                        'nct': 2,
                    }


        if np.isin(part, ['left', 'right']):
            part_eff = part
        elif part=='bulk':
            part_eff = self._bound
        else:
            part_eff = 'both'
        
        return self._hawkes_N_params_a_u + self._hawkes_N_params['u'][part_eff] + N_param_B[self._dist]






    def calc_part_bool(self, part=None):

        if part is None:
            part = 'all'

        part_bool = np.full(self._bdist.get_t_size(), False)

        if np.isin(part, ['both', 'bulk']):
            both_t = np.concatenate([self._hawkes_events_t[i,:self._hawkes_events_N[i]] for i in range(self._hawkes_events_N.size)])
            if part=='bulk':
                part_bool[:] = True
                part_bool[both_t] = False
            else:
                part_bool[both_t] = True
        elif np.isin(part, ['left', 'right']):
            if part=='left':
                i = 0
            elif part=='right':
                i = 1
            tail_i = np.copy(self._hawkes_events_t[i,:self._hawkes_events_N[i]])
            part_bool[tail_i] = True
        else:
            part_bool[:] = True

        return part_bool

    
    def calc_period_bool(self, period=None):

        if period is None:
            period = 'all'

        period_bool = np.full(self._bdist.get_t_size(), False)
        if period=='train':
            period_bool[self._train_t[0]:self._train_t[-1]] = True
        elif period=='forecast':
            period_bool[self._train_t[-1]:] = True
        else:
            period_bool[:] = True
        

        return period_bool


    def calc_part_period_bool(self, part=None, period=None):


        part_bool = self.calc_part_bool(part=part)
        period_bool = self.calc_period_bool(period=period)

        return np.logical_and(part_bool, period_bool)

        

        





    def calc_tt_hb_eval_abblr_atf_log_L_suff(self, part=None, period=None, flatten=None):

        if flatten is None:
            flatten=False

        log_L_suff = {suff: {dev: None for dev in ['N_data', 'N_params', 'log_L', 'AIC', 'BIC']} for suff in ['X']}
        log_L_suff['X']['log_L'] = self.calc_log_L_X(part=part, period=period)

        for suff in log_L_suff:

            log_L_suff[suff]['N_data'] = self.calc_N_M(part=part, period=period) 
            log_L_suff[suff]['N_params'] = self.calc_N_params(part=part) 


            log_L_suff[suff]['AIC'] = misc_fns.calc_AIC(N_params=log_L_suff[suff]['N_params'], log_L=log_L_suff[suff]['log_L'])
            log_L_suff[suff]['BIC'] = misc_fns.calc_BIC(N_data=log_L_suff[suff]['N_data'], N_params=log_L_suff[suff]['N_params'], log_L=log_L_suff[suff]['log_L'])

        if flatten:
            return {f"{val}_{suff}": log_L_suff[suff][val] for suff in log_L_suff for val in log_L_suff[suff]}
        else:
            return log_L_suff

  
    def calc_opt_dict(self):
        hb_dict =   {
                        "log_L_H_B": self._bdist._bulk_fit['res']['log_L'],
                        **{f"opt_{t}": self._bdist._bulk_fit['res']['opt_time'][t] for t in self._bdist._bulk_fit['res']['opt_time']}
                    }
        return hb_dict

 
    def set_F_X_2(self, hawkes):

        self.F_X_2 = self._bdist_2.cdf(hawkes._df[0].values)
        self.f_X_2 = self._bdist_2.pdf(hawkes._df[0].values)

        for i in range(2):
            self.F_X[hawkes._events["t"][i,:hawkes._events["N"][i]]] = i + ((-1)**i)*(1 - hawkes._events['F_M'][i,:hawkes._events["N"][i]]) * hawkes._t_series["p_event"][i,hawkes._events["t"][i,:hawkes._events["N"][i]]]
            self.f_X[hawkes._events["t"][i,:hawkes._events["N"][i]]] = hawkes._t_series["p_event"][i,hawkes._events["t"][i,:hawkes._events["N"][i]]] * scipy.stats.genpareto(c=hawkes._parameters.xi.get_value()[i],
                    scale=hawkes._events["sigma"][i,:hawkes._events["N"][i]]).pdf(hawkes._events['M'][i,:hawkes._events["N"][i]]) 
        
        self.log_L_2 = cvl.calc_log_L(f_X=self.f_X_2, train_t_end=hawkes._train_t[1])



    def set_vrs(self, hawkes, a_q, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):

        self._bdist.set_vrs(a_q=a_q, y=hawkes._df[0].values, train_t_end=self._train_t_end, 
                                            set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, 
                                            save_t=save_t)

    def set_vrs_eval(self, hawkes, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):

        self._bdist.vrs.set_vrs_eval(y=hawkes._df[0].values, train_t_end=self._train_t_end, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)


    def set_vrs_2(self, hawkes, a_q, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):

        self.vrs_2 = vares_3.vares_a(hawkes=hawkes, a_q=a_q, bdist=self._bdist_2, tdist=self._tdist_2, y=hawkes._df[0].values, train_t_end=self._train_t_end, 
                                            set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, 
                                            save_t=save_t)

    def set_vrs_2_eval(self, hawkes, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):

        self.vrs_2.set_vrs_eval(y=hawkes._df[0].values, train_t_end=self._train_t_end, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)




    
    def calc_vrs_eval(self, lead=None, atf=None):
        if lead is None:
            lead = dict()
        
        return self._bdist.vrs.calc_vrs_eval(lead={**lead, **self.calc_fit_emmdb_dict()}, atf=atf)

    def calc_fit_tt_hb_w_dict(self, page=None):

        if page is None:
            page = 'est'

        if page=='est':
            return self.calc_nct_est_dict()
        elif page=='se':
            return self.calc_nct_se_dict()



    def calc_fit_tt_hb_eval_w(self, lead=None):

        if lead is None:
                lead = dict()
        

        eval = self.calc_tt_hb_eval_abblr_atf_log_L()

        emmdb = self.calc_fit_emmdb_dict(),
        columns = [     
                        [k for k in lead],
                        [k for k in emmdb],
                        [k for k in eval.keys()],
                ]
        df_dict = {c: [] for c in np.concatenate(columns)}

        N = np.size(eval[columns[-1][0]])
        for k in columns[-1]:
            df_dict[k].append(eval[k])
        for c in range(2):
            for k in columns[0]:
                df_dict[k] = np.repeat(lead[k], N)

        return pd.DataFrame(df_dict)


    def calc_fit_tt_hb_w(self, lead=None, page=None):
        if lead is None:
            lead = dict()

        if page=='eval':
            
            return self.calc_tt_hb_eval(lead={**lead, **self.calc_fit_emmdb_dict()})
        else:
            

            hb_dict = self.calc_fit_tt_hb_w_dict(page=page)
            columns = [     
                            [k for k in lead],
                            [k for k in hb_dict.keys()],
                    ]
            df_dict = {c: [] for c in np.concatenate(columns)}
            
            for k in columns[-1]:
                df_dict[k].append(hb_dict[k])
            for k in columns[0]:
                df_dict[k] = np.repeat(lead[k], 1)

            return pd.DataFrame(df_dict)




def calc_hb_d(hawkes, dist, bound=None, set_fit=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):

    return {str(d): hawkes_bulk_bound(hawkes, dist=str(d), set_fit=set_fit, bound=bound, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t) for d in misc_fns.make_iterable_array(dist)}


def calc_hb_b(hawkes, bound, dist=None, set_fit=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None):

    return {str(b): hawkes_bulk(hawkes, dist=dist, bound=str(b), set_fit=set_fit, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t) for b in misc_fns.make_iterable_array(bound)}


#============
#============
#
# 2T-POT
#
#===========
#===========

class tt_pot(hawkes_threshold_opt):
    """Class for two-tailed peaks-over-threshold model group
    """
    def __init__(self, data, D_nodes=None, a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None, base_u_dir=None, exchanges=None, dt_dt_fine=None, train_t_end=None, meta_series=None, fit_fn_pre=None, fit_colval=None, opt_method=None, fix_a_lambda=None):
        """Constructor tt_pot class
        """

        if D_nodes is None:
            D_nodes = 2

        self._D_nodes = D_nodes


        if base_u_dir is None:
            base_u_dir = ['left', 'right']
        base_u_dir = misc_fns.make_iterable_array(base_u_dir)

        if u_dir is None:
           u_dir =  np.concatenate([base_u_dir for k in range(((len(base_u_dir)-1)+self.get_D_nodes())//2)])[:self.get_D_nodes()]
        
        

        self._tt_fit_time = {t: None for t in ["start", "end", "delta"]}

        self.set_fit_fn_pre(fit_fn_pre=fit_fn_pre, reset=True)
        self.set_fit_colval(fit_colval=fit_colval, reset=True)


        self.reset_rw(meta_series=meta_series)

        self.reset_params_threshold()
        self.set_tt(data=data, a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, exchanges=exchanges, dt_dt_fine=dt_dt_fine, train_t_end=train_t_end, opt_method=opt_method, fix_a_lambda=fix_a_lambda, reset=True)

    


    

    def set_a_u(self, a_u=None, reset=None, set_tt=None):

        self.a_u.set_value(value=a_u, reset=reset)

        if set_tt is None:
            set_tt = True

        if set_tt:
            for mode in self.tt:
                for mark in self.tt[mode]:
                    self.tt[mode][mark]._parameters.a_u.set_value(self.a_u.get_value(), reset=reset)


    def set_u_dir(self, u_dir=None, reset=None, set_tt=None):

        self.u_dir.set_value(value=u_dir, reset=reset)

        if set_tt is None:
            set_tt = True

        if set_tt:
            for mode in self.tt:
                for mark in self.tt[mode]:
                    self.tt[mode][mark]._parameters.u_dir.set_value(self.u_dir.get_value(), reset=reset)

    def set_u(self, u=None, reset=None, set_tt=None):

        self.u.set_value(value=u, reset=reset)

        if set_tt is None:
            set_tt = True
        
        if set_tt:
            for mode in self.tt:
                for mark in self.tt[mode]:
                    self.tt[mode][mark]._parameters.u.set_value(self.u.get_value(), reset=reset)

    def set_u_zero(self, u_zero=None, reset=None, set_tt=None):

        self.u_zero.set_value(value=u_zero, reset=reset)

        if set_tt is None:
            set_tt = True
        
        if set_tt:
            for mode in self.tt:
                for mark in self.tt[mode]:
                    self.tt[mode][mark]._parameters.u_zero.set_value(self.u_zero.get_value(), reset=reset)



    def set_params_threshold(self, a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None, 
                           reset=None, reset_a_u=None, reset_u_dir=None, reset_u=None, reset_u_zero=None,
                           set_tt=None, set_tt_a_u=None, set_tt_u_dir=None, set_tt_u=None, set_tt_u_zero=None,):
        
        resets = threshold.calc_resets_params_threshold(reset=reset, a_u=reset_a_u, u_dir=reset_u_dir, u=reset_u, u_zero=reset_u_zero)
        set_tts = threshold.calc_resets_params_threshold(reset=set_tt, a_u=set_tt_a_u, u_dir=set_tt_u_dir, u=set_tt_u, u_zero=set_tt_u_zero)

        self._set_params_threshold(a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, 
                           reset=reset, reset_a_u=reset_a_u, reset_u_dir=reset_u_dir, reset_u=reset_u, reset_u_zero=reset_u_zero)

        self.set_a_u(a_u=self.a_u.get_value(), reset=resets['a_u'], set_tt=set_tts['a_u'])
        self.set_u_dir(u_dir=self.u_dir.get_value(), reset=resets['u_dir'], set_tt=set_tts['u_dir'])
        self.set_u(u=self.u.get_value(), reset=resets['u'], set_tt=set_tts['u'])
        self.set_u_zero(u_zero=self.u_zero.get_value(), reset=resets['u_zero'], set_tt=set_tts['u_zero'])



    def reset_rw(self, meta_series=None):
        self._rw = dict()
        self._rw['meta'] = dict()
        self.set_meta_series(meta_series=meta_series)

        for cat in ['fit', 'vares']:
            self._rw[cat] = {rw: dict() for rw in ['r', 'w']}

    
        self._rw['fit']['w']['tt'] = {
                                        fn: pd.DataFrame() for fn in ['meta', 'est', 'se', 'eval', 'eval_static']
                                    }

        self._rw['fit']['w']['tt_hb'] = {
                                            fn: pd.DataFrame() for fn in ['est', 'se', 'eval']
                                        }
        self._rw['vares']['w']['tt_hb'] = {
                                            fn: pd.DataFrame() for fn in ['vrs']
                                        }



    def set_meta_series(self, meta_series=None):
        if meta_series is not None:
            self._rw['meta']['series'] = meta_series
        


    def set_tt(self, data, a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None, exchanges=None, dt_dt_fine=None, train_t_end=None, opt_method=None, fix_a_lambda=None, reset=False):


        self._exchanges = exchanges

        if dt_dt_fine is not None:
            self._dt_dt_fine = dt_dt_fine
        elif reset:
            self._dt_dt_fine = 24

        self.set_opt_settings(opt_method=opt_method, fix_a_lambda=fix_a_lambda, reset=True, set_tt=False)
        self.set_params_threshold(a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, reset=False, set_tt=False)
        self.tt = create_models_2T_POT_nt(data=data, a_u=self.a_u.get_value(), u_dir=self.u_dir.get_value(), u=self.u.get_value(), u_zero=self.u_zero.get_value(), exchanges=self._exchanges, dt_dt_fine=self._dt_dt_fine, train_t_end=train_t_end, opt_method=self.get_opt_method(), fix_a_lambda=self.get_fix_a_lambda())



    def set_opt_settings(self, opt_method=None, fix_a_lambda=None, 
                         reset=None, reset_opt_method=None, reset_fix_a_lambda=None,
                         set_tt=None, set_tt_opt_method=None, set_tt_fix_a_lambda=None, 
                         free=None):

        resets = hwks_fns.calc_resets_opt_setting(reset=reset, opt_method=reset_opt_method, fix_a_lambda=reset_fix_a_lambda)
        set_tts = hwks_fns.calc_resets_opt_setting(reset=set_tt, opt_method=set_tt_opt_method, fix_a_lambda=set_tt_fix_a_lambda)

        self.set_opt_method(opt_method=opt_method, reset=resets['opt_method'], set_tt=set_tts['opt_method'])
        self.set_fix_a_lambda(fix_a_lambda=fix_a_lambda, reset=resets['fix_a_lambda'], set_tt=set_tts['fix_a_lambda'], free=free)
        
        if set_tt is not None:
            set_tt_opt_method = set_tt
            set_tt_fix_a_lambda = set_tt
            
            

    def set_opt_method(self, opt_method=None, reset=None, set_tt=None):

        self._set_opt_method(opt_method=opt_method, reset=reset)

        if set_tt is None:
            set_tt = True

        if set_tt:
            for mode in self.tt:
                for mark in self.tt[mode]:
                    self.tt[mode][mark]._parameters.set_opt_method(opt_method=self.get_opt_method(), reset=reset)
    

    def set_fix_a_lambda(self, fix_a_lambda=None, reset=None, set_tt=None, free=None):

        self._set_fix_a_lambda(fix_a_lambda=fix_a_lambda, reset=reset)

        if set_tt is None:
            set_tt = True
        if free is None:
            free = True

        if set_tt:
            for mode in self.tt:
                for mark in self.tt[mode]:
                    self.tt[mode][mark]._parameters.set_fix_a_lambda(fix_a_lambda=self.get_fix_a_lambda(), reset=reset, free=free)


    

    def set_tt_fit(self, print_tt_fit_time=None, print_res_time=None, append_w=None, opt_method=None, fix_a_lambda=None, fp0=None):

        self.set_opt_settings(opt_method=opt_method, fix_a_lambda=fix_a_lambda, reset=True, set_tt=False)

        self._tt_fit = fit_models_2T_POT_nt(self.tt, print_tt_fit_time=print_tt_fit_time, print_res_time=print_res_time, opt_method=self.get_opt_method(), fix_a_lambda=self.get_fix_a_lambda(), fp0=fp0)
        self._tt_fit_time = self._tt_fit["meta_fit"]

        if append_w is None:
            append_w = True
        if append_w:
            w = self.calc_opt_w()
            for fn in w:
                self._rw['fit']['w']['tt'][fn] = self._rw['fit']['w']['tt'][fn].append(w[fn])

    def set_tt_bi_lr(self, fit_bi_lr=None, fp0_bi=None, print_res_time=None, append_w=None, opt_method=None, fix_a_lambda=None):
        for mode in ['bi_l', 'bi_r']:
            self.tt[mode] = copy.deepcopy(self.tt['bi'])
            for mark in self.tt[mode]:
                self.tt[mode][mark]._parameters.set_mode(mode=mode)

        if fit_bi_lr is None:
            fit_bi_lr = True
        if fp0_bi is None:
            fp0_bi = True
        if fit_bi_lr:
            for mode in ['bi_l', 'bi_r']:
                for mark in self.tt[mode]:
                    if fp0_bi:
                        fp0 = self.tt['bi'][mark]._parameters.flatten()
                    else:
                        fp0 = None
                    self.tt[mode][mark].set_parameters_estimate(flat_parameters_0=fp0, print_res_time=print_res_time, opt_method=opt_method, fix_a_lambda=fix_a_lambda)



        if append_w is None:
            append_w = True
        if append_w:
            w = self.calc_opt_w()
            for fn in w:
                self._rw['fit']['w']['tt'][fn] = self._rw['fit']['w']['tt'][fn].append(w[fn])


    def set_hb(self, dist=None, bound=None, mode_bound=None, set_fit=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, append_fit_tt_hb_w=None, append_vrs_tt_hb_w=None):
        if mode_bound is not None:
            self._mode_bound = mode_bound
        else:
            self._mode_bound = None

        for mode in self.tt:
            if self._mode_bound is not None:
                mbound = self._mode_bound[mode]
            else:
                mbound = bound
            for mark in self.tt[mode]:
                self.tt[mode][mark].set_hb(dist=dist, bound=mbound, set_fit=set_fit, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

        if append_fit_tt_hb_w is None:
            append_fit_tt_hb_w = True
        if append_fit_tt_hb_w is None:
            append_fit_tt_hb_w = True


        if append_fit_tt_hb_w:
            self.append_fit_tt_hb_w()
        if append_vrs_tt_hb_w and set_eval:
            self.append_vares_w()

        

    




    def set_hb_mp(self, dist=None, bound=None, mode_bound=None, set_fit=None, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, append_fit_tt_hb_w=None, append_vrs_tt_hb_w=None, factor=None, print_mp_time=None):


        print("set_tt_hb_vrs")
        if print_mp_time is None:
            print_mp_time = True
        hb_vrs_timer = misc_fns.timer(print_timer=print_mp_time)

        if set_fit is None:
            set_fit = True
        if set_eval is not None:
            set_eval = True
        if append_fit_tt_hb_w is not None:
            append_fit_tt_hb_w = True
        if append_vrs_tt_hb_w is not None:
            append_vrs_tt_hb_w = True
        if save_t is not None:
            save_t = False

        if mode_bound is not None:
            self._mode_bound = mode_bound
        else:
            self._mode_bound = None
        

        mm = [[mode, mark] for mode in self.tt for mark in self.tt[mode]]
        print(mm)

        for m in range(len(mm)):
            if self._mode_bound is not None:
                bound = self._mode_bound[mm[m][0]]
            self.tt[mm[m][0]][mm[m][1]].set_hb_mp_params(dist=dist, bound=bound, set_fit=set_fit, a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)


        fres, self._hb_vrs_time_run = misc_fns.mp_funs_factor(funs=[self.tt[mm[m][0]][mm[m][1]].set_hb_mp for m in range(len(mm))], 
                                    factor=factor, print_mp_time=print_mp_time)

        for m in range(len(mm)):
            self.tt[mm[m][0]][mm[m][1]]._hb = fres[m]
        

        if append_fit_tt_hb_w:
            self.append_fit_tt_hb_w()
        if append_vrs_tt_hb_w and set_eval:
            self.append_vares_w()

        hb_vrs_timer.stop()
        self._hb_vrs_time =  hb_vrs_timer.get_timer_dict()



    def set_vrs_mp(self, a_q=None, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, append_vrs_tt_hb_w=None, factor=None, print_mp_time=None):

        print("set_tt_hb_vrs")
        if print_mp_time is None:
            print_mp_time = True
        hb_vrs_timer = misc_fns.timer(print_timer=print_mp_time)

        if set_eval is not None:
            set_eval = True
        if append_vrs_tt_hb_w is not None:
            append_vrs_tt_hb_w = True
        if save_t is not None:
            save_t = False

        

        mm = [[mode, mark] for mode in self.tt for mark in self.tt[mode]]
        print(mm)

        for m in range(len(mm)):
            if self._mode_bound is not None:
                bound = self._mode_bound[mm[m][0]]
            self.tt[mm[m][0]][mm[m][1]].set_hb_mp_params(a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)


        fres, self._hb_vrs_time_run = misc_fns.mp_funs_factor(funs=[self.tt[mm[m][0]][mm[m][1]].set_vrs_mp for m in range(len(mm))], 
                                    factor=factor, print_mp_time=print_mp_time)

        for m in range(len(mm)):
            self.tt[mm[m][0]][mm[m][1]]._hb = fres[m]
        
        if append_vrs_tt_hb_w and set_eval:
            self.append_vares_w()

        hb_vrs_timer.stop()
        self._hb_vrs_time =  hb_vrs_timer.get_timer_dict()


    def set_vrs(self, a_q, set_eval=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, append_w=False):
        for mode in self.tt:
            for mark in self.tt[mode]:
                self.tt[mode][mark].set_vrs(a_q=a_q, set_eval=set_eval, CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

        if append_w is None:
            append_w = False
        if append_w and set_eval:
            self.append_vares_w()    
        


    def set_vrs_eval(self, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, append_w=None):
        for mode in self.tt:
            for mark in self.tt[mode]:
                self.tt[mode][mark].set_vrs_eval(CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, save_t=save_t)

        if append_w is None:
            append_w = True
        if append_w:
            self.append_vares_w()

    def append_fit_tt_hb_w(self):
        for fn in ['est', 'se', 'eval']:
            self._rw['fit']['w']['tt_hb'][fn] = self._rw['fit']['w']['tt_hb'][fn].append(self.calc_fit_tt_hb_w(lead=self._rw['meta']['series'], page=fn))

    def append_vares_w(self):
        for fn in ['vrs']:
            self._rw['vares']['w']['tt_hb'][fn] = self._rw['vares']['w']['tt_hb'][fn].append(self.calc_vrs_eval(lead=self._rw['meta']['series']))




    def calc_fit_tt_hb_w(self, lead=None, page=None):

        df = pd.DataFrame()
        for mode in self.tt:
            for mark in self.tt[mode]:
                df = df.append(self.tt[mode][mark].calc_fit_tt_hb_w(lead=lead, page=page))
        return df

    def calc_vrs_eval(self, lead=None, atf=None):
        df = pd.DataFrame()
        for mode in self.tt:
            for mark in self.tt[mode]:
                df = df.append(self.tt[mode][mark].calc_vrs_eval(lead=lead, atf=atf))
        return df
    # Checkpoint


    def set_fit_fn_pre(self, fit_fn_pre=None, reset=None):

        if reset is None:
            reset = False

        if fit_fn_pre is not None:
            self._fit_fn_pre = fit_fn_pre
        elif reset:
            self._fit_fn_pre = 'tt'


    def get_fit_fn_pre(self):

        return self._fit_fn_pre



    def set_fit_colval(self, fit_colval=None, reset=None):

        if reset is None:
            reset = False

        if fit_colval is not None:
            self._fit_colval = fit_colval
        elif reset:
            self._fit_colval = {}

    def get_fit_colval(self):

        return self._fit_colval


    def read_fit_tt(self, fit_dir, fit_fn_pre=None, fit_colval=None):

        self.set_fit_fn_pre(fit_fn_pre=fit_fn_pre, reset=False)
        self.set_fit_colval(fit_colval=fit_colval, reset=False)

 
        void = [f"{param}_{1+i}" for param in ["nu", "omega"] for i in range(2)]

        suf_list = ['meta', 'est', 'se']

        self._rw['fit']['r']['tt'] = {suf: pd.DataFrame() for suf in suf_list}
    
        for suf in suf_list:
            f_dir = f"{fit_dir}fit_{self.get_fit_fn_pre()}_{suf}.csv"
            if os.path.isfile(f_dir):
                self._rw['fit']['r']['tt'][suf] = misc_fns.index_df(pd.read_csv(f_dir, sep=','), colval=self.get_fit_colval())
                if suf == 'est':
                    self._rw['fit']['r']['tt']['est'][self.tt["bi"]["mark"]._parameters._fp_names] = self._rw['fit']['r']['tt']['est'][self.tt["bi"]["mark"]._parameters._fp_names].fillna(0)
                    self._rw['fit']['r']['tt']['est'][void] = np.nan

    def read_fit_tt_hb(self, fit_dir, fit_fn_pre=None, fit_colval=None):

        self.set_fit_fn_pre(fit_fn_pre=fit_fn_pre)
        self.set_fit_colval(fit_colval=fit_colval)


        suf_list = ['est', 'se']
        self._rw['fit']['r']['tt_hb'] = {suf: pd.DataFrame() for suf in suf_list}

        for suf in suf_list:
            f_dir = f"{fit_dir}fit_{self.get_fit_fn_pre()}_hb_{suf}.csv"
            if os.path.isfile(f_dir):
                self._rw['fit']['r']['tt_hb'][suf] = misc_fns.index_df(pd.read_csv(f_dir, sep=','), colval=self.get_fit_colval())


    def read_vares_tt_hb(self, vrs_dir, fit_fn_pre=None, fit_colval=None):

        self.set_fit_fn_pre(fit_fn_pre=fit_fn_pre)
        self.set_fit_colval(fit_colval=fit_colval)

        suf_list = ['vrs']
        self._rw['vares']['r']['tt_hb'] = {suf: pd.DataFrame() for suf in suf_list}

        for suf in suf_list:
            f_dir = f"{vrs_dir}vares_{self.get_fit_fn_pre()}_hb_{suf}.csv"
            if os.path.isfile(f_dir):
                self._rw['vares']['r']['tt_hb']['vrs'] = misc_fns.index_df(pd.read_csv(f_dir, sep=','), colval=self.get_fit_colval())



    def load_a_u(self, a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None, tt_read_params=None, tt_hb_read_params=None, tt_opt_method=None, tt_fix_a_lambda=None):

        if tt_read_params is None:
            tt_read_params = True
        if tt_hb_read_params is None:
            tt_hb_read_params = True



        self.set_opt_settings(opt_method=tt_opt_method, fix_a_lambda=tt_fix_a_lambda)

        self.set_params_threshold(a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, reset=False, set_tt=True)



        read = False
        if tt_read_params:

            exceed_param_colval = self.calc_params_threshold_dict(name_rm=['u', 'u_inclusive'])

            #elif np.isin(self._a_u, self._rw['fit']['r']['tt']["est"]["a_u_1"]):
            #if not np.isin("a_u_1", self._rw['fit']['r']['tt']["est"].columns):
            # print(f"a_u = {self._a_u} is in fit_tt_est")
            
            if not all(ele in self._rw['fit']['r']['tt']["est"].columns for ele in [p for p in exceed_param_colval]):
                print(f"Invalid fit_tt_est.csv file")
                print(f"Skipping parameter reads...")
            else:
                self.print_params_threshold()    
                if misc_fns.index_df(self._rw['fit']['r']['tt']["est"], colval=exceed_param_colval).shape[0]>0:
                    print(f"...are in fit_tt_est!")
                    print(f"Copying parameters and detecting events...")
                    read = True
                else:
                    print(f"...are NOT in fit_tt_est!")
                    print(f"Skipping parameter reads...")

        for mode in self.tt:
            for mark in self.tt[mode]:
                self.tt[mode][mark]._parameters.a_u.set_default(self.a_u.get_value())
                self.tt[mode][mark]._parameters.a_lambda.set_default(self.a_u.get_value())
                if read:
                    if False:
                        index = (self._rw['fit']['r']['tt']["est"]["a_u_1"]==self._a_u) & (self._rw['fit']['r']['tt']["est"]["mode"]==mode) & (self._rw['fit']['r']['tt']["est"]["mark"]==mark) & (self._rw['fit']['r']['tt']["est"]["fix_a_lambda"]==self.tt[mode][mark]._parameters.get_fix_a_lambda())
                        for param in ["a_u", "u_dir", "u", "u_zero", "a_lambda"]:
                            self.tt[mode][mark]._parameters.__dict__[param].set_value(self._rw['fit']['r']['tt']["est"][[f"{param}_{1+i}" for i in range(self.tt[mode][mark].get_D_nodes())]][index]._values)
                        self.tt[mode][mark].detect_events()
                        self.tt[mode][mark]._parameters._flat_parameters[self.tt[mode][mark]._parameters._flat_index['dict']['range']['a_lambda']] = self._rw['fit']['r']['tt']["est"][[f"a_lambda_{1+i}" for i in range(self.tt[mode][mark].get_D_nodes())]][index]._values
                        self.tt[mode][mark]._parameters.set_flat_parameters(self._rw['fit']['r']['tt']["est"][self.tt[mode][mark]._parameters._fp_names][index]._values[0])
                        self.tt[mode][mark].set_t_series()

                        for param in self.tt[mode][mark]._parameters._fitted:
                            self.tt[mode][mark]._parameters._fitted[param] = self._rw['fit']['r']['tt']["est"][param][index]._values[0]
                            self.tt[mode][mark]._parameters._fitted_se[param] = self._rw['fit']['r']['tt']["se"][param][index]._values[0]
                        for score in ["log_L", "N_params", "AIC", "BIC"]:
                            self.tt[mode][mark]._parameters._fitted_scores[score] = self._rw['fit']['r']['tt']["est"][score][index]._values[0]
                    else:

                        colval = {
                                    **self.tt[mode][mark]._parameters.calc_params_threshold_dict(name_rm=['u', 'u_inclusive']),
                                    **self.tt[mode][mark]._parameters.calc_opt_settings_dict(),
                                    **self.tt[mode][mark]._parameters.calc_mm_dict(),
                                }

                        for param in ["u", "a_lambda"]:
                            self.tt[mode][mark]._parameters.__dict__[param].set_value(misc_fns.index_df(self._rw['fit']['r']['tt']["est"], colval=colval)[[f"{param}_{1+i}" for i in range(self.tt[mode][mark].get_D_nodes())]]._values)

                        self.tt[mode][mark].detect_events()
                        self.tt[mode][mark]._parameters._flat_parameters[self.tt[mode][mark]._parameters._flat_index['dict']['range']['a_lambda']] = misc_fns.index_df(self._rw['fit']['r']['tt']["est"], colval=colval)[[f"a_lambda_{1+i}" for i in range(self.tt[mode][mark].get_D_nodes())]]._values[0]
                        self.tt[mode][mark]._parameters.set_flat_parameters(misc_fns.index_df(self._rw['fit']['r']['tt']["est"], colval=colval)[self.tt[mode][mark]._parameters._fp_names]._values[0])
                        self.tt[mode][mark].set_t_series()

                        for param in self.tt[mode][mark]._parameters._fitted:
                            self.tt[mode][mark]._parameters._fitted[param] = misc_fns.index_df(self._rw['fit']['r']['tt']["est"], colval=colval)[param]._values[0]
                            self.tt[mode][mark]._parameters._fitted_se[param] = misc_fns.index_df(self._rw['fit']['r']['tt']["se"], colval=colval)[param]._values[0]
                        for score in ["log_L_H_u", "N_params_H_u", "AIC_H_u", "BIC_H_u"]:
                            self.tt[mode][mark]._parameters._fitted_scores[score] = misc_fns.index_df(self._rw['fit']['r']['tt']["est"], colval=colval)[score]._values[0]
                else:
                    self.tt[mode][mark]._parameters.a_u.set_value(self.a_u.get_value())
                    self.tt[mode][mark].set_u()
                    self.tt[mode][mark].detect_events()

                    self.tt[mode][mark]._parameters.a_lambda.set_value(self.a_u.get_value())
                    self.tt[mode][mark]._parameters._flat_parameters[self.tt[mode][mark]._parameters._flat_index['dict']['range']['a_lambda']] = self.a_u.get_value()
                    

        
                

        if tt_hb_read_params:
            self.load_tt_hb()

                
        return read

    
    def load_tt_hb(self, fix_a_lambda=None):

        print('load_tt_hb')
        for mode in self.tt:
            for mark in self.tt[mode]:
                colval = {
                            **self.tt[mode][mark]._parameters.calc_params_threshold_dict(name_rm=['u', 'u_inclusive']),
                            **self.tt[mode][mark]._parameters.calc_opt_settings_dict(),
                            **self.tt[mode][mark]._parameters.calc_mm_dict(),
                        }
                if self.tt[mode][mark]._hb is not None:
                    for dist in self.tt[mode][mark]._hb.hb_d:
                        for bound in self.tt[mode][mark]._hb.hb_d[dist].hb_b:
                            index = misc_fns.get_index(df=self._rw['fit']['r']['tt_hb']["est"], 
                                                        colval={
                                                                    **colval,
                                                                    'dist': dist,
                                                                    'bound': bound,
                                                                })
                            print('read worked')
                            if np.any(index._values):
                                df_0=self._rw['fit']['r']['tt_hb']["est"]['df'][index]._values[0]
                                nc_0=self._rw['fit']['r']['tt_hb']["est"]['nc'][index]._values[0]
                                self.tt[mode][mark]._hb.hb_d[dist].hb_b[bound].load_bulk(hawkes=self.tt[mode][mark], df_0=df_0, nc_0=nc_0)


    def calc_tt_eval(self, meta_series=None, normalize=None, lags=None, boxpierce=None, head=None, pvalue=None, static=None, fix_a_lambda=None):

        if static is None:
            static=False
        

        if meta_series is None:
            meta_series = self._rw['meta']['series']
            #if np.isin('all_end', [key for key in meta_series]):
            #    meta_series.pop('all_end')


        d_w = dict()
        w = dict()

        w['meta'] = pd.DataFrame({
                                    **meta_series,
                                    **self.calc_params_threshold_dict(),
                                    **self.calc_opt_settings_dict(),
                                    **self._tt_fit_time,
                                }, index=[0])


        d_w['eval'] = pd.DataFrame()
        if static:
            if fix_a_lambda is None:
                fix_a_lambda = self.tt['bi']['mark']._parameters.get_fix_a_lambda()
            test_a_lambda = self.tt['bi']['mark']._tail_fit['fix_a_lambda'][fix_a_lambda]['train']['lambda']['a_lambda'][:,0]
            lead = {
                                **meta_series, 
                                **self.calc_params_threshold_dict(), 
                                **{f"test_a_lambda_{1+i}": test_a_lambda[i] for i in range(np.size(test_a_lambda))}
                            }

            d_w['eval'] = d_w['eval'].append(self.tt['bi']['mark'].calc_tt_eval(lead=lead, 
                                                                    normalize=normalize, lags=lags, boxpierce=boxpierce, head=head, pvalue=pvalue, static=static))
        else:
            for mode in self.tt:
                for mark in self.tt[mode]:
                    test_a_lambda = self.tt[mode][mark]._parameters.a_u.get_value()
                    lead = {
                                **meta_series, 
                                **self.tt[mode][mark]._parameters.calc_params_threshold_dict(), 
                                **self.tt[mode][mark]._parameters.calc_opt_settings_dict(), 
                                **self.tt[mode][mark]._parameters.calc_mm_dict(),
                                **{f"test_a_lambda_{1+i}": test_a_lambda[i] for i in range(np.size(test_a_lambda))},
                            }
                    d_w['eval'] = d_w['eval'].append(self.tt[mode][mark].calc_tt_eval(lead=lead, 
                                                                                        normalize=normalize, lags=lags, boxpierce=boxpierce, head=head, pvalue=pvalue, static=static))

        return d_w['eval']


    

    def calc_opt_w(self, meta_series=None):

        if meta_series is None:
            meta_series = self._rw['meta']['series']
            #if np.isin('all_end', [key for key in meta_series]):
            #    meta_series.pop('all_end')

        # Meta_settings

        d_w = dict()
        w = dict()

        meta_series_train = misc_fns.dict_key_rm(meta_series, keys_rm=['all_end'])

        
        w['meta'] = pd.DataFrame({
                                    **meta_series_train,
                                    **self.calc_params_threshold_dict(),
                                    **self.calc_opt_settings_dict(),
                                    #**self._tt_fit_time,
                                    **{f'meta_fit_{t}': self._tt_fit_time[t] for t in self._tt_fit_time},
                                }, index=[0])


        for fn in ['est', 'se']:
            d_w[fn] = hwks_fns.create_dict_2T_POT()

        for mode in d_w['est']:
            for mark in d_w['est'][mode]:

                res_time = {f'{step}_{t}': self.tt[mode][mark]._parameters.res_time[step][t] for step in self.tt[mode][mark]._parameters.res_time for t in self.tt[mode][mark]._parameters.res_time[step]}
                d_w['est'][mode][mark] =    {
                                                **meta_series_train,
                                                **self.tt[mode][mark]._parameters.calc_params_threshold_dict(),
                                                **self.tt[mode][mark]._parameters.calc_opt_settings_dict(),
                                                **self.tt[mode][mark]._parameters.calc_mm_dict(),
                                                **self.tt[mode][mark]._parameters._fitted,
                                                **self.tt[mode][mark]._parameters._fitted_scores,
                                                **res_time,
                                            }
                d_w['se'][mode][mark] =     {
                                                **meta_series_train,
                                                **self.tt[mode][mark]._parameters.calc_params_threshold_dict(),
                                                **self.tt[mode][mark]._parameters.calc_opt_settings_dict(),
                                                **self.tt[mode][mark]._parameters.calc_mm_dict(),
                                                **self.tt[mode][mark]._parameters._fitted_se,
                                            }


        for fn in ['est', 'se']:
            w[fn] = pd.DataFrame({
                        param: np.array([d_w[fn][mode][mark][param] for mode in d_w[fn] for mark in d_w[fn][mode]]) for param in d_w[fn]["bi"]["mark"]
                    })


        w['eval'] = self.calc_tt_eval(normalize=False, lags=5, boxpierce=False, pvalue=False, static=False)
        w['eval_static'] = self.calc_tt_eval(normalize=False, lags=5, boxpierce=False, pvalue=True, static=True)


        #for suf in ['start', 'end']:
        #    for pre in ['res', 'SLSQP']:
        #        w['est'][f"{pre}_{suf}"] = w['est'][f"{pre}_{suf}"].astype(dtype='datetime64[ns, UTC]')
        #        #w['meta'][f"{suf}"] = w['meta'][f"{suf}"].astype(dtype='datetime64[ns, UTC]')

        return w

    def print_tt_parameters_table(self, ci=None, mark=None):

        if ci is None:
            ci = False
        if mark is None:
            mark = 'mark'

        
        if ci:
            models = ['ci', 'ci_s']
        else:
            models = ['bi', 'bi_d']
        if mark == 'unmark':
            name_list = ['a_lambda', 'gamma', 'beta', 'xi', 'varsigma']
        else:
            name_list = ['a_lambda', 'gamma', 'beta', 'xi', 'varsigma', 'eta', 'alpha']

        
        for name in name_list:
            names = self.tt[models[0]][mark]._parameters.__dict__[name]._fp_names
            if np.ndim(names) == 1:
                names = [names]
            for i in range(len(names)):
                param_val = [[[self.tt[models[m]][mark]._parameters._fitted[names[i][j]], self.tt[models[m]][mark]._parameters._fitted_se[names[i][j]]] for j in range(2)] for m in range(len(models))]
                for m in range(2):
                    for j in range(2):
                        for k in range(2):
                            if param_val[m][j][k] is None:
                                param_val[m][j][k] = 0
                param = [[None for j in range(2)] for m in range(2)]
                for m in range(2):
                    for j in range(2):
                        if np.isnan(param_val[m][j][0]):
                            param[m][j] = "& "
                        else:
                            if param_val[m][j][0] == 0:
                                exp=0
                            else:
                                exp = int(np.floor(np.log(np.abs(param_val[m][j][0])) / np.log(10)))
                            est = param_val[m][j][0]*10**(-exp)
                            se = param_val[m][j][1]*10**(-exp)
                            if exp == 0:
                                param[m][j] = f"& ${est:.1f} \pm {se:.1f}$ "
                            else:
                                param[m][j] = f"& $({est:.1f} \pm {se:.1f}) \times 10^{{{exp}}}$ "
                            #param[m][j] = "& {:.4f} & ({:.4f}) ".format(param_val[m][j][0], param_val[m][j][1])
                print(f"$\\vect{{\\up{name}}}{hwks_fns.model_slash(name)}{hwks_fns.get_model_unit()[name]}$ " + param[0][0] + param[0][1] + param[1][0] + param[1][1] + "\\\\")


    def set_new_data(self, data):

        df = pd.DataFrame(np.transpose([data.values for i in range(2)]), index=data.index)
        for mode in self.tt:
            for mark in self.tt[mode]:
                self.tt[mode][mark].set_new_data(data=df)




def create_models_1T_POT(data, a_u=0.05, exchanges=None, dt_dt_fine=None):

    models = hwks_fns.create_dict_1T_POT()
    if type(data) != dict:
        data = {"process": data}

    if exchanges is not None:
        exchanges = [exchanges]

    for tail in models:
        for mark in models[tail]: 
            if tail=="abs":
                a_u *= 2
            models[tail][mark] = {period: hawkes_process(data=data[period], dt_dt_fine=dt_dt_fine, exchanges=exchanges,
                                                                        mark=mark, a_u=a_u, u_dir=tail) for period in data}

    return models


def create_models_2T_POT(data, a_u=0.05, exchanges=None, dt_dt_fine=None, opt_method=None, fix_a_lambda=None):

    models = hwks_fns.create_dict_2T_POT()
    if type(data) != dict:
        data = {"process": data}

    data_2 = {period: pd.DataFrame(data=np.resize(np.repeat(data[period].values,2), (data[period].size, 2)), index=data[period].index) for period in data}


    base = {period: hawkes_process(data=data_2[period], dt_dt_fine=dt_dt_fine, exchanges=exchanges, opt_method=opt_method, fix_a_lambda=fix_a_lambda,
                                                            mode="bi", mark="mark", a_u=a_u, u_dir=["left", "right"]) for period in data_2}

    for mode in models:
        for mark in models[mode]: 
            models[mode][mark] = copy.deepcopy(base)
            for period in models[mode][mark]:
                models[mode][mark][period]._parameters.set_mode(mode)
                models[mode][mark][period]._parameters.set_mark(mark)
                models[mode][mark][period]._parameters.set_fix_a_lambda()

    return models

def create_models_2T_POT_nt(data, a_u=None, u_dir=None, u_zero=None, u=None, u_inclusive=None, exchanges=None, dt_dt_fine=None, train_t_start=None, train_t_end=None, opt_method=None, fix_a_lambda=None):

    models = hwks_fns.create_dict_2T_POT()

    data_2 = pd.DataFrame(data=np.resize(np.repeat(data.values,2), (data.size, 2)), index=data.index)


    base = hawkes_process(data=data_2, dt_dt_fine=dt_dt_fine, exchanges=exchanges, opt_method=opt_method, fix_a_lambda=fix_a_lambda, 
                                                            mode="bi", mark="mark", 
                                                            a_u=a_u, u_dir=u_dir, u_zero=u_zero, u=u, u_inclusive=u_inclusive, 
                                                            train_t_start=train_t_start, train_t_end=train_t_end)

    for mode in models:
        for mark in models[mode]: 
            models[mode][mark] = copy.deepcopy(base)
            models[mode][mark]._parameters.set_mode(mode)
            models[mode][mark]._parameters.set_mark(mark)
            #models[mode][mark]._parameters.set_opt_settings(opt_method=opt_method, fix_a_lambda=fix_a_lambda)

    return models

def fit_models_2T_POT_nt(self, print_tt_fit_time=None, print_res_time=None, opt_method=None, fix_a_lambda=None, fp0=None):

    if print_tt_fit_time is None:
        print_tt_fit_time = True
    if print_res_time is None:
        print_res_time = True
    
    tt_fit_timer = misc_fns.timer(print_timer=print_tt_fit_time, print_head="fit_models_2T_POT")


    if fp0 is None:
        set_fp0 = True
        fp0 = {
                    mode:   {
                                mark: None for mark in self[mode]
                            } for mode in self
                }
        for mode in self:
            for mark in self[mode]:
                self[mode][mark].set_parameters_defaults()
                fp0[mode][mark] = self[mode][mark]._parameters.flatten_defaults()
    else:
        set_fp0 = False

    

    fp0_0 = copy.deepcopy(fp0)
    fp0_1 = copy.deepcopy(fp0)

    param_2T_bool = {tail: np.full(fp0["bi_d"]["unmark"].size, False) for tail in ["left", "right"]}
    for i in range(len(self["ci_s"]["unmark"]._parameters._flat_index["list"]["range"])):
        param_2T_bool["left"][self["ci_s"]["unmark"]._parameters._flat_index["list"]["range"][i][0]] = True
        param_2T_bool["right"][self["ci_s"]["unmark"]._parameters._flat_index["list"]["range"][i][-1]] = True


################################################################################

    def fit_mm(mode, mark):
        if print_res_time:
            print(mode, mark)
        self[mode][mark].set_parameters_estimate(flat_parameters_0=fp0[mode][mark], print_res_time=print_res_time, opt_method=opt_method, fix_a_lambda=fix_a_lambda)

##############################################################################

    fit_mm(mode="ci_s", mark="unmark")
    fit_mm(mode="bi_d", mark="unmark")
    if set_fp0:
        fp0["ci"]["unmark"] = self["ci_s"]["unmark"]._parameters.flatten()
    fit_mm(mode="ci", mark="unmark")
    if set_fp0:
        fp0["bi"]["unmark"] = self["ci"]["unmark"]._parameters.flatten()
    fit_mm(mode="bi", mark="unmark")

    if set_fp0:
        for mode in fp0:
            fp0[mode]["mark"] = self[mode]["unmark"]._parameters.flatten()

    fit_mm(mode="ci_s", mark="mark")
    fit_mm(mode="bi_d", mark="mark")
    if set_fp0:
        fp0["ci"]["mark"][self["bi_d"]["mark"]._parameters._flat_index["dict"]["range"]["eta"]] = self["bi_d"]["mark"]._parameters.flatten()[self["bi_d"]["mark"]._parameters._flat_index["dict"]["range"]["eta"]]
        fp0["ci"]["mark"][self["bi_d"]["mark"]._parameters._flat_index["dict"]["range"]["alpha"]] = self["bi_d"]["mark"]._parameters.flatten()[self["bi_d"]["mark"]._parameters._flat_index["dict"]["range"]["alpha"]]
    fit_mm(mode="ci", mark="mark")
    if set_fp0:
        fp0["bi"]["mark"][self["ci"]["mark"]._parameters._flat_index["dict"]["range"]["eta"]] = self["ci"]["mark"]._parameters.flatten()[self["ci"]["mark"]._parameters._flat_index["dict"]["range"]["eta"]]
        fp0["bi"]["mark"][self["ci"]["mark"]._parameters._flat_index["dict"]["range"]["alpha"]] = self["ci"]["mark"]._parameters.flatten()[self["ci"]["mark"]._parameters._flat_index["dict"]["range"]["alpha"]]
    fit_mm(mode="bi", mark="mark")

    for mode in self:
        for mark in self[mode]:
            fp0_1[mode][mark] = self[mode][mark]._parameters.flatten_defaults()


    tt_fit_timer.stop()

    fit_dict =  {
                    "meta_fit": tt_fit_timer.get_timer_dict(),
                    "fp0": fp0,
                    "fp0_0": fp0_0,
                    "fp0_1": fp0_1,
                }
    return fit_dict

def fit_models_2T_POT(self, print_tt_time=True, print_fit_time=True, print_res_time=True, fix_a_lambda=None):

    

    if print_tt_time:
        print("fit_models_2T_POT")
    tt_timer = misc_fns.timer(print_timer=print_tt_time)

    fit_time = {
                mode:   {
                            mark: None for mark in self[mode]
                        } for mode in self
                }

    fp0 = {
                mode:   {
                            mark: None for mark in self[mode]
                        } for mode in self
            }

    for mode in self:
        for mark in self[mode]:
            self[mode][mark]["train"].set_parameters_defaults()
            fp0[mode][mark] = self[mode][mark]["train"]._parameters.flatten_defaults()

    fp0_0 = copy.deepcopy(fp0)
    fp0_1 = copy.deepcopy(fp0)

    param_2T_bool = {tail: np.full(fp0["bi_d"]["unmark"].size, False) for tail in ["left", "right"]}
    for i in range(len(self["ci_s"]["unmark"]["train"]._parameters._flat_index["list"]["range"])):
        param_2T_bool["left"][self["ci_s"]["unmark"]["train"]._parameters._flat_index["list"]["range"][i][0]] = True
        param_2T_bool["right"][self["ci_s"]["unmark"]["train"]._parameters._flat_index["list"]["range"][i][-1]] = True


################################################################################

    def timed_fit(self, fp0=None, print_fit_time=True, print_res_time=True, fix_a_lambda=False):

        timer = misc_fns.timer(print_timer=print_fit_time)
        if print_fit_time:
            print("start:", time["start"])
        self.set_parameters_estimate(flat_parameters_0=fp0, print_res_time=print_res_time, fix_a_lambda=fix_a_lambda)
        timer.stop()
        return timer.get_timer_dict()

    def timed_fit_mm(mode, mark, print_fit_time=True, print_res_time=True, fix_a_lambda=False):
        if print_fit_time:
            print(mode, mark)
        fit_time[mode][mark] = timed_fit(self=self[mode][mark]["train"], fp0=fp0[mode][mark], fix_a_lambda=fix_a_lambda)

##############################################################################

    timed_fit_mm(mode="ci_s", mark="unmark")
    timed_fit_mm(mode="bi_d", mark="unmark")
    fp0["ci"]["unmark"] = self["ci_s"]["unmark"]["train"]._parameters.flatten()
    timed_fit_mm(mode="ci", mark="unmark")
    fp0["bi"]["unmark"] = self["ci"]["unmark"]["train"]._parameters.flatten()
    timed_fit_mm(mode="bi", mark="unmark")

    for mode in fp0:
        fp0[mode]["mark"] = self[mode]["unmark"]["train"]._parameters.flatten()

    timed_fit_mm(mode="ci_s", mark="mark")
    timed_fit_mm(mode="bi_d", mark="mark")
    fp0["ci"]["mark"][self["bi_d"]["mark"]["train"]._parameters._flat_index["dict"]["range"]["eta"]] = self["bi_d"]["mark"]["train"]._parameters.flatten()[self["bi_d"]["mark"]["train"]._parameters._flat_index["dict"]["range"]["eta"]]
    fp0["ci"]["mark"][self["bi_d"]["mark"]["train"]._parameters._flat_index["dict"]["range"]["alpha"]] = self["bi_d"]["mark"]["train"]._parameters.flatten()[self["bi_d"]["mark"]["train"]._parameters._flat_index["dict"]["range"]["alpha"]]
    timed_fit_mm(mode="ci", mark="mark")
    fp0["bi"]["mark"][self["ci"]["mark"]["train"]._parameters._flat_index["dict"]["range"]["eta"]] = self["ci"]["mark"]["train"]._parameters.flatten()[self["ci"]["mark"]["train"]._parameters._flat_index["dict"]["range"]["eta"]]
    fp0["bi"]["mark"][self["ci"]["mark"]["train"]._parameters._flat_index["dict"]["range"]["alpha"]] = self["ci"]["mark"]["train"]._parameters.flatten()[self["ci"]["mark"]["train"]._parameters._flat_index["dict"]["range"]["alpha"]]
    timed_fit_mm(mode="bi", mark="mark")

    for mode in self:
        for mark in self[mode]:
            fp0_1[mode][mark] = self[mode][mark]["train"]._parameters.flatten_defaults()

    tt_timer.stop()

    return fit_time, fp0, fp0_0, fp0_1


def set_a_u_periods(self, a_u):

    self["train"]._parameters.a_u.set_value(a_u)
    self["train"]._parameters.a_lambda.set_value(a_u)
    self["train"].set_u()
    for param in ["a_lambda", "a_u", "u"]:
        self["process"]._parameters.__dict__[param].set_value(self["train"]._parameters.__dict__[param].get_value())
    for period in self:
        self[period].detect_events()



@njit(   [
            nb.float64[:,:](nb.int64, nb.float64[:], nb.float64[:,:], nb.int32[:,:], nb.float64[:,:], nb.float64[:], nb.int32[:], nb.bool_, nb.float64[:], nb.float64[:], nb.float64[:]),
            nb.float64[:,:](nb.int64, nb.float64[:], nb.float64[:,:], nb.int32[:,:], nb.float64[:,:], nb.int32[:], nb.int32[:], nb.bool_, nb.float64[:], nb.float64[:], nb.float64[:]),
        ])
def lambda_forecast_fast_disc(D_nodes, mu, gamma, events_t, events_kappa, t, N, is_phi_exp, beta, nu, omega):

    chi = np.zeros((D_nodes, t.size), dtype=np.float64)

    for i in range(D_nodes):
        lags = np.outer(t, np.ones(N[i])) - events_t[i, 0:N[i]]
        if is_phi_exp:
            #chis_lags_events = np.exp(-(beta * lags)) * events_kappa[i, 0:N[i]]
            chis_lags_events = (np.exp(beta[i])-1.0)*np.exp(-(beta[i] * lags)) * events_kappa[i, 0:N[i]]

        else:
            #chis_lags_events = np.power((lags*nu[i] + 1.0), -(1.0+omega[i])) * events_kappa[i, 0:N[i]]

            chis_lags_events = (np.power(1-nu[i]*(lags-1.0), -omega[i]) - np.power(1-nu[i]*(lags), -omega[i])) * events_kappa[i, 0:N[i]]
            
            #np.power((lags*nu[i] + 1.0), -(1.0+omega[i])) * events_kappa[i, 0:N[i]]

        chi[i,:] = np.sum(chis_lags_events, axis=1)
    #chi = np.array(Parallel(n_jobs=num_cores)(delayed(self.calc_chi_node)(i, t, N) for i in range(self.get_D_nodes())))
    
    lambda_endo_t = np.ascontiguousarray(gamma) @ chi
    lambda_t = np.outer(mu, np.full(t.size, 1, dtype=np.int32)) + lambda_endo_t

    return lambda_t


@njit(   [
            nb.float64[:,:](nb.int64, nb.float64[:], nb.float64[:,:], nb.int32[:,:], nb.float64[:,:], nb.float64[:], nb.int32[:], nb.bool_, nb.float64[:], nb.float64[:], nb.float64[:]),
            nb.float64[:,:](nb.int64, nb.float64[:], nb.float64[:,:], nb.int32[:,:], nb.float64[:,:], nb.int32[:], nb.int32[:], nb.bool_, nb.float64[:], nb.float64[:], nb.float64[:]),
        ])
def lambda_forecast_fast_cont(D_nodes, mu, gamma, events_t, events_kappa, t, N, is_phi_exp, beta, nu, omega):

    chi = np.zeros((D_nodes, t.size), dtype=np.float64)

    for i in range(D_nodes):
        lags = np.outer(t, np.ones(N[i])) - events_t[i, 0:N[i]]
        if is_phi_exp:
            #chis_lags_events = np.exp(-(beta * lags)) * events_kappa[i, 0:N[i]]
            chis_lags_events = beta[i]*np.exp(-(beta[i] * lags)) * events_kappa[i, 0:N[i]]

        else:
            chis_lags_events = nu[i]*omega[i] * np.power((lags*nu[i] + 1.0), -(1.0+omega[i])) * events_kappa[i, 0:N[i]]

        chi[i,:] = np.sum(chis_lags_events, axis=1)
    #chi = np.array(Parallel(n_jobs=num_cores)(delayed(self.calc_chi_node)(i, t, N) for i in range(self.get_D_nodes())))
    
    lambda_endo_t = np.ascontiguousarray(gamma) @ chi
    lambda_t = np.outer(mu, np.full(t.size, 1, dtype=np.int32)) + lambda_endo_t

    return lambda_t

@njit(   [
            nb.float64[:,:](nb.int64, nb.float64[:], nb.float64[:,:], nb.int32[:,:], nb.float64[:,:], nb.float64[:], nb.int32[:], nb.bool_, nb.float64[:], nb.float64[:], nb.float64[:]),
            nb.float64[:,:](nb.int64, nb.float64[:], nb.float64[:,:], nb.int32[:,:], nb.float64[:,:], nb.int32[:], nb.int32[:], nb.bool_, nb.float64[:], nb.float64[:], nb.float64[:]),
        ])
def tilde_t_forecast_fast(D_nodes, mu, gamma, events_t, events_kappa, t, N, is_phi_exp, beta, nu, omega):

    int_chi = np.zeros((D_nodes, t.size), dtype=np.float64)

    for i in range(D_nodes):
        lags = np.outer(t, np.ones(N[i])) - events_t[i, 0:N[i]]
        if is_phi_exp:
            #int_chis_lags_events = np.exp(-(beta * lags)) * events_kappa[i, 0:N[i]]
            if beta[i]==0:
                int_chis_lags_events = 0.0 * lags * events_kappa[i, 0:N[i]]
            else:
                int_chis_lags_events = -np.exp(-(beta[i] * lags)) * events_kappa[i, 0:N[i]]
        else:
            if nu[i]==0:
                #int_chis_lags_events = 1.0 * lags
                int_chis_lags_events = 0.0 * lags  * events_kappa[i, 0:N[i]]
            elif omega[i]==0:
                #int_chis_lags_events = (1/(nu[i]))*np.log(lags*nu[i] + 1.0)
                int_chis_lags_events = 0.0 * lags  * events_kappa[i, 0:N[i]]
            else:
                int_chis_lags_events = -np.power((lags*nu[i] + 1.0), -(1.0+omega[i])) * events_kappa[i, 0:N[i]]

        int_chi[i,:] = np.sum(int_chis_lags_events, axis=1)
    #chi = np.array(Parallel(n_jobs=num_cores)(delayed(self.calc_chi_node)(i, t, N) for i in range(self.get_D_nodes())))
    
    tilde_t_endo_t = np.ascontiguousarray(gamma) @ int_chi
    tilde_t_t = np.outer(mu, t) + tilde_t_endo_t

    return tilde_t_t

@njit(   [
            nb.float64[:,:](nb.int64, nb.float64[:], nb.float64[:,:], nb.int32[:,:], nb.float64[:,:], nb.float64[:], nb.int32[:], nb.bool_, nb.float64[:], nb.float64[:], nb.float64[:]),
            nb.float64[:,:](nb.int64, nb.float64[:], nb.float64[:,:], nb.int32[:,:], nb.float64[:,:], nb.int32[:], nb.int32[:], nb.bool_, nb.float64[:], nb.float64[:], nb.float64[:]),
        ])
def lambda_forecast_fast_test(D_nodes, mu, gamma, events_t, events_kappa, t, N, is_phi_exp, beta, nu, omega):

    chi = np.zeros((D_nodes, t.size), dtype=np.float64)

    for i in range(D_nodes):
        chis_lags_events = np.outer(t, np.ones(N[i])) - events_t[i, 0:N[i]]
        if is_phi_exp:
            chis_lags_events = np.exp(-(beta[i] * chis_lags_events)) * events_kappa[i, 0:N[i]]
        else:
            chis_lags_events = np.power((chis_lags_events*nu + 1.0), -(1.0+omega)) * events_kappa[i, 0:N[i]]

        chi[i,:] = np.sum(chis_lags_events, axis=1)
    #chi = np.array(Parallel(n_jobs=num_cores)(delayed(self.calc_chi_node)(i, t, N) for i in range(self.get_D_nodes())))
    
    lambda_t = np.ascontiguousarray(gamma) @ chi + mu


    return lambda_t






@njit
def phi_exp(lag, beta, node=0):
    #return np.exp(-(beta * lag))
    return beta*np.exp(-(beta * lag))

@njit
def phi_pow(lag, nu, omega):
    return nu*omega * np.power((lag*nu + 1.0), -(1.0+omega))

@njit
def int_phi_exp(lag, beta, node=0):
    if beta == 0:
        #return 1.0 * lag
        return beta * lag
    else:
        #return -(1/beta)*np.exp(-(beta * lag))
        return -np.exp(-(beta * lag))

@njit
def int_phi_pow(lag, nu, omega):

    if nu == 0:
        #return 1.0 * lag
        return 0.0 * lag
    elif omega == 0:
        #return (1/(nu))*np.log(lag*nu + 1.0)
        return 0.0 * lag
    else:
        return -np.power((lag*nu + 1.0), -omega)