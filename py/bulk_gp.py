import numpy as np
import scipy
import numdifftools

import vares

import misc_functions as misc_fns
import dist_functions as dist_fns


class bulk_gp():
    """Class for parametric bulk distribution with Generalized Pareto distribution tails
    """
    
    def __init__(self, data=None, fit_method=None, train_t=None, bulk_bound=None, opt_method=None,
                        bulk_dist=None, bulk_loc=None, bulk_scale=None, bulk_df=None, bulk_nc=None,
                        gp_c=None, gp_loc=None, gp_scale=None, gp_weight=None, gp_sign=None):
        """Constructor unitnorm_gpd class
        """
        self.set_t_size(t_size=1)
        self._bulk_fit = None
        self.dist = None
        self.vrs = None

        self.kwds = dict()


        if data is None:
            self.set_bulk_gp(bulk_dist=bulk_dist, bulk_loc=bulk_loc, bulk_scale=bulk_scale, bulk_df=bulk_df, bulk_nc=bulk_nc,
                                gp_c=gp_c, gp_loc=gp_loc, gp_scale=gp_scale, gp_weight=gp_weight, gp_sign=gp_sign)
        else:

            if fit_method is None:
                fit_method = 'fit_gp'

            if fit_method=='fit_gp':
                self.fit_gp(data=data, train_t=train_t,
                                    bulk_dist=bulk_dist, bulk_loc=bulk_loc, bulk_scale=bulk_scale, bulk_df=bulk_df, bulk_nc=bulk_nc,
                                    gp_weight=gp_weight)
            elif fit_method=='fit_bulk':
                self.fit_bulk(data=data, train_t=train_t, opt_method=opt_method, bulk_bound=bulk_bound,
                                    bulk_dist=bulk_dist, bulk_df=bulk_df, bulk_nc=bulk_nc,
                                    gp_c=gp_c, gp_loc=gp_loc, gp_scale=gp_scale, gp_weight=gp_weight)


    def set_t_size(self, t_size=None):

        if t_size is not None:
            self._t_size = t_size


    def get_t_size(self):

        return self._t_size

    
    def fit_gp(self, data, a_u=None, train_t=None, 
                        bulk_dist=None, bulk_loc=None, bulk_scale=None, bulk_df=None, bulk_nc=None,
                        gp_weight=None):

        self.set_t_size(t_size=np.size(data))

        if gp_weight is None:
            gp_weight = 0
        gp_weight = p_rs(gp_weight)
        if train_t is None:
            train_t = np.array([0, data.size])
        self._train_t = train_t

        self.set_bulk(bulk_dist=bulk_dist, bulk_loc=bulk_loc, bulk_scale=bulk_scale, bulk_df=bulk_df, bulk_nc=bulk_nc)



        data_y = self.bulk_transform_y_x(x=data)
        gp_sign = p_rs(np.array([(-1)**(i+1) for i in range(2)]))

        if False:
            gp_loc_y = p_rs(np.quantile(data_y[train_t[0]:train_t[-1]], a_u_lr))
            gp_loc_x = self.bulk_transform_x_y(y=gp_loc_y)
        else:
            #gp_loc_x = p_rs(np.array([self._bulk.ppf(a_u_lr[i]) for i in range(2)]))
            gp_loc_x = p_rs(np.array([self._bulk.ppf(i - gp_sign[i,:]*gp_weight[i,:]) for i in range(2)]))
            gp_loc_y = p_rs(self.bulk_transform_y_x(x=gp_loc_x)[:,0])

        #gp_fit_y = [
        #            scipy.stats.genpareto.fit(np.abs(data_y[data_y < gp_loc_y[0]]), floc=np.abs(gp_loc_y[0])), 
        #            scipy.stats.genpareto.fit(np.abs(data_y[data_y > gp_loc_y[1]]), floc=np.abs(gp_loc_y[1])),
       #             ]

        
        data_y, tailsort = tailsort_fn(x=data_y, gp_loc=gp_loc_y, gp_sign=gp_sign)

        gp_fit_y = [scipy.stats.genpareto.fit(gp_sign[i]*data_y[self._train_t[0]:self._train_t[-1]][tailsort[i][self._train_t[0]:self._train_t[-1]]], floc=gp_sign[i]*gp_loc_y[i]) for i in range(2)]
        gp_y = [scipy.stats.genpareto(*gp_fit_y[i]) for i in range(2)]


        gp_c = np.array([gp_fit_y[i][0] for i in range(2)])
        gp_scale_y = np.array([gp_fit_y[i][2] for i in range(2)])
        gp_scale_x = np.outer(gp_scale_y, self._bulk_scale)
        #gp_scale_x = bulk_transform_x_y(y=gp_scale_y, bulk_scale=self._bulk_scale)


        self.set_gp(gp_c=gp_c, gp_loc=gp_loc_x, gp_scale=gp_scale_x, gp_weight=gp_weight)

    
    def fit_bulk(self, data, train_t=None, opt_method=None, bulk_bound=None,
                    bulk_dist=None, bulk_df=None, bulk_nc=None,
                    gp_c=None, gp_loc=None, gp_scale=None, gp_weight=None):

        self.set_t_size(t_size=np.size(data))

        if train_t is None:
            train_t = np.array([0, data.size])
        self._train_t = train_t

        self.set_gp(gp_c=gp_c, gp_loc=gp_loc, gp_scale=gp_scale, gp_weight=gp_weight)
        
        self._bulk_fit = bulk_fit(x=data, p=self._gp_weight, u=self._gp_loc, dist=bulk_dist, df_0=bulk_df, nc_0=bulk_nc, bound=bulk_bound, opt_method=opt_method, train_t=self._train_t)

        self.set_bulk(bulk_dist=self._bulk_fit['db']['dist'], 
                        bulk_loc=self._bulk_fit['ls']['est']['loc'], bulk_scale=self._bulk_fit['ls']['est']['scale'], 
                        bulk_df=self._bulk_fit['nct']['est']['df'], bulk_nc=self._bulk_fit['nct']['est']['nc'])


 
    def set_y_dist(self):

        gp_c_y = self._gp_c[:,0]
        gp_loc_y = p_rs(self.bulk_transform_y_x(self._gp_loc))[:,0]
        gp_scale_y = p_rs(self._gp_scale/self._bulk_scale)[:,0]

        self._y_dist = bulk_gp(bulk_dist=self._bulk_dist, bulk_loc=0, bulk_scale=1, bulk_df=self._bulk_df, bulk_nc=self._bulk_nc,
                        gp_c=gp_c_y, gp_loc=gp_loc_y, gp_scale=gp_scale_y)


    def calc_gp_dict(self):

        N_dim = np.shape(self._gp_c)[0]
        gp_dict = { 
                        **{f'xi_y_{1+k}': self._gp_c[k,0] for k in range(N_dim)},
                        **{f'u_y_{1+k}': self._gp_loc[k,0] for k in range(N_dim)},
                        **{f'sigma_y_{1+k}': self._gp_scale[k,0] for k in range(N_dim)},
                    }

        return gp_dict


    def bulk_transform_x_y(self, y):
        return bulk_transform_x_y(y=y, bulk_loc=self._bulk_loc, bulk_scale=self._bulk_scale)

    def bulk_transform_y_x(self, x):
        return bulk_transform_y_x(x=x, bulk_loc=self._bulk_loc, bulk_scale=self._bulk_scale)

        
        
    def set_bulk_gp(self, bulk_dist=None, bulk_loc=None, bulk_scale=None, bulk_df=None, bulk_nc=None,
                            gp_c=None, gp_loc=None, gp_scale=None, gp_weight=None, gp_sign=None):

        self.set_bulk(bulk_dist=bulk_dist, bulk_loc=bulk_loc, bulk_scale=bulk_scale, bulk_df=bulk_df, bulk_nc=bulk_nc)
        self.set_gp(gp_c=gp_c, gp_loc=gp_loc, gp_scale=gp_scale, gp_weight=gp_weight, gp_sign=gp_sign)

    def set_bulk(self, bulk_dist=None, bulk_loc=None, bulk_scale=None, bulk_df=None, bulk_nc=None):
        if bulk_dist is None:
            bulk_dist = 'norm'
        if bulk_loc is None:
            bulk_loc = 0
        if bulk_scale is None:
            bulk_scale = 1
        if bulk_df is None:
            bulk_df = 8
        if bulk_nc is None:
            bulk_nc = 0
        
        self._bulk_dist = bulk_dist
        self._bulk_loc = bulk_loc
        self._bulk_scale = bulk_scale
        self._bulk_df = bulk_df
        self._bulk_nc = bulk_nc

        if self._bulk_dist=='nct':
            self._bulk = scipy.stats.nct(loc=self._bulk_loc, scale=self._bulk_scale, df=self._bulk_df, nc=self._bulk_nc)
        else:
            self._bulk_nc=0
            if self._bulk_dist=='t':
                self._bulk = scipy.stats.t(loc=self._bulk_loc, scale=self._bulk_scale, df=self._bulk_df)
            else:
                self._bulk_df = np.infty
                if self._bulk_dist=='norm':
                    self._bulk = scipy.stats.norm(loc=self._bulk_loc, scale=self._bulk_scale)

        self.kwds['loc'] = self._bulk_loc
        self.kwds['scale'] = self._bulk_scale
        self.kwds['df'] = self._bulk_df
        self.kwds['nc'] = self._bulk_nc



    def set_gp(self, gp_c=None, gp_loc=None, gp_scale=None, gp_weight=None, gp_sign=None):

        if gp_c is None:
            gp_c = 0.25
        if gp_loc is None:
            gp_loc = np.array([-1, 1])
        if gp_scale is None:
            gp_scale = 1
        if gp_sign is None:
            gp_sign = np.array([-1,1], dtype=int)

        self._gp_c = p_rs(gp_c)
        self._gp_loc = p_rs(gp_loc)
        self._gp_scale = p_rs(gp_scale)
        self._gp_sign = p_rs(gp_sign)

        if gp_weight is None:
            self._gp_weight = np.array([[0],[1]]) + np.array([[1],[-1]]) * self._bulk.cdf(self._gp_loc)
        else:
            self._gp_weight = p_rs(gp_weight)

        self._gp = [scipy.stats.genpareto(c=self._gp_c[i,:], loc=self._gp_sign[i,:]*self._gp_loc[i,:], scale=self._gp_scale[i,:]) for i in range(2)]


    def tailsort(self, x):

        return tailsort_fn(x, self._gp_loc, self._gp_sign)

        #x = misc_fns.make_iterable_array(x)
        #tailsort = [None for i in range(2)]
        #for i in range(2):
        #    tailsort[i] = self._gp_sign[i,:] * x > self._gp_sign[i,:] * self._gp_loc[i, :]

        #return x, tailsort

    def get_N_dist(self):

        return np.max([
                        np.size(self._bulk_loc), np.size(self._bulk_scale), np.size(self._bulk_df), np.size(self._bulk_nc),
                        np.shape(self._gp_c)[1], np.shape(self._gp_loc)[1], np.shape(self._gp_scale)[1], 
                        ])



    def get_sub_dist(self, k_0=None, k_F=None):

        if k_0 is None:
            k_0 = 0
        if k_F is None:
            k_F = k_0 + 1

        N_dist = self.get_N_dist()


        if np.size(self._bulk_loc) == N_dist:
            bulk_loc = self._bulk_loc[k_0:k_F]
        else:
            bulk_loc = self._bulk_loc
        if np.size(self._bulk_scale) == N_dist:
            bulk_scale = self._bulk_scale[k_0:k_F]
        else:
            bulk_scale = self._bulk_scale
        if np.size(self._bulk_df) == N_dist:
            bulk_df = self._bulk_df[k_0:k_F]
        else:
            bulk_df = self._bulk_df
        if np.size(self._bulk_nc) == N_dist:
            bulk_nc = self._bulk_nc[k_0:k_F]
        else:
            bulk_nc = self._bulk_nc

        if np.shape(self._gp_c)[1] == N_dist:
            gp_c = self._gp_c[:, k_0:k_F]
        else:
            gp_c = self._gp_c
        if np.shape(self._gp_loc)[1] == N_dist:
            gp_loc = self._gp_loc[:, k_0:k_F]
        else:
            gp_loc = self._gp_loc
        if np.shape(self._gp_scale)[1] == N_dist:
            gp_scale = self._gp_scale[:, k_0:k_F]
        else:
            gp_scale = self._gp_scale


        sub_dist = bulk_gp(bulk_dist=self._bulk_dist, bulk_loc=bulk_loc, bulk_scale=bulk_scale, bulk_df=bulk_df, bulk_nc=bulk_nc,
                                gp_c=gp_c, gp_loc=gp_loc, gp_scale=gp_scale)

        return sub_dist




    def rvs(self, N=None):

        if N is None:
            N = self.get_N_dist()

        rvs_F = scipy.stats.uniform().rvs(N)
        return self.ppf(rvs_F)



    def pdf(self, x):

        x, tailsort = self.tailsort(x)
        
        pdf = self._bulk.pdf(x) * (1 - np.sum(self._gp_weight, axis=0))

        for i in range(2):
            pdf_gp = self._gp[i].pdf(self._gp_sign[i,:]*x) * self._gp_weight[i,:]
            pdf[tailsort[i]] = pdf_gp[tailsort[i]]

        return pdf


    def cdf(self, x):

        x, tailsort = self.tailsort(x)
        
        cdf = self._bulk.cdf(x)

        for i in range(2):
            cdf_gp = i - self._gp_sign[i,:] * (1 - self._gp[i].cdf(self._gp_sign[i,:]*x)) * self._gp_weight[i,:]
            cdf[tailsort[i]] = cdf_gp[tailsort[i]]

        return cdf


    def ppf(self, F):

        F = misc_fns.make_iterable_array(F)
        
        x = self._bulk.ppf(F)
        x, tailsort = self.tailsort(x)

        for i in range(2):
            x_gp = self._gp_sign[i,:] * self._gp[i].ppf(1 + self._gp_sign[i,:] * ((F - i) / self._gp_weight[i,:]))
            x[tailsort[i]] = x_gp[tailsort[i]]

        return x


    def xint(self, lb=None, ub=None):

        N = self.get_N_dist()

        F_lub = np.full((2, N), np.nan)
        x_lub = np.full((2, N), np.infty)
        x_lub[0,:] *= -1

        if lb is not None:
            x_lub[0,:] = lb
        if ub is not None:
            x_lub[1,:] = ub

        for i in range(2):
            F_lub[i,:] = self.cdf(x_lub[i,:])

        gp_loc_F = self._bulk.cdf(self._gp_loc)

        lb_piece = np.sum(F_lub[0,:] > gp_loc_F, axis=0)
        ub_piece = np.sum(F_lub[1,:] > gp_loc_F, axis=0)


        tailsort = np.array([[np.logical_and(lb_piece==i, ub_piece==j) for j in range(3)] for i in range(3)])


        weight = np.full((3, N), np.nan)
        piece_xint = np.full((3, N), np.nan)

        weight[0,:] = gp_loc_F[0,:]
        weight[1,:] = np.diff(gp_loc_F, axis=0)
        weight[2,:] = 1-gp_loc_F[1,:]
        for i in range(2):
            weight[:1+i,np.sum(tailsort[1+i,:,:], axis=0, dtype=bool)] = 0
            weight[1+i:,np.sum(tailsort[:,i,:], axis=0, dtype=bool)] = 0
        weight[0,np.sum(tailsort[0,:,:], axis=0, dtype=bool)] -= F_lub[0,np.sum(tailsort[0,:,:], axis=0, dtype=bool)]
        weight[2,np.sum(tailsort[:,2,:], axis=0, dtype=bool)] -= 1 - F_lub[1,np.sum(tailsort[:,2,:], axis=0, dtype=bool)]
        weight[1,np.sum(tailsort[1,:,:], axis=0, dtype=bool)] -= (F_lub[0,np.sum(tailsort[1,:,:], axis=0, dtype=bool)] - gp_loc_F[0,np.sum(tailsort[1,:,:], axis=0, dtype=bool)])
        weight[1,np.sum(tailsort[:,1,:], axis=0, dtype=bool)] -= -(F_lub[1,np.sum(tailsort[:,1,:], axis=0, dtype=bool)] - gp_loc_F[1,np.sum(tailsort[:,1,:], axis=0, dtype=bool)])

        #bulk_F = np.copy(weight[1,:])

        gp_lub = np.full((2, 2, N), np.nan)
        for i in range(2):
            minus = (-1)**(i+1)
            for j in range(2):
                gp_lub[i,(1-i)+minus*j,:] =  minus*(x_lub[j,:] - self._gp_loc[i,:])
        gp_lub[gp_lub<=0] = 0


        for i in range(2):
            minus = (-1)**(i+1)
            piece_xint[2*i,:] = self._gp_loc[i,:] + minus*xint_gpd(lb=gp_lub[i,0,:], ub=gp_lub[i,1,:], xi=self._gp_c[i,:], sigma=self._gp_scale[i,:])

        bulk_lub = np.copy(x_lub)
        if np.shape(self._gp_loc[0]) == N:
            bulk_lub[0,np.sum(tailsort[0,:,:], axis=0, dtype=bool)] = self._gp_loc[0,np.sum(tailsort[0,:,:], axis=0, dtype=bool)]
            #bulk_lub[0,np.sum(tailsort[:,1,:], axis=0, dtype=bool)] = self._gp_loc[0,np.sum(tailsort[:,1,:], axis=0, dtype=bool)]
        else:
            bulk_lub[0,np.sum(tailsort[0,:,:], axis=0, dtype=bool)] = self._gp_loc[0,:]
            #bulk_lub[0,np.sum(tailsort[:,1,:], axis=0, dtype=bool)] = self._gp_loc[0,:]
        if np.shape(self._gp_loc[1]) == N:
            bulk_lub[1,np.sum(tailsort[:,2,:], axis=0, dtype=bool)] = self._gp_loc[1,np.sum(tailsort[:,2,:], axis=0, dtype=bool)]
        else:
            bulk_lub[1,np.sum(tailsort[:,2,:], axis=0, dtype=bool)] = self._gp_loc[1,:]
            #bulk_lub[0,np.sum(tailsort[:,1,:], axis=0, dtype=bool)] = self._gp_loc[0,:]
        
        piece_xint[1,:] = misc_fns.bound_xint_dist(dist=self._bulk_dist, lb=bulk_lub[0,:], ub=bulk_lub[1,:], loc=self._bulk_loc, scale=self._bulk_scale, df=self._bulk_df, nc=self._bulk_nc) / weight[1,:]
        #piece_xint[1,bulk_F!=0] /= bulk_F[bulk_F!=0] 

        weight /= np.sum(weight, axis=0)

        piece_xint[weight==0] = 0

        xint = np.sum(weight*piece_xint, axis=0)

        #return xint, piece_xint, weight, gp_lub, bulk_lub, tailsort, gp_loc_F
        return xint

    def Q(self, a_q=None):

        if a_q is None:
            a_q = 0.025

        a_q_lr = np.array([a_q, 1-a_q])
        Q = np.array([self.ppf(a_q_lr[i]) for i in range(2)])

        return Q

    def E(self, a_q=None):

        Q = self.Q(a_q=a_q)
        E = np.array([self.xint(ub=Q[0,:]), self.xint(lb=Q[1,:])])

        return E


    def set_vrs(self, a_q, set_eval=None, y=None, train_t_end=None, CT_Q=None, CT_E=None, S_Q=None, S_E=None, save_t=None, t_start=None, t_end=None, period=None):

        self.vrs = vares.vares_a(a_q=a_q, bdist=self, set_eval=set_eval, 
                                    y=y, train_t_end=train_t_end, 
                                    CT_Q=CT_Q,  CT_E=CT_E, S_Q=S_Q, S_E=S_E, 
                                    save_t=save_t, t_start=t_start, t_end=t_end, period=period)

        
    
#============
#============
#
# Functions
#
#===========
#===========


def p_rs(p):
    if np.ndim(p) == 0:
        p_rs = np.array([[p] for i in range(2)])
    elif np.ndim(p) == 1:
        if np.size(p) == 2:
            p_rs = np.transpose([p])
        else:
            p_rs = np.outer(np.ones(2, dtype=int), p)
    else:
        p_rs = p
    return p_rs


def bulk_transform_x_y(y, bulk_loc=0, bulk_scale=1):
    if len(np.shape(y)) > 1:
        return np.outer(np.ones(np.shape(y)[0], dtype=int), bulk_loc) + np.outer(y, bulk_scale)
    elif np.size(y) > 1:
        return np.outer(np.ones(np.size(y), dtype=int), bulk_loc) + np.outer(y, bulk_scale)
    else:
        return bulk_loc + bulk_scale*y


def bulk_transform_y_x(x, bulk_loc=0, bulk_scale=1):
    if len(np.shape(x)) > 1:
        return (x - np.outer(np.ones(np.shape(x)[0], dtype=int), bulk_loc))/bulk_scale
    elif np.size(x) > 1:
        return (x - np.outer(np.ones(np.size(x), dtype=int), bulk_loc))/bulk_scale
    else:
        return (x-bulk_loc)/bulk_scale

def tailsort_fn(x, gp_loc, gp_sign):

    x = misc_fns.make_iterable_array(x)
    tailsort = [None for i in range(2)]
    for i in range(2):
        tailsort[i] = gp_sign[i,:] * x > gp_sign[i,:] * gp_loc[i, :]

    return x, tailsort

def gpd_mean_excess(m, xi, sigma):

    return m + (m*xi + sigma)/(1-xi)

def gpd_cdf(x, xi, sigma):

    z = x/sigma
    is_zero = xi == 0
    if np.any(is_zero):
        F = 1 - np.exp(-z)
        if np.all(is_zero) == False:
            not_zero = np.logical_not(is_zero)
            F[not_zero] = 1 - (1 + xi[not_zero] * z[not_zero])**(-(1/xi[not_zero]))
    else:
        F = 1 - (1 + xi * z)**(-(1/xi))

    return F
        

def xint_gpd(lb=None, ub=None, xi=None, sigma=None):

    if lb is None or np.all(lb==-np.infty):
        lb=0
        p_02 = 1
    else:
        p_02 = (1-gpd_cdf(x=lb, xi=xi, sigma=sigma))
    if ub is None or np.all(ub==np.infty):
        p_12 = 0
        me_ub = 0
    else:
        p_12 = (1-gpd_cdf(x=ub, xi=xi, sigma=sigma))
        me_ub = gpd_mean_excess(m=ub, xi=xi, sigma=sigma)
    
    p_01 = p_02 - p_12
    ub_excess_frac = misc_fns.make_iterable_array(p_12*me_ub)
    ub_excess_frac[p_12==0] = 0
    

    return (p_02*gpd_mean_excess(m=lb, xi=xi, sigma=sigma) - ub_excess_frac)/p_01

#============
# bulk_fit
#===========

def bulk_parameters(p, u, dist="norm", df=1E+18, nc=0, bound=None):

    p = p_rs(p)
    u = p_rs(u)
    F_p = np.copy(p)
    if bound=="left":
        F_p[1,:] = 1 - p[0,:]
    elif bound=="right":
        F_p[0,:] = p[1,:]
        F_p[1,:] = 1 - p[1,:]
    else:
        F_p[1,:] = 1 - p[1,:]

    if dist=="norm":
        x_p = scipy.stats.norm().ppf(F_p)
        df = np.nan
        nc = np.nan
    elif dist=="t":
        x_p = scipy.stats.t(df=df).ppf(F_p)
        nc = np.nan
    elif dist=="nct":
        x_p = scipy.stats.nct(df=df, nc=nc).ppf(F_p)

    scale = np.diff(u, axis=0)[0]/np.diff(x_p, axis=0)[0]
    loc = (np.sum(u, axis=0) - scale*np.sum(x_p, axis=0))/2

    return loc, scale, df, nc


def bulk_norm_dev(x, p, u, bulk_bool, bound=None):
    loc, scale, df, nc = bulk_parameters(p=p, u=u, dist="norm", bound=bound)
    dist = scipy.stats.norm(loc=loc[bulk_bool],
                            scale=scale[bulk_bool])
    return -2*np.sum(np.log(dist.pdf(x[bulk_bool])))

def bulk_t_dev(param, x, p, u, bulk_bool, bound=None):
    loc, scale, df, nc = bulk_parameters(p=p, u=u, dist="t", df=param[0], bound=bound)
    dist = scipy.stats.t(df=df,
                            loc=loc[bulk_bool],
                            scale=scale[bulk_bool])



    return -2*np.sum(np.log(dist.pdf(x[bulk_bool])))

def bulk_nct_dev(param, x, p, u, bulk_bool, bound=None):
    loc, scale, df, nc = bulk_parameters(p=p, u=u, dist="nct", df=param[0], nc=param[1], bound=bound)
    dist = scipy.stats.nct(df=df, nc=nc,
                            loc=loc[bulk_bool],
                            scale=scale[bulk_bool])

    return -2*np.sum(np.log(dist.pdf(x[bulk_bool])))


def get_bulk_bool(x, u):

    bulk_bool = ~np.logical_or(x < u[0], x > u[1])

    return bulk_bool

def bulk_fit(x, p, u, dist=None, df_0=None, nc_0=None, bound=None, opt_method=None, train_t=None):

    if dist is None:
        dist = 'norm'
    if bound is None:
        bound = 'both'
    if df_0 is None:
        df_0 = 6
    if nc_0 is None:
        nc_0 = 0
    if train_t is None:
        train_t = [0, x.size]
    if opt_method is None:
        opt_method = 'SLSQP'

    bulk_bool = get_bulk_bool(x=x, u=u)

    train_bulk_bool = np.copy(bulk_bool)
    train_bulk_bool[:train_t[0]] = False
    train_bulk_bool[train_t[-1]:] = False

    nct_bounds = np.array([
                            [2.1, 100],
                            [-3, 3]
                        ])


    se = np.full(2, np.nan)
    res = None


    

    if dist=="norm":
        def dev_fun(param=None):
            return bulk_norm_dev(x, p, u, train_bulk_bool, bound)
        param_0 = np.array([])
    elif dist=="t":
        def dev_fun(param):
            return bulk_t_dev(param, x, p, u, train_bulk_bool, bound)
        param_0 = np.array([df_0])
        nct_bounds_lb = nct_bounds[0,0]
        nct_bounds_ub = nct_bounds[0,1]
    elif dist=='nct': 
        def dev_fun(param):
            return bulk_nct_dev(param, x, p, u, train_bulk_bool, bound)
        param_0 = np.array([df_0, nc_0])
        nct_bounds_lb = nct_bounds[:,0]
        nct_bounds_ub = nct_bounds[:,1]
    
    if opt_method == 'load':
        loc, scale, df, nc = bulk_parameters(p=p, u=u, dist=dist, df=df_0, nc=nc_0, bound=bound)
        log_L = -dev_fun(param=param_0)/2
    
    else:
        if dist=="norm":
            fitted_success=None
            SLSQP_timer = misc_fns.timer(reset_start=True)
            loc, scale, df, nc = bulk_parameters(p=p, u=u, dist=dist, bound=bound)
            log_L = -bulk_norm_dev(x, p, u, train_bulk_bool, bound=bound)/2
        else:
            fitted_success=True
            


            SLSQP_timer = misc_fns.timer()
            res = scipy.optimize.minimize(dev_fun, x0=param_0, method=opt_method,
                                            bounds=scipy.optimize.Bounds(lb=nct_bounds_lb, ub=nct_bounds_ub))
            SLSQP_timer.stop()

            if dist=="t":
                loc, scale, df, nc = bulk_parameters(p=p, u=u, dist=dist, df=res.x[0], bound=bound)
            elif dist=='nct':
                loc, scale, df, nc = bulk_parameters(p=p, u=u, dist=dist, df=res.x[0], nc=res.x[1], bound=bound)


            
            hess_success = True
            try:
                hess = numdifftools.Hessian(dev_fun, full_output=True, method="central", step=res.x/1E6)(res.x)[0]
            except IndexError as err:
                hess_success=False
                fitted_success=False
            if hess_success:
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

                    'db': {'dist': dist, 'bound': bound},
                    'ls': {'est': {'loc': loc, 'scale': scale}},
                    'nct':  {
                                'est': {'df': df, 'nc': nc},
                                'se': {'df': se[0], 'nc': se[1]},
                            },
                    'res':  {
                                'log_L': log_L,
                                'SLSQP_time': SLSQP_timer.get_timer_dict(),
                                'res': res,
                            },
                    'res_t':  {
                                'train_t': train_t,
                            },
                }


    return ret_dict





