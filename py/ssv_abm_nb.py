
import numpy as np
#import scipy
import numba as nb
from numba import jit, njit
#import numba_scipy






@njit(nb.float64(nb.bool_[:]))
def calc_A_n(N_s_a_bool):

    # return 1.0  # A_n mod
    return np.sqrt(np.sum(N_s_a_bool))


@njit(nb.float64[:,:](nb.float64[:,:]))
def diag_mirror(M):

    return M + M.T


@njit(   [
            nb.float64[:,:](nb.int64, nb.float64, nb.float64[:,:], nb.float64[:,:]),
            nb.float64[:,:](nb.int64, nb.float64, nb.int32[:,:], nb.int32[:,:])
        ])
def calc_Hess_F_fast_2(T, sigma_F, diag, diag_1):

    D = 1/(sigma_F**2)
    
    return -2*D*diag + D*diag_1


@njit([nb.float64[:](
                        nb.float64[:], nb.float64
                    ),
    nb.float64[:,:](
                    nb.float64[:,:], nb.float64
                )
                    ])
def calc_n_t_s(a_t_s, beta=1.0):    # A_n mod


    safe_a_t_s = a_t_s - np.max(a_t_s)

    n_t_s_t = np.exp(beta*safe_a_t_s)
    return n_t_s_t/np.sum(n_t_s_t, axis=0)




def calc_n_t_s_2d(a_t_s, beta=1.0):    # A_n mod


    safe_a_t_s = a_t_s - np.max(a_t_s, axis=0)

    n_t_s_t = np.exp(beta*safe_a_t_s)
    return n_t_s_t/np.sum(n_t_s_t, axis=0)




# test_int64
@njit([nb.float64(
                    nb.float64, nb.int64, nb.float64, nb.float64
                    ),
    nb.float64[:](
                    nb.float64[:], nb.int64, nb.float64, nb.float64
                )
                    ])
def rho_njit(x, type_int=1, gamma_1=0.0, gamma_2=1.0):

    rho = x
        
    if type_int==0:
        rho = 0.0*x
    elif type_int==1:
        rho = x
    elif type_int==2:
        rho =  x + gamma_1*(x**3)
    elif type_int==3:
        rho =  np.tanh(gamma_1*x)
    elif type_int==4:
        rho = np.tanh(gamma_1*(1 - np.sqrt(x)/gamma_2))

    return rho



@njit(nb.int64(
                    nb.float64,  nb.float64, nb.int64,
                    ),
                    )
def calc_tbar_alpha(eta=1.0, alpha=0.99, tbar=1):

    if eta==0:
        return tbar
    elif eta==1:
        return 1
    else:
        return 1 + int(np.log(1-alpha)/np.log(1-eta))
    

def expon_window(X, tbar=None, eta=None, alpha=None):

    T = np.size(X)
    

    if eta is None:
        eta = 0


    if tbar is None:
        tbar = calc_tbar_alpha(eta=eta, alpha=alpha, tbar=T)

    if eta==1:
        expon_kernel = np.zeros(tbar)
        expon_kernel[-1] = 1.0
    else:
        expon_kernel = (1-eta)**np.flip(np.arange(tbar))
    norm_expon_kernel = np.flip(np.cumsum(np.flip(expon_kernel)))




@njit(nb.types.Tuple((
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], 
                    ))(
                        nb.bool_, nb.int64, nb.int64,
                        nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.float64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64, nb.float64[:], 
                        nb.int64, nb.float64,
                        nb.bool_[:], nb.bool_[:], nb.int32[:],
                        nb.float64[:,:], nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], 
                    ))
def run_p_t_njit(sim, T_buff, T_1, 
                psi, delta_s, gamma_1_s, gamma_2_s, sigma_s, eta_s, eta_varsigma, varsigma_0,
                beta, eta_w, alpha_0_s, alpha_n, alpha_w, lambda_w_s, 
                G_sigma, A_G,
                N_s_a_bool, N_s_e_bool, rho_type_int_s, 
                nepsilon_t_s, p_F, 
                p, r, varsigma_t, D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s):

    N_s = np.shape(rho_t_s)[0]


    psi_2 = psi**2

    update_n_t_s = np.any(N_s_e_bool)

    A_n = calc_A_n(N_s_a_bool=N_s_a_bool)  # A_n mod



    exp_gamma_2_s2 = np.exp(gamma_2_s[2])



    for t in range(T_buff,T_1):


        a_t_s[:,t] = alpha_0_s + alpha_n*n_t_s[:,t-1] + alpha_w*nu_t_s[:,t-1]

        if update_n_t_s:
            #n_t_s[N_s_e_bool,t] = calc_n_t_s(a_t_s=a_t_s[N_s_e_bool,t], beta=beta)  # A_n mod
            n_t_s[N_s_e_bool,t] = A_n * calc_n_t_s(a_t_s=a_t_s[N_s_e_bool,t], beta=beta)

        S_t_s[0,t] = (p_F[t-1] - p[t-1])/psi
        S_t_s[1,t] = (p[t-1] - p[t-2])/psi

        

        if varsigma_t[t-1]==0:
            S_t_s[2,t] = 0.0
        else:
            S_t_s[2,t] = 1 - np.abs(r[t-1]/psi)/(exp_gamma_2_s2*varsigma_t[t-1])

            #S_t_s[2,t] = (r[t-1] - rbar_t)**2 / (psi_2*A_G)

        M_t_s[:,t] = eta_s * S_t_s[:,t] + (1-eta_s) * M_t_s[:,t-1]

        for s in range(N_s):
            rho_t_s[s,t] = rho_njit(x=M_t_s[s,t], type_int=rho_type_int_s[s], gamma_1=gamma_1_s[s], gamma_2=gamma_2_s[s])


        if sim:
            D_t_s[:,t] = delta_s*rho_t_s[:,t] + sigma_s*nepsilon_t_s[:,t]   
            r[t] = psi*(n_t_s[N_s_a_bool,t] @ D_t_s[N_s_a_bool,t])  # A_n mod
            p[t] = p[t-1] + r[t]

        
        if t==T_buff:
            varsigma_t[t] =  eta_varsigma * np.abs(r[t]/psi) + (1-eta_varsigma) * varsigma_0
        else:
            varsigma_t[t] =  eta_varsigma * np.abs(r[t]/psi) + (1-eta_varsigma) * varsigma_t[t-1]



        #g_t_s[:,t] = (r[t]/psi) * (delta_s * rho_t_s[:,t-1] + G_sigma * sigma_s * nepsilon_t_s[:,t-1])
        g_t_s[:,t] = (r[t]/psi) * (delta_s * rho_t_s[:,t-1] + G_sigma * sigma_s * nepsilon_t_s[:,t-1])/A_G  # A_n mod
        w_t_s[:,t] = eta_w*g_t_s[:,t] + (1-eta_w)*w_t_s[:,t-1]
        nu_t_s[:,t] = w_t_s[:,t] * np.exp(lambda_w_s*(w_t_s[:,t] < 0))


    return (p, r, varsigma_t, D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s)




@njit(nb.float64[:](
                        nb.float64[:], 
                        nb.int64, nb.int64,
                        nb.float64,
                        nb.float64,
                    ))
def append_Hess_F_k_factor(row,
                        k, T, 
                        factor=1.0,
                        epsilon=0.0):





        if k<T-1:
            row[k+1] += factor
        if k>0:
           row[k-1] += factor

        row[k] += epsilon-2.0*factor
        
        return row




@njit(nb.float64[:](
                        nb.int64, nb.int64,
                        nb.float64,
                        nb.float64,
                    ))
def calc_Hess_F_k_factor(k, T, 
                        factor=1.0,
                        epsilon=0.0):


    return append_Hess_F_k_factor(np.zeros(T),
                        k, T, 
                        factor,
                        epsilon)


@njit(nb.float64[:,:](
                        nb.float64[:,:],
                        nb.int64,
                        nb.float64,
                        nb.float64,
                    ))
def append_Hess_F_nb_factor(Hess,
                        T, 
                        factor=1.0,
                        epsilon=0.0):

        for k in range(T):
            append_Hess_F_k_factor(Hess[k,:], k, T, 
                        factor=factor,
                        epsilon=epsilon)

        return Hess


@njit(nb.float64[:,:](
                        nb.int64,
                        nb.float64,
                        nb.float64,
                    ))
def calc_Hess_F_nb_factor(
                        T, 
                        factor=1.0,
                        epsilon=0.0):

        Hess = np.zeros((T,T))

        append_Hess_F_nb_factor(Hess, T, 
                        factor=factor,
                        epsilon=epsilon)

        return Hess



@njit(nb.float64[:,:](
                        nb.float64[:,:],
                        nb.int64,
                        nb.float64, nb.float64,
                        nb.float64,
                    ))
def append_Hess_F_nb(Hess,
                        T, 
                        psi, sigma_F,
                        epsilon=0.0):

        factor = (psi*sigma_F)**(-2)
        for k in range(T):
            append_Hess_F_k_factor(Hess[k,:], k, T, 
                        factor=factor,
                        epsilon=epsilon)

        return Hess


@njit(nb.float64[:,:](
                        nb.int64,
                        nb.float64, nb.float64,
                        nb.float64,
                    ))
def calc_Hess_F_nb(T, 
                    psi, sigma_F,
                    epsilon=0.0):

    return append_Hess_F_nb(Hess=np.zeros((T,T)), 
                            T=T, 
                            psi=psi, sigma_F=sigma_F, 
                            epsilon=epsilon)



@njit(nb.float64[:](
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                    ))
def calc_Hess_D_K_kp1_factors(
                                delta_f, 
                                n_t_f, 
                                sigma_t_D_2, rho_prime_k, rho_prime2_k, D_mu,
                            ):
    
    return -(((rho_prime_k*delta_f*n_t_f)**2 - rho_prime2_k*delta_f*D_mu)/ sigma_t_D_2) 


@njit(nb.float64[:](
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                    ))
def calc_Hess_D_K_kp2_factors(
                                n_t_f, 
                                sigma_t_D_2, D_mu, dr_mu, ns2_m_1, nu_w,
                            ):
    
    return (nu_w * (ns2_m_1*(1 - (D_mu**2)/sigma_t_D_2) - ((D_mu*dr_mu)/sigma_t_D_2)) * n_t_f)

@njit(nb.float64[:](
                        nb.float64, 
                        nb.float64,
                        nb.float64[:], nb.float64[:], 
                    ))
def calc_Hess_D_K_kp0_factors(
                                beta, 
                                Q_q, 
                                rho_prime2_k, r_1):
    
    return -beta * Q_q * rho_prime2_k[1:] * r_1[:-1]



@njit(nb.float64[:,:](
                        nb.float64[:,:], 
                        nb.int64,
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                    ))
def append_Hess_D_N_nb(Hess, 
                        T, 
                        delta_f, 
                        n_t_f, 
                        sigma_t_D_2, rho_prime_k, rho_prime2_k, D_mu,
                        ):
    
    kp1_factors = calc_Hess_D_K_kp1_factors(
                                delta_f=delta_f, 
                                n_t_f=n_t_f, 
                                sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, D_mu=D_mu,
                            )

    for k in range(T-1):

        k_prime = T-(2+k)

        Hess[k_prime,k_prime] += kp1_factors[1+k_prime]

    return Hess

@njit(nb.float64[:,:](
                        nb.int64,
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                    ))
def calc_Hess_D_N_nb(
                        T, 
                        delta_f, 
                        n_t_f, 
                        sigma_t_D_2, rho_prime_k, rho_prime2_k, D_mu,
                        ):
    
    return append_Hess_D_N_nb(Hess=np.zeros((T,T)), 
                        T=T, 
                        delta_f=delta_f, 
                        n_t_f=n_t_f, 
                        sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, D_mu=D_mu,
                        )


@njit(nb.float64[:,:](
                        nb.float64[:,:], 
                        nb.int64,
                        nb.float64, 
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64, 
                    ))
def append_Hess_D_B_K_nb(Hess, 
                        T, 
                        beta, 
                        Q_q, 
                        n_t_f, 
                        sigma_t_D_2, rho_prime2_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                        eta_w=1.0):

    b_Q = beta * Q_q

    if b_Q!=0:

        Reta_w = 1 - eta_w


        kp0_factors = calc_Hess_D_K_kp0_factors(
                                beta=beta, 
                                Q_q=Q_q, 
                                rho_prime2_k=rho_prime2_k, r_1=r_1)
        

        kp2_factors = calc_Hess_D_K_kp2_factors(
                                n_t_f=n_t_f, 
                                sigma_t_D_2=sigma_t_D_2, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w,
                            )
        
        kp2_factor = 0.0
        

        for k in range(T-1):

            k_prime = T-(2+k)
            
            if k_prime < T-2:
                kp2_factor =  kp2_factors[2+k_prime] + Reta_w * kp2_factor

            Hess[k_prime,k_prime] += kp0_factors[k_prime]*kp2_factor

    return Hess


@njit(nb.float64[:,:](
                        nb.int64,
                        nb.float64, 
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                        nb.float64,
                    ))
def calc_Hess_D_B_K_nb(T, 
                        beta, 
                        Q_q, 
                        n_t_f, 
                        sigma_t_D_2, rho_prime2_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                        eta_w=1.0):

    return append_Hess_D_B_K_nb(Hess=np.zeros((T,T)), 
                                T=T, 
                                beta=beta, 
                                Q_q=Q_q, 
                                n_t_f=n_t_f, 
                                sigma_t_D_2=sigma_t_D_2, rho_prime2_k=rho_prime2_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1, 
                                eta_w=eta_w)




@njit(nb.float64[:,:](
                        nb.float64[:,:], 
                        nb.int64,
                        nb.float64, 
                        nb.float64, 
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64, 
                    ))
def append_Hess_D_K_nb(Hess, 
                        T, 
                        delta_f, 
                        beta, 
                        Q_q, 
                        n_t_f, 
                        sigma_t_D_2, rho_prime_k, rho_prime2_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                        eta_w=1.0):


    Reta_w = 1 - eta_w

    kp1_factors = calc_Hess_D_K_kp1_factors(
                                delta_f=delta_f, 
                                n_t_f=n_t_f, 
                                sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, D_mu=D_mu,
                            )


    kp0_factors = calc_Hess_D_K_kp0_factors(
                                beta=beta, 
                                Q_q=Q_q, 
                                rho_prime2_k=rho_prime2_k, r_1=r_1)
        

    kp2_factors = calc_Hess_D_K_kp2_factors(
                            n_t_f=n_t_f, 
                            sigma_t_D_2=sigma_t_D_2, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w,
                        )
    kp1_factor = 0.0
    kp2_factor = 0.0

    for k in range(T-1):

        k_prime = T-(2+k)


        kp1_factor = kp1_factors[1+k_prime]

        if beta!=0:
            if k_prime < T-2:
                kp2_factor =  kp2_factors[2+k_prime] + Reta_w * kp2_factor

        Hess[k_prime,k_prime] += kp1_factor + kp0_factors[k_prime]*kp2_factor

    return Hess


@njit(nb.float64[:,:](
                        nb.int64,
                        nb.float64, 
                        nb.float64, 
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64,
                    ))
def calc_Hess_D_K_nb(T, 
                        delta_f, 
                        beta, 
                        Q_q, 
                        n_t_f, 
                        sigma_t_D_2, rho_prime_k, rho_prime2_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                        eta_w=1.0):

    return append_Hess_D_K_nb(Hess=np.zeros((T,T)), 
                                T=T, 
                                delta_f=delta_f, 
                                beta=beta, 
                                Q_q=Q_q, 
                                n_t_f=n_t_f, 
                                sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1, 
                                eta_w=eta_w)


@njit(nb.float64[:,:](
                        nb.float64[:,:], nb.int64, 
                        nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64
                    ))
def append_Hess_D_K_nb_alt(Hess, T, 
                        beta, delta_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, rho_prime2_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=1.0):


    Reta_w = 1 - eta_w


    k_factors = -beta * Q_q * rho_prime2_k[1:] * r_1[:-1]
    kp1_factor = 0.0
    kp2_factor = 0.0

    kp1_factors = -(((rho_prime_k*delta_f*n_t_f)**2 - rho_prime2_k*delta_f*D_mu)/ sigma_t_D_2) 
    kp2_factors = (nu_w * (ns2_m_1*(1 - (D_mu**2)/sigma_t_D_2) - ((D_mu*dr_mu)/sigma_t_D_2)) * n_t_f)

    diag = np.diag(Hess)

    for k in range(T-1):

        k_prime = T-(2+k)


        kp1_factor = kp1_factors[1+k_prime]

        if beta!=0:
            if k_prime < T-2:
                kp2_factor =  kp2_factors[2+k_prime] + Reta_w * kp2_factor

        diag[k_prime] += kp1_factor + k_factors[k_prime]*kp2_factor

    np.fill_diagonal(Hess, diag)

    return Hess

@njit(nb.float64[:,:](
                        nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64
                    ))
def calc_Hess_D_K_nb_alt(T, 
                        beta, delta_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, rho_prime2_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=1.0):

    return append_Hess_D_K_nb_alt(np.zeros((T,T)), T, 
                        beta, delta_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, rho_prime2_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=eta_w)



@njit(nb.float64(
                        nb.int64, nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64
                    ))
def calc_Hess_D_K_kk(k, T, 
                        beta, delta_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, rho_prime2_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=1.0):
        


        k_factor = -beta * Q_q * (rho_prime2_k[k]*r_1[k])
        kp2_factor = 0.0
        kp1_factor = 0.0

        if k<T-1:
            kp1_factor = -(((rho_prime_k[k+1]*delta_f*n_t_f[k+1])**2 - rho_prime2_k[k+1]*delta_f*D_mu[k+1])/ sigma_t_D_2[k+1]) 
            if k<T-2:

                ts = np.arange(T)
                Reta_w_pow = ts-(2+k)
                if eta_w==1:
                    Reta_w_l = 1.0*(Reta_w_pow==0)
                else:
                    ln_Reta = np.log(1-eta_w)
                    Reta_w_l = (Reta_w_pow>=0)*np.exp(ln_Reta*(Reta_w_pow))

                kp2_factor = np.sum(Reta_w_l[2+k:]*nu_w[2+k:] * (ns2_m_1[2+k:]*(1 - (D_mu[2+k:]**2)/sigma_t_D_2[2+k:]) - ((D_mu[2+k:]*dr_mu[2+k:])/sigma_t_D_2[2+k:])) * n_t_f[2+k:])
                
                #kp2_factor = np.flip(np.cumsum(np.flip(Reta_w_l*nu_w * (ns2_m_1*(1 - (D_mu**2)/sigma_t_D_2) - ((D_mu*dr_mu)/sigma_t_D_2)) * n_t_f)))[2+k]
                #np.flip(np.cumsum(np.flip(eta_l[k+2]*nu_w[k+2] * (ns2_m_1[k+2]*(1 - (D_mu[k+2]**2)/sigma_t_D_2[k+2]) - ((D_mu[k+2]*dr_mu[k+2])/sigma_t_D_2[k+2])) * n_t_f[k+2])))

        return kp1_factor + k_factor*kp2_factor




@njit(nb.float64[:](
                        nb.int64, nb.int64,
                        nb.float64, nb.float64, nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64, nb.float64, 
                    ))
def calc_Hess_F_D_K_k(k, T, 
                        beta, psi, sigma_F, delta_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, rho_prime2_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=1.0, epsilon=0.0):

        factor = (psi*sigma_F)**(-2)
        row = calc_Hess_F_k_factor(k=k, T=T, factor=factor, epsilon=epsilon)

        row[k] += calc_Hess_D_K_kk(k=k, T=T, 
                        beta=beta, delta_f=delta_f, n_t_f=n_t_f, Q_q=Q_q, nu_w=nu_w, D_mu=D_mu, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, ns2_m_1=ns2_m_1, dr_mu=dr_mu, r_1=r_1, 
                        eta_w=eta_w)

        
        return row



@njit(nb.float64[:](
                        nb.int64, nb.int64,
                        nb.float64, nb.float64, nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64, nb.float64, 
                    ))
def calc_Hess_F_D_K_k_alt(k, T, 
                        beta, psi, sigma_F, delta_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, rho_prime2_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=1.0, epsilon=0.0):


        row = np.zeros(T)

        ts = np.arange(T)

        

        inv_sigma_F_2 = (psi*sigma_F)**(-2)

        Reta_w_pow = ts-(2+k)
        if eta_w==1:
            Reta_w_l = 1.0*(Reta_w_pow==0)
        else:
            ln_Reta = np.log(1-eta_w)
            Reta_w_l = (Reta_w_pow>=0)*np.exp(ln_Reta*(Reta_w_pow))


        k_factor = -beta * Q_q * (rho_prime2_k[k]*r_1[k])
        kp2_factor = 0.0
        kp1_factor = 0.0


        if k<T-1:
            row[k+1] = inv_sigma_F_2
            kp1_factor = -(((rho_prime_k[k+1]*delta_f*n_t_f[k+1])**2 - rho_prime2_k[k+1]*delta_f*D_mu[k+1])/ sigma_t_D_2[k+1]) 
            if k<T-2:

                kp2_factor = np.sum(Reta_w_l[2+k:]*nu_w[2+k:] * (ns2_m_1[2+k:]*(1 - (D_mu[2+k:]**2)/sigma_t_D_2[2+k:]) - ((D_mu[2+k:]*dr_mu[2+k:])/sigma_t_D_2[2+k:])) * n_t_f[2+k:])
                
                #kp2_factor = np.flip(np.cumsum(np.flip(Reta_w_l*nu_w * (ns2_m_1*(1 - (D_mu**2)/sigma_t_D_2) - ((D_mu*dr_mu)/sigma_t_D_2)) * n_t_f)))[2+k]
                #np.flip(np.cumsum(np.flip(eta_l[k+2]*nu_w[k+2] * (ns2_m_1[k+2]*(1 - (D_mu[k+2]**2)/sigma_t_D_2[k+2]) - ((D_mu[k+2]*dr_mu[k+2])/sigma_t_D_2[k+2])) * n_t_f[k+2])))

        if k>0:
           row[k-1] = inv_sigma_F_2

        row[k] = epsilon-2.0*inv_sigma_F_2 + kp1_factor + k_factor*kp2_factor

        
        return row



@njit(nb.float64[:,:](
                        nb.float64[:,:], 
                        nb.int64,
                        nb.float64, nb.float64, nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                        nb.float64, 
                    ))
def append_Hess_F_D_N_nb(Hess, 
                        T, 
                        psi, sigma_F, delta_f, 
                        n_t_f, 
                        sigma_t_D_2, rho_prime_k, rho_prime2_k, D_mu,
                        epsilon=0.0):

    Hess = append_Hess_F_nb(Hess=Hess, 
                                T=T, 
                                psi=psi, sigma_F=sigma_F, 
                                epsilon=epsilon)
    Hess = append_Hess_D_N_nb(Hess=Hess, 
                                T=T, 
                                delta_f=delta_f, 
                                n_t_f=n_t_f, 
                                sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, D_mu=D_mu,
                               )

    return Hess


@njit(nb.float64[:,:](
                        nb.int64,
                        nb.float64, nb.float64, nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                        nb.float64, 
                    ))
def calc_Hess_F_D_N_nb(
                        T, 
                        psi, sigma_F, delta_f, 
                        n_t_f, 
                        sigma_t_D_2, rho_prime_k, rho_prime2_k, D_mu,
                        epsilon=0.0):
    
    return append_Hess_F_D_N_nb(Hess=np.zeros((T,T)), 
                                T=T, 
                                psi=psi, sigma_F=sigma_F, delta_f=delta_f, 
                                n_t_f=n_t_f, 
                                sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, D_mu=D_mu,
                                epsilon=epsilon
                               )




@njit(nb.float64[:,:](
                        nb.float64[:,:], 
                        nb.int64,
                        nb.float64, nb.float64, nb.float64, 
                        nb.float64, 
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64, nb.float64, 
                    ))
def append_Hess_F_D_K_nb(Hess, 
                        T, 
                        psi, sigma_F, delta_f, 
                        beta, 
                        Q_q, 
                        n_t_f, 
                        sigma_t_D_2, rho_prime_k, rho_prime2_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                        eta_w=1.0, epsilon=1.0):

    Hess = append_Hess_F_nb(Hess=Hess, 
                                T=T, 
                                psi=psi, sigma_F=sigma_F, 
                                epsilon=epsilon)
    Hess = append_Hess_D_K_nb(Hess=Hess, 
                                T=T, 
                                delta_f=delta_f, 
                                beta=beta, 
                                Q_q=Q_q, 
                                n_t_f=n_t_f, 
                                sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1, 
                                eta_w=eta_w)

    return Hess


@njit(nb.float64[:,:](
                        nb.int64,
                        nb.float64, nb.float64, nb.float64, 
                        nb.float64, 
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64, nb.float64, 
                    ))
def calc_Hess_F_D_K_nb(T, 
                        psi, sigma_F, delta_f, 
                        beta, 
                        Q_q, 
                        n_t_f, 
                        sigma_t_D_2, rho_prime_k, rho_prime2_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                        eta_w=1.0, epsilon=1.0):

    
    return append_Hess_F_D_K_nb(Hess=np.zeros((T,T)), 
                                T=T,  
                                psi=psi, sigma_F=sigma_F, delta_f=delta_f, 
                                beta=beta,
                                Q_q=Q_q, 
                                n_t_f=n_t_f, 
                                sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1, 
                                eta_w=eta_w, epsilon=epsilon)





@njit(nb.float64[:,:](
                        nb.float64[:,:], nb.int64,
                        nb.float64, nb.float64, nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64, nb.float64, 
                    ))
def append_Hess_F_D_K_nb_old(Hess, T, 
                        beta, psi, sigma_F, delta_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, rho_prime2_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=1.0, epsilon=0.0):

    for k in range(T):
        Hess[k,:] += calc_Hess_F_D_K_k(k=k, T=T, 
                        beta=beta, psi=psi, sigma_F=sigma_F, delta_f=delta_f, n_t_f=n_t_f, Q_q=Q_q, nu_w=nu_w, D_mu=D_mu, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, ns2_m_1=ns2_m_1, dr_mu=dr_mu, r_1=r_1, 
                        eta_w=eta_w, epsilon=epsilon)

    return Hess


@njit(nb.float64[:,:](
                        nb.int64,
                        nb.float64, nb.float64, nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64, nb.float64, 
                    ))
def calc_Hess_F_D_K_nb_old(T, 
                        beta, psi, sigma_F, delta_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, rho_prime2_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=1.0, epsilon=0.0):

    
    return append_Hess_F_D_K_nb_old(np.zeros((T,T)), T, 
                        beta, psi, sigma_F, delta_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, rho_prime2_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=eta_w, epsilon=epsilon)









@njit(nb.float64[:](
                        nb.float64[:],
                        nb.int64, nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64,
                    ))
def append_Hess_D_B_L_k(row,
                        k, T, 
                        beta, delta_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=1.0):


        ts = np.arange(T)

        if eta_w==1:
            Reta_w_l = 1.0*(ts==0)
        else:
            ln_Reta = np.log(1-eta_w)
            Reta_w_l = (ts>=0)*np.exp(ln_Reta*ts)



        if k < T-1:
            row_full = -(beta * delta_f * Q_q) * r_1[k]*rho_prime_k[1+k]  * nu_w[2:]*((1/sigma_t_D_2) * ((n_t_f**2)*(2*beta*D_mu*ns2_m_1 + rho_prime_k*dr_mu) - n_t_f*(1-n_t_f)*rho_prime_k*D_mu))[2:] 
            row[1+k:-1] = row_full[k:] * Reta_w_l[:-(2+k)]
            if k > 0:
                row[:k] = np.flip(row_full[:k] * Reta_w_l[:k])
        
        return row

@njit(nb.float64[:](
                        
                        nb.int64, nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64,
                    ))
def calc_Hess_D_B_L_k(k, T, 
                        beta, delta_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=1.0):

    return append_Hess_D_B_L_k(np.zeros(T),
                        k, T, 
                        beta, delta_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=eta_w)


@njit(nb.float64[:,:](
                        nb.float64[:,:],
                        nb.int64,
                        nb.float64, 
                        nb.float64, 
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64,
                    ))
def append_Hess_D_B_L_nb(Hess,
                        T, 
                        delta_f, 
                        beta, 
                        Q_q, 
                        n_t_f, 
                        sigma_t_D_2, rho_prime_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                        eta_w=1.0):


        if beta!=0:
            ts = np.arange(T)

            if eta_w==1:
                Reta_w_l = 1.0*(ts==0)
            else:
                ln_Reta = np.log(1-eta_w)
                Reta_w_l = np.exp(ln_Reta*ts)


            row_factor = -(beta * delta_f * Q_q)* r_1[:-2]*rho_prime_k[1:-1]
            column_t_factor = nu_w[2:]*((1/sigma_t_D_2) * ((n_t_f**2)*(2*beta*D_mu*ns2_m_1 + rho_prime_k*dr_mu) - n_t_f*(1-n_t_f)*rho_prime_k*D_mu))[2:] 

            for k in range(T-1):
                Hess[k,1+k:-1] = row_factor[k] * column_t_factor[k:] * Reta_w_l[:-(2+k)]
                Hess[1+k:-1,k] = Hess[k,1+k:-1]
        
        return Hess


@njit(nb.float64[:,:](
                        nb.int64,
                        nb.float64, 
                        nb.float64, 
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64,
                    )) 
def calc_Hess_D_B_L_nb(T, 
                        delta_f, 
                        beta, 
                        Q_q, 
                        n_t_f, 
                        sigma_t_D_2, rho_prime_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                        eta_w=1.0):

    
    return append_Hess_D_B_L_nb(Hess=np.zeros((T,T)),
                                T=T, 
                                delta_f=delta_f, 
                                beta=beta, 
                                Q_q=Q_q, 
                                n_t_f=n_t_f, 
                                sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1, 
                                eta_w=eta_w)


@njit(nb.float64[:,:](
                        nb.float64[:,:], nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64,
                    )) 
def append_Hess_D_B_L_nb_old(Hess, T, 
                        beta, delta_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=1.0):

    

    for k in range(T-1):
        Hess[k,:] = append_Hess_D_B_L_k(Hess[k,:], k=k, T=T, 
                        beta=beta, delta_f=delta_f, n_t_f=n_t_f, Q_q=Q_q, nu_w=nu_w, D_mu=D_mu, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, ns2_m_1=ns2_m_1, dr_mu=dr_mu, r_1=r_1, 
                        eta_w=eta_w)

    return Hess


@njit(nb.float64[:,:](
                        nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64,
                    )) 
def calc_Hess_D_B_L_nb_old(T, 
                        beta, delta_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=1.0):

    
    return append_Hess_D_B_L_nb_old(np.zeros((T,T)), T, 
                        beta, delta_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=eta_w)



@njit([nb.types.UniTuple(nb.float64[:],2)(
                        nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64
                    ),
                    nb.types.UniTuple(nb.float64[:],2)(
                        nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]
                    )
                    ])
def calc_Hess_D_B_Sigma_nb_Sl1Sl2_t_factor(sigma_f, n_t_f, D_mu, sigma_t_D_2, ns2_m_1, dr_mu, factor=1.0):


    Sl1 = factor * ((-2*(n_t_f**3)*(((sigma_f**2)/sigma_t_D_2)**2) + 2*n_t_f*((sigma_f**2)/sigma_t_D_2) - (1-n_t_f)) + ((D_mu**2)/sigma_t_D_2) * (4*(n_t_f**3)*(((sigma_f**2)/sigma_t_D_2)**2) - 2*n_t_f*(1-4*n_t_f)*((sigma_f**2)/sigma_t_D_2) + (1+n_t_f)) + ((dr_mu**2)/sigma_t_D_2) + ((D_mu*dr_mu)/sigma_t_D_2) * (- (1-2*n_t_f)))

    Sl2 = factor * (((D_mu*dr_mu)/sigma_t_D_2) * 2*n_t_f*ns2_m_1)

    return Sl1, Sl2

@njit([nb.types.UniTuple(nb.float64[:],2)(
                        nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64
                    ),
        nb.types.UniTuple(nb.float64[:],2)(
                        nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]
                    )
                    ])
def calc_Hess_D_B_Sigma_nb_Sl1Sl2_t_nu_w_n_t_f_factor(sigma_f, n_t_f, D_mu, sigma_t_D_2, ns2_m_1, dr_mu, nu_w, factor=1.0):


    return calc_Hess_D_B_Sigma_nb_Sl1Sl2_t_factor(sigma_f=sigma_f, n_t_f=n_t_f, D_mu=D_mu, sigma_t_D_2=sigma_t_D_2, ns2_m_1=ns2_m_1, dr_mu=dr_mu, factor=factor*(nu_w**2)*n_t_f)





@njit(nb.float64(
                        nb.int64, nb.int64, nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]
                    ))
def calc_Hess_D_B_Sigma_Sl1_fast_kl(k, l, T, sigma_f, eta_w, n_t_f, nu_w, D_mu, sigma_t_D_2, dr_mu):


        ts = np.arange(T)

        if k>l:
            kl2 = k+2
        else:
            kl2 = l+2
        

        Sl1 = np.zeros((4, T))




        Reta_w_pow = 2*ts-(k+l+4)
        if eta_w==1:
            Reta_w_t = 1.0*(Reta_w_pow==0)
        else:
            ln_Reta = np.log(1-eta_w)
            Reta_w_t = (Reta_w_pow>=0)*np.exp(ln_Reta*(Reta_w_pow))
            #Reta_w_l = np.exp(-ln_Reta*ts)

        Sl1[0,:] = (-2*(n_t_f**3)*(((sigma_f**2)/sigma_t_D_2)**2) + 2*n_t_f*((sigma_f**2)/sigma_t_D_2) - (1-n_t_f))
        Sl1[1,:] = ((D_mu**2)/sigma_t_D_2) * (4*(n_t_f**3)*(((sigma_f**2)/sigma_t_D_2)**2) - 2*n_t_f*(1-4*n_t_f)*((sigma_f**2)/sigma_t_D_2) + (1+n_t_f))
        Sl1[2,:] = ((dr_mu**2)/sigma_t_D_2)
        Sl1[3,:] = ((D_mu*dr_mu)/sigma_t_D_2) * (- (1-2*n_t_f))


        if eta_w==1: 

            return (nu_w[kl2]**2)*n_t_f[kl2]*np.sum(Sl1, axis=0)[kl2]
        else:
           return np.sum(Reta_w_t[kl2:] * (nu_w[kl2:]**2) * n_t_f[kl2:] * np.sum(Sl1, axis=0)[kl2:])





@njit(nb.float64(
                        nb.int64, nb.int64, nb.int64,
                        nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]
                    ))
def calc_Hess_D_B_Sigma_Sl2_fast_kl(k, l, T, eta_w, n_t_f, nu_w, D_mu, sigma_t_D_2, ns2_m_1, dr_mu):


        ts = np.arange(T)

        if k>l:
            kl2 = k+2
        else:
            kl2 = l+2


        ts = np.arange(T)
        Reta_w_pow = 2*ts-(k+l+4)
        if eta_w==1:
            Reta_w_t = 1.0*(Reta_w_pow==0)
        else:
            ln_Reta = np.log(1-eta_w)
            Reta_w_t = (Reta_w_pow>=0)*np.exp(ln_Reta*(Reta_w_pow))



        Sl2 = (((D_mu*dr_mu)/sigma_t_D_2) * 2*n_t_f*ns2_m_1)

        if eta_w==1: 
            return (nu_w[kl2]**2)*n_t_f[kl2]*Sl2[kl2]
        else:
            return np.sum(Reta_w_t[kl2:] * (nu_w[kl2:]**2) * n_t_f[kl2:] * Sl2[kl2:])





@njit(nb.float64(
                        nb.int64, nb.int64, nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64,
                    ))
def calc_Hess_D_B_Sigma_kl(k, l, T, 
                        beta, sigma_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=1.0):





        Sl1 = calc_Hess_D_B_Sigma_Sl1_fast_kl(k, l, T, sigma_f, eta_w, n_t_f, nu_w, D_mu, sigma_t_D_2, dr_mu)
        Sl2 = calc_Hess_D_B_Sigma_Sl2_fast_kl(k, l, T, eta_w, n_t_f, nu_w, D_mu, sigma_t_D_2, ns2_m_1, dr_mu)

        #sumSl1[:2+k] = sumSl1[2+k]
        #sumSl2[:2+k] = sumSl2[2+k]

        if k < T-1 and l < T-1:
            return - ((beta**2) * (Q_q**2)) * r_1[k]*r_1[l] * (
                                                                        (rho_prime_k[1+k]*rho_prime_k[1+l]) * Sl1
                                                                        + (rho_prime_k[1+k]+rho_prime_k[1+l]) * Sl2
                                                                    )
        else:
            return 0.0



@njit(nb.float64[:](
                        nb.int64, nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64,
                    ))
def calc_Hess_D_B_Sigma_kl_k(k, T, 
                        beta, sigma_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=1.0):


        row = np.zeros(T)

        if k < T-1:
            for l in range(T-1):
                row[l] = calc_Hess_D_B_Sigma_kl(k=k, l=l, T=T, 
                        beta=beta, sigma_f=sigma_f, n_t_f=n_t_f, Q_q=Q_q, nu_w=nu_w, D_mu=D_mu, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, ns2_m_1=ns2_m_1, dr_mu=dr_mu, r_1=r_1, 
                        eta_w=eta_w)


        return row



@njit(nb.float64[:,:](
                        nb.int64, 
                        nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64,
                    ))
def calc_Hess_D_B_Sigma_kl_nb(T, 
                        beta, sigma_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=1.0):

    
    H = np.zeros((T,T))
    for k in range(T-1):
        H[k,:] = calc_Hess_D_B_Sigma_kl_k(k=k, T=T, 
                        beta=beta, sigma_f=sigma_f, n_t_f=n_t_f, Q_q=Q_q, nu_w=nu_w, D_mu=D_mu, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, ns2_m_1=ns2_m_1, dr_mu=dr_mu, r_1=r_1, 
                        eta_w=eta_w)

    return H



@njit(nb.float64[:](nb.int64, nb.float64, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]))
def calc_Hess_D_B_Sigma_sumSl1_fast(T, sigma_f, eta_w, n_t_f, eta_l, nu_w, D_mu, sigma_t_D_2, dr_mu):


        Sl1 = np.zeros((4, T))

        Sl1[0,:] = (-2*(n_t_f**3)*(((sigma_f**2)/sigma_t_D_2)**2) + 2*n_t_f*((sigma_f**2)/sigma_t_D_2) - (1-n_t_f))
        Sl1[1,:] = ((D_mu**2)/sigma_t_D_2) * (4*(n_t_f**3)*(((sigma_f**2)/sigma_t_D_2)**2) - 2*n_t_f*(1-4*n_t_f)*((sigma_f**2)/sigma_t_D_2) + (1+n_t_f))
        Sl1[2,:] = ((dr_mu**2)/sigma_t_D_2)
        Sl1[3,:] = ((D_mu*dr_mu)/sigma_t_D_2) * (- (1-2*n_t_f))


        if eta_w==1: 

            return (nu_w**2)*n_t_f*np.sum(Sl1, axis=0)
        else:
            return np.flip(np.cumsum(np.flip(((eta_l*nu_w)**2) * n_t_f * np.sum(Sl1, axis=0))))



@njit(nb.float64[:](nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])) 
def calc_Hess_D_B_Sigma_sumSl2_fast(eta_w, n_t_f, eta_l, nu_w, D_mu, sigma_t_D_2, ns2_m_1, dr_mu):


        Sl2 = ((D_mu*dr_mu)/sigma_t_D_2) * 2*n_t_f*ns2_m_1

        if eta_w==1: 
            return (nu_w**2)*n_t_f*Sl2
        else:
            return np.flip(np.cumsum(np.flip(((eta_l*nu_w)**2) * n_t_f * Sl2)))




@njit(nb.float64[:](
                        nb.int64, nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]
                    ))
def calc_Hess_D_B_Sigma_sumSl1_fast_k(k, T, sigma_f, eta_w, n_t_f, nu_w, D_mu, sigma_t_D_2, dr_mu):


        

        Sl1 = np.zeros((4, T))

        ts = np.arange(T)
        Reta_w_pow = ts-(2+k)
        if eta_w==1:
            Reta_w_t_2 = 1.0*(Reta_w_pow==0)
        else:
            ln_Reta = np.log(1-eta_w)
            Reta_w_t_2 = (Reta_w_pow>=0)*np.exp(ln_Reta*(2*Reta_w_pow+k))

            #Reta_w_l = np.exp(-ln_Reta*ts)

        Sl1[0,:] = (-2*(n_t_f**3)*(((sigma_f**2)/sigma_t_D_2)**2) + 2*n_t_f*((sigma_f**2)/sigma_t_D_2) - (1-n_t_f))
        Sl1[1,:] = ((D_mu**2)/sigma_t_D_2) * (4*(n_t_f**3)*(((sigma_f**2)/sigma_t_D_2)**2) - 2*n_t_f*(1-4*n_t_f)*((sigma_f**2)/sigma_t_D_2) + (1+n_t_f))
        Sl1[2,:] = ((dr_mu**2)/sigma_t_D_2)
        Sl1[3,:] = ((D_mu*dr_mu)/sigma_t_D_2) * (- (1-2*n_t_f))


        if eta_w==1: 

            Sl1_ret = (nu_w**2)*n_t_f*np.sum(Sl1, axis=0)
        else:
            Sl1_ret = np.flip(np.cumsum(np.flip(Reta_w_t_2 * (nu_w**2) * n_t_f * np.sum(Sl1, axis=0))))

        Sl1_ret[:2+k] = Sl1_ret[2+k]

        return Sl1_ret



@njit(nb.float64[:](
                        nb.int64, nb.int64,
                        nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]
                    ))
def calc_Hess_D_B_Sigma_sumSl2_fast_k(k, T, eta_w, n_t_f, nu_w, D_mu, sigma_t_D_2, ns2_m_1, dr_mu):


        ts = np.arange(T)
        Reta_w_pow = ts-(2+k)
        if eta_w==1:
            Reta_w_t_2 = 1.0*(Reta_w_pow==0)
        else:
            ln_Reta = np.log(1-eta_w)
            Reta_w_t_2 = (Reta_w_pow>=0)*np.exp(ln_Reta*(2*Reta_w_pow+k))



        Sl2 = ((D_mu*dr_mu)/sigma_t_D_2) * 2*n_t_f*ns2_m_1

        if eta_w==1: 
            Sl2_ret = (nu_w**2)*n_t_f*Sl2
        else:
            Sl2_ret = np.flip(np.cumsum(np.flip(Reta_w_t_2 * (nu_w**2) * n_t_f * Sl2)))

        Sl2_ret[:2+k] = Sl2_ret[2+k]

        return Sl2_ret





@njit(nb.types.UniTuple(nb.float64[:],2)(
                        nb.int64, nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]
                    ))
def calc_Hess_D_B_Sigma_sumSl1Sl2_fast_k(k, T, sigma_f, eta_w, n_t_f, nu_w, D_mu, sigma_t_D_2, ns2_m_1, dr_mu):


        

        Sl1 = np.zeros((4, T))

        ts = np.arange(T)
        Reta_w_pow = ts-(2+k)
        if eta_w==1:
            Reta_w_t_2 = 1.0*(Reta_w_pow==0)
        else:
            ln_Reta = np.log(1-eta_w)
            Reta_w_t_2 = (Reta_w_pow>=0)*np.exp(ln_Reta*(2*Reta_w_pow+k))

            #Reta_w_l = np.exp(-ln_Reta*ts)

        Sl1[0,:] = (-2*(n_t_f**3)*(((sigma_f**2)/sigma_t_D_2)**2) + 2*n_t_f*((sigma_f**2)/sigma_t_D_2) - (1-n_t_f))
        Sl1[1,:] = ((D_mu**2)/sigma_t_D_2) * (4*(n_t_f**3)*(((sigma_f**2)/sigma_t_D_2)**2) - 2*n_t_f*(1-4*n_t_f)*((sigma_f**2)/sigma_t_D_2) + (1+n_t_f))
        Sl1[2,:] = ((dr_mu**2)/sigma_t_D_2)
        Sl1[3,:] = ((D_mu*dr_mu)/sigma_t_D_2) * (- (1-2*n_t_f))

        Sl2 = ((D_mu*dr_mu)/sigma_t_D_2) * 2*n_t_f*ns2_m_1


        if eta_w==1: 

            Sl1_ret = (nu_w**2)*n_t_f*np.sum(Sl1, axis=0)
            Sl2_ret = (nu_w**2)*n_t_f*Sl2
        else:
            Sl1_ret = np.flip(np.cumsum(np.flip(Reta_w_t_2 * (nu_w**2) * n_t_f * np.sum(Sl1, axis=0))))
            Sl2_ret = np.flip(np.cumsum(np.flip(Reta_w_t_2 * (nu_w**2) * n_t_f * Sl2)))

        Sl1_ret[:2+k] = Sl1_ret[2+k]
        Sl2_ret[:2+k] = Sl2_ret[2+k]

        return Sl1_ret, Sl2_ret





@njit(nb.float64[:](
                        nb.int64, nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64,
                    ))
def calc_Hess_D_B_Sigma_k(k, T, 
                        beta, sigma_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=1.0):


        row = np.zeros(T)
        ts = np.arange(T)


        sumSl1Sl2 = calc_Hess_D_B_Sigma_sumSl1Sl2_fast_k(k, T, sigma_f, eta_w, n_t_f, nu_w, D_mu, sigma_t_D_2, ns2_m_1, dr_mu)

        #sumSl1[:2+k] = sumSl1[2+k]
        #sumSl2[:2+k] = sumSl2[2+k]

        if k < T-1:
            row[:-2] = - ((beta**2) * (Q_q**2)) * r_1[k]*r_1[:-2] * (
                                                                        (rho_prime_k[1+k]*rho_prime_k[1:-1]) * sumSl1Sl2[0][2:]
                                                                        + (rho_prime_k[1+k]+rho_prime_k[1:-1]) * sumSl1Sl2[1][2:]
                                                                    )



        #if k < T-1:
        #    row[:-2] = - ((beta**2) * (Q_q**2)) * r_1[k]*r_1[:-2] * (
        #                                                                (rho_prime_k[1+k]*rho_prime_k[1:-1]) * (sumSl1[2:])[klmax[:-2]]
        #                                                                + (rho_prime_k[1+k]+rho_prime_k[1:-1]) * (sumSl2[2:])[klmax[:-2]]
        #                                                            )


        return row



@njit(nb.float64[:,:](
                        nb.int64, 
                        nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64,
                    ))
def calc_Hess_D_B_Sigma_nb_old(T, 
                        beta, sigma_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, ns2_m_1, dr_mu, r_1, 
                        eta_w=1.0):

    
    H = np.zeros((T,T))
    for k in range(T):
        H[k,:] = calc_Hess_D_B_Sigma_k(k=k, T=T, 
                        beta=beta, sigma_f=sigma_f, n_t_f=n_t_f, Q_q=Q_q, nu_w=nu_w, D_mu=D_mu, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, ns2_m_1=ns2_m_1, dr_mu=dr_mu, r_1=r_1, 
                        eta_w=eta_w)

    return H


@njit(nb.types.UniTuple(nb.float64[:],2)(
                        nb.int64, nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:]
                    ))
def calc_Hess_D_B_Sigma_sumSl1Sl2_alt_k(k, T, sigma_f, eta_w, n_t_f, nu_w, D_mu, sigma_t_D_2, ns2_m_1, dr_mu, Reta_w_lt_0):




        Sl1 = np.zeros((4, T))
        Sl1[0,:] = (-2*(n_t_f**3)*(((sigma_f**2)/sigma_t_D_2)**2) + 2*n_t_f*((sigma_f**2)/sigma_t_D_2) - (1-n_t_f))
        Sl1[1,:] = ((D_mu**2)/sigma_t_D_2) * (4*(n_t_f**3)*(((sigma_f**2)/sigma_t_D_2)**2) - 2*n_t_f*(1-4*n_t_f)*((sigma_f**2)/sigma_t_D_2) + (1+n_t_f))
        Sl1[2,:] = ((dr_mu**2)/sigma_t_D_2)
        Sl1[3,:] = ((D_mu*dr_mu)/sigma_t_D_2) * (- (1-2*n_t_f))

        Sl2 = ((D_mu*dr_mu)/sigma_t_D_2) * 2*n_t_f*ns2_m_1


        Sl1_ret = np.zeros(T)
        Sl2_ret = np.zeros(T)


        if eta_w==1: 

            Sl1_ret[k] = ((nu_w[2+k]**2)*n_t_f[2+k]*np.sum(Sl1[:,2+k]))
            Sl2_ret[k] = (nu_w[2+k]**2)*n_t_f[2+k]*Sl2[2+k]
        else:

            ts = np.arange(T)

            ln_Reta = np.log(1-eta_w)
            Reta_w_subk =  np.exp(ln_Reta*(1+ts[:k]))


            if k>0:
                Reta_w_lt_0 = np.ascontiguousarray(Reta_w_lt_0[:-k,:-k])
            else:
                Reta_w_lt_0 = np.ascontiguousarray(Reta_w_lt_0)



            
            #lnReta_w_lt_0 = 1.0*np.add.outer(-ts[:-2]*ln_Reta, 2*ts[:-2]*ln_Reta)
            #Reta_w_lt_0 = np.triu(np.exp(lnReta_w_lt_0))

            #Reta_w_lt_0 = np.triu(np.exp(np.add.outer(-ts[:-2]*ln_Reta, 2*ts[:-2]*ln_Reta)))

            

            Sl1_ret[k:-2] = Reta_w_lt_0 @ ((nu_w**2) * n_t_f * np.sum(Sl1, axis=0))[2+k:]
            Sl2_ret[k:-2] = Reta_w_lt_0 @ ((nu_w**2) * n_t_f * Sl2)[2+k:]

            Sl1_ret[:k] = Sl1_ret[k] * Reta_w_subk
            Sl2_ret[:k] = Sl2_ret[k] * Reta_w_subk

        return Sl1_ret, Sl2_ret


@njit(nb.float64[:](
                        nb.int64, nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], 
                        nb.float64,
                    ))
def calc_Hess_D_B_Sigma_k_alt(k, T, 
                        beta, sigma_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, ns2_m_1, dr_mu, Reta_w_lt_0, r_1, 
                        eta_w=1.0):


        row = np.zeros(T)
        ts = np.arange(T)


        sumSl1Sl2 = calc_Hess_D_B_Sigma_sumSl1Sl2_alt_k(k, T, sigma_f, eta_w, n_t_f, nu_w, D_mu, sigma_t_D_2, ns2_m_1, dr_mu, Reta_w_lt_0)

        #sumSl1[:2+k] = sumSl1[2+k]
        #sumSl2[:2+k] = sumSl2[2+k]

        if k < T-1:
            row[:-2] = - ((beta**2) * (Q_q**2)) * r_1[k]*r_1[:-2] * (
                                                                        (rho_prime_k[1+k]*rho_prime_k[1:-1]) * sumSl1Sl2[0][2:]
                                                                        + (rho_prime_k[1+k]+rho_prime_k[1:-1]) * sumSl1Sl2[1][2:]
                                                                    )



        #if k < T-1:
        #    row[:-2] = - ((beta**2) * (Q_q**2)) * r_1[k]*r_1[:-2] * (
        #                                                                (rho_prime_k[1+k]*rho_prime_k[1:-1]) * (sumSl1[2:])[klmax[:-2]]
        #                                                                + (rho_prime_k[1+k]+rho_prime_k[1:-1]) * (sumSl2[2:])[klmax[:-2]]
        #                                                            )


        return row



@njit(nb.float64[:,:](
                        nb.int64, 
                        nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], 
                        nb.float64,
                    ))
def calc_Hess_D_B_Sigma_nb_alt_k(T, 
                        beta, sigma_f, n_t_f, Q_q, nu_w, D_mu, sigma_t_D_2, rho_prime_k, ns2_m_1, dr_mu, Reta_w_lt_0, r_1, 
                        eta_w=1.0):

    
    H = np.zeros((T,T))
    for k in range(T):
        H[k,:] = calc_Hess_D_B_Sigma_k_alt(k=k, T=T, 
                        beta=beta, sigma_f=sigma_f, n_t_f=n_t_f, Q_q=Q_q, nu_w=nu_w, D_mu=D_mu, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, ns2_m_1=ns2_m_1, dr_mu=dr_mu, Reta_w_lt_0=Reta_w_lt_0, r_1=r_1, 
                        eta_w=eta_w)

    return H








@njit([nb.types.UniTuple(nb.float64[:,:],2)(
                        nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64
                    ),
        nb.types.UniTuple(nb.float64[:,:],2)(
                        nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]
                    )]
                    )
def calc_Hess_D_B_Sigma_nb_sumSl1Sl2_factor(T, sigma_f, eta_w, n_t_f, nu_w, D_mu, sigma_t_D_2, ns2_m_1, dr_mu, factor=1.0):

        ts = np.arange(T)

        Reta_w = 1 - eta_w

        ln_Reta_w_l_k = np.zeros(2*(T-2))
        ln_Reta_w_l_k[:(T-2):] = np.exp(np.log(Reta_w)*np.flip(ts[:-2]))
        


        Sl1Sl2 = calc_Hess_D_B_Sigma_nb_Sl1Sl2_t_nu_w_n_t_f_factor(sigma_f=sigma_f, n_t_f=n_t_f, D_mu=D_mu, sigma_t_D_2=sigma_t_D_2, ns2_m_1=ns2_m_1, dr_mu=dr_mu, nu_w=nu_w, factor=factor)


        Sl1_ret = np.zeros((T,T))
        Sl2_ret = np.zeros((T,T))

        for k in range(T-2):
            k_prime = T-(3+k)
            Sl1_ret[k_prime, :T-2] = ln_Reta_w_l_k[k:k+(T-2)] * Sl1Sl2[0][k_prime] + Reta_w * Sl1_ret[k_prime+1, :T-2]
            Sl2_ret[k_prime, :T-2] = ln_Reta_w_l_k[k:k+(T-2)] * Sl1Sl2[1][k_prime] + Reta_w * Sl2_ret[k_prime+1, :T-2]

        
        return Sl1_ret, Sl2_ret







@njit([nb.float64[:,:](
                        nb.float64[:,:], nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64
                    ),
        nb.float64[:,:](
                        nb.float64[:,:], nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]
                    )]
                    )
def append_Hess_D_B_Sigma_nb_factor_alt(Hess, T, sigma_f, eta_w, n_t_f, nu_w, D_mu, sigma_t_D_2, rho_prime_k, ns2_m_1, dr_mu, r_1, factor=1.0):

        ts = np.arange(T)

        Reta_w = 1 - eta_w

        ln_Reta_w_l_k = np.zeros(2*(T-2))

        if eta_w==1:
            ln_Reta_w_l_k[T-3] = 1
        else:
            ln_Reta_w_l_k[:(T-2)] = np.exp(np.log(Reta_w)*np.flip(ts[:-2]))
        


        Sl1Sl2 = calc_Hess_D_B_Sigma_nb_Sl1Sl2_t_nu_w_n_t_f_factor(sigma_f=sigma_f, n_t_f=n_t_f, D_mu=D_mu, sigma_t_D_2=sigma_t_D_2, ns2_m_1=ns2_m_1, dr_mu=dr_mu, nu_w=nu_w, factor=factor)


        Sl1_row = np.zeros(T-2)
        Sl2_row = np.zeros(T-2)

        for k in range(T-2):
            k_prime = T-(3+k)
            Sl1_row = ln_Reta_w_l_k[k:k+(T-2)] * Sl1Sl2[0][2+k_prime] + Reta_w * Sl1_row
            Sl2_row = ln_Reta_w_l_k[k:k+(T-2)] * Sl1Sl2[1][2+k_prime] + Reta_w * Sl2_row

            Hess[k_prime, :T-2] += (r_1[k_prime]*r_1[:-2])*((rho_prime_k[1+k_prime]*rho_prime_k[1:-1])*Sl1_row + (rho_prime_k[1+k_prime]+rho_prime_k[1:-1])*Sl2_row)

        
        return Hess

@njit([nb.float64[:,:](
                        nb.float64[:,:], 
                        nb.int64,
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64, nb.float64
                    ),
        nb.float64[:,:](
                        nb.float64[:,:], 
                        nb.int64,
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64, nb.float64[:]
                    )]
                    )
def append_Hess_D_B_Sigma_nb_factor(Hess, 
                                     T, 
                                     sigma_f, 
                                     n_t_f, 
                                     sigma_t_D_2, rho_prime_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                                     eta_w=1.0, factor=1.0):

        

        Sl1Sl2 = calc_Hess_D_B_Sigma_nb_Sl1Sl2_t_nu_w_n_t_f_factor(sigma_f=sigma_f, n_t_f=n_t_f, D_mu=D_mu, sigma_t_D_2=sigma_t_D_2, ns2_m_1=ns2_m_1, dr_mu=dr_mu, nu_w=nu_w, factor=factor)


        Sl1_row = np.zeros(T-2)
        Sl2_row = np.zeros(T-2)

        if eta_w==1:

            diag = (r_1[:-2]**2)*((rho_prime_k[1:-1]**2)*Sl1Sl2[0][2:] + (2*rho_prime_k[1:-1])*Sl1Sl2[1][2:])
            for k in range(T-2):
                Hess[k, k] += diag[k]

        else:
            ts = np.arange(T)
            Reta_w = 1 - eta_w
            ln_Reta_w_l_k = np.zeros(2*(T-2))
            ln_Reta_w_l_k[:(T-2)] = np.exp(np.log(Reta_w)*np.flip(ts[:-2]))

            for k in range(T-2):
                k_prime = T-(3+k)
                Sl1_row = ln_Reta_w_l_k[k:k+(T-2)] * Sl1Sl2[0][2+k_prime] + Reta_w * Sl1_row
                Sl2_row = ln_Reta_w_l_k[k:k+(T-2)] * Sl1Sl2[1][2+k_prime] + Reta_w * Sl2_row

                Hess[k_prime, :T-2] += (r_1[k_prime]*r_1[:-2])*((rho_prime_k[1+k_prime]*rho_prime_k[1:-1])*Sl1_row + (rho_prime_k[1+k_prime]+rho_prime_k[1:-1])*Sl2_row)

        
        return Hess


@njit([nb.float64[:,:](
                        nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64
                    ),
        nb.float64[:,:](
                        nb.int64,
                        nb.float64, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]
                    )]
                    )
def calc_Hess_D_B_Sigma_nb_factor_alt(T, sigma_f, eta_w, n_t_f, nu_w, D_mu, sigma_t_D_2, rho_prime_k, ns2_m_1, dr_mu, r_1, factor=1.0):

        return append_Hess_D_B_Sigma_nb_factor_alt(np.zeros((T,T)), T, sigma_f, eta_w, n_t_f, nu_w, D_mu, sigma_t_D_2, rho_prime_k, ns2_m_1, dr_mu, r_1, factor)

        


@njit(nb.float64[:,:](
                        nb.float64[:,:], 
                        nb.int64,
                        nb.float64, 
                        nb.float64, 
                        nb.float64,
                        nb.float64[:],
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                        nb.float64, 
                    ))
def append_Hess_D_B_Sigma_nb_alt(Hess, 
                           T, 
                            sigma_f, 
                            beta, 
                            Q_q, 
                            n_t_f, 
                            sigma_t_D_2, rho_prime_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                            eta_w=1.0):

    factor = -(beta*Q_q)**(2)

    if factor==0:
        return Hess
    else:
        return append_Hess_D_B_Sigma_nb_factor_alt(Hess, T, sigma_f, eta_w, n_t_f, nu_w, D_mu, sigma_t_D_2, rho_prime_k, ns2_m_1, dr_mu, r_1, factor)


@njit(nb.float64[:,:](
                        nb.int64,
                        nb.float64, 
                        nb.float64, 
                        nb.float64,
                        nb.float64[:],
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                        nb.float64, 
                    ))
def calc_Hess_D_B_Sigma_nb_alt(T, 
                            sigma_f, 
                            beta, 
                            Q_q, 
                            n_t_f, 
                            sigma_t_D_2, rho_prime_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                            eta_w=1.0):

    factor = -(beta*Q_q)**(2)

    return append_Hess_D_B_Sigma_nb_factor_alt(Hess=np.zeros((T,T)), 
                                        T=T, 
                                        sigma_f=sigma_f, 
                                        n_t_f=n_t_f, 
                                        sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1, 
                                        eta_w=eta_w, factor=factor)





@njit(nb.float64[:,:](
                        nb.float64[:,:], 
                        nb.int64,
                        nb.float64, 
                        nb.float64, 
                        nb.float64,
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64,
                    ))
def append_Hess_D_B_Sigma_nb(Hess, 
                                T, 
                                sigma_f, 
                                beta, 
                                Q_q, 
                                n_t_f, 
                                sigma_t_D_2, rho_prime_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                                eta_w=1.0):

    factor = -(beta*Q_q)**(2)
    if factor==0:
        return Hess
    else:
        return append_Hess_D_B_Sigma_nb_factor(Hess, 
                                            T=T, 
                                                sigma_f=sigma_f, 
                                                n_t_f=n_t_f, 
                                                sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1, 
                                                eta_w=eta_w, factor=factor)


@njit(nb.float64[:,:](
                        nb.int64,
                        nb.float64, 
                        nb.float64, 
                        nb.float64,
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64,
                    ))
def calc_Hess_D_B_Sigma_nb(
                            T, 
                                sigma_f, 
                                beta, 
                                Q_q, 
                                n_t_f, 
                                sigma_t_D_2, rho_prime_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                                eta_w=1.0,
                            ):

    factor = -(beta*Q_q)**(2)
    return append_Hess_D_B_Sigma_nb_factor(Hess=np.zeros((T,T)), 
                                            T=T, 
                                            sigma_f=sigma_f, 
                                            n_t_f=n_t_f, 
                                            sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1, 
                                            eta_w=eta_w, factor=factor)


def calc_Hess_D_B_Sigma_nb_np_factor(T, 
                        sigma_f, 
                        n_t_f, 
                        sigma_t_D_2, rho_prime_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                        eta_w=1.0, factor=1.0):


    sumSl1Sl2 = calc_Hess_D_B_Sigma_nb_sumSl1Sl2_factor(T=T, sigma_f=sigma_f, eta_w=eta_w, n_t_f=n_t_f, nu_w=nu_w, D_mu=D_mu, sigma_t_D_2=sigma_t_D_2, ns2_m_1=ns2_m_1, dr_mu=dr_mu, factor=factor)

    S = np.zeros((T,T))

    S[:-2,:-2] = np.outer(r_1[:-2],  r_1[:-2]) * (
                                                        np.multiply.outer(rho_prime_k[1:-1], rho_prime_k[1:-1]) * sumSl1Sl2[0][:-2,:-2]
                                                        + np.add.outer(rho_prime_k[1:-1], rho_prime_k[1:-1]) * sumSl1Sl2[1][:-2,:-2]
                                                    )
    
    
    return S




@njit(nb.float64[:,:](
                        nb.float64[:,:], 
                        nb.int64,
                        nb.float64, nb.float64, 
                        nb.float64, 
                        nb.float64,
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64,
                    ))
def append_Hess_D_B_nb(Hess,  
                        T, 
                        delta_f, sigma_f, 
                        beta, 
                        Q_q, 
                        n_t_f, 
                        sigma_t_D_2, rho_prime_k, rho_prime2_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                        eta_w=1.0):

        

        Hess = append_Hess_D_B_K_nb(Hess=np.zeros((T,T)), 
                                T=T, 
                                beta=beta, 
                                Q_q=Q_q, 
                                n_t_f=n_t_f, 
                                sigma_t_D_2=sigma_t_D_2, rho_prime2_k=rho_prime2_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1, 
                                eta_w=eta_w)
        
        Hess = append_Hess_D_B_L_nb(Hess=Hess,
                                    T=T, 
                                    delta_f=delta_f, 
                                    beta=beta, 
                                    Q_q=Q_q, 
                                    n_t_f=n_t_f, 
                                    sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1, 
                                    eta_w=eta_w)
        Hess = append_Hess_D_B_Sigma_nb(Hess=Hess,
                                        T=T, 
                                        sigma_f=sigma_f, 
                                        beta=beta, 
                                        Q_q=Q_q, 
                                        n_t_f=n_t_f, 
                                        sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1, 
                                    eta_w=eta_w)

        return Hess



@njit(nb.float64[:,:](
                        nb.int64,
                        nb.float64, nb.float64, 
                        nb.float64, 
                        nb.float64,
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64,
                    ))
def calc_Hess_D_B_nb( 
                        T, 
                        delta_f, sigma_f, 
                        beta, 
                        Q_q, 
                        n_t_f, 
                        sigma_t_D_2, rho_prime_k, rho_prime2_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                        eta_w=1.0):
    
    return append_Hess_D_B_nb(Hess=np.zeros((T,T)),  
                        T=T, 
                        delta_f=delta_f, sigma_f=sigma_f, 
                        beta=beta, 
                        Q_q=Q_q, 
                        n_t_f=n_t_f, 
                        sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1, 
                        eta_w=eta_w)





@njit(nb.float64[:,:](
                        nb.float64[:,:], 
                        nb.int64,
                        nb.float64, nb.float64, nb.float64, nb.float64, 
                        nb.float64, 
                        nb.float64,
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64, nb.float64
                    ))
def append_Hess_nb(Hess,  
                        T, 
                        psi, sigma_F, delta_f, sigma_f, 
                        beta, 
                        Q_q, 
                        n_t_f, 
                        sigma_t_D_2, rho_prime_k, rho_prime2_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                        eta_w=1.0, epsilon=0.0):

        Hess = append_Hess_F_D_K_nb(Hess=Hess,
                                    T=T,  
                                    psi=psi, sigma_F=sigma_F, delta_f=delta_f, 
                                    beta=beta,
                                    Q_q=Q_q, 
                                    n_t_f=n_t_f, 
                                    sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1, 
                                    eta_w=eta_w, epsilon=epsilon)
        if beta!=0:
            Hess = append_Hess_D_B_L_nb(Hess=Hess,
                                        T=T, 
                                        delta_f=delta_f, 
                                        beta=beta, 
                                        Q_q=Q_q, 
                                        n_t_f=n_t_f, 
                                        sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1, 
                                        eta_w=eta_w)
            Hess = append_Hess_D_B_Sigma_nb(Hess=Hess,
                                            T=T, 
                                            sigma_f=sigma_f, 
                                            beta=beta, 
                                            Q_q=Q_q, 
                                            n_t_f=n_t_f, 
                                            sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1, 
                                        eta_w=eta_w)

        return Hess


   
@njit(nb.float64[:,:](
                        nb.int64,
                        nb.float64, nb.float64, nb.float64, nb.float64, 
                        nb.float64, 
                        nb.float64,
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.float64, nb.float64
                    ))
def calc_Hess_nb(T, 
                    psi, sigma_F, delta_f, sigma_f, 
                    beta, 
                    Q_q, 
                    n_t_f, 
                    sigma_t_D_2, rho_prime_k, rho_prime2_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                    eta_w=1.0, epsilon=0.0):

    return append_Hess_nb(np.zeros((T,T)), 
                            T, 
                            psi, sigma_F, delta_f, sigma_f, 
                            beta, 
                            Q_q, 
                            n_t_f, 
                            sigma_t_D_2, rho_prime_k, rho_prime2_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1, 
                            eta_w=eta_w, epsilon=epsilon)


@njit(nb.float64[:](
                        nb.float64[:], 
                        nb.int64,
                        nb.float64[:],
                        nb.float64, nb.float64, 
                    ))
def append_Jac_F_nb(Jac, 
                        T, 
                        p_F,
                        psi, sigma_F):

    inv_sigma_F_2 = (psi*sigma_F)**(-2)
        
    Jac[:-1] = -(2*p_F[1:-1] - p_F[:-2] - p_F[2:]) * inv_sigma_F_2
    Jac[-1] = -(p_F[-1] - p_F[-2]) * inv_sigma_F_2


    return Jac


@njit(nb.float64[:](
                        nb.int64,
                        nb.float64[:],
                        nb.float64, nb.float64,
                    ))
def calc_Jac_F_nb(T, 
                        p_F,
                        psi, sigma_F):

    return append_Jac_F_nb(np.zeros(T), T, p_F, psi, sigma_F)


@njit(nb.float64[:](
                        nb.float64[:],
                        nb.int64,
                        nb.float64,
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))
def append_Jac_D_N_nb(Jac, 
                        T, 
                        delta_f, 
                        n_t_f,
                        sigma_t_D_2, rho_prime_k, D_mu):

    
    Jac[:-1] += (delta_f*D_mu*n_t_f*rho_prime_k/sigma_t_D_2)[1:]

    return Jac

@njit(nb.float64[:](
                        nb.int64,
                        nb.float64,
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))
def calc_Jac_D_N_nb( 
                        T, 
                        delta_f, 
                        n_t_f,
                        sigma_t_D_2, rho_prime_k, D_mu):

    return append_Jac_D_N_nb(Jac=np.zeros(T), 
                        T=T, 
                        delta_f=delta_f, 
                        n_t_f=n_t_f,
                        sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu)


@njit(nb.float64[:](
                        nb.float64[:], 
                        nb.int64,
                        nb.float64[:],
                        nb.float64, nb.float64, nb.float64,
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))
def append_Jac_F_D_N_nb(Jac, 
                        T, 
                        p_F, 
                        psi, sigma_F, delta_f, 
                        n_t_f,
                        sigma_t_D_2, rho_prime_k, D_mu):

    Jac = append_Jac_F_nb(Jac=Jac, T=T, p_F=p_F, psi=psi, sigma_F=sigma_F)
    Jac = append_Jac_D_N_nb(Jac=Jac, 
                                    T=T, 
                                    delta_f=delta_f, 
                                    n_t_f=n_t_f,
                                    sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu)

    return Jac


@njit(nb.float64[:](
                        nb.int64,
                        nb.float64[:],
                        nb.float64, nb.float64, nb.float64,
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))
def calc_Jac_F_D_N_nb( 
                        T, 
                        p_F, 
                        psi, sigma_F, delta_f, 
                        n_t_f,
                        sigma_t_D_2, rho_prime_k, D_mu):

    return append_Jac_F_D_N_nb(np.zeros(T), 
                                    T=T, 
                                    p_F=p_F,
                                    psi=psi, sigma_F=sigma_F, delta_f=delta_f, 
                                    n_t_f=n_t_f,
                                    sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu)



@njit(nb.float64[:](
                        nb.float64[:],
                        nb.int64,
                        nb.float64, nb.float64, 
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))
def append_Jac_D_B_nb(Jac, 
                        T, 
                        beta, eta_w, 
                        Q_q,
                        n_t_f,
                        sigma_t_D_2, rho_prime_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1):

    if beta!=0:
        k_factors = beta * rho_prime_k[1:-1] * Q_q * r_1[:-2]
        e_factors = (nu_w*n_t_f*((D_mu*dr_mu/sigma_t_D_2) + ns2_m_1*(((D_mu**2)/sigma_t_D_2)-1)))[2:]

        e_factor = 0.0

        Reta_w = 1 - eta_w

        for k in range(T-2):
            k_prime = T-(3+k)
            e_factor = e_factors[k_prime] + Reta_w * e_factor
            Jac[k_prime] += k_factors[k_prime]*e_factor

    return Jac

@njit(nb.float64[:](
                        nb.int64,
                        nb.float64, nb.float64, 
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))
def calc_Jac_D_B_nb(
                        T, 
                        beta, eta_w, 
                        Q_q,
                        n_t_f,
                        sigma_t_D_2, rho_prime_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1):
    
    return append_Jac_D_B_nb(Jac=np.zeros(T), 
                        T=T, 
                        beta=beta, eta_w=eta_w, 
                        Q_q=Q_q,
                        n_t_f=n_t_f,
                        sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1)


@njit(nb.float64[:](
                        nb.float64[:],
                        nb.int64,
                        nb.float64,
                        nb.float64, nb.float64, 
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))
def append_Jac_D_nb(Jac, 
                        T, 
                        delta_f,
                        beta, eta_w, 
                        Q_q,
                        n_t_f,
                        sigma_t_D_2, rho_prime_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1):

    Jac = append_Jac_D_N_nb(Jac=Jac, 
                        T=T, 
                        delta_f=delta_f, 
                        n_t_f=n_t_f,
                        sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu)
    
    Jac = append_Jac_D_B_nb(Jac, 
                        T=T, 
                        beta=beta, eta_w=eta_w, 
                        Q_q=Q_q,
                        n_t_f=n_t_f,
                        sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1)

    return Jac


@njit(nb.float64[:](
                        nb.int64,
                        nb.float64,
                        nb.float64, nb.float64, 
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))
def calc_Jac_D_nb(T, 
                        delta_f, 
                        beta, eta_w, 
                        Q_q,
                        n_t_f,
                        sigma_t_D_2, rho_prime_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1):

    
    return append_Jac_D_nb(np.zeros(T), 
                                T, 
                                delta_f, 
                                beta, eta_w, 
                                Q_q,
                                n_t_f,
                                sigma_t_D_2, rho_prime_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1)




@njit(nb.float64[:](
                        nb.float64[:], 
                        nb.int64,
                        nb.float64[:],
                        nb.float64, nb.float64, nb.float64,
                        nb.float64, nb.float64, 
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))
def append_Jac_nb(Jac, 
                        T, 
                        p_F, 
                        psi, sigma_F, delta_f, 
                        beta, eta_w, 
                        Q_q,
                        n_t_f,
                        sigma_t_D_2, rho_prime_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1):

    Jac = append_Jac_F_D_N_nb(np.zeros(T), 
                                    T=T, 
                                    p_F=p_F,
                                    psi=psi, sigma_F=sigma_F, delta_f=delta_f, 
                                    n_t_f=n_t_f,
                                    sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu)
    
    Jac = append_Jac_D_B_nb(Jac, 
                        T=T, 
                        beta=beta, eta_w=eta_w, 
                        Q_q=Q_q,
                        n_t_f=n_t_f,
                        sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1)

    return Jac

@njit(nb.float64[:](
                        nb.int64, 
                        nb.float64[:],
                        nb.float64, nb.float64, nb.float64,
                        nb.float64, nb.float64, 
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))
def calc_Jac_nb(T, 
                    p_F, 
                    psi, sigma_F, delta_f, 
                    beta, eta_w, 
                    Q_q,
                    n_t_f,
                    sigma_t_D_2, rho_prime_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
                    ):

    return append_Jac_nb(np.zeros(T), 
                                    T=T, 
                                    p_F=p_F,
                                    psi=psi, sigma_F=sigma_F, delta_f=delta_f, 
                                    beta=beta, eta_w=eta_w, 
                                    Q_q=Q_q,
                                    n_t_f=n_t_f,
                                    sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1)



@njit(nb.float64[:](
                        nb.float64[:], 
                        nb.int64,
                        nb.float64[:],
                        nb.float64, nb.float64, nb.float64,
                        nb.float64, nb.float64, 
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))
def append_Jac_nb_old(Jac, 
                        T, 
                        p_F, 
                        psi, sigma_F, delta_f, 
                        beta, eta_w, 
                        Q_q,
                        n_t_f,
                        sigma_t_D_2, rho_prime_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1):

    Jac = append_Jac_F_nb(Jac=Jac, T=T, p_F=p_F, psi=psi, sigma_F=sigma_F)
    Jac = append_Jac_D_nb(Jac=Jac, 
                                    T=T, 
                                    delta_f=delta_f, 
                                    beta=beta, eta_w=eta_w, 
                                    Q_q=Q_q,
                                    n_t_f=n_t_f,
                                    sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1)

    return Jac

@njit(nb.float64[:](
                        nb.int64, 
                        nb.float64[:],
                        nb.float64, nb.float64, nb.float64,
                        nb.float64, nb.float64, 
                        nb.float64, 
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))
def calc_Jac_nb_old(T, 
                    p_F, 
                    psi, sigma_F, delta_f, 
                    beta, eta_w, 
                    Q_q,
                    n_t_f,
                    sigma_t_D_2, rho_prime_k, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
                    ):

    return append_Jac_nb_old(np.zeros(T), 
                                    T=T, 
                                    p_F=p_F,
                                    psi=psi, sigma_F=sigma_F, delta_f=delta_f, 
                                    beta=beta, eta_w=eta_w, 
                                    Q_q=Q_q,
                                    n_t_f=n_t_f,
                                    sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1)






@njit(nb.float64[:](
                        nb.bool_[:],
                        nb.float64[:], 
                        nb.float64[:,:], nb.float64[:,:],
                        nb.int64,
                    ))
def calc_mu_t_D(N_s_a_bool, delta_s, rho_t_s, n_t_s, T_0=0): # A_n mod

    return delta_s[N_s_a_bool] @ (rho_t_s[N_s_a_bool,T_0:] * n_t_s[N_s_a_bool,T_0:])

@njit(nb.float64[:](
                        nb.bool_[:],
                        nb.float64[:], 
                        nb.float64[:,:],
                        nb.int64,
                    ))
def calc_sigma_t_D_2(N_s_a_bool, sigma_s, n_t_s, T_0=0): # A_n mod


    return (sigma_s[N_s_a_bool]**2) @ (n_t_s[N_s_a_bool,T_0:]**2)


@njit(nb.float64[:](
                        nb.bool_[:],
                        nb.float64[:], 
                        nb.float64[:,:],
                        nb.int64,
                    ))
def calc_sigma_t_D(N_s_a_bool, sigma_s, n_t_s, T_0=0): # A_n mod

    return np.sqrt(calc_sigma_t_D_2(N_s_a_bool=N_s_a_bool, sigma_s=sigma_s, n_t_s=n_t_s, T_0=T_0))

@njit(nb.float64[:](
                        nb.int64,
                        nb.float64, nb.float64, nb.float64, 
                        nb.float64[:,:], nb.float64[:,:],
                        nb.int64, nb.int64,
                    ))
def calc_rho_prime_k(T, 
                     psi, gamma_1_f, gamma_2_f, 
                     M_t_s, rho_t_s, 
                     rho_type_int_f=1, T_0=0):

    


    if psi==0:
        rho_prime_k = np.zeros(T)
    else:
        inv_psi = psi**(-1)
        if rho_type_int_f==1:
            rho_prime_k =  np.repeat(inv_psi, T)
        elif rho_type_int_f==2:
            rho_prime_k = inv_psi*(1 + 3*gamma_1_f*(M_t_s[0,T_0:]**2))
        elif rho_type_int_f==3:
            rho_prime_k = (inv_psi*gamma_1_f)*(1 - (rho_t_s[0,T_0:]**2))
        elif rho_type_int_f==4:
            rho_prime_k = (-inv_psi*gamma_1_f/gamma_2_f)*(1 - (rho_t_s[0,T_0:]**2))/(2*np.sqrt(M_t_s[0,T_0:]))
        else:
            rho_prime_k = None

    return rho_prime_k

@njit(nb.float64[:](
                        nb.int64,
                        nb.float64, nb.float64, nb.float64, 
                        nb.float64[:,:], nb.float64[:,:],
                        nb.int64, nb.int64,
                    ))
def calc_rho_prime2_k(T, 
                      psi, gamma_1_f, gamma_2_f, 
                      M_t_s, rho_t_s, 
                      rho_type_int_f=1, T_0=0):




    if psi==0:
        rho_prime2_k = np.zeros(T)
    else:
        inv_psi_2 = psi**(-2)
        if rho_type_int_f==1:
            rho_prime2_k =  np.zeros(T)
        elif rho_type_int_f==2:
            rho_prime2_k =  (inv_psi_2*6*gamma_1_f)*M_t_s[0,T_0:]
        elif rho_type_int_f==3:
            rho_prime2_k = -(inv_psi_2*2*(gamma_1_f**2))*rho_t_s[0,T_0:]*(1 - (rho_t_s[0,T_0:]**2))
        elif rho_type_int_f==4:
            rho_prime2_k = (-inv_psi_2*((gamma_1_f/gamma_2_f)**2)/8) * (M_t_s[0,T_0:]**(-3/2)) * (1 + 4*np.sqrt(M_t_s[0,T_0:])*rho_t_s[0,T_0:]) * (1 - (rho_t_s[0,T_0:]**2)) 


    return rho_prime2_k




@njit(nb.float64[:](
                        nb.int64,
                        nb.float64[:],
                        nb.float64[:],
                        nb.float64[:],
                        nb.int64
                    ))
def calc_ns2_m_1(T, 
                    sigma_s, 
                    n_t_f, 
                    sigma_t_D_2,
                    T_0=0):
     
    if np.all(sigma_s==0):
        ns2_m_1 = np.repeat(-1.0, T)
    else:
        ns2_m_1 = n_t_f[T_0:]*((sigma_s[0]**2)/sigma_t_D_2) - 1.0

    return ns2_m_1




@njit(nb.float64[:](
                        nb.int64,
                        nb.float64, nb.float64[:],
                        nb.int64,
                    ))
def calc_nu_w(T, 
                    lambda_w_f, w_t_f, 
                    T_0=0):
     
    nu_w = np.zeros(T)
    nu_w[1:] = np.exp(lambda_w_f*(w_t_f[T_0:-1] < 0))

    return nu_w


@njit(nb.float64[:](
                        nb.int64,
                        nb.float64[:],
                        nb.int64,
                    ))
def calc_r_1(T, 
                    r,
                    T_0=0):
     
    r_1 = np.zeros(T)
    r_1[:-1] = r[T_0:-1]

    return r_1




@njit(nb.types.Tuple((
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]
                    ))(
                        nb.int64, 
                        nb.float64[:], nb.float64[:], 
                        nb.bool_[:],
                        nb.float64, nb.float64, nb.float64[:], nb.float64[:], nb.float64, nb.float64, nb.float64[:], 
                        nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                        nb.int64, nb.int64
                    ))
def calc_Hcomp_p_p_F(T, 
                        r, r_F,
                        N_s_a_bool, 
                        psi, mu_F, delta_s, sigma_s, gamma_1_f, gamma_2_f, lambda_w_s, 
                        M_t_s, rho_t_s, n_t_s, w_t_s, 
                        mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
                        rho_type_int_f=1, T_0=0):




    mu_t_D = calc_mu_t_D(N_s_a_bool=N_s_a_bool, delta_s=delta_s, rho_t_s=rho_t_s, n_t_s=n_t_s, T_0=T_0)     # A_n mod
    sigma_t_D_2 = calc_sigma_t_D_2(N_s_a_bool=N_s_a_bool, sigma_s=sigma_s, n_t_s=n_t_s, T_0=T_0)    # A_n mod

    rho_prime_k = calc_rho_prime_k(T=T, psi=psi, gamma_1_f=gamma_1_f, gamma_2_f=gamma_2_f, M_t_s=M_t_s, rho_t_s=rho_t_s, rho_type_int_f=rho_type_int_f, T_0=T_0)
    rho_prime2_k = calc_rho_prime2_k(T=T, psi=psi, gamma_1_f=gamma_1_f, gamma_2_f=gamma_2_f, M_t_s=M_t_s, rho_t_s=rho_t_s, rho_type_int_f=rho_type_int_f, T_0=T_0)

    F_mu = (r_F[T_0:]/psi) - mu_F
    D_mu = r[T_0:]/psi - mu_t_D 
    dr_mu = N_s_a_bool[0]*delta_s[0]*rho_t_s[0,T_0:] - mu_t_D


    ns2_m_1 = calc_ns2_m_1(T, sigma_s=sigma_s, n_t_f=n_t_s[0,:], sigma_t_D_2=sigma_t_D_2, T_0=T_0)

    #if np.all(sigma_s==0):
    #    ns2_m_1 = np.repeat(-1, T)
    #else:
    #    ns2_m_1 = n_t_s[0,T_0:]*((sigma_s[0]**2)/sigma_t_D_2) - 1

    #nu_w[0] = 0.0
    #nu_w[1:] = np.exp(lambda_w_s[0]*(w_t_s[0,T_0:-1] < 0))

    
    #r_1[-1] = 0.0
    #r_1[:-1] = r[1+T_0:]

    nu_w = calc_nu_w(T=T, lambda_w_f=lambda_w_s[0], w_t_f=w_t_s[0,:], T_0=T_0)
    r_1 = calc_r_1(T=T, r=r, T_0=T_0)


    return (mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1)






@njit(nb.float64(
                        nb.float64, 
                        nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))
def calc_log_L_FD_t(sigma_F, sigma_t_D_2, D_mu, F_mu):



    sigma_F_2 = sigma_F**2


    return -(1/2)*np.sum(((F_mu**2)/sigma_F_2) + ((D_mu**2)/sigma_t_D_2) + np.log(sigma_t_D_2))


@njit(nb.float64(       
                        nb.int64,
                        nb.float64, nb.float64, 
                    ))
def calc_log_L_FD_c(T, psi, sigma_F):



    return -T * np.sum(np.array([1,1,2]) * np.log(np.array([2*np.pi, sigma_F, psi])))



@njit(nb.float64(
                        nb.int64,
                        nb.float64, nb.float64, 
                        nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))
def calc_log_L_FD(T, 
                     psi, sigma_F, 
                     sigma_t_D_2, D_mu, F_mu):

    return calc_log_L_FD_c(T=T, psi=psi, sigma_F=sigma_F) + calc_log_L_FD_t(sigma_F=sigma_F, sigma_t_D_2=sigma_t_D_2, D_mu=D_mu, F_mu=F_mu)






@njit(nb.float64(
                        nb.int64,
                        nb.float64, nb.float64, 
                        nb.float64[:], 
                    ))
def calc_log_L_F(T, 
                     psi, sigma_F, 
                     F_mu):

    return  -(1/2) * (T*np.sum(np.array([1,2,2]) * np.log(np.array([2*np.pi, psi, sigma_F]))) + np.sum(((F_mu**2)/(sigma_F**2))))


@njit(nb.float64(
                        nb.int64,
                        nb.float64, 
                        nb.float64[:], nb.float64[:],
                    ))
def calc_log_L_D(T, 
                     psi,
                     sigma_t_D_2, D_mu):

    return  -(1/2) * (T*np.sum(np.array([1,2]) * np.log(np.array([2*np.pi, psi]))) + np.sum(((D_mu**2)/sigma_t_D_2) + np.log(sigma_t_D_2)))
    



@njit(nb.float64(
                        nb.int64,
                        nb.float64, 
                        nb.float64, 
                        nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))
def calc_log_L_FD_alt(T, 
                     psi, 
                     sigma_F, 
                     sigma_t_D_2, D_mu, F_mu):

    return calc_log_L_F(T=T, psi=psi, sigma_F=sigma_F, F_mu=F_mu) + calc_log_L_D(T=T, psi=psi, sigma_t_D_2=sigma_t_D_2, D_mu=D_mu)


@njit(nb.float64(
                        nb.float64[:,:]
                    ))
def calc_log_L_H(Hess):

    return -(1/2)*np.linalg.slogdet(Hess)[1]


@njit(nb.float64[:](
                            nb.int64,
                            nb.float64[:,:],
                            nb.float64, nb.float64,
                            nb.float64[:], nb.float64[:], nb.float64[:], 
                        ))
def calc_log_L_FDH(T, 
                        Hess,
                        psi, sigma_F, 
                        sigma_t_D_2, D_mu, F_mu, 
                        ):
    
    return np.array([calc_log_L_F(T=T, psi=psi, sigma_F=sigma_F, F_mu=F_mu),
                            calc_log_L_D(T=T, psi=psi, sigma_t_D_2=sigma_t_D_2, D_mu=D_mu),
                            calc_log_L_H(Hess=Hess)
                        ])


@njit(nb.float64(
                        nb.int64,
                        nb.float64[:,:],
                        nb.float64, nb.float64,
                        nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))
def calc_log_L_LA(T, 
                        Hess,
                        psi, sigma_F, 
                        sigma_t_D_2, D_mu, F_mu, 
                        ):


    return calc_log_L_FD(T=T, 
                            psi=psi, sigma_F=sigma_F, 
                            sigma_t_D_2=sigma_t_D_2, D_mu=D_mu, F_mu=F_mu) + calc_log_L_H(Hess=Hess)


@njit(nb.types.Tuple((nb.float64, nb.float64[:]))(
                            nb.int64,
                            nb.float64[:,:],
                            nb.float64, nb.float64,
                            nb.float64[:], nb.float64[:], nb.float64[:], 
                        ))
def calc_log_L_all(T, 
                        Hess,
                        psi, sigma_F, 
                        sigma_t_D_2, D_mu, F_mu, 
                        ):

    log_L_FDH = calc_log_L_FDH(T=T, 
                        Hess=Hess,
                        psi=psi, sigma_F=sigma_F, 
                        sigma_t_D_2=sigma_t_D_2, D_mu=D_mu, F_mu=F_mu, 
                        )
    log_L_LA = np.sum(log_L_FDH)

    return log_L_LA, log_L_FDH



@njit([nb.types.Tuple((
                        nb.float64[:,:], nb.float64[:], 
                        nb.float64, nb.float64[:],
                        nb.float64[:], nb.float64[:], 
                        nb.float64[:], 
                        nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))(
                        nb.int64, 
                        nb.int64, nb.int64, nb.int64,
                        nb.float64[:,:], nb.float64[:], 
                        nb.float64, nb.float64[:],
                        nb.float64[:], nb.float64[:], 
                        nb.float64[:], nb.float64[:], 
                        nb.float64[:], 
                        nb.bool_[:], nb.bool_[:], nb.int32[:],
                        nb.float64, nb.float64, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.float64, 
                        nb.float64,  nb.float64, nb.float64[:], nb.float64, nb.float64, nb.float64[:], 
                        nb.float64,  nb.float64,  nb.float64, 
                        nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.bool_, nb.bool_, nb.bool_,
                    )]
                    )
def sim_p_t(
                T, T_0, T_1, T_buff,
                Hess, Jac, 
                log_L_LA, log_L_FDH,
                p, r, 
                p_F, r_F,
                varsigma_t,
                N_s_a_bool, N_s_e_bool, rho_type_int_s, 
                psi, mu_F, sigma_F, delta_s, gamma_1_s, gamma_2_s, sigma_s, eta_s, eta_varsigma, varsigma_0,
                beta, eta_w, alpha_0_s, alpha_n, alpha_w, lambda_w_s, 
                G_sigma, A_G, Q_q, 
                D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s, 
                mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
                sim=True, calc_log_L=True, Hess_F_D_N=False):




    #rho_type_int_s = np.array([_rho_type_int[rho_type] for rho_type in self._parameters.rho_type_s.get_value()], dtype=int)


    #if sim:
    #    T_0 = self.get_T_buff()
    #else:
    #    T_0 = T_0
    #T_1 = self.get_T_tot_buff()

    (
          p, r, 
          varsigma_t,
          D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s
           ) = run_p_t_njit(sim=sim, T_buff=T_buff, T_1=T_1, 
                                                                                                psi=np.float64(psi), 
                                                                                                delta_s=delta_s, 
                                                                                                gamma_1_s=gamma_1_s, 
                                                                                                gamma_2_s=gamma_2_s, 
                                                                                                sigma_s=sigma_s, 
                                                                                                eta_s=eta_s, 
                                                                                                eta_varsigma=np.float64(eta_varsigma), 
                                                                                                varsigma_0=np.float64(varsigma_0), 
                                                                                                beta=np.float64(beta), 
                                                                                                eta_w=np.float64(eta_w), 
                                                                                                alpha_0_s=alpha_0_s, 
                                                                                                alpha_n=np.float64(alpha_n), 
                                                                                                alpha_w=np.float64(alpha_w), 
                                                                                                lambda_w_s=lambda_w_s, 
                                                                                                G_sigma=np.float64(G_sigma),
                                                                                                A_G=np.float64(A_G),  
                                                                                                N_s_a_bool=N_s_a_bool, 
                                                                                                N_s_e_bool=N_s_e_bool, 
                                                                                                rho_type_int_s=rho_type_int_s, 
                                                                                                nepsilon_t_s=nepsilon_t_s, 
                                                                                                p_F=p_F, 
                                                                                                p=p, 
                                                                                                r=r, 
                                                                                                varsigma_t=varsigma_t,
                                                                                                D_t_s=D_t_s, 
                                                                                                a_t_s=a_t_s, 
                                                                                                n_t_s=n_t_s, 
                                                                                                S_t_s=S_t_s, 
                                                                                                M_t_s=M_t_s, 
                                                                                                rho_t_s=rho_t_s, 
                                                                                                g_t_s=g_t_s, 
                                                                                                w_t_s=w_t_s, 
                                                                                                nu_t_s=nu_t_s
                                                                                                )


    #(mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1) = 

    (mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1) = calc_Hcomp_p_p_F(T=T, 
                                                                                                                        r=r, r_F=r_F,
                                                                                                                        N_s_a_bool=N_s_a_bool, 
                                                                                                                        psi=psi, mu_F=mu_F, delta_s=delta_s, sigma_s=sigma_s, gamma_1_f=np.float64(gamma_1_s[0]), gamma_2_f=np.float64(gamma_2_s[0]), lambda_w_s=lambda_w_s, 
                                                                                                                        M_t_s=M_t_s, rho_t_s=rho_t_s, n_t_s=n_t_s, w_t_s=w_t_s, 
                                                                                                                        mu_t_D=mu_t_D, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, F_mu=F_mu, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1,
                                                                                                                        rho_type_int_f=np.int64(rho_type_int_s[0]), T_0=T_0)
    
    
    if calc_log_L:                                                
        if Hess_F_D_N:

            Hess = calc_Hess_F_D_N_nb( 
                                T=T, 
                                psi=psi, sigma_F=sigma_F, delta_f=delta_s[0],
                                n_t_f=n_t_s[0,T_0:], 
                                sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, D_mu=D_mu,
                                epsilon=0.0
                            )
            

            Jac = calc_Jac_F_D_N_nb(
                                    T=T, 
                                    p_F=p_F[T_0-1:],
                                    psi=psi, sigma_F=sigma_F, delta_f=delta_s[0],  
                                    n_t_f=n_t_s[0,T_0:],
                                    sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu,
                                )


        else:
            Hess = calc_Hess_nb( 
                                T=T, 
                                psi=psi, sigma_F=sigma_F, delta_f=delta_s[0], sigma_f=sigma_s[0], 
                                beta=beta, 
                                Q_q=Q_q, 
                                n_t_f=n_t_s[0,T_0:], 
                                sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1, 
                                eta_w=eta_w, epsilon=0.0
                            )
            

            Jac = calc_Jac_nb(
                                    T=T, 
                                    p_F=p_F[T_0-1:],
                                    psi=psi, sigma_F=sigma_F, delta_f=delta_s[0], 
                                    beta=beta, eta_w=eta_w,  
                                    Q_q=Q_q, 
                                    n_t_f=n_t_s[0,T_0:],
                                    sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1, 
                                )
            

        
        log_L_LA, log_L_FDH = calc_log_L_all(T=T,
                                            Hess=Hess, 
                                            psi=psi, sigma_F=sigma_F, 
                                            sigma_t_D_2=sigma_t_D_2, D_mu=D_mu, F_mu=F_mu)
    



    return (
            Hess, Jac, 
            log_L_LA, log_L_FDH,
            p, r, 
            varsigma_t,
            D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s,
            mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1
            )



@njit(nb.float64[:,:](
                        nb.float64[:,:],
                    ))
def inv_A(A):
     
    return np.linalg.inv(A)


@njit(nb.float64[:](
                        nb.float64[:],
                    ))
def nb_diff(vector):
     
    return vector[1:] - vector[:-1]



@njit([nb.types.Tuple((
                        nb.float64[:,:], nb.float64[:], 
                        nb.float64, nb.float64[:], 
                        nb.float64[:], nb.float64[:], 
                        nb.float64[:], nb.float64[:], 
                        nb.float64[:], 
                        nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))(
                        nb.float64[:], 
                        nb.float64[:], nb.float64[:], 
                        nb.float64,
                        nb.int64, nb.int64, nb.int64,
                        nb.float64[:,:], nb.float64[:], 
                        nb.float64, nb.float64[:],
                        nb.float64[:], nb.float64[:], 
                        nb.float64[:], 
                        nb.bool_[:], nb.bool_[:], nb.int32[:],
                        nb.float64, nb.float64, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.float64, 
                        nb.float64,  nb.float64, nb.float64[:], nb.float64, nb.float64, nb.float64[:], 
                        nb.float64, nb.float64, nb.float64,
                        nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.int64, nb.bool_, nb.bool_, nb.bool_
                    )]
                    )
def set_p_F(new_p_F,
            p_F, r_F, p_0_F, 
            T, T_1, T_buff,
            Hess, Jac, 
            log_L_LA, log_L_FDH,
                p, r, 
                varsigma_t,
                N_s_a_bool, N_s_e_bool, rho_type_int_s,
                psi, mu_F, sigma_F, delta_s, gamma_1_s, gamma_2_s, sigma_s, eta_s, eta_varsigma, varsigma_0,
                beta, eta_w, alpha_0_s, alpha_n, alpha_w, lambda_w_s, 
                G_sigma, A_G, Q_q, 
                D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s, 
                mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
            T_0=0, run_p_t=True, calc_log_L=True, Hess_F_D_N=False):




    p_F[T_0:] = new_p_F
    p_F[:T_0] = p_0_F


    #r_F = np.zeros(p_F.size, dtype=np.float64)
    r_F[1:] = nb_diff(vector=p_F)
    r_F[0] = 0.0


    if run_p_t:
        (
            Hess, Jac, 
            log_L_LA, log_L_FDH,
            p, r, 
            varsigma_t,
            D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s,
            mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
            ) = sim_p_t(
                                                    T=T, T_0=T_0, T_1=T_1, T_buff=T_buff,
                                                    Hess=Hess, Jac=Jac, 
                                                    log_L_LA=log_L_LA, log_L_FDH=log_L_FDH,
                                                    p=p, r=r, 
                                                    p_F=p_F, r_F=r_F,
                                                    varsigma_t=varsigma_t,
                                                    N_s_a_bool=N_s_a_bool, N_s_e_bool=N_s_e_bool, rho_type_int_s=rho_type_int_s, 
                                                    psi=psi, mu_F=mu_F, sigma_F=sigma_F, delta_s=delta_s, gamma_1_s=gamma_1_s, gamma_2_s=gamma_2_s, sigma_s=sigma_s, eta_s=eta_s, eta_varsigma=eta_varsigma, varsigma_0=varsigma_0,
                                                    beta=beta, eta_w=eta_w, alpha_0_s=alpha_0_s, alpha_n=alpha_n, alpha_w=alpha_w, lambda_w_s=lambda_w_s, 
                                                    G_sigma=G_sigma, A_G=A_G, Q_q=Q_q,
                                                    D_t_s=D_t_s, a_t_s=a_t_s, n_t_s=n_t_s, S_t_s=S_t_s, M_t_s=M_t_s, rho_t_s=rho_t_s, g_t_s=g_t_s, w_t_s=w_t_s, nu_t_s=nu_t_s, nepsilon_t_s=nepsilon_t_s, 
                                                    mu_t_D=mu_t_D, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, F_mu=F_mu, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1,
                                                    sim=False, calc_log_L=calc_log_L, Hess_F_D_N=Hess_F_D_N)

    return (
                Hess, Jac, 
                log_L_LA, log_L_FDH,
                p, r, 
                p_F, r_F,
                varsigma_t,
                D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s,
                mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
            )







@njit(nb.types.Tuple((
                        nb.float64[:], nb.float64[:], nb.float64[:], 
                        #nb.int64, nb.float64, nb.float64, #nb.bool_,
                        nb.float64[:], nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:,:],
                        nb.int64, nb.float64, nb.float64,
                        nb.int64, nb.float64, nb.float64,
                        nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], 
                        nb.bool_, nb.bool_,
                        #nb.float64[:], nb.int64, nb.float64, nb.float64, nb.float64[:],
                        #nb.int64, nb.float64[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.float64[:], nb.float64[:,:], nb.float64[:], nb.bool_, nb.bool_,
                        #nb.int64, nb.int64, nb.int32[:], nb.float64[:,:], nb.float64[:,:], # track
                        nb.float64[:,:], nb.float64[:],
                        nb.float64, nb.float64[:],
                        nb.float64[:], nb.float64[:], 
                        nb.float64[:], nb.float64[:],
                        nb.float64[:],
                        nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))(
                        nb.int64, 
                        nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.int64, nb.int64, nb.float64, nb.float64, #nb.bool_,
                        nb.float64[:], nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:,:],
                        nb.int64, nb.float64, nb.float64,
                        nb.int64, nb.float64, nb.float64,
                        nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], 
                        nb.bool_, nb.bool_,
                        #nb.float64[:], nb.int64, nb.float64, nb.float64, nb.float64[:],
                        #nb.int64, nb.float64[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.float64[:], nb.float64[:,:], nb.float64[:], nb.bool_, nb.bool_,
                        #nb.int64, nb.int64, nb.int32[:], nb.float64[:,:], nb.float64[:,:], # track
                        nb.float64[:,:], nb.float64[:], 
                        nb.float64, nb.float64[:],
                        nb.int64, nb.int64, nb.int64,
                        nb.float64[:], nb.float64[:], 
                        nb.float64[:], nb.float64[:], 
                        nb.float64[:],
                        nb.float64, 
                        nb.bool_[:], nb.bool_[:], nb.int32[:],
                        nb.float64, nb.float64, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.float64,
                        nb.float64,  nb.float64, nb.float64[:], nb.float64, nb.float64, nb.float64[:], 
                        nb.float64,  nb.float64,  nb.float64, 
                        nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.bool_, nb.bool_, nb.bool_, nb.bool_, nb.bool_,
                    )
                    )
def calc_p_F_opt(
                T, 
                p_F_0, p_F_0_opt_last_1, p_F_0_set,
                N_k, N_k_trunc, epsilon, step_norm_cutoff, #check_opt_last
                p_F_opt, p_F_end, 
                log_L_k, step_norm_k, p_F_k,
                k_opt, log_L_opt, step_norm_opt,
                k_end, log_L_end, step_norm_end,
                p_F_k0, p_F_k1, A, B,
                use_opt_last, succeed,
                #p_F_0, N_k, epsilon, step_norm_cutoff, p_F_0_opt_last_1,
                #k_opt, p_F_k, p_F_k0, p_F_k1, p_F_opt, log_L_k, log_L_opt, step_norm_k, A, B, succeed, use_opt_last,
                #N_track, track_i, track_k_opt, track_log_L_k, track_step_norm_k,
                Hess, Jac, 
                log_L_LA, log_L_FDH,
                T_1, T_0, T_buff,
                p, r, 
                p_F, r_F,
                varsigma_t,
                p_0_F,
                N_s_a_bool, N_s_e_bool, rho_type_int_s, 
                psi, mu_F, sigma_F, delta_s, gamma_1_s, gamma_2_s, sigma_s, eta_s, eta_varsigma, varsigma_0,
                beta, eta_w, alpha_0_s, alpha_n, alpha_w, lambda_w_s, 
                G_sigma, A_G, Q_q, 
                D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s, 
                mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
                check_opt_last=False, overwrite_a=True, overwrite_b=True, break_nan=False, Hess_F_D_N=False):

    T_sqrt = np.sqrt(T)









    p_F_k0 =  np.copy(p_F_0)
    p_F_k1 = np.copy(p_F_0)

    p_F_k[0,:] = p_F_k1


    p_F_opt = np.copy(p_F_k0)





    (
        Hess, Jac, 
        log_L_LA, log_L_FDH,
        p, r, 
        p_F, r_F,
        varsigma_t,
        D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s,
        mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
            ) = set_p_F(new_p_F=p_F_k1, 
                                    p_F=p_F, r_F=r_F, p_0_F=p_0_F,
                                    T=T, T_1=T_1, T_buff=T_buff,
                                    Hess=Hess, Jac=Jac, 
                                    log_L_LA=log_L_LA, log_L_FDH=log_L_FDH,
                                    p=p, r=r, 
                                    varsigma_t=varsigma_t,
                                    N_s_a_bool=N_s_a_bool, N_s_e_bool=N_s_e_bool, rho_type_int_s=rho_type_int_s, 
                                    psi=psi, mu_F=mu_F, sigma_F=sigma_F, delta_s=delta_s, gamma_1_s=gamma_1_s, gamma_2_s=gamma_2_s, sigma_s=sigma_s, eta_s=eta_s, eta_varsigma=eta_varsigma, varsigma_0=varsigma_0,
                                    beta=beta, eta_w=eta_w, alpha_0_s=alpha_0_s, alpha_n=alpha_n, alpha_w=alpha_w, lambda_w_s=lambda_w_s, 
                                    G_sigma=G_sigma, A_G=A_G, Q_q=Q_q,
                                    D_t_s=D_t_s, a_t_s=a_t_s, n_t_s=n_t_s, S_t_s=S_t_s, M_t_s=M_t_s, rho_t_s=rho_t_s, g_t_s=g_t_s, w_t_s=w_t_s, nu_t_s=nu_t_s, nepsilon_t_s=nepsilon_t_s, 
                                    mu_t_D=mu_t_D, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, F_mu=F_mu, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1,
                                    T_0=T_0, run_p_t=True, calc_log_L=True, Hess_F_D_N=Hess_F_D_N)
    
    k_opt = 0
    p_F_opt = np.copy(p_F_0)
    log_L_opt = log_L_LA


    if check_opt_last:
        (
            Hess, Jac, 
            log_L_LA, log_L_FDH,
            p, r, 
            p_F, r_F,
            varsigma_t,
            D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s,
            mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
                ) = set_p_F(new_p_F=p_F_0_opt_last_1, 
                                        p_F=p_F, r_F=r_F, p_0_F=p_0_F,
                                        T=T, T_1=T_1, T_buff=T_buff,
                                        Hess=Hess, Jac=Jac, 
                                        log_L_LA=log_L_LA, log_L_FDH=log_L_FDH,
                                        p=p, r=r, 
                                        varsigma_t=varsigma_t,
                                        N_s_a_bool=N_s_a_bool, N_s_e_bool=N_s_e_bool, rho_type_int_s=rho_type_int_s, 
                                        psi=psi, mu_F=mu_F, sigma_F=sigma_F, delta_s=delta_s, gamma_1_s=gamma_1_s, gamma_2_s=gamma_2_s, sigma_s=sigma_s, eta_s=eta_s, eta_varsigma=eta_varsigma, varsigma_0=varsigma_0,
                                        beta=beta, eta_w=eta_w, alpha_0_s=alpha_0_s, alpha_n=alpha_n, alpha_w=alpha_w, lambda_w_s=lambda_w_s, 
                                        G_sigma=G_sigma, A_G=A_G, Q_q=Q_q,
                                        D_t_s=D_t_s, a_t_s=a_t_s, n_t_s=n_t_s, S_t_s=S_t_s, M_t_s=M_t_s, rho_t_s=rho_t_s, g_t_s=g_t_s, w_t_s=w_t_s, nu_t_s=nu_t_s, nepsilon_t_s=nepsilon_t_s, 
                                        mu_t_D=mu_t_D, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, F_mu=F_mu, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1,
                                        T_0=T_0, run_p_t=True, calc_log_L=True, Hess_F_D_N=Hess_F_D_N)
        if not np.isnan(log_L_LA):
            use_opt_last = log_L_LA > log_L_opt
            if use_opt_last:
                p_F_opt = np.copy(p_F_0_opt_last_1)
                log_L_opt = log_L_LA
            else:
                (
                    Hess, Jac, 
                    log_L_LA, log_L_FDH,
                    p, r, 
                    p_F, r_F,
                    varsigma_t,
                    D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s,
                    mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
                        ) = set_p_F(new_p_F=p_F_0, 
                                                p_F=p_F, r_F=r_F, p_0_F=p_0_F,
                                                T=T, T_1=T_1, T_buff=T_buff,
                                                Hess=Hess, Jac=Jac, 
                                                log_L_LA=log_L_LA, log_L_FDH=log_L_FDH,
                                                p=p, r=r, 
                                                varsigma_t=varsigma_t,
                                                N_s_a_bool=N_s_a_bool, N_s_e_bool=N_s_e_bool, rho_type_int_s=rho_type_int_s, 
                                                psi=psi, mu_F=mu_F, sigma_F=sigma_F, delta_s=delta_s, gamma_1_s=gamma_1_s, gamma_2_s=gamma_2_s, sigma_s=sigma_s, eta_s=eta_s, eta_varsigma=eta_varsigma, varsigma_0=varsigma_0,
                                                beta=beta, eta_w=eta_w, alpha_0_s=alpha_0_s, alpha_n=alpha_n, alpha_w=alpha_w, lambda_w_s=lambda_w_s, 
                                                G_sigma=G_sigma, A_G=A_G, Q_q=Q_q,
                                                D_t_s=D_t_s, a_t_s=a_t_s, n_t_s=n_t_s, S_t_s=S_t_s, M_t_s=M_t_s, rho_t_s=rho_t_s, g_t_s=g_t_s, w_t_s=w_t_s, nu_t_s=nu_t_s, nepsilon_t_s=nepsilon_t_s, 
                                                mu_t_D=mu_t_D, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, F_mu=F_mu, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1,
                                                T_0=T_0, run_p_t=True, calc_log_L=True, Hess_F_D_N=Hess_F_D_N)
                
    p_F_0_set = np.copy(p_F_opt)
    
    p_F_k[0,:] = np.copy(p_F_opt)
    log_L_k[0] = log_L_LA

    p_F_k0 = np.copy(p_F_opt)
    p_F_k1 = np.copy(p_F_opt)

    for k in range(1,N_k):
        p_F_k0 = np.copy(p_F_k1)

        A = -np.copy(Hess)
        np.fill_diagonal(A, np.diag(A)-epsilon)
        B = (A @ p_F_k0) + Jac
        p_F_k1 = np.linalg.solve(A, B)
        #p_F_k1 = scipy.linalg.solve(A, B, assume_a="sym", overwrite_a=overwrite_a, overwrite_b=overwrite_b)
        step_norm_k[k] = np.sqrt(np.sum((p_F_k1 - p_F_k0)**2))/T_sqrt

        #A = np.copy(Hess)
        #np.fill_diagonal(A, np.diag(A)+epsilon)
        #B = np.ascontiguousarray(np.linalg.inv(A)) @ np.ascontiguousarray(Jac)
        #p_F_k1 = p_F_k0 + B
        #step_norm_k[k] = np.sqrt(np.sum((B)**2))/T_sqrt


        (
        Hess, Jac, 
        log_L_LA, log_L_FDH,
        p, r, 
        p_F, r_F,
        varsigma_t,
        D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s,
        mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
            ) = set_p_F(new_p_F=p_F_k1, 
                                    p_F=p_F, r_F=r_F, p_0_F=p_0_F,
                                    T=T, T_1=T_1, T_buff=T_buff,
                                    Hess=Hess, Jac=Jac, 
                                    log_L_LA=log_L_LA, log_L_FDH=log_L_FDH,
                                    p=p, r=r, 
                                    varsigma_t=varsigma_t,
                                    N_s_a_bool=N_s_a_bool, N_s_e_bool=N_s_e_bool,
                                    psi=psi, mu_F=mu_F, sigma_F=sigma_F, delta_s=delta_s, gamma_1_s=gamma_1_s, gamma_2_s=gamma_2_s, sigma_s=sigma_s, eta_s=eta_s, eta_varsigma=eta_varsigma, varsigma_0=varsigma_0,
                                    beta=beta, eta_w=eta_w, alpha_0_s=alpha_0_s, alpha_n=alpha_n, alpha_w=alpha_w, lambda_w_s=lambda_w_s, 
                                    G_sigma=G_sigma, A_G=A_G, Q_q=Q_q,
                                    rho_type_int_s=rho_type_int_s, 
                                    D_t_s=D_t_s, a_t_s=a_t_s, n_t_s=n_t_s, S_t_s=S_t_s, M_t_s=M_t_s, rho_t_s=rho_t_s, g_t_s=g_t_s, w_t_s=w_t_s, nu_t_s=nu_t_s, nepsilon_t_s=nepsilon_t_s, 
                                    mu_t_D=mu_t_D, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, F_mu=F_mu, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1,
                                    T_0=T_0, run_p_t=True, calc_log_L=True, Hess_F_D_N=Hess_F_D_N)
        


        log_L_k[k] = log_L_LA
        p_F_k[k,:] = p_F_k1

        if log_L_k[k] > log_L_opt:
            k_opt = k
            log_L_opt = log_L_k[k_opt]
            p_F_opt = np.copy(p_F_k1)

        if step_norm_cutoff>0:
            if step_norm_k[k]<step_norm_cutoff:
                break

        if N_k_trunc>0:
            if k - k_opt >= N_k_trunc:
                break

        if np.any(np.isnan(Jac)) or np.any(np.isnan(Hess)):
            succeed = False
            if break_nan:
                log_L_opt = np.nan
                break
    


    k_end = k
    log_L_end = log_L_k[k_end]
    step_norm_end = step_norm_k[k_end]
    p_F_end = np.copy(p_F_k1)


    #x_iK = np.copy(x_i1)
    if succeed or not break_nan:
        p_F_0_opt_last_1 = np.copy(p_F_opt)
        if k!=k_opt:
            (
                Hess, Jac, 
                log_L_LA, log_L_FDH,
                p, r, 
                p_F, r_F,
                varsigma_t,
                D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s,
                mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
                    ) = set_p_F(new_p_F=p_F_opt, 
                                            p_F=p_F, r_F=r_F, p_0_F=p_0_F,
                                            T=T, T_1=T_1, T_buff=T_buff,
                                            Hess=Hess, Jac=Jac, 
                                            log_L_LA=log_L_LA, log_L_FDH=log_L_FDH,
                                            p=p, r=r, 
                                            varsigma_t=varsigma_t,
                                            N_s_a_bool=N_s_a_bool, N_s_e_bool=N_s_e_bool,
                                            psi=psi, mu_F=mu_F, sigma_F=sigma_F, delta_s=delta_s, gamma_1_s=gamma_1_s, gamma_2_s=gamma_2_s, sigma_s=sigma_s, eta_s=eta_s, eta_varsigma=eta_varsigma, varsigma_0=varsigma_0,
                                            beta=beta, eta_w=eta_w, alpha_0_s=alpha_0_s, alpha_n=alpha_n, alpha_w=alpha_w, lambda_w_s=lambda_w_s, 
                                            G_sigma=G_sigma, A_G=A_G, Q_q=Q_q,
                                            rho_type_int_s=rho_type_int_s, 
                                            D_t_s=D_t_s, a_t_s=a_t_s, n_t_s=n_t_s, S_t_s=S_t_s, M_t_s=M_t_s, rho_t_s=rho_t_s, g_t_s=g_t_s, w_t_s=w_t_s, nu_t_s=nu_t_s, nepsilon_t_s=nepsilon_t_s, 
                                            mu_t_D=mu_t_D, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, F_mu=F_mu, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1,
                                            T_0=T_0, run_p_t=True, calc_log_L=True, Hess_F_D_N=Hess_F_D_N)
            
    

    
    


    return (
            p_F_0, p_F_0_opt_last_1, p_F_0_set,
            #N_k, epsilon, step_norm_cutoff,
            p_F_opt, p_F_end, 
            log_L_k, step_norm_k, p_F_k,
            k_opt, log_L_opt, step_norm_opt,
            k_end, log_L_end, step_norm_end,
            p_F_k0, p_F_k1, A, B,
            use_opt_last, succeed,
            #p_F_0, N_k, epsilon, step_norm_cutoff, p_F_0_opt_last_1, 
            #k_opt, p_F_k, p_F_k0, p_F_k1, p_F_opt, log_L_k, log_L_opt, step_norm_k, A, B, succeed, use_opt_last,
            #N_track, track_i, track_k_opt, track_log_L_k, track_step_norm_k,
            Hess, Jac, 
            log_L_LA, log_L_FDH,
            p, r, 
            p_F, r_F,
            varsigma_t,
            D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s,
            mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
            )


@njit(nb.types.Tuple((
                        nb.float64[:], nb.float64[:], nb.float64[:], 
                        #nb.int64, nb.float64, nb.float64, #nb.bool_,
                        nb.float64[:], nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:,:],
                        nb.int64, nb.float64, nb.float64,
                        nb.int64, nb.float64, nb.float64,
                        nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], 
                        nb.bool_, nb.bool_,
                        #nb.float64[:], nb.int64, nb.float64, nb.float64, nb.float64[:],
                        #nb.int64, nb.float64[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.float64[:], nb.float64[:,:], nb.float64[:], nb.bool_, nb.bool_,
                        #nb.int64, nb.int64, nb.int32[:], nb.float64[:,:], nb.float64[:,:], # track
                        nb.float64[:,:], nb.float64[:],
                        nb.float64, nb.float64[:],
                        nb.float64[:], nb.float64[:], 
                        nb.float64[:], nb.float64[:],
                        nb.float64[:],
                        nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                    ))(
                        nb.int64, 
                        nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.int64, nb.int64, nb.float64, nb.float64, #nb.bool_,
                        nb.float64[:], nb.float64[:], 
                        nb.float64[:], nb.float64[:], nb.float64[:,:],
                        nb.int64, nb.float64, nb.float64,
                        nb.int64, nb.float64, nb.float64,
                        nb.float64[:], nb.float64[:], nb.float64[:,:], nb.float64[:], 
                        nb.bool_, nb.bool_,
                        #nb.float64[:], nb.int64, nb.float64, nb.float64, nb.float64[:],
                        #nb.int64, nb.float64[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.float64[:], nb.float64[:,:], nb.float64[:], nb.bool_, nb.bool_,
                        #nb.int64, nb.int64, nb.int32[:], nb.float64[:,:], nb.float64[:,:], # track
                        nb.float64[:,:], nb.float64[:], 
                        nb.float64, nb.float64[:],
                        nb.int64, nb.int64, nb.int64,
                        nb.float64[:], nb.float64[:], 
                        nb.float64[:], nb.float64[:], 
                        nb.float64[:],
                        nb.float64, 
                        nb.bool_[:], nb.bool_[:], nb.int32[:],
                        nb.float64, nb.float64, nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.float64,
                        nb.float64,  nb.float64, nb.float64[:], nb.float64, nb.float64, nb.float64[:], 
                        nb.float64,  nb.float64,  nb.float64, 
                        nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], 
                        nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], 
                        nb.bool_, nb.bool_, nb.bool_, nb.bool_, nb.bool_,
                    )
                    )
def calc_p_F_opt_no_term(
                T, 
                p_F_0, p_F_0_opt_last_1, p_F_0_set,
                N_k, N_k_trunc, epsilon, step_norm_cutoff, #check_opt_last
                p_F_opt, p_F_end, 
                log_L_k, step_norm_k, p_F_k,
                k_opt, log_L_opt, step_norm_opt,
                k_end, log_L_end, step_norm_end,
                p_F_k0, p_F_k1, A, B,
                use_opt_last, succeed,
                #p_F_0, N_k, epsilon, step_norm_cutoff, p_F_0_opt_last_1,
                #k_opt, p_F_k, p_F_k0, p_F_k1, p_F_opt, log_L_k, log_L_opt, step_norm_k, A, B, succeed, use_opt_last,
                #N_track, track_i, track_k_opt, track_log_L_k, track_step_norm_k,
                Hess, Jac, 
                log_L_LA, log_L_FDH,
                T_1, T_0, T_buff,
                p, r, 
                p_F, r_F,
                varsigma_t,
                p_0_F,
                N_s_a_bool, N_s_e_bool, rho_type_int_s, 
                psi, mu_F, sigma_F, delta_s, gamma_1_s, gamma_2_s, sigma_s, eta_s, eta_varsigma, varsigma_0,
                beta, eta_w, alpha_0_s, alpha_n, alpha_w, lambda_w_s, 
                G_sigma, A_G, Q_q, 
                D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s, 
                mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
                check_opt_last=False, overwrite_a=True, overwrite_b=True, break_nan=False, Hess_F_D_N=False):

    T_sqrt = np.sqrt(T)









    p_F_k0 =  np.copy(p_F_0)
    p_F_k1 = np.copy(p_F_0)

    p_F_k[0,:] = p_F_k1


    p_F_opt = np.copy(p_F_k0)





    (
        Hess, Jac, 
        log_L_LA, log_L_FDH,
        p, r, 
        p_F, r_F,
        varsigma_t,
        D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s,
        mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
            ) = set_p_F(new_p_F=p_F_k1, 
                                    p_F=p_F, r_F=r_F, p_0_F=p_0_F,
                                    T=T, T_1=T_1, T_buff=T_buff,
                                    Hess=Hess, Jac=Jac, 
                                    log_L_LA=log_L_LA, log_L_FDH=log_L_FDH,
                                    p=p, r=r, 
                                    varsigma_t=varsigma_t,
                                    N_s_a_bool=N_s_a_bool, N_s_e_bool=N_s_e_bool, rho_type_int_s=rho_type_int_s, 
                                    psi=psi, mu_F=mu_F, sigma_F=sigma_F, delta_s=delta_s, gamma_1_s=gamma_1_s, gamma_2_s=gamma_2_s, sigma_s=sigma_s, eta_s=eta_s, eta_varsigma=eta_varsigma, varsigma_0=varsigma_0,
                                    beta=beta, eta_w=eta_w, alpha_0_s=alpha_0_s, alpha_n=alpha_n, alpha_w=alpha_w, lambda_w_s=lambda_w_s, 
                                    G_sigma=G_sigma, A_G=A_G, Q_q=Q_q,
                                    D_t_s=D_t_s, a_t_s=a_t_s, n_t_s=n_t_s, S_t_s=S_t_s, M_t_s=M_t_s, rho_t_s=rho_t_s, g_t_s=g_t_s, w_t_s=w_t_s, nu_t_s=nu_t_s, nepsilon_t_s=nepsilon_t_s, 
                                    mu_t_D=mu_t_D, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, F_mu=F_mu, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1,
                                    T_0=T_0, run_p_t=True, calc_log_L=True, Hess_F_D_N=Hess_F_D_N)
    
    k_opt = 0
    p_F_opt = np.copy(p_F_0)
    log_L_opt = log_L_LA


    if check_opt_last:
        (
            Hess, Jac, 
            log_L_LA, log_L_FDH,
            p, r, 
            p_F, r_F,
            varsigma_t,
            D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s,
            mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
                ) = set_p_F(new_p_F=p_F_0_opt_last_1, 
                                        p_F=p_F, r_F=r_F, p_0_F=p_0_F,
                                        T=T, T_1=T_1, T_buff=T_buff,
                                        Hess=Hess, Jac=Jac, 
                                        log_L_LA=log_L_LA, log_L_FDH=log_L_FDH,
                                        p=p, r=r, 
                                        varsigma_t=varsigma_t,
                                        N_s_a_bool=N_s_a_bool, N_s_e_bool=N_s_e_bool, rho_type_int_s=rho_type_int_s, 
                                        psi=psi, mu_F=mu_F, sigma_F=sigma_F, delta_s=delta_s, gamma_1_s=gamma_1_s, gamma_2_s=gamma_2_s, sigma_s=sigma_s, eta_s=eta_s, eta_varsigma=eta_varsigma, varsigma_0=varsigma_0,
                                        beta=beta, eta_w=eta_w, alpha_0_s=alpha_0_s, alpha_n=alpha_n, alpha_w=alpha_w, lambda_w_s=lambda_w_s, 
                                        G_sigma=G_sigma, A_G=A_G, Q_q=Q_q,
                                        D_t_s=D_t_s, a_t_s=a_t_s, n_t_s=n_t_s, S_t_s=S_t_s, M_t_s=M_t_s, rho_t_s=rho_t_s, g_t_s=g_t_s, w_t_s=w_t_s, nu_t_s=nu_t_s, nepsilon_t_s=nepsilon_t_s, 
                                        mu_t_D=mu_t_D, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, F_mu=F_mu, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1,
                                        T_0=T_0, run_p_t=True, calc_log_L=True, Hess_F_D_N=Hess_F_D_N)
        if not np.isnan(log_L_LA):
            use_opt_last = log_L_LA > log_L_opt
            if use_opt_last:
                p_F_opt = np.copy(p_F_0_opt_last_1)
                log_L_opt = log_L_LA
            else:
                (
                    Hess, Jac, 
                    log_L_LA, log_L_FDH,
                    p, r, 
                    p_F, r_F,
                    varsigma_t,
                    D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s,
                    mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
                        ) = set_p_F(new_p_F=p_F_0, 
                                                p_F=p_F, r_F=r_F, p_0_F=p_0_F,
                                                T=T, T_1=T_1, T_buff=T_buff,
                                                Hess=Hess, Jac=Jac, 
                                                log_L_LA=log_L_LA, log_L_FDH=log_L_FDH,
                                                p=p, r=r, 
                                                varsigma_t=varsigma_t,
                                                N_s_a_bool=N_s_a_bool, N_s_e_bool=N_s_e_bool, rho_type_int_s=rho_type_int_s, 
                                                psi=psi, mu_F=mu_F, sigma_F=sigma_F, delta_s=delta_s, gamma_1_s=gamma_1_s, gamma_2_s=gamma_2_s, sigma_s=sigma_s, eta_s=eta_s, eta_varsigma=eta_varsigma, varsigma_0=varsigma_0,
                                                beta=beta, eta_w=eta_w, alpha_0_s=alpha_0_s, alpha_n=alpha_n, alpha_w=alpha_w, lambda_w_s=lambda_w_s, 
                                                G_sigma=G_sigma, A_G=A_G, Q_q=Q_q,
                                                D_t_s=D_t_s, a_t_s=a_t_s, n_t_s=n_t_s, S_t_s=S_t_s, M_t_s=M_t_s, rho_t_s=rho_t_s, g_t_s=g_t_s, w_t_s=w_t_s, nu_t_s=nu_t_s, nepsilon_t_s=nepsilon_t_s, 
                                                mu_t_D=mu_t_D, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, F_mu=F_mu, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1,
                                                T_0=T_0, run_p_t=True, calc_log_L=True, Hess_F_D_N=Hess_F_D_N)
                
    p_F_0_set = np.copy(p_F_opt)
    
    p_F_k[0,:] = np.copy(p_F_opt)
    log_L_k[0] = log_L_LA

    p_F_k0 = np.copy(p_F_opt)
    p_F_k1 = np.copy(p_F_opt)

    for k in range(1,N_k):
        p_F_k0 = np.copy(p_F_k1)

        A = -np.copy(Hess)
        np.fill_diagonal(A, np.diag(A)-epsilon)
        B = (A @ p_F_k0) + Jac
        p_F_k1 = np.linalg.solve(A, B)
        #p_F_k1 = scipy.linalg.solve(A, B, assume_a="sym", overwrite_a=overwrite_a, overwrite_b=overwrite_b)
        step_norm_k[k] = np.sqrt(np.sum((p_F_k1 - p_F_k0)**2))/T_sqrt

        #A = np.copy(Hess)
        #np.fill_diagonal(A, np.diag(A)+epsilon)
        #B = np.ascontiguousarray(np.linalg.inv(A)) @ np.ascontiguousarray(Jac)
        #p_F_k1 = p_F_k0 + B
        #step_norm_k[k] = np.sqrt(np.sum((B)**2))/T_sqrt


        (
        Hess, Jac, 
        log_L_LA, log_L_FDH,
        p, r, 
        p_F, r_F,
        varsigma_t,
        D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s,
        mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
            ) = set_p_F(new_p_F=p_F_k1, 
                                    p_F=p_F, r_F=r_F, p_0_F=p_0_F,
                                    T=T, T_1=T_1, T_buff=T_buff,
                                    Hess=Hess, Jac=Jac, 
                                    log_L_LA=log_L_LA, log_L_FDH=log_L_FDH,
                                    p=p, r=r, 
                                    varsigma_t=varsigma_t,
                                    N_s_a_bool=N_s_a_bool, N_s_e_bool=N_s_e_bool,
                                    psi=psi, mu_F=mu_F, sigma_F=sigma_F, delta_s=delta_s, gamma_1_s=gamma_1_s, gamma_2_s=gamma_2_s, sigma_s=sigma_s, eta_s=eta_s, eta_varsigma=eta_varsigma, varsigma_0=varsigma_0,
                                    beta=beta, eta_w=eta_w, alpha_0_s=alpha_0_s, alpha_n=alpha_n, alpha_w=alpha_w, lambda_w_s=lambda_w_s, 
                                    G_sigma=G_sigma, A_G=A_G, Q_q=Q_q,
                                    rho_type_int_s=rho_type_int_s, 
                                    D_t_s=D_t_s, a_t_s=a_t_s, n_t_s=n_t_s, S_t_s=S_t_s, M_t_s=M_t_s, rho_t_s=rho_t_s, g_t_s=g_t_s, w_t_s=w_t_s, nu_t_s=nu_t_s, nepsilon_t_s=nepsilon_t_s, 
                                    mu_t_D=mu_t_D, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, F_mu=F_mu, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1,
                                    T_0=T_0, run_p_t=True, calc_log_L=True, Hess_F_D_N=Hess_F_D_N)
        


        log_L_k[k] = log_L_LA
        p_F_k[k,:] = p_F_k1

        if log_L_k[k] > log_L_opt:
            k_opt = k
            log_L_opt = log_L_k[k_opt]
            p_F_opt = np.copy(p_F_k1)

        #if step_norm_k[k]<step_norm_cutoff:
        #    break

        #if k - k_opt >= N_k_trunc:
        #    break

        if np.any(np.isnan(Jac)) or np.any(np.isnan(Hess)):
            succeed = False
            if break_nan:
                log_L_opt = np.nan
                break
    


    k_end = k
    log_L_end = log_L_k[k_end]
    step_norm_end = step_norm_k[k_end]
    p_F_end = np.copy(p_F_k1)


    #x_iK = np.copy(x_i1)
    if succeed or not break_nan:
        p_F_0_opt_last_1 = np.copy(p_F_opt)
        if k!=k_opt:
            (
                Hess, Jac, 
                log_L_LA, log_L_FDH,
                p, r, 
                p_F, r_F,
                varsigma_t,
                D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s,
                mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
                    ) = set_p_F(new_p_F=p_F_opt, 
                                            p_F=p_F, r_F=r_F, p_0_F=p_0_F,
                                            T=T, T_1=T_1, T_buff=T_buff,
                                            Hess=Hess, Jac=Jac, 
                                            log_L_LA=log_L_LA, log_L_FDH=log_L_FDH,
                                            p=p, r=r, 
                                            varsigma_t=varsigma_t,
                                            N_s_a_bool=N_s_a_bool, N_s_e_bool=N_s_e_bool,
                                            psi=psi, mu_F=mu_F, sigma_F=sigma_F, delta_s=delta_s, gamma_1_s=gamma_1_s, gamma_2_s=gamma_2_s, sigma_s=sigma_s, eta_s=eta_s, eta_varsigma=eta_varsigma, varsigma_0=varsigma_0,
                                            beta=beta, eta_w=eta_w, alpha_0_s=alpha_0_s, alpha_n=alpha_n, alpha_w=alpha_w, lambda_w_s=lambda_w_s, 
                                            G_sigma=G_sigma, A_G=A_G, Q_q=Q_q,
                                            rho_type_int_s=rho_type_int_s, 
                                            D_t_s=D_t_s, a_t_s=a_t_s, n_t_s=n_t_s, S_t_s=S_t_s, M_t_s=M_t_s, rho_t_s=rho_t_s, g_t_s=g_t_s, w_t_s=w_t_s, nu_t_s=nu_t_s, nepsilon_t_s=nepsilon_t_s, 
                                            mu_t_D=mu_t_D, sigma_t_D_2=sigma_t_D_2, rho_prime_k=rho_prime_k, rho_prime2_k=rho_prime2_k, F_mu=F_mu, D_mu=D_mu, dr_mu=dr_mu, ns2_m_1=ns2_m_1, nu_w=nu_w, r_1=r_1,
                                            T_0=T_0, run_p_t=True, calc_log_L=True, Hess_F_D_N=Hess_F_D_N)
            
    

    
    


    return (
            p_F_0, p_F_0_opt_last_1, p_F_0_set,
            #N_k, epsilon, step_norm_cutoff,
            p_F_opt, p_F_end, 
            log_L_k, step_norm_k, p_F_k,
            k_opt, log_L_opt, step_norm_opt,
            k_end, log_L_end, step_norm_end,
            p_F_k0, p_F_k1, A, B,
            use_opt_last, succeed,
            #p_F_0, N_k, epsilon, step_norm_cutoff, p_F_0_opt_last_1, 
            #k_opt, p_F_k, p_F_k0, p_F_k1, p_F_opt, log_L_k, log_L_opt, step_norm_k, A, B, succeed, use_opt_last,
            #N_track, track_i, track_k_opt, track_log_L_k, track_step_norm_k,
            Hess, Jac, 
            log_L_LA, log_L_FDH,
            p, r, 
            p_F, r_F,
            varsigma_t,
            D_t_s, a_t_s, n_t_s, S_t_s, M_t_s, rho_t_s, g_t_s, w_t_s, nu_t_s, nepsilon_t_s,
            mu_t_D, sigma_t_D_2, rho_prime_k, rho_prime2_k, F_mu, D_mu, dr_mu, ns2_m_1, nu_w, r_1,
            )