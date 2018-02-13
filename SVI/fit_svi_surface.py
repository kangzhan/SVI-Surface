import pandas as pd
from scipy.optimize import  bisect
import scipy.stats
import datetime as datetime
from scipy import interpolate
import numpy as np

from svi_raw import svi_raw
from SVINelderMeadOptimization import g,w,sum_negatives

from svi_calibration_util import svi_calibration_util
import matplotlib.pyplot as plt
from svi_convert import svi_convertparameters
from scipy.optimize import minimize
N = scipy.stats.norm(0, 1).cdf

def eurocall(sigma,*args):
    S=args[0]
    E=args[1]
    r=args[2]
    T=args[3]
    real_price=args[4]
    d_1 = (np.log(S / E) + (r + 0.5 * sigma ** 2)) / (sigma * np.sqrt(T))
    d_2 = (np.log(S / E) + (r - 0.5 * sigma ** 2)) / (sigma * np.sqrt(T))
    return S * N(d_1) - E * np.exp(-r * T) * N(d_2)- real_price
def europut(sigma,*args):
    S = args[0]
    E = args[1]
    r = args[2]
    T = args[3]
    real_price = args[4]
    d_1 = (np.log(S / E) + (r + 0.5 * sigma ** 2)) / (sigma * np.sqrt(T))
    d_2 = (np.log(S / E) + (r - 0.5 * sigma ** 2)) / (sigma * np.sqrt(T))
    return -S * N(-d_1) + E * np.exp(-r * T) * N(-d_2)- real_price


def calcvol(s, e, rate, t, p, callput):
    zeros=[]
    if callput == '0':

        # zero=newton(call.sse,0.05,maxiter=100000,tol=1e-5)
        for i in range(len(p)):
            zero = bisect(eurocall, -10, 1000000, (s, e[i], rate, t, p[i]))
            zeros.append(zero)
        return np.array(zeros)

    else:

        # zero=newton(put.sse,0.05,maxiter=100000,tol=1e-5)
        for i in range(len(p)):
            zero = bisect(europut, -10, 1000000,(s,e[i],rate,t,p[i]))
            zeros.append(zero)
        return np.array(zeros)




def generateRandomStartValues(lb, ub):
    lb[np.abs(lb) > 1000] = -1000
    ub[np.abs(ub) > 1000] = 1000
    param0 = lb + np.random.random(len(lb)) * (ub - lb)
    return param0



def fit_svi_surface(implied_volatility,log_moneyness, maturity ):

    total_implied_variance = np.array(implied_volatility ** 2 * maturity)

    maturities = np.array(sorted(list(set(maturity))))
    T=len(maturities)
    theta = np.zeros(T)
    parameters = np.zeros([5, T])

    for t in np.arange(T - 1, -1, -1):
        pos = maturity == maturities[t]

        log_moneyness_t = np.array(log_moneyness[pos])
        k=np.arange(-1,1,0.005)
        total_implied_variance_t = total_implied_variance[pos]

        if t == T - 1:

            # slice_before = []
            slice_after = []

        else:
            # slice_before=[]
            param_after = parameters[:, t+1]
            slice_after = svi_raw(k, param_after, maturities[t + 1])

        print('================================starting fitting================================================')

        # parameters[:, t] = get_svi_optimal_params_non_calendar(slice_after,[list(log_moneyness_t),
        #                                                                     total_implied_variance_t],maturities[t])
        parameters[:, t] = svi_calibration_util(slice_after,[log_moneyness_t, total_implied_variance_t],
                                                               maturities[t])

        # penalize butterfly arbitrage
        # a, b, rho, m, sigma    shunxu
        params_old=parameters[:, t]
        if (g(params_old)[0]<0).any():
            parameters_bnd=svi_convertparameters(params_old,'raw','jumpwing',maturities[t])
            parameters_bnd[3]=parameters_bnd[2] + 2 * parameters_bnd[1]
            parameters_bnd[4]=parameters_bnd[0]*4*parameters_bnd[2]*parameters_bnd[3]/(parameters_bnd[2]+parameters_bnd[3])**2
            parameters_bnd_raw=svi_convertparameters(parameters_bnd,'jumpwing','raw',maturities[t])
            def recalib(param):
                norm=sum((w(log_moneyness_t,params_old)-w(log_moneyness_t,param))**2)+0.00001*g(param)[1]

                return norm

            a, b, rho, m, sigma=params_old
            a_bnds, b_bnds, rho_bnds, m_bnds, sigma_bnds=parameters_bnd


            bnds=((min(a,a_bnds),max(a,a_bnds)), (min(b,b_bnds),max(b,b_bnds)),(min(rho,rho_bnds),max(rho,rho_bnds)),
                  (min(m,m_bnds),max(m,m_bnds)),(min(sigma,sigma_bnds),max(sigma,sigma_bnds)))
            res=minimize(recalib,(np.array(parameters_bnd_raw)+np.array(params_old))/2,bounds=bnds,method='SLSQP',tol=1e-9)
            param_new=res.x
            parameters[:, t]=param_new



#penalize calendar arbitrage and butterfly arbitrage

        if slice_after != [] and np.array((svi_raw(k, parameters[:, t], maturities[t]) - slice_after) > 0).any():
            print(np.array((svi_raw(k, parameters[:, t], maturities[t]) - slice_after) > 0))
            print('eliminating calendar arb')
            param_no_calendar_arb=parameters[:, t]

            # a:param_no_calendar_arb[0]
            # param_no_calendar_arb[0] = param_no_calendar_arb[0] + 0.5
            # while np.array((svi_raw(k, param_no_calendar_arb, maturities[t]) - slice_after) > 0).any():
            #     param_no_calendar_arb[0] = param_no_calendar_arb[0] - 0.5
            def recalib_calendar(param):
                norm = sum((w(log_moneyness_t, parameters[:, t]) - w(log_moneyness_t, param)) ** 2) + 0.0001 * sum_negatives(np.array(svi_raw(k, param, maturities[t]) - slice_after))++0.00001*g(param)[1]
                return norm

            res2 = minimize(recalib_calendar,param_no_calendar_arb, method='SLSQP', tol=1e-9)
            parameters[:, t] = res2.x

        print(parameters[:, t])
        theta[t] =svi_raw(0, parameters[:, t],maturities[t])












        plt.plot(g(parameters[:, t])[0],'r--')

        plt.plot([0]*600,'b')
        plt.show()

        a , b ,  rho, m,sigma=parameters[:, t]
        x_range = np.arange(min(log_moneyness_t) - 0.005, max(log_moneyness_t) + 0.02, 0.1 / 100)
        tv_svi2 =  a + b * (rho * (x_range - m) + np.sqrt((x_range - m) ** 2 + sigma ** 2))

        plt.figure()
        plt.plot(np.exp(log_moneyness_t), implied_volatility[pos], 'ro')
        plt.plot(np.exp(x_range), np.sqrt(tv_svi2/maturities[t]), 'b--')

        plt.show()

    return parameters,theta





















