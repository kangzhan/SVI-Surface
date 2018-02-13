from scipy.optimize import minimize,fmin
import numpy as np
import math
from svi_raw import  svi_raw
def w(k,param):

    a_star, b_star, rho_star, m_star, sigma_star = param
    return a_star + b_star * (rho_star * (k - m_star) + np.sqrt((k - m_star) ** 2 + sigma_star ** 2))

def g(gk_param):
    k = np.arange(-1, 1, 0.005)
    a_star, b_star, rho_star, m_star, sigma_star = gk_param
    wk = (a_star + b_star * (rho_star * (k - m_star) + np.sqrt((k - m_star) ** 2 + sigma_star ** 2)))

    w_first_deri = b_star * (rho_star + (k - m_star) / np.sqrt((k - m_star) ** 2 + sigma_star ** 2))
    w_second_deri = b_star * ((np.sqrt((k - m_star) ** 2 + sigma_star ** 2) - (k - m_star) ** 2 / np.sqrt(
        (k - m_star) ** 2 + sigma_star ** 2)) / ((k - m_star) ** 2 + sigma_star ** 2))
    result = (1 - k * w_first_deri / 2 / wk) ** 2 - w_first_deri ** 2 / 4 * (1.0 / wk + 0.25) + w_second_deri / 2
    return [result,np.abs(sum(result[result<0]))]


def sum_negatives(arr):
    if type(arr)!=np.array:
        arr=np.array(arr)
    return sum(arr[arr>0])
#
#
# class SVINelderMeadOptimization:
#
#
#     def __init__(self,ttm,data,init_adc,init_msigma,tol):
#         self.ttm = ttm
#         self.init_msigma = init_msigma
#         self.init_adc = init_adc
#         self.tol = tol
#         self.data = data
#         self.minima_flag=0
#
#
#
#     def outter_fun(self,params):
#         m,sigma = params
#
#         adc_0 = self.init_adc
#         def inner_fun(params):
#             a,d,c = params
#             sum = 0.0
#             for i,xi in enumerate(self.data[0]):
#                 yi = (xi - m)/sigma
#                 f_msigma = (a + d*yi + c * math.sqrt((yi**2 + 1)) - self.data[1][i])**2
#                 sum += f_msigma
#
#             # parameters = np.zeros(5)
#             # parameters[0] = np.divide(a, self.ttm)
#             # parameters[1] = np.divide(c, (sigma * self.ttm))
#             # parameters[2] = m
#             # parameters[3] = np.divide(d, c)
#             # parameters[4] = sigma
#             return sum
#         #print(m,sigma)
#         # Constraints: 0 <= c <=4sigma; |d| <= c and |d| <= 4sigma - c; 0 <= a <= max{vi}
#         #print("m",m,";\tsigma",sigma)
#         # bnds = ((None,max(self.data[1])),(-4*sigma,4*sigma),(1e-10, 4*sigma))
#         bnds = ((-1, max(self.data[1])), (-4 * sigma, 4 * sigma), (0, 4 * sigma))
#         #bnds = ((None, None), (-4 * sigma, 4 * sigma), (0, 4 * sigma))
#         #b = np.array(bnds,float)
#         cons = ({'type': 'ineq', 'fun': lambda x: x[2] - abs(x[1])},
#                 {'type': 'ineq', 'fun': lambda x: 4 * sigma - x[2] - abs(x[1])})
#         inner_res = minimize(inner_fun,adc_0,method='SLSQP',constraints=cons)
#         #inner_res = minimize(inner_fun, adc_0, method='SLSQP', tol=1e-6)
#         a_star,d_star,c_star = inner_res.x
#         #global _a_star,_d_star,_c_star
#         self._a_star, self._d_star, self._c_star = inner_res.x
#         #print(a_star,d_star,c_star)
#         sum = 0.0
#         for i,xi in enumerate(self.data[0]):
#             yi = (xi - m)/sigma
#             f_msigma = (a_star + d_star*yi + c_star * math.sqrt((yi**2 + 1)) - self.data[1][i])**2
#             sum += f_msigma
#         # parameters = np.zeros(5)
#         # parameters[0] = np.divide(a_star, self.ttm)
#         # parameters[1] = np.divide(c_star, (sigma * self.ttm))
#         # parameters[2] = m
#         # parameters[3] = np.divide(d_star, c_star)
#         # parameters[4] = sigma
#
#
#         return sum
#
#     def optimization(self):
#         calibrated_params=self.search_over_epsilon()
#
#         return calibrated_params[self.minima_flag]
#
#     def search_over_epsilon(self):
#         sigma_range=np.arange(1e-10,1,0.2)
#         m_range=np.arange(-1,1,0.2)
#         flag=0
#         objs=[]
#         calibrated_params=[]
#         for i in sigma_range:
#             for j in m_range:
#                 outter_res = minimize(self.outter_fun, np.array([j,i]), method='Nelder-Mead', tol=self.tol)
#                 m_star, sigma_star = outter_res.x
#                 obj = outter_res.fun
#                 calibrated_params.append(np.array([self._a_star, self._d_star, self._c_star, m_star, sigma_star]))
#                 objs.append(obj)
#                 print (obj)
#                 if flag>0 and objs[flag]<objs[flag-1]:
#                     self.minima_flag = flag
#                 flag += 1
#         return calibrated_params
#                 # print('a_star,d_star,c_star, m_star, sigma_star: ', calibrated_params)
#
#



class SVINelderMeadOptimization:


    def __init__(self,slice_after,data,init_adc,init_msigma,tol):
        self.init_msigma = init_msigma
        self.init_adc = init_adc
        self.tol = tol
        self.data = data
        self.slice_after=slice_after

    def outter_fun(self,params):
        m,sigma = params
        sigma = max(1e-10,sigma)
        adc_0 = self.init_adc
        def inner_fun(params):
            a,d,c = params
            sum = 0.0
            for i,xi in enumerate(self.data[0]):
                yi = (xi - m)/sigma
                f_msigma = (a + d*yi + c * math.sqrt((yi**2 + 1)) - self.data[1][i])**2
                sum += f_msigma
            return sum
        #print(m,sigma)
        # Constraints: 0 <= c <=4sigma; |d| <= c and |d| <= 4sigma - c; 0 <= a <= max{vi}
        #print("m",m,";\tsigma",sigma)
        bnds = ((1e-10,max(self.data[1])),(-4*sigma,4*sigma),(0, 4*sigma))

        b = np.array(bnds,float)
        cons = (
            {'type':'ineq','fun': lambda x: x[2] - abs(x[1])},
            {'type':'ineq','fun': lambda x: 4*sigma - x[2] - abs(x[1])}
        )
        #inner_res = minimize(inner_fun,adc_0,method='SLSQP',bounds = bnds,constraints=cons, tol=1e-6)
        inner_res = minimize(inner_fun, adc_0, method='SLSQP', constraints=cons,tol=1e-6)
        a_star,d_star,c_star = inner_res.x
        #global _a_star,_d_star,_c_star
        self._a_star, self._d_star, self._c_star = inner_res.x
        #print(a_star,d_star,c_star)
        sum = 0.0

        for i,xi in enumerate(self.data[0]):
            yi = (xi - m)/sigma
            f_msigma = (a_star + d_star*yi + c_star * math.sqrt((yi**2 + 1)) - self.data[1][i])**2
            sum += f_msigma
        return sum

    def optimization(self):
        outter_res = minimize(self.outter_fun, self.init_msigma, method='Nelder-Mead', tol=self.tol)
        m_star,sigma_star = outter_res.x
        #print(outter_res.x)
        #print(outter_res)
        obj = outter_res.fun
        #print(_a_star,_d_star,_c_star,m_star,sigma_star)
        # SVI parameters: a, b, sigma, rho, m
        calibrated_params = [self._a_star, self._d_star, self._c_star,m_star,sigma_star]
        return calibrated_params

