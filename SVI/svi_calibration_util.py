from SVINelderMeadOptimization import SVINelderMeadOptimization
import numpy as np


def svi_calibration_util(slice_after,data,ttm,sim_no = 10):
    logMoneynesses = data[0]
    totalvariance = data[1]
    calibrated_params = []
    sse_list = []
    params=[]

    for i in range(15):
        # ms_0=[0.01,0.01]
        ms_0 = np.random.randint(0, 100, 2) / 100  # 参数初始值，可调
        adc_0=[0.01,0.01,0.01]
        # ms_0 =[0.01,0.01]

        # nm = SVINelderMeadOptimization(ttm,data,adc_0,ms_0,1e-8)
        nm = SVINelderMeadOptimization(slice_after, data, adc_0, ms_0, 1e-8)
        calibrated_params = nm.optimization()
        _a_star, _d_star, _c_star, m_star, sigma_star = calibrated_params

        sse = 0.0
        for i, m in enumerate(logMoneynesses):
            tv = totalvariance[i]
            y_1 = np.divide((m - m_star), sigma_star)
            tv_1 = _a_star + _d_star * y_1 + _c_star * np.sqrt(y_1 ** 2 + 1)
            sse += (tv - tv_1) ** 2
        sse_list.append(sse)
        params.append(calibrated_params)
    calibrated_params=params[np.argmin(sse_list)]



    _a_star, _d_star, _c_star, m_star, sigma_star = calibrated_params
    # a_star = np.divide(_a_star, ttm)
    # b_star = np.divide(_c_star, (sigma_star * ttm))
    a_star=_a_star
    b_star= np.divide(_c_star, sigma_star )
    rho_star = np.divide(_d_star, _c_star)
    final_parames = [a_star, b_star, rho_star, m_star, sigma_star]

    # x_range = np.arange(min(logMoneynesses) - 0.005, max(logMoneynesses) + 0.02, 0.1 / 100)
    # tv_svi2 = np.multiply(
    #     a_star + b_star * (rho_star * (x_range - m_star) + np.sqrt((x_range - m_star) ** 2 + sigma_star ** 2)), ttm)
    #

    # plt.figure()
    # plt.plot(logMoneynesses, totalvariance, 'ro')
    # plt.plot(x_range, tv_svi2, 'b--')
    # plt.show()

    return final_parames



