import numpy as np
from svi_raw import svi_raw
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from SVINelderMeadOptimization import g,w,sum_negatives
from svi_convert import svi_convertparameters
from scipy.optimize import  bisect
import scipy.stats
from fit_svi_surface import eurocall,calcvol


def svi_interpolation(spot,log_moneyness, tau_interp, interest_interp, parameters, theta,
        maturities):

    # forward_theta, interest_rate_theta

    # close = forward_interp*np.exp(-interest_interp * tau_interp)
    if np.isin(tau_interp, maturities):
        total_implied_variance = svi_raw(log_moneyness, parameters[:, maturities == tau_interp], tau_interp)
        implied_volatility = np.sqrt(total_implied_variance / tau_interp)


    elif min(maturities) < tau_interp and tau_interp < max(maturities):
        theta_interp = interp1d(maturities, theta)(tau_interp)
        idx=np.argmin(abs(maturities-tau_interp))
        if maturities[idx] < tau_interp:
            idx = idx + 1
        alpha_t = (np.sqrt(theta[idx]) - np.sqrt(theta_interp)) / (np.sqrt(theta[idx]) - np.sqrt(theta[idx - 1]))


        parameters_jw_fromer = svi_convertparameters(parameters[:, idx - 1], 'raw', 'jumpwing', maturities[idx - 1])
        parameters_jw_after = svi_convertparameters(parameters[:, idx], 'raw', 'jumpwing', maturities[idx])

        param_interp_jw = alpha_t * parameters_jw_fromer + (1 - alpha_t) * parameters_jw_after


        param_interp=svi_convertparameters(param_interp_jw, 'jumpwing', 'raw', tau_interp)

        total_implied_variance = svi_raw(log_moneyness, param_interp, tau_interp)

        implied_volatility = np.sqrt(total_implied_variance / tau_interp)

    elif   tau_interp < maturities[0]:
         # extrapolation for small maturities
        theta_interp = interp1d(maturities, theta,fill_value='extrapolate')(tau_interp)

        strike_1 = spot * np.exp(log_moneyness)
        call_price_1 = np.array([x if x>=0 else 0 for x in (spot - strike_1)])

        idx=0
        total_implied_variance_2 = svi_raw(log_moneyness, parameters[:, idx], maturities[idx])
        implied_volatility_2 = np.sqrt(total_implied_variance_2 / maturities[idx])
        strike_2 = spot*np.exp(interest_interp*maturities[0]) * np.exp(log_moneyness)
        call_price_2 = eurocall(implied_volatility_2,spot, strike_2, interest_interp, maturities[idx],0)


        forward_interp=interp1d(np.array([0, maturities[0]]), [spot,spot*np.exp(interest_interp*maturities[0])])(tau_interp)
        alpha_t = (np.sqrt(theta[idx]) - np.sqrt(theta_interp)) / np.sqrt(theta[idx])
        K_t = forward_interp * np.exp(log_moneyness)
        call_price = K_t* (alpha_t * call_price_1/ strike_1 + (1 - alpha_t) * call_price_2/ strike_2)
        implied_volatility = calcvol(spot, K_t, interest_interp, tau_interp, call_price, '0')
        total_implied_variance = np.array(implied_volatility)** 2 * tau_interp




    # extrapolation for large maturities

    else:
        theta_interp = interp1d(maturities, theta,fill_value='extrapolate')(tau_interp)

        total_implied_variance = svi_raw(log_moneyness, parameters[:, -1], maturities[-1])
        total_implied_variance = total_implied_variance + theta_interp - theta[-1]
        implied_volatility = np.sqrt(total_implied_variance / tau_interp)









    return total_implied_variance




