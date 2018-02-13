import numpy as np
def svi_raw(k, param, tau):
    a = param[0]
    b = param[1]
    m = param[2]
    rho = param[3]
    sigma = param[4]
    totalvariance = (a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma **2)))

    return totalvariance
