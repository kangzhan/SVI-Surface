import numpy as np
def svi_convertparameters(param_old,trans_from,to, tau):
    if trans_from=='raw' and to =='jumpwing':
    # from raw to jumpwing
        a = param_old[0]
        b = param_old[1]
        rho= param_old[2]
        m = param_old[3]
        sigma = param_old[4]
        w = a + b * (-rho * m + np.sqrt(m**2 + sigma**2))
        #w = w*tau





        v = w / tau
        psi = 1 / np.sqrt(w) * b / 2 * (-m /  np.sqrt(m**2 + sigma**2) + rho)
        p = 1 /  np.sqrt(w) * b * (1 - rho)
        c = 1 /  np.sqrt(w) * b * (1 + rho)
        vt = 1 / tau * (a + b * sigma *  np.sqrt(1 - rho **2))



        param_new = np.array([v,psi,p,c,vt])

    # from jumpwing to raw
    elif trans_from=='jumpwing' and to =='raw':
        v = param_old[0]
        psi = param_old[1]
        p =param_old[2]
        c = param_old[3]
        vt =param_old[4]
        w = v * tau

        b = np.sqrt(w) / 2 * (c + p)
        rho = 1 - p * np.sqrt(w) / b
        beta = rho - 2 * psi * np.sqrt(w)/b
        alpha = np.sign(beta) * np.sqrt(1 / beta **2 - 1)
        m = (v - vt) * tau / (b * (-rho + np.sign(alpha) *  np.sqrt(1 + alpha**2) - alpha *  np.sqrt(1 - rho**2)))
        if m == 0:
            sigma = (vt * tau - w) / b / ( np.sqrt(1 - rho**2) - 1)
        else:
            sigma = alpha * m
        a = vt * tau - b * sigma *  np.sqrt(1 - rho ** 2)
        if sigma < 0:
            sigma = 0
        param_new = np.array([a,b,rho,m,sigma])
    else:
        raise('INVALID INPUT')



    return param_new
