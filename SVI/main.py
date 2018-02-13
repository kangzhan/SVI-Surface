import numpy as np
import pandas as pd
from fit_svi_surface import fit_svi_surface
from svi_interpolation import svi_interpolation
d=pd.read_csv('C:\\Users\\DELL\\Desktop\\call.csv')
spot = 3.171
rf = 0.03
strikes = d['exe_price'].values
impliedVols = d['impvol'].values

ttm = d['ttm'].values
maturities = np.array(sorted(list(set(ttm))))

logMoneynesses=np.log(strikes / spot * np.exp(rf * ttm))
print(impliedVols )
print(impliedVols**2*0.665)
params,theta= fit_svi_surface(impliedVols,logMoneynesses,ttm)

# print (svi_interpolation(logMoneynesses,0.5,0.03,params,theta,maturities))
print(params)