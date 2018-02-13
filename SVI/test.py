# from SVINelderMeadOptimization import g
# import matplotlib.pyplot as plt
# param=[ 0.00316341 , 0.08230402, -0.71575497, -0.02899962,  0.04797578]
#
# plt.plot(g(param)[0])
# plt.plot([0]*len(g(param)),'y')
# plt.show()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fit_svi_surface import fit_svi_surface
from svi_interpolation import svi_interpolation
from svi_raw import svi_raw
d=pd.read_csv('C:\\Users\\DELL\\Desktop\\call.csv')
spot = 3.171
rf = 0.03
strikes = d['exe_price'].values
impliedVols = d['impvol'].values
ttm = d['ttm'].values
logMoneynesses=np.log(strikes / spot * np.exp(rf * ttm))
maturities = np.array(sorted(list(set(ttm))))

params=np.array([[ -6.12602361e-04  ,-7.68306690e-03 , -1.41268209e-03 ,  1.27275581e-02],
 [  3.28095217e-02 ,  7.23893507e-02  , 1.00896003e-01 ,  1.13173980e-01],
 [ -3.81018899e-01  ,-4.26001565e-01 , -5.45357904e-01  ,-7.19617321e-01],
 [ -2.32187306e-02 , -4.54770989e-02  ,-2.26383083e-02 ,  1.24983464e-02],
 [  3.77894587e-02  , 1.33798429e-01 ,  8.35171426e-02  , 1.07997580e-01]])
k=np.arange(-1,1,0.005)
theta=np.zeros(4)
for i in range(4):
    theta[i]=svi_raw(0,params[:,i],maturities[i])

#log_moneyness, tau_interp, interest_interp, parameters, theta,maturities, forward_theta, interest_rate_theta

tau_interp=0.04
plt.plot(k,svi_interpolation(spot,k,tau_interp,0.03,params,theta,maturities),'orange',label=str(tau_interp))



plt.plot(k,svi_raw(k,params[:,0],maturities[0]),'blue',label='0.090')
plt.plot(k,svi_raw(k,params[:,1],maturities[1]),'green',label='0.167')
# plt.plot(k,svi_raw(k,params[:,2],maturities[2]),'blue',label='0.416')
# plt.plot(k,svi_raw(k,params[:,3],maturities[3]),'purple',label='0.665')


#
# plt.plot(k,np.sqrt(svi_raw(k,params[:,0],maturities[0])/maturities[0]),'r')
# plt.plot(k,np.sqrt(svi_raw(k,params[:,1],maturities[1])/maturities[1]),'g')
# plt.plot(k,np.sqrt(svi_raw(k,params[:,2],maturities[2])/maturities[2]),'b')
# plt.plot(k,np.sqrt(svi_raw(k,params[:,3],maturities[3])/maturities[3]),'y')
plt.legend(loc=0)

fig1=plt.figure()
ax=Axes3D(fig1)
x=np.arange(0.01,0.5,0.01)
y=np.arange(-1,1,0.005)

#
# surface=black_var_surface.blackVol(x,y)**2*x
#
# ax.plot_surface(x,y,surface,rstride=1,cstride=1,cmap='rainbow')

# surface=svi_interpolation(spot, y, x, 0.03, params, theta, maturities)**2*x
surface=[]
for i in range(len(x)):

    surface.append(svi_interpolation(spot, y, x[i], 0.03, params, theta, maturities))
y,x=np.meshgrid(y,x)
ax.plot_surface(y,x,np.array(surface))
# print([len(surface),len(surface[0])])
# print([len(x),len(x[0])])
# print([len(y),len(y[0])])
plt.show()