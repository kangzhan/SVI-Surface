import QuantLib as ql
import pandas as pd
import numpy as np
from svi_raw import svi_raw
from mpl_toolkits.mplot3d import Axes3D
from fit_svi_surface import fit_svi_surface
# def to_ql_dates(datetime_dates):
#     ql_dates = []
#     for d in datetime_dates:
#         dt = ql.Date(d.day,d.month,d.year)
#         ql_dates.append(dt)
#     return ql_dates
import matplotlib.pyplot as plt
evalDate = ql.Date(26, 1, 2018)
calendar = ql.China()
daycounter = ql.ActualActual()

d=pd.read_csv('C:\\Users\\DELL\\Desktop\\call.csv')
spot = 3.171
rf = 0.03
strikes = d['exe_price'].values
impliedVols = d['impvol'].values
ttm = d['ttm'].values
logMoneynesses=np.log(strikes / spot * np.exp(rf * ttm))
maturities = np.array(sorted(list(set(ttm))))
k=np.arange(-1,1,0.005)
# params=np.array([[ -6.12602361e-04  ,-7.68306690e-03 , -1.41268209e-03 ,  1.27275581e-02],
#  [  3.28095217e-02 ,  7.23893507e-02  , 1.00896003e-01 ,  1.13173980e-01],
#  [ -3.81018899e-01  ,-4.26001565e-01 , -5.45357904e-01  ,-7.19617321e-01],
#  [ -2.32187306e-02 , -4.54770989e-02  ,-2.26383083e-02 ,  1.24983464e-02],
#  [  3.77894587e-02  , 1.33798429e-01 ,  8.35171426e-02  , 1.07997580e-01]])
params,theta= fit_svi_surface(impliedVols,logMoneynesses,ttm)
implied_vols = ql.Matrix(len(k), len(maturities))
volset=[]
for i in range(4):
    volset.append(np.sqrt(svi_raw(k,params[:,i],maturities[i])/maturities[i]))
for i in range(implied_vols.rows()):
    for j in range(implied_vols.columns()):
        implied_vols[i][j] = volset[j][i]


black_var_surface = ql.BlackVarianceSurface(evalDate, calendar, [ql.Date(28,2,2018),ql.Date(28,3,2018),ql.Date(27,6,2018),ql.Date(26,9,2018)], k,
                                                    implied_vols, daycounter)
#
# plt.figure(figsize=(16,9))
# for i in np.arange(0.01,0.5,0.01):
#     tau_interp=i
#     plt.legend()
#     plt.plot(k,np.array([black_var_surface.blackVol(tau_interp,i) for i in k])**2*tau_interp ,label=str(tau_interp))
#
# plt.plot(k,svi_raw(k,params[:,0],maturities[0]),'black',label=0.090)
# plt.legend(loc=0)
# # plt.plot(k,svi_raw(k,params[:,1],maturities[1]),'green')
# plt.show()
fig1=plt.figure()
ax=Axes3D(fig1)
x=np.arange(0.01,0.5,0.01)
y=np.arange(-1,1,0.005)
x,y=np.meshgrid(x,y)
#
# surface=black_var_surface.blackVol(x,y)**2*x
#
# ax.plot_surface(x,y,surface,rstride=1,cstride=1,cmap='rainbow')
surface=[]
for i in range(len(x)):
    surface.append([])
    for j in range(len(x[i])):

        surface[i].append(black_var_surface.blackVol(x[i][j],y[i][j])**2*x[i][j])
print(x)
print(y)

print(surface)
ax.plot_surface(y,x,np.array(surface))
print([len(surface),len(surface[0])])
print([len(x),len(x[0])])
print([len(y),len(y[0])])
plt.show()