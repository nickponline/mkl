from scipy.optimize import minimize, rosen, rosen_der
import cloud

def func(x0):
    res = minimize(rosen, x0, method='Nelder-Mead')
    return res

x0 = [1.3, 0.7, 0.8, 1.9, 1.2]

# Local
res = func(x0)
print res.x

# Remote 
jid = cloud.call(func, x0)
print jid
res = cloud.result(jid)
print res.x
