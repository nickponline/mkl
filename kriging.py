from kernel_learning import *
from DIRECT import solve

# gaussian_process_surrogate(fun, search="ei")
# gaussian_process_surrogate(fun, search="pi")

gaussian_process_surrogate_2d(fun_2d, search="ei")
gaussian_process_surrogate_2d(fun_2d, search="pi")

#gaussian_process_surrogate_nd(fun)
#gaussian_process_surrogate_nd(fun_5d, 5, search='ei')