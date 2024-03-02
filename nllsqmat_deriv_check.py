import numpy as np
import sys
sys.path.append("..")
from LineSearchOpt import *

n = 144 #features 784=28*28 (image size)
m = 60 #examples 60,000
p = 10 #classes (one for each integer 0-9)
X = np.random.rand(n, p) #n,p not n,m?
Y = np.random.rand(m, n)
C = np.random.rand(m, p)
CT = C.transpose()
XT = X.transpose()
YT = Y.transpose()
sigma = np.tanh #activation function

# evaluate objective function
def eval_objfun( X, Y, C, flag="d2f" ):
    # evaluate objective function
    f = 0.5*np.square(np.linalg.norm( sigma(np.matmul(Y, X)) - C ))

    if flag == "f":
        return f

    # evaluate gradient
    df = np.matmul( np.matmul( Y, 1 - np.square(sigma(np.matmul(Y, X)))), \
                    sigma(np.matmul(Y, X)) - C)

    if flag == "df":
        return f,df

    #n = A.shape[0];
    # evaluate hessian
    d2f = np.matmul( np.matmul( np.matmul(Y, Y), -2* sigma(np.matmul(Y, X)) + 2* (sigma(np.matmul(Y, X)))**3), \
                     sigma(np.matmul(Y, X)) - C)\
            + np.square(np.matmul( Y, 1 - np.square(sigma(np.matmul(Y, X)))))

    return f,df,d2f #returns tuple


# initialize class
opt = Optimize();

# define function handle
fctn = lambda x, flag: eval_objfun( X, Y, C, flag )

# set objective function
opt.set_objfctn( fctn )

# perform derivative check
opt.deriv_check( X )
