#2
import numpy as np
import sys
sys.path.append("..")
from LineSearchOpt import *

n = 10 #col A
m = 12 #rows A
A = np.random.rand(m, n)
L = np.random.rand(n, n)
x = np.random.rand(n)
b = np.random.rand(m)
AT = A.transpose()
LT = L.transpose()
xT = x.transpose()
bT = b.transpose()

# evaluate objective function
def eval_objfun( A, L, x, b, beta, flag="d2f" ):

    # evaluate objective function
    f = 0.5*np.square(np.linalg.norm( np.sin(np.matmul(A, x)) - b ))\
            + beta*0.5*np.square(np.linalg.norm(np.matmul(L, x)))

    if flag == "f":
        return f

    # evaluate gradient
    df = np.matmul( np.matmul( AT, np.diag( np.cos( np.matmul(A, x)))), \
                    np.sin(np.matmul(A, x)) - b)\
                    + beta * np.matmul( LT, np.matmul(L, x))

    if flag == "df":
        return f,df

    # evaluate hessian
    d2f = np.matmul( AT, A) * ( 1 - 2 * np.matmul(np.sin(np.matmul(A, x)),\
                                    np.sin(np.matmul(A, x)).transpose())\
                                + np.matmul( bT, np.sin(np.matmul(A, x))))\
            + beta * np.matmul( LT, L)

    return f,df,d2f


# initialize class
opt = Optimize();

# define function handle
fctn = lambda x, flag: eval_objfun( A, L, x, b, 0.1, flag )

# set objective function
opt.set_objfctn( fctn )

# perform derivative check
opt.deriv_check( x )

