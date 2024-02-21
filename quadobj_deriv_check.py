import numpy as np
import sys
sys.path.append("..")
from LineSearchOpt import *

n = 10 #dimension
Q = np.random.rand(n, n)
x = np.random.rand(n)
b = np.random.rand(n)
c = np.random.rand(1)
QT = Q.transpose()
xT = x.transpose()
bT = b.transpose()

############Code block below uses rlsq_deriv_check.py#########################
# evaluate objective function
def eval_objfun(Q, x, b, c, flag="d2f" ):
    # compute residual
    #r = np.matmul( A, x ) - y

    # evaluate objective function
    #f = 0.5*np.inner( r, r ) + alpha*0.5*np.inner( x, x )
    f = 0.5 * (np.matmul(np.matmul(xT, Q), x)) + np.matmul(bT, x) + c

    if flag == "f":
        return f

    # evaluate gradient
    #AT = A.transpose()
    #derivative(f) (row vector)
    #gradient(f) (column vector):
    df = 0.5 * np.matmul((QT + Q), x) + b #derived by hand

    if flag == "df":
        return f,df

    n = np.shape(Q)[0]  #code change from Chameleon (dimension of square matrix?)
    # evaluate hessian
    d2f = 0.5 * (QT + Q) #derived by hand

    return f,df,d2f;

# initialize class
opt = Optimize()

# define function handle https://docs.python.org/3/tutorial/controlflow.html
fctn = lambda x, flag: eval_objfun(Q, x, b, c, flag)

# set objective function
opt.set_objfctn(fctn)  #fctn is f,df, or d2f depending on flag arg used above?

# perform derivative check
opt.deriv_check(x)

############Code block above uses rlsq_deriv_check.py#########################