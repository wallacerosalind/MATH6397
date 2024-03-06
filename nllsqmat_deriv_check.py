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


df_temp = np.zeros((n, p))
val = 0 #declare and initiate variable
tanval = 0 #declare and initiate variable to hold sum (sigma/tanh input)
#construct df_temp by assigning df/dx(ij) to ith row, jth col
for i in range(1,n):
    for j in range(1,p):
        for k in range(1,m):
            for l in range(1,n):
                    tanval +=  Y[k,l] * X[l,j]
            val += Y[k,i] * (1-np.square(sigma(tanval))) * (sigma(tanval) - C[k,j]
        df_temp[i,j] = val



# evaluate objective function
def eval_objfun( X, Y, C, flag="d2f" ):
    # evaluate objective function
    f = 0.5*np.square(np.linalg.norm( sigma(np.matmul(Y, X)) - C ))

    if flag == "f":
        return f

    # evaluate gradient
    df =

    if flag == "df":
        return f,df

    #n = A.shape[0];
    # evaluate hessian
    #d2f =
    return f,df#,d2f #returns tuple

# initialize class
opt = Optimize();

# define function handle
fctn = lambda x, flag: eval_objfun( X, Y, C, flag )

# set objective function
opt.set_objfctn( fctn )

# perform derivative check
opt.deriv_check( X )
