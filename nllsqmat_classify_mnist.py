#3b
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
sys.path.append("../data")
from LineSearchOpt import *
from Data import *

#populate df_temp matrix
df_temp = np.zeros((n, p))
val = 0 #declare and initiate variable
tanval = 0 #declare and initiate variable to hold sum (sigma/tanh input)
#construct df_temp by assigning df/dx(ij) to ith row, jth col
for i in range(1,n):
    for j in range(1,p):
        for k in range(1,m):
            for l in range(1,n):
                    tanval +=  Y[k,l] * X[l,j]
            val += Y[k,i] * (1-np.square(sigma(tanval))) * (sigma(tanval) - C[k,j])
        df_temp[i,j] = val



# evaluate objective function
def eval_objfun( X, Y, C, flag="df" ):
    # evaluate objective function
    f = 0.5*np.square(np.linalg.norm( sigma(np.matmul(Y, X)) - C ))

    if flag == "f":
        return f

    # evaluate gradient
    df = df_temp.reshape(n*p)

    if flag == "df":
        return f,df

    #n = A.shape[0];
    # evaluate hessian
    d2f = np.identity(X.shape[0]*X.shape[1]) #placeholder since not computing Hessian
    return f,df,d2f #returns tuple


n = 128; # problem dimension

# initialize classes
opt = Optimize()
dat = Data()

A = dat.get_spd_mat( n )

xtrue = np.random.rand( n )

# compute right hand side
y = np.matmul( A, xtrue )

# define function handle
fctn = lambda x, flag: eval_objfctn( A, x, y, 1e-6, flag )

# initial guess
x = np.zeros( n )

# set parameters
opt.set_objfctn( fctn )
opt.set_maxiter( 1 ) #3b:run for 1 iter then report accuracy for each training and test datasets. Then set to 100 iters

# execture solver (gsc)
xgd = opt.run( x, "gdsc" )

# execture solver (newton)
xnt = opt.run( x, "newton" )


z = np.linspace( 0, 1, n)
plt.plot( z, xgd, marker="1", linestyle='', markersize=12)
plt.plot( z, xnt, marker="2", linestyle='', markersize=12)
plt.plot( z, xtrue )
plt.legend(['gradient descent', 'newton', r'$x^\star$'], fontsize="20")
plt.show()
