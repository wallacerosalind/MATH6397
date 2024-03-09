#3b
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
sys.path.append("../data")
from LineSearchOpt import *
from Data import *

n = 784 #features 784=28*28 (image size)
m = 60 #examples 60,000
p = 10 #classes (one for each integer 0-9)
#X = np.random.rand(n, p)
#Y = np.random.rand(m, n)
#C = np.random.rand(m, p)
sigma = np.tanh #activation function

# evaluate objective function
def eval_objfun( X, Y, C, flag="df" ):
    # evaluate objective function
    f = 0.5*np.square(np.linalg.norm( sigma(np.matmul(Y, X)) - C ))

    if flag == "f":
        return f

    # evaluate gradient
    # populate df_temp, the gradient matrix of f
    df_temp = np.zeros((n, p))
    val = 0  # declare and initiate variable
    tanval = 0  # declare and initiate variable to hold sum (sigma/tanh input)
    # construct df_temp by assigning df/dx(ij) to ith row, jth col
    for i in range(1, n):  # i=row
        for j in range(1, p):  # j=col
            for k in range(1, m):
                for l in range(1, n):
                    tanval += Y[k, l] * X[l, j]
                val += Y[k, i] * (1 - np.square(sigma(tanval))) * (sigma(tanval) - C[k, j])
            df_temp[i, j] = val
    df = df_temp.reshape(n*p) #df_temp is an nxp matrix. This line converts df_temp to n*p vector and assigns this vector to df

    if flag == "df":
        return f,df

    # evaluate hessian
    d2f = np.identity(X.shape[0]*X.shape[1]) #placeholder since not computing Hessian
    return f,df,d2f #returns tuple

# initialize classes
opt = Optimize()
dat = Data()
Y,C,L = dat.read_mnist("test")
print(Y.shape[0]) #view the row dimension of Y
print(Y.shape[1]) #view the col dimension of Y
Xtrue = np.random.rand(n, p)

# define function handle
fctn = lambda Xtrue, flag: eval_objfun( Xtrue, Y, C, "df" )

# initial guess
X = np.zeros(Y.shape[1]*C.shape[1])

# set parameters
opt.set_objfctn( fctn )
opt.set_maxiter( 1 ) #3b:run for 1 iter then report accuracy for each training and test datasets. Then set to 100 iters

# execture solver (gsc)
xgd = opt.run( X, "gdsc" )

# execture solver (newton)
#xnt = opt.run( x, "newton" )


z = np.linspace( 0, 1, n)
plt.plot( z, xgd, marker="1", linestyle='', markersize=12)
#plt.plot( z, xnt, marker="2", linestyle='', markersize=12)
plt.plot( z, Xtrue )
plt.legend(['gradient descent', 'newton', r'$x^\star$'], fontsize="20")
plt.show()
