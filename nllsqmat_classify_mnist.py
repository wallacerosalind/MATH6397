#3b
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
sys.path.append("../data")
from LineSearchOpt import *
from Data import *
#import autograd as ag

n = 784 #features 784=28*28 (image size)
m = 20 #examples 60,000
p = 10 #classes (one for each integer 0-9)
#X = np.random.rand(n, p)
#Y = np.random.rand(m, n)
#C = np.random.rand(m, p)
sigma = np.tanh #activation function

# evaluate objective function
def eval_objfun( X, Y, C, flag="df" ):
    #ensure that X is in matrix form (to allow matmul within f)
    X = X.reshape(Y.shape[1] , C.shape[1], order = 'F')
    # evaluate objective function
    f = 0.5*np.square(np.linalg.norm( sigma(np.matmul(Y, X)) - C ))

    if flag == "f":
        return f

    # evaluate gradient
    # populate df_temp, the gradient matrix of f
        # df = np.matmul(Y.transpose(), np.matmul(np.ones((Y.shape[0],X.shape[1])) - np.square(sigma(np.matmul(Y, X))), sigma(np.matmul(Y, X)) - C ))
    ones = np.ones(np.matmul(Y, X).shape)
    tanYX = sigma(np.matmul(Y, X))
    dtanYX = ones - tanYX ** 2
    """
    df_temp = np.zeros((Y.shape[1], C.shape[1]))
    val = 0  # declare and initiate variable
    tanval = 0  # declare and initiate variable to hold sum (sigma/tanh input)
    # construct df_temp by assigning df/dx(ij) to ith row, jth col
    for i in range(0, Y.shape[1]-1):  # i=row; n
        for j in range(0, C.shape[1]-1):  # j=col; p
            for k in range(0, Y.shape[0]-1): #m
                for l in range(0, Y.shape[1]-1):
                    tanval += Y[k, l] * X[l, j]
                    #if tanval != 0:   #debug complete: tanval noteq zero
                        #print('tanval = ')
                        #print(tanval)
                val += Y[k, i] * (1 - (sigma(tanval))**2) * (sigma(tanval) - C[k, j])
                #if val != 0:
                 #   print('val = ')
                  #  print(val)
            df_temp[i, j] = val
    """

    df = df.reshape(-1, order='F') #df_temp is an nxp matrix. This line converts df_temp to n*p vector and assigns this vector to df
    if flag == "df":
        return f,df

    # evaluate hessian
    #d2f = np.identity(X.shape[0]*X.shape[1]) #placeholder since not computing Hessian
    #return f,df,d2f #returns tuple

# initialize classes
opt = Optimize()
dat = Data()
fctn = lambda X, flag: eval_objfun( X, Y, C, flag)
#Y,C,L = dat.read_mnist("test")  #comment out either test or train line
Y,C,L = dat.read_mnist("train",m)  #m=Y.shape[0] rather than m=10k or 60k for faster testing
#print(Y.shape)
#print(C.shape)
Xtrue = np.random.rand(Y.shape[1], C.shape[1])
# initial guess
X = np.zeros(Y.shape[1]*C.shape[1])
# set parameters
opt.set_objfctn( fctn )
opt.set_maxiter( 10 ) #3b:run for 1 iter then report accuracy for each training and test datasets. Then set to 100 iters
# execture solver (gsc)
xgd = opt.run( X, "gdsc" )

# execture solver (newton)
#xnt = opt.run( X, "newton" )
#z = np.linspace( 0, 1, n)
z = np.linspace( 0, 1, xgd.shape[0]) #.shape[] returns int; .shape returns tuple
#print(xgd.shape)#debug
plt.plot( z, xgd, marker="1", linestyle='', markersize=12)
#plt.plot( z, xnt, marker="2", linestyle='', markersize=12)
Xtrue = Xtrue.reshape(Xtrue.shape[0]*Xtrue.shape[1], order = 'F')
plt.plot( z, Xtrue)
#plt.legend(['gradient descent', 'newton', r'$x^\star$'], fontsize="20")
plt.legend(['gradient descent', r'$x^\star$'], fontsize="20")
plt.show()
