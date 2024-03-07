#1b
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from LineSearchOpt import *
from Data import *

spd = Data() #initiate Data class to use spd function for spd Q
n = 100 #dimension
Q = np.random.rand(n, n)    #comment out one of the Qs
#Q = spd.get_spd_mat( n )
x = np.random.rand(n)
b = np.random.rand(n)
c = np.random.rand(1)
QT = Q.transpose()
xT = x.transpose()
bT = b.transpose()


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

    return f,df,d2f



#n = 512; # problem dimension (10 is placeholder. Test random positive integers?)
#A = np.random.rand( n, n )
xtrue = np.random.rand( n )

# compute right hand side
y = np.matmul( Q, xtrue )

# initialize class
opt = Optimize();


# define function handle
#fctn = lambda x, flag: eval_objfun( A, x, y, 0.03, flag )
fctn = lambda x, flag: eval_objfun(Q, x, b, c, flag)
opt.set_objfctn( fctn )

bound = np.zeros(2) #returns 2-dim array of zeroes
b = 5

bound[0] = -b # lower bound
bound[1] =  b # upper bound
m = 100 # number of samples

# number of random perturbations
ntrials = 10

g = np.zeros([m,ntrials])

# draw random perturbations
for i in range(ntrials):
    # draw a random point
    x = np.random.rand( n )
    # compute 1d function along line: g(t) = f( x + t v )
    g[:,i] = opt.cvx_check( x, bound, m )


# plot
t = np.linspace( bound[0], bound[1], m )
plt.plot( t, g )
plt.show()