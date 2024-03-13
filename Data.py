#copy from Chameleon
import numpy as np
import os
from scipy.linalg import qr
from scipy.sparse import csr_matrix
import requests
import gzip
import sys
sys.path.append("data")

class Data:

    def __init__(self):
        self._files = []
        return

    def get_spd_mat( self, n ):
        """
        get_spd_mat generates a random spd matrix of size n x n
        """

        # compute random orthogonal matrix
        H = np.random.randn( n, n )
        Q, R = qr(H)

        # compute spd matrix
        S = np.diag( np.logspace( 2, -2, n ))
        A = np.matmul( S, Q.T )
        A = np.matmul( Q, A )

        return A



    def read_mnist( self, flag="train", m = 0 ):
        """
        read_mnist read the mnist data set

        The MNIST dataset was constructed from two datasets of the US National Institute of Standards and Technology (NIST). The training set consists of handwritten digits from 250 different people, 50 percent high school students, and 50 percent employees from the Census Bureau. Note that the test set contains handwritten digits from different people following the same split.

The MNIST dataset is publicly available at https://yann.lecun.com/exdb/mnist/ and consists of the following four parts: - Training set images: train-images-idx3-ubyte.gz (9.9 MB, 47 MB unzipped, and 60,000 samples) - Training set labels: train-labels-idx1-ubyte.gz (29 KB, 60 KB unzipped, and 60,000 labels) - Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 7.8 MB, unzipped and 10,000 samples) - Test set labels: t10k-labels-idx1-ubyte.gz (5 KB, 10 KB unzipped, and 10,000 labels)
        """

        if flag == "train":
            # training data
            self._files.append( "train-images-idx3-ubyte.gz" ) # 60,000 samples
            self._files.append( "train-labels-idx1-ubyte.gz" ) # 60,000 labels
            if m == 0:
                m = 60000
        else:
            # testing data
            self._files.append( "t10k-images-idx3-ubyte.gz" ) # 10,000 samples
            self._files.append( "t10k-labels-idx1-ubyte.gz" ) # 10,000 labels
            if m == 0:
                m = 10000

        self._download_mnist();

        n = 28 # image size

        # read the images
        with gzip.open(self._files[0]) as fstream:
            fstream.read(16);
            buffer = fstream.read( n*n*m )
            #print(buffer)
            Y = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
            Y = Y.reshape(m, n*n)
            Y = Y / 255.0
            #print('Y')
            #print(Y)

        # read the labels
        with gzip.open(self._files[1]) as fstream:
            fstream.read(8);
            buffer = fstream.read( m )
            L = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)

        rows = np.array(range(0,m))
        cols = L
        data = np.ones(m)

        # create binary matrix for 10 classes
        C = csr_matrix((data, (rows,cols)), shape=(m,10)).toarray()

        return Y,C,L



    def _download_mnist( self ):
        """
        _download_mnist download the mnist data set
        """
        url = "http://yann.lecun.com/exdb/mnist/";

        i = 0
        while i < len( self._files ):
            filename = url + self._files[i]
            xfile = filename.split("/")[-1]
            #xfile = "./data/" + xfile

            # if file does not exist, download
            if not os.path.isfile( xfile ):
                with open(xfile, "wb") as f:
                    r = requests.get( filename )
                    f.write(r.content)
            else:
                print("file", xfile, "already exists")
            i += 1

        return



    def check_class( self, Cpred, Ctrue ):
        """
        check_class given data classification Cpred and known classes Ctrue,
        give percentage of correct classifications and indicices of
        correct classifications
        """

        # get ids for prediction and true classes; corresponds to predicted
        # numbers
        pred_class = np.argmax( Cpred, axis=1 )
        true_class = np.argmax( Ctrue, axis=1 )

        ids = pred_class == true_class

        n = np.sum( ids )
        pct = float(n)/float(len(true_class))

        print( "prediction accuracy: ", 100*pct, "( {:e} )".format(pct) )

        return pct, n