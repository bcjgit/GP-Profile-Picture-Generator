#! /usr/bin/env python3
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
import matplotlib.pyplot as plot
from scipy.sparse.linalg import spsolve
import argparse


def draw_gp(n=100):
    # Construct L as the standard five-point stencil approximation to the laplacian 
    c = [2, -1]
    c.extend([0 for x in range(n-2)])
    T = linalg.toeplitz(c)
    L = sparse.kron(T,sparse.eye(n)) + sparse.kron(sparse.eye(n),T)

    # Solving the system Lx = w where w ~ N(0,I) <=> drawing from GP
    w = np.random.normal(size=(n**2,1))
    x = spsolve(L,w, use_umfpack=True)

    # Reshape result
    x = np.reshape(x,(n,n))
    return x


def main():
    # Parse dimension of (square) image from command line along with verbose and save flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", help="Dimension of square GP image",type=int)
    parser .add_argument("--verbose",help="Plot the gp",action="store_true")
    parser.add_argument("--save",help="Save gp draw to .png",action="store_true")
    args = parser.parse_args()

    # Draw from gp
    gp = draw_gp(args.dim)

    # Plot and/or save image per user input
    plot.imshow(gp,cmap='summer')
    if args.save:
        plot.imsave('github_picture.png',gp ,cmap='summer')
    if args.verbose:
        plot.show()



if __name__ == "__main__":
    main()
