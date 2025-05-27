"""
This is the main program code that sets up the topology optimisation problem.
This optimisation tries to minimize the global compliance and thus maximizing
the stiffness.

Bram Lagerweij
Aerospace Structures and Materials Department TU Delft
2018
"""

# importing external modules
import time
import math

# importing custom modules
from loads import *
from constraints import DensityConstraint
from fesolvers import FESolver, CvxFEA, CGFEA, PiNN_FEA
from topopt import Topopt
from plotting import Plot


# material properties
# young = 1
# poisson = 0.3

# material properties (Stainless steel)
young = 210
poisson = 0.3

# constraints
Emin = 1e-9
volfrac = 0.4
move = 1

# mesh dimensions
nelx = 200
nely = 50

# optimizer parameters
penal = 3.0
rmin = 3
filt = 'sensitivity'
maxiter = 200
delta = 0#0.005

# plotting and printing options
verbose = True
plotting = True
save_plot = False
history = True

# constraints object created
den_con = DensityConstraint(nelx, nely, move, volume_frac=volfrac)

# loading case object, other classes can be selected and created
load = HalfBeam(nelx, nely, young, Emin, poisson)

# FEA object is generated, other solvers can be selected and created
gtsolver = FESolver(verbose=verbose)
fesolver = PiNN_FEA(verbose=verbose)

# create optimizer object and initialise the problem
optimizer = Topopt(den_con, load, fesolver, verbose=verbose)

# execute the optimization
t = time.time()
x, x_history = optimizer.layout(penal, rmin, delta, maxiter, filt, history)
print('Elapsed time is: ', time.time() - t, 'seconds.')

# plotting
pl = Plot(load, title='Cantilever beam example 200x50 elementen')
pl.boundary(load)
pl.loading(load)

if history:
    for i in x_history:
        pl.add(i, animated=True)
    pl.save('video')

pl.add(x, animated=False)

if save_plot:
    pl.save('figure')

if plotting:
    pl.show()
