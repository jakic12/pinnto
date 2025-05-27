"""
Finite element solvers for the displacement from stiffness matrix and force
vector. This version of the code is meant for global compliance minimization.

Bram Lagerweij
Aerospace Structures and Materials Department TU Delft
2018
"""

# Importing modules parent class
import numpy as np
from scipy.sparse import coo_matrix

# Importing linear algabra solver for the CvxFEA class
import cvxopt
import cvxopt.cholmod

# Importing linear algabra solver for the SciPyFEA class
from scipy.sparse.linalg import spsolve

# Imporning linear algabla conjugate gradient solver
from scipy.sparse.linalg import cg
from scipy.sparse import diags

import matplotlib.pyplot as plt

from torch import nn
import torch
import torch.nn.functional as F
from collections import namedtuple


class FESolver(object):
    """
    This parent FEA class can only assemble the global stiffness matrix and
    exclude all fixed degrees of freedom from it. This stiffenss csc-sparse
    stiffness matrix is assebled in the gk_freedof method. This
    class solves the FE problem with a sparse LU-solver based upon umfpack.
    This solver is slow and inefficient. It is however more robust.

    Parameters
    ----------
    verbose : bool, optional
        False if the FEA should not print updates

    Attributes
    ---------
    verbose : bool
        False if the FEA does not print updates.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose

    # finite element computation for displacement
    def displace(self, load, x, ke, kmin, penal, iter_):
        """
        FE solver based upon the sparse SciPy solver that uses umfpack.

        Parameters
        ----------
        load : object, child of the Loads class
            The loadcase(s) considerd for this optimisation problem.
        x : 2-D array size(nely, nelx)
            Current density distribution.
        ke : 2-D array size(8, 8)
            Local fully dense stiffnes matrix.
        kmin : 2-D array size(8, 8)
            Local stiffness matrix for an empty element.
        penal : float
            Material model penalisation (SIMP).

        Returns
        -------
        u : 1-D array len(max(edof)+1)
            Displacement of all degrees of freedom
        """
        freedofs = np.array(load.freedofs())
        nely, nelx = x.shape

        f_free = load.force()[freedofs]
        k_free = self.gk_freedofs(load, x, ke, kmin, penal)

        # solving the system f = Ku with scipy
        u = np.zeros(load.dim*(nely+1)*(nelx+1))
        u[freedofs] = spsolve(k_free, f_free)

        return u

    # sparce stiffness matix assembly
    def gk_freedofs(self, load, x, ke, kmin, penal, iter_):
        """
        Generates the global stiffness matrix with deleted fixed degrees of
        freedom. It generates a list with stiffness values and their x and y
        indices in the global stiffness matrix. Some combination of x and y
        appear multiple times as the degree of freedom might apear in multiple
        elements of the FEA. The SciPy coo_matrix function adds them up at the
        background.

        Parameters
        ----------
        load : object, child of the Loads class
            The loadcase(s) considerd for this optimisation problem.
        x : 2-D array size(nely, nelx)
            Current density distribution.
        ke : 2-D array size(8, 8)
            Local fully dense stiffnes matrix.
        kmin : 2-D array size(8, 8)
            Local stiffness matrix for an empty element.
        penal : float
            Material model penalisation (SIMP).

        Returns
        -------
        k : 2-D sparse csc matrix
            Global stiffness matrix without fixed degrees of freedom.
        """
        freedofs = np.array(load.freedofs())
        nelx = load.nelx
        nely = load.nely

        edof, x_list, y_list = load.edof()

        #  SIMP - Ee(xe) = Emin + x^p (E-Emin)
        kd = x.T.reshape(nelx*nely, 1, 1) ** penal  # knockdown factor
        value_list = ((np.tile(kmin, (nelx*nely, 1, 1)) + np.tile(ke-kmin, (nelx*nely, 1, 1))*kd)).flatten()

        # coo_matrix sums duplicated entries and sipmlyies slicing
        dof = load.dim*(nelx+1)*(nely+1)
        k = coo_matrix((value_list, (y_list, x_list)), shape=(dof, dof)).tocsc()
        k = k[freedofs, :][:, freedofs]

        return k


class CvxFEA(FESolver):
    """
    This parent FEA class is used to assemble the global stiffness matrix while
    this class solves the FE problem with a Supernodal Sparse Cholesky
    Factorization.

    Attributes
    --------
    verbose : bool
        False if the FEA should not print updates.
    """
    def __init__(self, verbose=False):
        super().__init__(verbose)

    # finite element computation for displacement
    def displace(self, load, x, ke, kmin, penal, iter_):
        """
        FE solver based upon a Supernodal Sparse Cholesky Factorization. It
        requires the instalation of the cvx module. [1]_

        Parameters
        ----------
        load : object, child of the Loads class
            The loadcase(s) considerd for this optimisation problem.
        x : 2-D array size(nely, nelx)
            Current density distribution.
        ke : 2-D array size(8, 8)
            Local fully dense stiffnes matrix.
        kmin : 2-D array size(8, 8)
            Local stiffness matrix for an empty element.
        penal : float
            Material model penalisation (SIMP).

        Returns
        -------
        u : 1-D array len(max(edof))
            Displacement of all degrees of freedom

        References
        ---------
        .. [1] Y. Chen, T. A. Davis, W. W. Hager, S. Rajamanickam, "Algorithm
            887: CHOLMOD, Supernodal Sparse Cholesky Factorization and
            Update/Downdate", ACM Transactions on Mathematical Software, 35(3),
            22:1-22:14, 2008.
        """
        freedofs = np.array(load.freedofs())
        nely, nelx = x.shape

        f = load.force()
        B_free = cvxopt.matrix(f[freedofs])

        k_free = self.gk_freedofs(load, x, ke, kmin, penal).tocoo()
        k_free = cvxopt.spmatrix(k_free.data, k_free.row, k_free.col)

        u = np.zeros(load.dim*(nely+1)*(nelx+1))

        # setting up a fast cholesky decompositon solver
        cvxopt.cholmod.linsolve(k_free, B_free)
        u[freedofs] = np.array(B_free)[:, 0]

        return u

DirichletPINN_params = namedtuple('DirichletPINN_params', ['u0', 'u1', 'v0', 'v1'])
example = DirichletPINN_params(
    lambda y: 0,
    lambda y: 0,
    lambda x: torch.sin(2 * np.pi * x),
    lambda x: torch.sin(2 * np.pi * x),
)

class DirichletPINN(nn.Module):
    def __init__(self, hidden_shape, arr_params, activation=nn.Sigmoid()):
        super(DirichletPINN, self).__init__()
        self.nn_shape = [2, *hidden_shape, 2]
        self.nn_layers = nn.ModuleList()
        self.params = arr_params
     
        for i in range(len(self.nn_shape)-1):
            self.nn_layers.append(nn.Linear(self.nn_shape[i], self.nn_shape[i+1]))
            self.nn_layers.append(activation)

        # remove the last activation
        self.nn_layers = self.nn_layers[:-1]

    
    def forward(self, X):
        """
        Forward pass through the neural network.
        Multiply the outputs with a*(1-a)*out[a] to make sure that the
        dirichled boundary conditions are satisfied.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, 2) where the first column
            is x and the second column is y.
        """
        model_out = X
        for layer in self.nn_layers:
            model_out = layer(model_out)

        out = torch.zeros(X.shape[0], self.nn_shape[-1])
        for i in range(self.nn_shape[-1]):
            u0, u1, v0, v1 = self.params[i]
            # Trial solution is in the form of y_{i,t}(x,y) = h1_i(x,y) + model_i(x,y)*h2_i(x,y)
            # h1_i(0,y) = u0_i(y)
            # h1_i(1,y) = u1_i(y)
            # h1_i(x,0) = v0_i(x)
            # h1_i(x,1) = v1_i(x)
            h1 = lambda x,y: (1 - x) * u0(y) + x * u1(y) + (1 - y) * v0(x) + y * v1(x)
            h2 = lambda x,y: x * (1 - x) * y * (1 - y)
            out[:, i] = h1(X[:, 0], X[:, 1]) + model_out[:, i] * h2(X[:, 0], X[:, 1])
        
        return out

def plot_mixed_arr(arr, load, title="", iter_=None):
    # Plot the mixed array of size (dim * nely+1 * nelx+1)
    # Reshape the array to (nely+1, nelx+1)
    idx = np.arange(arr.shape[0])
    arr1 = arr[idx % load.dim == 0].reshape((load.nelx+1, load.nely+1)).T
    arr2 = arr[idx % load.dim == 1].reshape((load.nelx+1, load.nely+1)).T
    plt.imshow(arr1, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f"{title} x-dim")
    plt.savefig(f"plots/x_dim_iter_{iter_}.png")

    plt.clf()
    # Plot the second dimension
    plt.imshow(arr2, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f"{title} y-dim")
    plt.savefig(f"plots/y_dim_iter_{iter_}.png")
    plt.clf()
    

class PiNN_FEA(FESolver):
    """
    This parent FEA class is used to assemble the global stiffness matrix while
    this class solves the FE Energy based physics informed neural network (PINN).

    Attributes
    --------
    verbose : bool
        False if the FEA should not print updates.
    """
    def __init__(self, verbose=False, epochs=1000, learning_rate=0.001, early_stopping=50):
        super().__init__(verbose)

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
    
    def calculate_energy_loss(self, load, points, output, force_in_x, force_in_y, ext_forces, density):
        """
        The loss function is the potential energy of the system.
        
        Parameters
        ----------
        load : object, child of the Loads class
            The loadcase(s) considerd for this optimisation problem.
        points : torch.Tensor
            Input tensor of shape (batch_size, 2) where the first column
            is x and the second column is y.
        output : torch.Tensor
            Output tensor of shape (batch_size, 2) where the first column
            is u and the second column is v.
        force_in_x : torch.Tensor
            Tensor of whape (num_points,2) of external forces (only points that are free in the x direction).
        force_in_y : torch.Tensor
            Tensor of whape (num_points,2) of external forces (only points that are free in the y direction).
        ext_forces : torch.Tensor
            External forces of shape (num_points, 2) where the first column
            is fx and the second column is fy.
        density : torch.Tensor
            Density of the material of shape (batch_size, ).
        Returns
        -------
        loss : torch.Tensor
            The loss value.
        """
        u = torch.tensor(np.zeros((load.nely+1, load.nelx+1)), dtype=torch.float32)
        v = torch.tensor(np.zeros((load.nely+1, load.nelx+1)), dtype=torch.float32)

        # points_int = np.array(points.detach().numpy()).astype(int)


        u[points[:, 1].int(), points[:, 0].int()] = output[:, 0]
        v[points[:, 1].int(), points[:, 0].int()] = output[:, 1]

        # Calculate the gradient of the output with respect to the input
        grad_ux = torch.autograd.grad(outputs=output[:, 0], inputs=points,
                              grad_outputs=torch.ones_like(output[:, 0]),
                              create_graph=True)[0]

        grad_uy = torch.autograd.grad(outputs=output[:, 1], inputs=points,
                                    grad_outputs=torch.ones_like(output[:, 1]),
                                    create_graph=True)[0]
        
        uxx = grad_ux[:, 0]
        uxy = grad_ux[:, 1]
        uyx = grad_uy[:, 0]
        uyy = grad_uy[:, 1]

        eps_xx = uxx
        eps_yy = uyy
        eps_xy = 0.5 * (uxy + uyx)

        E = load.young
        nu = load.poisson

        lambda_ = E * nu / ((1 + nu)*(1-nu))
        mu = E / (2 * (1 + nu))

        sigma_xx = lambda_ * (eps_xx + eps_yy) + 2 * mu * eps_xx
        sigma_yy = lambda_ * (eps_xx + eps_yy) + 2 * mu * eps_yy
        sigma_xy = 2 * mu * eps_xy

        # Max pooling on the density matrix
        node_density_matrix = F.max_pool2d(density.unsqueeze(0), kernel_size=2, stride=1, padding=1)[0]

        # Calculate the internal strain energy
        Ein = (0.5 * (sigma_xx * eps_xx + sigma_yy * eps_yy + 0.5 * sigma_xy * eps_xy) * node_density_matrix[points[:, 1].int(), points[:, 0].int()].reshape(-1)).mean()

        # Calculate the external work done by the loads
        u_ext = u[force_in_x[:, 1], force_in_x[:, 0]]
        v_ext = v[force_in_y[:, 1], force_in_y[:, 0]]

        # Magnitude of the external force in the x direction in the points that are free in the x direction
        mag_force_in_x = ext_forces[load.node(force_in_x[:, 1], force_in_x[:, 0])]
        density_in_x = node_density_matrix[force_in_x[:, 1], force_in_x[:, 0]]

        # Magnitude of the external force in the y direction in the points that are free in the y direction
        mag_force_in_y = ext_forces[load.node(force_in_y[:, 1], force_in_y[:, 0])+1]
        density_in_y = node_density_matrix[force_in_y[:, 1], force_in_y[:, 0]]

        Eex = torch.mean(mag_force_in_x * u_ext) + torch.mean(mag_force_in_y * v_ext)

        return Ein - Eex

    # finite element computation for displacement
    def displace(self, load, x, ke, kmin, penal, iter_):
        """
        FE solver based upon a Supernodal Sparse Cholesky Factorization. It
        requires the instalation of the cvx module. [1]_

        Parameters
        ----------
        load : object, child of the Loads class
            The loadcase(s) considerd for this optimisation problem.
        x : 2-D array size(nely, nelx)
            Current density distribution.
        ke : 2-D array size(8, 8)
            Local fully dense stiffnes matrix.
        kmin : 2-D array size(8, 8)
            Local stiffness matrix for an empty element.
        penal : float
            Material model penalisation (SIMP).

        Returns
        -------
        u : 1-D array len(max(edof))
            Displacement of all degrees of freedom
        """

        density = torch.tensor(x)**penal

        fixed_points = np.array(load.fixdofs())

        fixed_in_x_idx = torch.tensor(load.elpos(fixed_points[fixed_points % load.dim == 1]//2))
        fixed_in_y_idx = torch.tensor(load.elpos((fixed_points[fixed_points % load.dim == 0]-1)//2))

        print(fixed_in_x_idx)

        # h1_i(0,y) = u0_i(y)
        # h1_i(1,y) = u1_i(y)
        # h1_i(x,0) = v0_i(x)
        # h1_i(x,1) = v1_i(x)

        # boundary conditions for the x direction
        u_fixed = DirichletPINN_params(
            lambda y: 1 - torch.isin((y * (load.nely+1)).int(), (fixed_in_x_idx[fixed_in_x_idx[:, 0] == 0, 1]).int()).int(),
            lambda y: 1 - torch.isin((y * (load.nely+1)).int(), (fixed_in_x_idx[fixed_in_x_idx[:, 0] == 1, 1]).int()).int(),
            lambda x: 1 - torch.isin((x * (load.nely+1)).int(), (fixed_in_x_idx[fixed_in_x_idx[:, 1] == 0, 0]).int()).int(),
            lambda x: 1 - torch.isin((x * (load.nely+1)).int(), (fixed_in_x_idx[fixed_in_x_idx[:, 1] == 1, 0]).int()).int(),
        )
        v_fixed = DirichletPINN_params(
            lambda y: 1 - torch.isin((y * (load.nely+1)).int(), (fixed_in_y_idx[fixed_in_y_idx[:, 0] == 0, 1]).int()).int(),
            lambda y: 1 - torch.isin((y * (load.nely+1)).int(), (fixed_in_y_idx[fixed_in_y_idx[:, 0] == 1, 1]).int()).int(),
            lambda x: 1 - torch.isin((x * (load.nely+1)).int(), (fixed_in_y_idx[fixed_in_y_idx[:, 1] == 0, 0]).int()).int(),
            lambda x: 1 - torch.isin((x * (load.nely+1)).int(), (fixed_in_y_idx[fixed_in_y_idx[:, 1] == 1, 0]).int()).int(),
        )

        # test boundary conditions
        # x = torch.linspace(0, 1, 100).unsqueeze(1)
        # plt.plot(x, u_fixed.u0(x), label='u0')
        # plt.plot(x, u_fixed.u1(x), label='u1')
        # plt.plot(x, v_fixed.v0(x), label='v0')
        # plt.plot(x, v_fixed.v1(x), label='v1')
        # plt.xlabel('x')
        # plt.ylabel('Boundary condition')
        # plt.title('Boundary conditions')
        # plt.legend()
        # plt.show()

    
        #model = DirichletPINN([80,80,80,80,80,80,80,80], [u_fixed, v_fixed])
        model = DirichletPINN([80,80,80,80,80,80,80,80], [u_fixed, v_fixed])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))#self.learning_rate)
        # x = torch.linspace(0, 1, 100).unsqueeze(1).requires_grad_(True)
        # y = torch.linspace(0, 1, 100).unsqueeze(1).requires_grad_(True)
        # points = torch.cat([x, y], dim=1)
        
        free_points = np.array(load.freedofs())

        free_in_x_idx = free_points[free_points % load.dim == 0]
        free_in_y_idx = free_points[free_points % load.dim == 1]

        # Some points can be free in only one direction so they may not be in the
        # free_points array.
        all_free_points = np.unique(np.concat([free_in_x_idx, free_in_y_idx-1]))

        free_in_x = load.elpos(free_in_x_idx//2)
        free_in_y = load.elpos((free_in_y_idx-1)//2)
        force_arr = torch.tensor(np.array(load.force()))

        u = np.zeros(load.dim*(load.nely+1)*(load.nelx+1))
        points_x = torch.linspace(0, load.nelx, load.nelx+1)
        points_y = torch.linspace(0, load.nely, load.nely+1)
        grid_y, grid_x = torch.meshgrid(points_y, points_x, indexing='ij')
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()
        x = grid_x / (load.nelx + 1)
        y = grid_y / (load.nely + 1)

        best_loss = np.inf
        early_stopping_counter = 0
        for i in range(self.epochs):
            optimizer.zero_grad()
            points = torch.stack([x, y], dim=1).requires_grad_(True)
            
            output = model(points / torch.tensor([load.nelx+1, load.nely+1]))
            loss = self.calculate_energy_loss(load, points, output, free_in_x, free_in_y, force_arr, density)

            loss.backward()
            print(f"Epoch {i+1}/{self.epochs}, Loss: {loss.item()}")
            optimizer.step()

            if loss < best_loss:
                best_loss = loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

                if early_stopping_counter >= self.early_stopping:
                    print(f"Early stopping at epoch {i}")
                    break
        
        grid_x = np.array(grid_x.int().detach().numpy())
        grid_y = np.array(grid_y.int().detach().numpy())

        u[load.node(grid_x, grid_y)*2+1] = output[:, 0].detach().numpy()
        u[load.node(grid_x, grid_y)*2] = output[:, 1].detach().numpy()
        
        plot_mixed_arr(u, load, "Displacement field", iter_)

        return u


class CGFEA(FESolver):
    """
    This parent FEA class can assemble the global stiffness matrix and this
    class solves the FE problem with a sparse solver based upon a
    preconditioned conjugate gradient solver. The preconditioning is based
    upon the inverse of the diagonal of the stiffness matrix.

    Attributes
    ----------
    verbose : bool
        False if the FEA should not print updates.
    ufree_old : array len(freedofs)
        Displacement field of previous CG iteration
    """
    def __init__(self, verbose=False):
        super().__init__(verbose)
        self.ufree_old = None

    # finite element computation for displacement
    def displace(self, load, x, ke, kmin, penal, iter_):
        """
        FE solver based upon the sparse SciPy solver that uses a preconditioned
        conjugate gradient solver, preconditioning is based upon the inverse
        of the diagonal of the stiffness matrix. Currently the relative
        tolerance is hardcoded as 1e-3.

        Recomendations

        - Make the tolerance change over the iterations, low accuracy is
          required for first itteration, more accuracy for the later ones.
        - Add more advanced preconditioner.
        - Add gpu accerelation.

        Parameters
        ----------
        load : object, child of the Loads class
            The loadcase(s) considerd for this optimisation problem.
        x : 2-D array size(nely, nelx)
            Current density distribution.
        ke : 2-D array size(8, 8)
            Local fully dense stiffnes matrix.
        kmin : 2-D array size(8, 8)
            Local stiffness matrix for an empty element.
        penal : float
            Material model penalisation (SIMP).

        Returns
        -------
        u : 1-D array len(max(edof)+1)
            Displacement of all degrees of freedom
        """
        freedofs = np.array(load.freedofs())
        nely, nelx = x.shape

        f_free = load.force()[freedofs]
        k_free = self.gk_freedofs(load, x, ke, kmin, penal)

        # Preconditioning
        L = diags(1/k_free.diagonal())

        # solving the system f = Ku with cg implementation
        u = np.zeros(load.dim*(nely+1)*(nelx+1))
        u[freedofs], info = cg(k_free, f_free, x0=self.ufree_old, tol=1e-3, M=L)

        # update uold
        self.ufree_old = u[freedofs]

        if self.verbose is True:
            if info > 0:
                print('convergence to tolerance not achieved after ', info, ' itrations')
            if info < 0:
                print('Illegal input or breakdown ', info)

        return u


if __name__ == '__main__':
    example = DirichletPINN_params(
        lambda y: 0,
        lambda y: 0,
        lambda x: torch.sin(2 * np.pi * x),
        lambda x: torch.sin(2 * np.pi * x),
    )

    model = DirichletPINN([200], [example, example])
    x = torch.linspace(0, 1, 100).unsqueeze(1).requires_grad_(True)
    y = torch.linspace(0, 1, 100).unsqueeze(1).requires_grad_(True)
    points = torch.cat([x, y], dim=1)
    output = model(points)
    print(output)
# torch.autograd.grad(outputs=output, inputs=x, grad_outputs=torch.ones_like(output), create_graph=True)[0]

# x = torch.linspace(0, 1, 100).unsqueeze(1).requires_grad_(True)
# y = torch.linspace(0, 1, 100).unsqueeze(1).requires_grad_(True)
# points = torch.cat([x, y], dim=1)
# output = model(points)