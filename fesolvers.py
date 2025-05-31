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

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        k_free = self.gk_freedofs(load, x, ke, kmin, penal, iter_)

        # solving the system f = Ku with scipy
        u = np.zeros(load.dim*(nely+1)*(nelx+1))
        u[freedofs] = spsolve(k_free, f_free)

        return u

    # sparce stiffness matix assembly
    def gk_freedofs(self, load, x, ke, kmin, penal, iter_=None):
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

FixedBoundaries = namedtuple('FixedBoundaries', ['left', 'right', 'bottom', 'top'])
class DirichletPINN(nn.Module):
    def __init__(self, hidden_shape, arr_params, activation=nn.Sigmoid(), boundary_slope=20):
        super(DirichletPINN, self).__init__()
        self.nn_shape = [2, *hidden_shape, 2]
        self.nn_layers = nn.ModuleList()
        self.params = arr_params
     
        for i in range(len(self.nn_shape)-1):
            self.nn_layers.append(nn.Linear(self.nn_shape[i], self.nn_shape[i+1]))
            self.nn_layers.append(activation)

        # remove the last activation
        self.nn_layers = self.nn_layers[:-1]
        self.boundary_slope = boundary_slope

    def fix_boundary(self, x):
        return 2/(1 + torch.exp(-self.boundary_slope * x)) - 1

    
    def visualize_params(self):
        x = torch.linspace(0, 1, 100)
        y = torch.linspace(0, 1, 100)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)

        out = torch.zeros(points.shape[0], self.nn_shape[-1])
        for i in range(self.nn_shape[-1]):
            left, right, bottom, top = self.params[i]
            out[:, i] += 1
            if left:
                out[:, i] *= self.fix_boundary(points[:, 0])
            if right:
                out[:, i] *= self.fix_boundary(1 - points[:, 0])
            if bottom:
                out[:, i] *= self.fix_boundary(points[:, 1])
            if top:
                out[:, i] *= self.fix_boundary(1 - points[:, 1])
            
        
        plt.figure(figsize=(10, 5))
        for i in range(self.nn_shape[-1]):
            plt.subplot(1, self.nn_shape[-1], i+1)
            plt.imshow(out[:, i].reshape(X.shape).T.detach().cpu().numpy(), cmap='viridis', extent=(0, 1, 0, 1))
            plt.title(f'Output {i}')
            plt.colorbar()
        
        plt.tight_layout()
        plt.show()

    def forward_only_network(self, X):
        """
        Forward pass through the neural network without applying the boundary conditions.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, 2) where the first column
            is x and the second column is y.
        """
        model_out = X
        for layer in self.nn_layers:
            model_out = layer(model_out)
        return model_out

    
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
        model_out = self.forward_only_network(X)

        out = torch.zeros(X.shape[0], self.nn_shape[-1]).to(device=device)
        for i in range(self.nn_shape[-1]):
            left, right, bottom, top = self.params[i]
            out[:, i] = model_out[:, i]
            if left:
                out[:, i] *= self.fix_boundary(X[:, 0])
            if right:
                out[:, i] *= self.fix_boundary(1 - X[:, 0])
            if bottom:
                out[:, i] *= self.fix_boundary(1 - X[:, 1])
            if top:
                out[:, i] *= self.fix_boundary(X[:, 1])
        
        return out

def plot_mixed_arr(arr, load, title=""):
    idx = np.arange(arr.shape[0])
    arr1 = arr[idx % load.dim == 0].reshape((load.nelx+1, load.nely+1)).T
    arr2 = arr[idx % load.dim == 1].reshape((load.nelx+1, load.nely+1)).T

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(arr1, cmap='viridis')
    axs[0].set_title(f"{title} x-dim")
    axs[1].imshow(arr2, cmap='viridis')
    axs[1].set_title(f"{title} y-dim")
    for ax in axs.flat:
        fig.colorbar(ax.images[0], ax=ax, orientation='vertical')
    plt.tight_layout()
    plt.show()

def plot_displacement(arr, load, title="", iter_=None, gt=None):
    # Plot the mixed array of size (dim * nely+1 * nelx+1)
    # Reshape the array to (nely+1, nelx+1)
    idx = np.arange(arr.shape[0])
    arr1 = arr[idx % load.dim == 0].reshape((load.nelx+1, load.nely+1)).T
    arr2 = arr[idx % load.dim == 1].reshape((load.nelx+1, load.nely+1)).T

    arr1_gt = gt[idx % load.dim == 0].reshape((load.nelx+1, load.nely+1)).T
    arr2_gt = gt[idx % load.dim == 1].reshape((load.nelx+1, load.nely+1)).T

    fig, axs = plt.subplots(2, 3, figsize=(15, 5))

    axs[0, 0].imshow(arr1, cmap='viridis')
    axs[0, 0].set_title(f"{title} x-dim")
    
    axs[1,0].imshow(arr2, cmap='viridis')
    axs[1,0].set_title(f"{title} y-dim")
    axs[0,1].imshow(arr1_gt, cmap='viridis')
    axs[0,1].set_title(f"GT {title} x-dim")
    axs[1,1].imshow(arr2_gt, cmap='viridis')
    axs[1,1].set_title(f"GT {title} y-dim")
    axs[0,2].imshow(arr1 - arr1_gt, cmap='viridis')
    axs[0,2].set_title(f"Difference {title} x-dim")
    axs[1,2].imshow(arr2 - arr2_gt, cmap='viridis')
    axs[1,2].set_title(f"Difference {title} y-dim")

    for ax in axs.flat:
        fig.colorbar(ax.images[0], ax=ax, orientation='vertical')
    plt.tight_layout()
    fig.savefig(f"plots/{title}_iter_{iter_}.png")
    plt.close(fig)


    

class PiNN_FEA(FESolver):
    """
    This parent FEA class is used to assemble the global stiffness matrix while
    this class solves the FE Energy based physics informed neural network (PINN).

    Attributes
    --------
    verbose : bool
        False if the FEA should not print updates.
    """
    def __init__(self, verbose=False, epochs=4000, learning_rate=0.0001, early_stopping=500, gt_solver=None, Emin=1e-9, plotting=False):
        super().__init__(verbose)

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.gt_solver = gt_solver
        self.Emin = Emin
        
        if self.gt_solver is None:
            self.gt_solver = FESolver(verbose=verbose)

        self.plotting = plotting
        if self.plotting:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            self.imshowU = axs[0].imshow(np.zeros((100+1, 100+1)), cmap='viridis')
            axs[0].set_title(f"u")
            self.imshowV = axs[1].imshow(np.zeros((100+1, 100+1)), cmap='viridis')
            axs[1].set_title(f"v")
            plt.show(block=False)
        
    def plot_uv_lattice_non_block(self, arr, load):
        if not self.plotting:
            return
        
        idx = np.arange(arr.shape[0])
        arr1 = arr[idx % load.dim == 0].reshape((load.nelx+1, load.nely+1)).T
        arr2 = arr[idx % load.dim == 1].reshape((load.nelx+1, load.nely+1)).T
        
        self.imshowU.set_data(arr1)
        self.imshowV.set_data(arr2)

        self.imshowU.autoscale()
        self.imshowV.autoscale()

        plt.pause(0.0001)
    
    def calculate_energy_loss(self, load, points, output, ext_forces, density):
        """
        The loss function is the potential energy of the system.

        ARRAYS ARE FLATTENED USING THE FORTRAN COLUMN MAJOR ORDERING!!!!
        
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
        u = torch.tensor(np.zeros((load.nely+1, load.nelx+1)), dtype=torch.float32).to(device=device)
        v = torch.tensor(np.zeros((load.nely+1, load.nelx+1)), dtype=torch.float32).to(device=device)

        points_int = points.int().to(device=device)

        u[points_int[:, 1], points_int[:, 0]] = output[:, 0]
        v[points_int[:, 1], points_int[:, 0]] = output[:, 1]

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

        # Max pooling on the density matrix
        node_density_matrix = F.max_pool2d(density.unsqueeze(0), kernel_size=2, stride=1, padding=1)[0].to(device=device)
        density_list = node_density_matrix[points_int[:, 1], points_int[:, 0]]

        # SIMP penalisation
        E = self.Emin + (load.young - self.Emin) * density_list # density WAS ALREADY PENALIZED

        nu = load.poisson
        lambda_ = E * nu / ((1 + nu)*(1-nu))
        mu = E / (2 * (1 + nu))
        
        # PINNTO
        sigma_xx = lambda_ * (eps_xx + eps_yy) + 2 * mu * eps_xx
        sigma_yy = lambda_ * (eps_xx + eps_yy) + 2 * mu * eps_yy
        sigma_xy = 2 * mu * eps_xy

        # Calculate the internal strain energy
        Ein = (0.5 * (sigma_xx * eps_xx + sigma_yy * eps_yy + 2*sigma_xy * eps_xy)).sum()

        uv_lattice = torch.zeros(load.dim*(load.nely+1)*(load.nelx+1)).to(device=device)
        idxs = np.array(range((load.nely+1)*(load.nelx+1)))
        uv_lattice[idxs*2] = u.T.reshape(-1)
        uv_lattice[idxs*2+1] = v.T.reshape(-1)

        if self.plotting:
            with torch.no_grad():
                self.plot_uv_lattice_non_block(uv_lattice.cpu(), load)

        ext_forces = ext_forces.to(device=device)
        
        Eex = (ext_forces * uv_lattice).sum()

        return Ein - Eex

    # finite element computation for displacement
    def displace(self, load, density_orig, ke, kmin, penal, iter_):
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

        density = torch.tensor(density_orig)**penal

        u_fixed = FixedBoundaries(left=True, right=False, bottom=False, top=False)
        v_fixed = FixedBoundaries(left=True, right=False, bottom=False, top=False)

    
        #model = DirichletPINN([80,80,80,80,80,80,80,80], [u_fixed, v_fixed])
        model = DirichletPINN([80,80,80,80,80,80,80,80], [u_fixed, v_fixed], activation=nn.Tanh()).to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        # x = torch.linspace(0, 1, 100).unsqueeze(1).requires_grad_(True)
        # y = torch.linspace(0, 1, 100).unsqueeze(1).requires_grad_(True)
        # points = torch.cat([x, y], dim=1)
        
        force_arr = torch.tensor(np.array(load.force()))

        points_x = torch.linspace(0, load.nelx, load.nelx+1)
        points_y = torch.linspace(0, load.nely, load.nely+1)
        grid_y, grid_x = torch.meshgrid(points_y, points_x, indexing='ij')
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()

        best_loss = np.inf
        early_stopping_counter = 0

        random_sample_n=128

        diff_plot = plt.plot([], [], label='Max abs ground truth diff')[0]
        loss_plot = plt.plot([], [], label='Loss')[0]
        plt.legend()
        plt.show(block=False)
        loss_history = []
        diff_history = []

        u_gt = self.gt_solver.displace(load, density_orig, ke, kmin, penal, iter_)

        progress_bar = tqdm(range(self.epochs), desc="Training Progress", unit="epoch")
        for i in progress_bar:
            optimizer.zero_grad()

            points = torch.stack([grid_x, grid_y], dim=1).requires_grad_(True).to(device=device)

            # Randomly sample points on the unit square
            #points = (torch.rand((random_sample_n, 2))*torch.tensor([load.nelx+1, load.nely+1])).requires_grad_(True).to(device=device)
            
            output = model(points / torch.tensor([load.nelx+1, load.nely+1]).to(device=device)).to(device=device)
            loss = self.calculate_energy_loss(load, points, output, force_arr, density)

            loss.backward()

            optimizer.step()

            if loss < best_loss:
                best_loss = loss.item()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

                if early_stopping_counter >= self.early_stopping:
                    print(f"Early stopping at epoch {i}")
                    break

            u = np.zeros(load.dim*(load.nely+1)*(load.nelx+1))
            grid_x_int = np.array(points[:, 0].cpu().int().detach().numpy())
            grid_y_int = np.array(points[:, 1].cpu().int().detach().numpy())
            u[load.node(grid_x_int, grid_y_int)*2] = output[:, 0].detach().cpu().numpy()
            u[load.node(grid_x_int, grid_y_int)*2+1] = output[:, 1].detach().cpu().numpy()
            progress_bar.set_postfix({"abs(u - gt)":np.abs(u - u_gt).max(), "Loss": f"{loss.item():.10f}", "Best loss": f"{best_loss:.10f}"})

            loss_history.append(loss.item())
            diff_history.append(np.abs(u - u_gt).max())

            if i % 100 == 0:
                loss_plot.set_xdata(np.arange(len(loss_history)))
                loss_plot.set_ydata(loss_history)

                diff_plot.set_xdata(np.arange(len(diff_history)))
                diff_plot.set_ydata(diff_history)

                plt.xlim(0, len(loss_history))
                plt.ylim(min(loss_history), max(diff_history))
                plt.pause(0.0001)
        
        points = torch.stack([grid_x, grid_y], dim=1).requires_grad_(True).to(device=device)
        output = model(points / torch.tensor([load.nelx+1, load.nely+1]).to(device=device)).to(device=device)

        u = np.zeros(load.dim*(load.nely+1)*(load.nelx+1))
        
        grid_x = np.array(grid_x.int().detach().numpy())
        grid_y = np.array(grid_y.int().detach().numpy())

        u[load.node(grid_x, grid_y)*2] = output[:, 0].detach().cpu().numpy()
        u[load.node(grid_x, grid_y)*2+1] = output[:, 1].detach().cpu().numpy()

        plot_displacement(u, load, "Displacement field", iter_, gt=u_gt)

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
