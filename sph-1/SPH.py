"""
Basic SPH simulation code in Python.

Based on guidance by Philip Mocz @ https://pmocz.github.io/.
"""
import numpy as np

class SPH:
    def W(self,xs,ys,zs,h):
        """
        3D Gaussian Smoothing Kernel for constant smoothing length
        xs : matrix of x positions for all particles
        ys : matrix of y positions for all particles
        zs : matrix of z positions for all particles
        h  : smoothing length
        """
        # Magnitude of position vectors
        r = np.sqrt(xs*xs + ys*ys + zs*zs)

        # Evaluated smoothing function
        w = np.power((1 / h * (np.sqrt(np.pi))),3) * np.exp(-(r*r)/(h*h))
    
        return w
    
    def grad_W(self,xs,ys,zs,h):
        """
        Gradient of 3D Gaussian Smoothing Kernel for constant smoothing length
        xs : matrix of x positions for all particles
        ys : matrix of y positions for all particles
        zs : matrix of z positions for all particles
        h  : smoothing length
        """
        # Magnitude of position vectors
        r = np.sqrt(xs*xs + ys*ys + zs*zs)

        n = -2 * np.exp(-(r*r)/(h*h)) / np.power(h,5) / np.power(np.pi,1.5)

        # Evaluated gradient in 3 dimensions
        wxs = n * xs
        wys = n * ys
        wzs = n * zs

        return wxs, wys, wzs

    def pairwise_seperations(self,ri : np.ndarray, rj : np.ndarray):
        """
        Vectorized computation of pairwise distances between points.

        ris : M x 3 matrix of positions
        rjs : M x 3 matrix of positions
        """
        M = ri.shape(0)
        N = rj.shape(0)

        # Positions ri = (x,y,z)
        rix = ri[:,0].reshape((M,1))
        riy = ri[:,1].reshape((M,1))
        riz = ri[:,2].reshape((M,1))

        # Positions rj = (x,y,z)
        rjx = rj[:,0].reshape((N,1))
        rjy = rj[:,1].reshape((N,1))
        rjz = rj[:,2].reshape((N,1))

        # Matrices to store pairwise particle separations: ri - rj
        # M x N matrices of separations
        dxs = rix - rjx.T
        dys = riy - rjy.T
        dzs = riz - rjz.T

        return dxs, dys, dzs

    def get_density(self,rs : np.ndarray, pos : np.ndarray, m):
        """
        Get density at sampled locations from SPH particle distributions.
        r  : M x 3 matrix of sampled locations
        pos: N x 3 matrix of SPH particle current positions
        m  : mass of particle
        h  : smoothing length
        """
        M = rs.shape[0] # Number of locations
        dx, dy, dz = self.pairwise_seperations(rs, pos)

        # M x 1 vector of accelarations
        rho = np.sum(m * self.W(dx, dy, dz, h), 1).reshape((M,1))

        return rho

    def get_pressure(self,rho : np.ndarray, k,n):
        """
        Pressure from density using the equation of state.

        rho: vector of densities
        k  : equation of state constant
        n  : polytropic index
        p  : pressure
        """
        P = k * np.power(rho, 1 + 1/n)

        return P

    def get_accelaration(self,pos : np.ndarray, vs : np.ndarray, m, h, k ,n, lmbda, nu):
        """
        Calculate accelaration on each SPH particles.

        pos  : N x 3 matrix of positions
        vs   : N x 3 matrix of velocities
        m    : particle mass
        h    : smoothing length
        k    : equation of state constant
        n    : polytropic index
        lmbda: external force constant
        nu   : fluid viscosity
        a    : N x 3 matrix of accelerations
        """
        N = pos.shape[0]

        # Get densitites at the positions of the particles
        rho = self.get_density(pos, pos, m, h)

        # Get the pressures
        P = self.get_pressure(rho, k, n)

        # Get pairwise distances and gradients
        dx, dy, dz = self.pairwise_seperations(pos, pos)
        dWx, dWy, dWz = self.grad_W(dx, dy, dz, h)

        # Add Pressure contribution to accelarations
        axs = -np.sum(m * (np.power(P/rho,2) + P.T/np.power(rho.T,2)) * dWx, 1).reshape((N,1))
        ays = -np.sum(m * (np.power(P/rho,2) + P.T/np.power(rho.T,2)) * dWy, 1).reshape((N,1))
        azs = -np.sum(m * (np.power(P/rho,2) + P.T/np.power(rho.T,2)) * dWz, 1).reshape((N,1))

        # Stack accelaration components into a matrix
        a : np.ndarray = np.hstack((axs,ays,azs))

        # Add external potential force and viscosity
        a : np.ndarray = a + (lmbda * pos - nu * vs)

        return a

    






        






