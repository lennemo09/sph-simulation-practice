import numpy as np
from SPH import SPH
from scipy.special import gamma

class Parameters:
    def __init__(self, N, t, tEnd, dt, M, R, h, k, n, nu, visualize=True):
        self.N = N
        self.t = t
        self.tEnd = tEnd
        self.dt = dt
        self.M = M
        self.R = R
        self.h = h
        self.k = k
        self.n = n
        self.nu = nu
        self.visualize = visualize
        self.lmbda = self.get_lambda()
        self.m = self.get_mass()

    def get_lambda(self):
        return 2*self.k*(1+self.n)*np.power(np.pi,(-3/(2*self.n))) * (self.M*gamma(5/2+self.n)/np.power(self.R,3)/np.power(gamma(1+self.n)),(1/self.n)) / (self.R*self.R)

    def get_mass(self):
        return self.M / self.N

class SPHSimulator:
    def __init__(self, params : Parameters):
        self.params : Parameters = params

        # number of timesteps
        self.Nt = int(np.ceil(params.tEnd/params.dt))

        self.m = params
    
    def main(self, pos, vs):
        """
        Main loop of the simulation.
        
        The positions and velocities are updated using a leap-frog scheme ('kick-drift-kick'). 
        For each timestep Δt, each particle receives a half-step 'kick':
            vi = vi + ai * dt/2
        Followed by a full-step drift:
            ri = ri + dt * vi
        And finally another half-step kick.

        pos : matrix of positions of all particles.
        vs  : matrix of velocities of all particles.
        """
        params = self.params
        # Initial accelarations (gravitational)
        acc = SPH.get_accelaration(pos, vs, params.m, params.h, params.k, params.n, params.lmbda, params.nu)
        for i in range(self.Nt):
            # Half-step kick:
            # Velocity += accelaration times half the timestep.
            vs += acc * params.dt/2 

            # Drift
            pos += vs * params.dt

            # Update accelarations
            acc = SPH.get_accelaration(pos, vs, params.m, params.h, params.k, params.n, params.lmbda, params.nu)

            # Half-step kick:
            vs += acc * params.dt / 2

            # Increment timestep
            params.t += params.dt
            

