"""
Basic SPH simulation code in Python.

Based on guidance by Philip Mocz @ https://pmocz.github.io/.
"""
import numpy as np

class SPH:
    def W(xs,ys,zs,h):
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
    
    def gradW(xs,ys,zs,h):
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


        






