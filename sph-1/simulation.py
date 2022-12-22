import numpy as np
from SPH import SPH
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio
import re
import os

FIG_SIZE = (4,5)
FIG_DPI =220
CMAP = mpl.colormaps['turbo']
XLIM1= (-1.4,1.4)
YLIM1= (-1.4,1.4)
ZLIM1= (-1.4,1.4)

XLIM2 = (0,1)
YLIM2 = (0,3)

FRAMETIME = 0.001

IMG_DIR = "./sims/"
GIF_DIR = "./anims/"

PLOT_SAMPLES = 100

Z_MIN_SIZE = 5
Z_MAX_SIZE = 25

BG_COLOUR = (1,1,1)

SPH = SPH()

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def png_to_vid(dir):
	images = []
	sorted_dir = os.listdir(dir)[:]
	sort_nicely(sorted_dir)
	for file_name in sorted_dir:
		if file_name.endswith('.png'):
			file_path = os.path.join(dir,file_name)
			images.append(np.array(imageio.imread(file_path)))
			print(f"Added frame: {file_path}")
	
	# Pause for 10 frames before loop
	for _ in range(20):
		images.append(imageio.imread(file_path))

	count = len(os.listdir(GIF_DIR))
	writer = imageio.get_writer(f'{GIF_DIR}/sim_{count+1}.mp4', fps=20)
	for frame in images:
		# print(frame.shape)
		writer.append_data(frame)
	writer.close()

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
		return 2*self.k*(1+self.n)*np.power(np.pi,(-3/(2*self.n))) * np.power(self.M*gamma(5/2+self.n)/np.power(self.R,3)/gamma(1+self.n),(1/self.n)) / (self.R*self.R)

	def get_mass(self):
		# Mass of each particle is uniformly distributed by star mass
		# Mass of star / Number of particles
		return self.M / self.N

class SPHSimulator:
	def __init__(self, params : Parameters):
		self.params : Parameters = params

		# number of timesteps
		self.Nt = int(np.ceil(params.tEnd/params.dt))

		self.m = params
	
	def main(self, pos, vs, save_anim=True):
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
		acc = SPH.get_accelaration(
				pos = pos, 
				vs = vs, 
				m = params.m, 
				h = params.h, 
				k = params.k, 
				n = params.n, 
				lmbda = params.lmbda, 
				nu = params.nu
			)
		
		# Pyplot initialization
		fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
		grid = plt.GridSpec(3,1, wspace=0, hspace=0.3)
		ax1 = plt.subplot(grid[0:2,0], projection='3d')
		ax2 = plt.subplot(grid[2,0])

		rr = np.zeros((PLOT_SAMPLES,3)) 	 # PLOT_SAMPLES x 3 matrix of zeros
		rlin = np.linspace(0,1,PLOT_SAMPLES) # PLOT_SAMPLES values between 0 and 1
		rr[:,0] = rlin # Set first column of rr to rlin

		# Analytical solution for density for reference line
		rho_analytic = (params.R*params.R - rlin*rlin) *  params.lmbda/(4*params.k)

		for i in range(self.Nt):
			# Half-step kick:
			# Velocity += accelaration times half the timestep.
			vs += acc * params.dt/2 

			# Drift
			pos += vs * params.dt

			# Update accelarations
			acc = SPH.get_accelaration(
				pos = pos, 
				vs = vs, 
				m = params.m, 
				h = params.h, 
				k = params.k, 
				n = params.n, 
				lmbda = params.lmbda, 
				nu = params.nu
			)

			# Half-step kick:
			vs += acc * params.dt / 2

			# Increment timestep
			params.t += params.dt
			# Size is inverse of z:
			zmin = pos[:,2].min()
			zmax = pos[:,2].max()

			zmid = (zmax-zmin)/2

			# Get densities for plotting
			rhos = SPH.get_density(pos, pos, params.m, params.h)

			# Real-time visualization
			if params.visualize or (i == self.Nt-1):
				plt.sca(ax1) # Set ax1 as new plot
				plt.cla() # Clear current plot

				cval = np.minimum((rhos-3)/3,1).flatten() # Color for each particle with their density, clip at 1.
				z_shifted = pos[:,2] - zmid
				z = pos[:,2]
				z_new_min = z.min()
				z_new_max = z.max()
				sizes = (((z - z_new_min) / (z_new_max - z_new_min) ) * (Z_MAX_SIZE - Z_MIN_SIZE) + Z_MIN_SIZE).flatten()
				ax1.scatter3D(pos[:,0],pos[:,1],pos[:,2],c=cval,cmap=CMAP, alpha=0.6, edgecolors='none')

				ax1.set(xlim=XLIM1,ylim=YLIM1,zlim=ZLIM1)
				ax1.set_aspect('auto','box')
				ax1.set_proj_type('persp')

				ax1.set_xticks([-1,0,1])
				ax1.set_yticks([-1,0,1])
				ax1.set_zticks([-1,0,1])

				ax1.set_facecolor('black') # Background colour
				ax1.set_facecolor(BG_COLOUR)
				ax1.set_title(f't={params.t:.2f}')

				# make the panes transparent
				ax1.xaxis.set_pane_color(BG_COLOUR)
				ax1.yaxis.set_pane_color(BG_COLOUR)
				ax1.zaxis.set_pane_color(BG_COLOUR)
				# make the grid lines transparent
				ax1.xaxis._axinfo["grid"]['color'] =  BG_COLOUR
				ax1.yaxis._axinfo["grid"]['color'] =  BG_COLOUR
				ax1.zaxis._axinfo["grid"]['color'] =  BG_COLOUR

				plt.sca(ax2)
				plt.cla()
				ax2.set(xlim=XLIM2,ylim=YLIM2)
				ax2.set_aspect(0.1)
				plt.plot(rlin, rho_analytic, color='gray', linewidth=2)

				# Plot labels and legend
				plt.sca(ax2)
				plt.xlabel('Radius')
				plt.ylabel('Density (radial)')

				rho_radial = SPH.get_density(rr, pos, params.m, params.h)

				plt.plot(rlin, rho_radial, color='blue')

				# Save fig
				plt.savefig(f'{IMG_DIR}{i}.png',dpi=240)
				plt.pause(FRAMETIME)
			
		if save_anim:
			png_to_vid(IMG_DIR)
			print("Saved animation as MP4.")

		# Show
		plt.show()

		return 0

if __name__ == "__main__":
	params = Parameters(
		N 	= 400,      		# Number of particles
		t 	= 0,      		# Initial time
		tEnd= 16,  			# End time
		dt 	= 0.04,  		# Time increment (timestep)
		M 	= 2,      	# Star mass
		R 	= 0.75,   		# Star radius
		h 	= 0.1,    		# Kernel smoothing length
		k 	= 0.1,    		# Equation of state constant
		n 	= 1,      		# Polytropic index
		nu 	= 1,     		# Damping coefficient (viscosity)
		visualize = True	# Enable realtime visualization
	)

	np.random.seed(42)

	# Cartesian coords matrix shape (Nx3 matrix)
	matrix_shape = (params.N,3)

	# Generate random positions for N particles 
	pos = np.random.randn(matrix_shape[0],matrix_shape[1])

	# All particles have zero initial velocity
	vs = np.zeros(matrix_shape)

	for pi in range(len(pos)):
		p_pos = pos[pi]

		if np.random.rand() < 0.5:
			velocity = np.array([p_pos[1], - p_pos[0], 0])
		else:
			velocity = np.array([- p_pos[1], p_pos[0], 0])

		vs[pi] = velocity * np.random.randint(1,50)

	simulator = SPHSimulator(params)
	simulator.main(pos, vs)



			

