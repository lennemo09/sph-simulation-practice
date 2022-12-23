import numpy as np
from SPH import SPH
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio
import re
import os
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from custom_cmap import astro_heat

FIG_SIZE = (7,7)
FIG_DPI =220
CMAP = astro_heat
XLIM1= (-1.4,1.4)
YLIM1= (-1.4,1.4)
ZLIM1= (-1.4,1.4)

XLIM2 = (0,1)
YLIM2 = (0,5)

P_x_samples, P_y_samples, P_z_samples = (25, 25, 25)

FRAMETIME = 0.001

IMG_DIR = "./sims/"
GIF_DIR = "./anims/"

PLOT_SAMPLES = 100

Z_MIN_SIZE = 5
Z_MAX_SIZE = 50

BG_COLOUR = (.1,.1,.1)
ELEMENT_COLOUR = (1,1,1)


mpl.rcParams['text.color'] = ELEMENT_COLOUR
mpl.rcParams['axes.labelcolor'] = ELEMENT_COLOUR
mpl.rcParams['xtick.color'] = ELEMENT_COLOUR
mpl.rcParams['ytick.color'] = ELEMENT_COLOUR
mpl.rcParams['axes.edgecolor'] = ELEMENT_COLOUR

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
	
	for frame in images[1:]:
		writer.append_data(frame)
	writer.close()

class Parameters:
	def __init__(self,
		N 	= 400,      		# Number of particles
		t 	= 0,      		# Initial time
		tEnd= 18,  			# End time
		dt 	= 0.04,  		# Time increment (timestep)
		M 	= 2,      	# Star mass
		R 	= 0.75,   		# Star radius
		h 	= 0.1,    		# Kernel smoothing length
		k 	= 0.1,    		# Equation of state constant
		n 	= 1,      		# Polytropic index
		nu 	= 1,     		# Damping coefficient (viscosity)
		visualize = True	# Enable realtime visualization
	):
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
		fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI, facecolor=BG_COLOUR)
		fig.suptitle(r'Star formation with $N$={} particles (coloured by particle density $\rho_i$){}$\Delta t$={}; $\nu$={}; $h$={}'.format(params.N,'\n',params.dt,params.nu,params.h))
		grid = plt.GridSpec(3,2, wspace=0.6, hspace=0.6)
		ax1 = plt.subplot(grid[0:2,0], projection='3d') # 3D view
		ax2 = plt.subplot(grid[2,0]) # Radial density plot
		ax3 = plt.subplot(grid[2,1]) # 2D view
		ax4 = plt.subplot(grid[0:2,1], projection='3d') # Pressure plot

		plot2 = ax3.scatter([0,1],[1,0],s=1,c=[0,1],cmap=CMAP, alpha=1, edgecolors='none')
		divider = make_axes_locatable(ax3)
		cax = divider.append_axes('right', size=0.05, pad=0.05)
		cbar = fig.colorbar(plot2, cax=cax, orientation='vertical')
		cbar.set_ticks([])

		rr = np.zeros((PLOT_SAMPLES,3)) 	 # PLOT_SAMPLES x 3 matrix of zeros
		rlin = np.linspace(0,1,PLOT_SAMPLES) # PLOT_SAMPLES values between 0 and 1
		rr[:,0] = rlin # Set first column of rr to rlin

		# Points to sample pressure
		px = np.linspace(-1, 1, P_x_samples)
		py = np.linspace(-1, 1, P_y_samples)
		pz = np.linspace(-1, 1, P_z_samples)
		pxv, pyv, pzv = np.meshgrid(px, py, pz)
		pressure_samples = np.vstack([pxv.ravel(), pyv.ravel(), pzv.ravel()]).T

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
				ax1.set_box_aspect([1,1,1])
				ax1.set_proj_type('ortho')

				ax1.set_xticks([-1,0,1])
				ax1.set_yticks([-1,0,1])
				ax1.set_zticks([-1,0,1])

				ax1.set_facecolor('black') # Background colour
				ax1.set_facecolor(BG_COLOUR)
				ax1.set_title(r'$t$={:.2f}'.format(params.t))

				# make the panes transparent
				ax1.xaxis.set_pane_color(BG_COLOUR)
				ax1.yaxis.set_pane_color(BG_COLOUR)
				ax1.zaxis.set_pane_color(BG_COLOUR)
				# make the grid lines transparent
				ax1.xaxis._axinfo["grid"]['color'] =  BG_COLOUR
				ax1.yaxis._axinfo["grid"]['color'] =  BG_COLOUR
				ax1.zaxis._axinfo["grid"]['color'] =  BG_COLOUR
				
				plt.sca(ax3)
				plt.cla()
			
				ax3.scatter(pos[:,0],pos[:,1],s=sizes,c=cval,cmap=CMAP, alpha=0.6, edgecolors='none')
				ax3.set_aspect('equal','box')
				ax3.set(xlim=XLIM1,ylim=ZLIM1)
				ax3.set_xticks([-1,0,1])
				ax3.set_yticks([-1,0,1])
				ax3.set_facecolor('black')
				ax3.set_facecolor(BG_COLOUR)
				ax3.set_title(r'$(x,y)$ view (line-of-sight along $z$)')

				plt.sca(ax2)
				plt.cla()
				ax2.set(xlim=XLIM2,ylim=YLIM2)
				ax2.set_aspect(0.1)
				ax2.plot(rlin, rho_analytic, color='#000969', linewidth=2,linestyle='dashed')
				ax2.set_facecolor('black') # Background colour
				ax2.set_facecolor(BG_COLOUR)

				# Plot labels and legend
				ax2.set_xlabel('Radius')
				ax2.set_ylabel('Density (radial)')

				rho_radial = SPH.get_density(rr, pos, params.m, params.h)

				ax2.plot(rlin, rho_radial, color='#ff1fda')

				# Pressure plot
				plt.sca(ax4)
				plt.cla()
				rhos_grid_sample = SPH.get_density(pressure_samples, pos, params.m, params.h)
				pressures = SPH.get_pressure(rhos_grid_sample, params.k, params.n)
				pressures_in_grid = np.copy(pressure_samples)
				pressures_in_grid[:,2] = pressures.ravel()
				pressures_df = pd.DataFrame(pressures_in_grid, columns = ['x','y','P'])
				pressures_df = pressures_df.groupby(['x','y'])['P'].sum().reset_index().to_numpy()

				ax4.plot_trisurf(pressures_df[:,0],pressures_df[:,1],pressures_df[:,2], cmap=CMAP, linewidth=0, antialiased=True)
				ax4.set(xlim=XLIM1,ylim=YLIM1,zlim=(0,2))
				ax4.set_box_aspect([1,1,1])
				ax4.set_proj_type('ortho')

				ax4.set_xticks([-1,0,1])
				ax4.set_yticks([-1,0,1])
				ax4.set_zticks([])

				ax4.set_facecolor('black') # Background colour
				ax4.set_facecolor(BG_COLOUR)
				ax4.set_title(r'Total pressure over{}$(x,y)$ plane: $\Sigma_z\rho$'.format('\n'))

				# make the panes transparent
				ax4.xaxis.set_pane_color(BG_COLOUR)
				ax4.yaxis.set_pane_color(BG_COLOUR)
				ax4.zaxis.set_pane_color(BG_COLOUR)
				# make the grid lines transparent
				ax4.xaxis._axinfo["grid"]['color'] =  BG_COLOUR
				ax4.yaxis._axinfo["grid"]['color'] =  BG_COLOUR
				ax4.zaxis._axinfo["grid"]['color'] =  BG_COLOUR

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
		N 	= 1200,      		# Number of particles
		t 	= 0,      		# Initial time
		tEnd= 28,  			# End time
		dt 	= 0.04,  		# Time increment (timestep)
		M 	= 2,      	# Star mass
		R 	= 1,   		# Star radius
		h 	= 0.2,    		# Kernel smoothing length
		k 	= 0.1,    		# Equation of state constant
		n 	= 1,      		# Polytropic index
		nu 	= 0.5,     		# Damping coefficient (viscosity)
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

		# if np.random.rand() < 0.5:
		# 	velocity = np.array([p_pos[1], - p_pos[0], 0])
		# else:
		# 	velocity = np.array([- p_pos[1], p_pos[0], 0])
		velocity = np.array([p_pos[1], - p_pos[0], 0])
		vs[pi] = velocity * np.random.randint(10,25)

	simulator = SPHSimulator(params)
	simulator.main(pos, vs)



			

