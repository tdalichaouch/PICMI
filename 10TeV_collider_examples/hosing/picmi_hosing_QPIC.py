
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os, sys, shutil
import importlib
import json

sys.path.append(os.path.abspath('../../picmi_scripts'))
from UTILITY_QUICKPIC import cst  # importing constants
#################### MODIFY THESE PARMATERS ####################

#################### simulation reference density ####################
n0 = 1e16 * 1e6 # number density in units of 1/m^3
wp = np.sqrt(cst.q_e**2 * n0/(cst.ep0 * cst.m_e))
kp = wp/cst.c
######################################################################

#################### grid params (in units of c/wp) ####################
zmin,zmax = -6.6,4.62
xmin, xmax = -9.395847, 9.395847
ymin, ymax = -9.395847, 9.395847

nx, ny, nz = 512, 512, 512
#################### time step params (in units of wp^-1) ####################
dt_qpic = 10
ndump_diag = 10


#################### MPI_config ####################
QPIC_nodes = [32,4] # QPAD cpu split

#################### drive beam params ####################
beam1_charge = 3.193e-9 # charge [C]
beam1_sigmas = [7.285e-6, 7.285e-6, 2.549e-5] # sigmas [m]
beam1_centroid_position = [0,0,0] # centroid at (0,0,0) 
beam1_macroparticles = 2e6

#################### trailing beam (#2) params ####################
beam2_charge = 0.956e-9 # charge [C]
beam2_sigmas = [7.285e-6, 7.285e-6, 1.274e-05] #sigmas [m]
beam2_centroid_position = [2e-6, 0, -3e-4] # offset 2 micron in x, behind driver by 300 um
beam2_macroparticles = 1e6

#################### common beam params (shared by both beams) ####################
gamma = 19569.47 # Energy [10 GeV] 

# matched condition
beta, alpha = np.sqrt(2 * gamma)/kp, 0 
rms_vel = [beam1_sigmas[0] * gamma/beta, beam1_sigmas[1] * gamma/beta, 0] 




# Pre-ionized He gas profile
n_plasma = n0
z = np.array([0,2]) # distance in meters
gas_density = np.array([n0, n0]) # gas density profile as a func of z
ppc_plasma = [2,2] # (2,2) ppc in (r,z)


# PATH to QuickPIC executables (qpic.e)
path_to_qpic = 'dd'  
assert path_to_qpic, Exception('Need to specify path_to_qpic')
#################### END OF MODIFICATION SECTION ####################

#################### mkdir sim folder ####################
sim_dir= 'hosing_sims/QPIC_sim'
os.makedirs(sim_dir, exist_ok = True)




#################### Run QPAD (sim #1) ####################

def run_qpad_sim1(z, gas_density):
	print('Constructing QuickPIC Hosing simulation')
	from UTILITY_QUICKPIC import QUICKPIC_sim
	sim1 = QUICKPIC_sim(n0)
	kp = sim1.kp
	wp = sim1.wp
	sim1.init_grid(nx = nx, ny= ny, nz = nz, xmin = xmin/kp, xmax = xmax/kp, ymin = ymin/kp, ymax = ymax/kp, zmin = zmin/kp, zmax = zmax/kp)

	# add drive bunch
	sim1.add_gaussian_electron_bunch(beam1_charge, beam1_sigmas, n_macroparticles = beam1_macroparticles, \
		bunch_rms_velocity = rms_vel, bunch_centroid_velocity = [0,0,gamma], alpha = alpha)
	# add trailing bunch
	sim1.add_gaussian_electron_bunch(beam2_charge, beam2_sigmas, n_macroparticles = beam2_macroparticles,  \
		bunch_rms_velocity = rms_vel, bunch_centroid_velocity = [0,0,gamma], bunch_centroid_position = beam2_centroid_position,\
									 alpha = alpha)

	# add pre-ionized plasma
	sim1.add_longitudinal_plasma(z, gas_density, ppc=ppc_plasma) # only consider first level of Hydrogen
	
	# QPAD diag info
	sim1.add_field_diagnostics(data_list = ['Ex', 'Ez', 'rho'], period = ndump_diag, slice_ind = 256) # 
	sim1.add_particle_diagnostics(period = ndump_diag, psample =1) # sample every 10 particles

	zsim1 = 20000.1/kp # 1 meter

	print('Running 1st leg QuickPIC sim (output in ' + sim_dir+ '/output.txt)...')
	sim1.run_simulation(dt = dt_qpic/wp, tmax = (zsim1 + 0.01)/cst.c, nodes = QPIC_nodes, sim_dir = sim_dir, path_to_qpic = path_to_qpic)
	
print()
run_qpad_sim1(z,gas_density)




