import sys
import os 
import numpy as np
import h5py
import matplotlib.pyplot as plt
import shutil
import importlib
import json

sys.path.append(os.path.abspath('../../picmi_scripts'))
from UTILITY_QPAD import cst  # importing constants

#################### simulation reference density ####################
n0 = 1e16 * 1e6 # number density in units of 1/m^3
wp = np.sqrt(cst.q_e**2 * n0/(cst.ep0 * cst.m_e))
kp = wp/cst.c


#################### grid params (in units of c/wp) ####################
zmin,zmax = -6.6, 3
rmin, rmax = 0, 9.395847
nr, nz = 256, 438
n_modes = 1


#################### time step params (in units of 1/wp) ####################
dt_qpad = 10
ndump_diag = 10 # dump every 10 timsteps


#################### MPI_config ####################
QPAD_nodes = [8,16] # QPAD cpu split


#################### drive beam params ####################
beam1_charge = 3.193e-9 # charge [C]
beam1_sigmas = [7.285e-6, 7.285e-6, 2.549e-5] # sigmas [m]
beam1_centroid_position = [0,0,0] # centroid at (0,0,0) 


#################### trailing beam (#2) params ####################
beam2_charge = 0.956e-9 # charge [C]
beam2_sigmas = [7.285e-6, 7.285e-6, 1.274e-05] #sigmas [m]
beam2_centroid_position = [2e-6, 0, -3e-4] # offset 2 micron in x, behind driver by 300 um


#################### common beam params (shared by both beams) ####################
gamma = 19569.47 # Energy [10 GeV] 

# matched condition
beta, alpha = np.sqrt(2 * gamma)/kp, 0 
rms_vel = [beam1_sigmas[0] * gamma/beta, beam1_sigmas[1] * gamma/beta, 0] 

# ppc of the beams
ppc_beam, num_theta_beam = [2, 1, 2], 8  # (2,2,8) ppc along (r,z,phi)


#################### Plasma profile ####################
z = np.array([0,2]) # distance in meters
gas_density = np.array([n0, n0]) # gas density profile as a func of z
ppc_plasma, num_theta_plasma  = [4, 1], 8 # 4 ppc in r, 8 in phi


# PATH to QPAD executables (qpad.e)
path_to_qpad = 'd'  
assert path_to_qpad, Exception('Need to specify path_to_qpad')

#################### mkdir sim folder ####################
sim_dir= 'hosing_sims/QPAD_sim'
os.makedirs(sim_dir, exist_ok = True)


#################### Run QPAD (sim #1) ####################
def run_qpad_sim1(z, gas_density):
	print('Constructing QPAD Hosing simulation')
	from UTILITY_QPAD import QPAD_sim
	sim1 = QPAD_sim(n0)
	kp, wp = sim1.kp, sim1.wp 
	sim1.init_grid(nr = nr, nz = nz, rmin = 0, rmax = rmax/kp, zmin = zmin/kp, zmax = zmax/kp, n_modes = n_modes)

	# add drive bunch
	sim1.add_gaussian_electron_bunch(beam1_charge, beam1_sigmas, ppc = ppc_beam, num_theta = num_theta_beam, \
		bunch_rms_velocity = rms_vel, bunch_centroid_velocity = [0, 0, gamma], alpha = alpha)
	# add trailing bunch
	sim1.add_gaussian_electron_bunch(beam2_charge, beam2_sigmas,ppc = ppc_beam,  num_theta = num_theta_beam, \
		bunch_rms_velocity = rms_vel, bunch_centroid_velocity = [0, 0, gamma], bunch_centroid_position = beam2_centroid_position,\
									 alpha = alpha)

	# add pre-ionized plasma
	sim1.add_longitudinal_plasma(z, gas_density, ppc=ppc_plasma, num_theta = num_theta_plasma) # only consider first level of Hydrogen
	
	# QPAD diag info
	sim1.add_field_diagnostics(data_list = ['Ez', 'Er', 'Bphi', 'rho'], period = ndump_diag) # 
	sim1.add_particle_diagnostics(period = ndump_diag, psample =1) # sample every particle

	zsim1 = 20000.1/kp # 1 meter

	print('Running 1st leg QPAD sim (output in ' + sim_dir+ '/output.txt)...')
	sim1.run_simulation(dt = dt_qpad/wp, tmax = (zsim1 + 0.01)/cst.c, nodes = QPAD_nodes, sim_dir = sim_dir, path_to_qpad = path_to_qpad)

run_qpad_sim1(z,gas_density)



