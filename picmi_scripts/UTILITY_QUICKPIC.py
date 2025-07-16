
import picmi_quickpic as picmi
import numpy as np
from pmd_beamphysics import ParticleGroup
import h5py
from scipy.optimize import fsolve
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.constants import physical_constants
import subprocess, os 
import importlib
cst = picmi.constants

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


class QUICKPIC_sim:


	"""
	Constructor

	Parameters
	----------
	n0: float
		Normalizing Density in units of m^{-3}

	"""
	def __init__(self, n0 = 1e17 * 1e6):
		self.n0 = n0 
		self.wp = np.sqrt(cst.q_e**2 * self.n0/(cst.ep0 * cst.m_e))
		self.kp = self.wp/cst.c

		self.P = None
		self.layouts, self.species_list = [],[]
		self.simulation, self.solver, self.grid = None, None, None
		self.if_beam, self.part_diags, self.field_diags = [], [], []
		importlib.reload(picmi)
		

	"""
	Initialize Grid Paramters

	Parameters
	----------
	nx, ny,  nz: integer
		Number of grid cells along x, y and z, respectfully. 

	zmin, zmax: float
		Grid bounds along z.

	xmin, xmax: float
		Grid bounds along x. 

	ymin, ymax: float
		Grid bounds along y. 

		
	"""
	def init_grid(self,nx = None, ny = None, nz = None, zmin = None, zmax = None, xmin = 0, xmax = None, ymin = None, ymax = None):
		self.grid = picmi.Cartesian3DGrid(
			number_of_cells           = [nx, ny, nz],
			lower_bound               = [xmin, ymin, zmin],
			upper_bound               = [xmax, ymax, zmax],
			lower_boundary_conditions = ['dirichlet', 'dirichlet', 'open'],
			upper_boundary_conditions = ['dirichlet', 'dirichlet', 'open'],
			moving_window_velocity    = [0, 0, cst.c])


		solver_dict = {}
		self.solver = picmi.ElectromagneticSolver( grid = self.grid , **solver_dict)

	"""
	Add QPAD Beam File

	Parameters
	----------
	qpad_file_in: string 
		path to qpad file

	qpic_file_out:string
		path to written file for QuickPIC
	
	scale_q_fac: float, default = 1.0
		factor to scale charge q (generally required when importing from QPAD)

	offset: vector of length 3 of floats, default = [0, 0, 0]
		amount to shift macroparticle positions
		
	"""
	def add_qpad_file_bunch(self, qpad_file_in, qpic_file_out, scale_q_fac =1.0, directory = '.', z_select=False, offset = [0, 0, 0]):
		assert self.grid is not None, Exception("Must initialize grid before adding bunch")
		
		hf = h5py.File('./'+qpad_file_in, 'r')
		if(len(hf['x1'].shape) ==0):
			z,r,x,y = np.array([0]),np.array([1]), np.array([0]), np.array([1])
			q = np.array([0])
			pz,px,py = np.array([0]),np.array([0]),np.array([0])
		else:
			x,y,z = hf['x1'][:], hf['x2'][:], hf['x3'][:]
			px,py,pz = hf['p1'][:],hf['p2'][:], hf['p3'][:]
			q = hf['q'][:]
		time = hf.attrs['TIME'][0]
		hf.close()

		x = x + offset[0] * self.kp
		y = y + offset[1] * self.kp
		z = z + offset[2] * self.kp
		# px = 0 * px
		# py = 0 * py
		# print(np.average(px),np.average(py), np.average(pz))
		# print(np.std(px),np.std(py), np.std(pz))
		hf = h5py.File(directory + '/' + qpic_file_out, 'w')
		hf.create_dataset('x1', data=x)
		hf.create_dataset('x2', data=y)
		hf.create_dataset('x3', data=-1 * z)
		# px *= 0
		# py *= 0
		hf.create_dataset('p1', data=px)
		hf.create_dataset('p2', data=py)
		hf.create_dataset('p3', data=pz)
		hf.create_dataset('q',data=-np.abs(q)*scale_q_fac)
		hf.close()
		self.species_list.append(picmi.Species( particle_type = 'electron', 
			initial_distribution = picmi.FileDistribution(qpic_file_out)))
		self.if_beam.append(True)
		self.layouts.append(picmi.FileLayout(grid = self.grid))

	"""
	Add OSIRIS Beam File

	Parameters
	----------
	osiris_file_in: string 
		path to OSIRIS file

	qpic_file_out:string
		path to written file for OSIRIS
	
	scale_q_fac: float, default = 1.0
		factor to scale charge q (generally required when importing from OSIRIS)

		
	"""
	def add_osiris_file_bunch(self, osiris_file_in, qpic_file_out, scale_q_fac =1.0, directory = '.', z_select=False):
		assert self.grid is not None, Exception("Must initialize grid before adding bunch")
		
		hf = h5py.File('./'+osiris_file_in, 'r')
		if(len(hf['x1'].shape) ==0):
			z,r,x,y = np.array([0]),np.array([1]), np.array([0]), np.array([1])
			q = np.array([0])
			pz,px,py = np.array([0]),np.array([0]),np.array([0])
		else:
			z,r,x,y = hf['x1'][:], hf['x2'][:], hf['x3'][:],hf['x4'][:]
			pz, px, py = hf['p1'][:],hf['p2'][:], hf['p3'][:]
			q = hf['q'][:]
		time = hf.attrs['TIME'][0]
		hf.close()
		# px = 0 * px
		# py = 0 * py
		# print(np.average(px),np.average(py), np.average(pz))
		# print(np.std(px),np.std(py), np.std(pz))
		hf = h5py.File(directory + '/' + qpic_file_out, 'w')
		hf.create_dataset('x1', data=x)
		hf.create_dataset('x2', data=y)
		hf.create_dataset('x3', data=z-time)
		# px *= 0
		# py *= 0
		hf.create_dataset('p1', data=px)
		hf.create_dataset('p2', data=py)
		hf.create_dataset('p3', data=pz)
		hf.create_dataset('q',data=-np.abs(q)*scale_q_fac)
		hf.close()
		self.species_list.append(picmi.Species( particle_type = 'electron', 
			initial_distribution = picmi.FileDistribution(qpic_file_out)))
		self.if_beam.append(True)
		self.layouts.append(picmi.FileLayout(grid = self.grid))
	
	"""
	Add openPMD Beam File

	Parameters
	----------
	pmd_file_in: string 
		path to openPMD file

	qpic_file_out:string
		path to written file for OSIRIS
	
	scale_q_fac: float, default = 1.0
		factor to scale charge q (generally required when importing from OSIRIS)

	op: function, default = np.median
		shifts beam based on op on coordinates

	z_select: bool,default = False
		if false, z is calculated from 'time' dataset

		
	"""
	def add_openpmd_file_bunch(self, pmd_file_in, qpic_file_out, op = np.median, directory = '.', z_select=False):
		assert self.grid is not None, Exception("Must initialize grid before adding OpenPMD bunch")
		
		## simulation paramters ##
		P = ParticleGroup(pmd_file_in)
		P = P[P.status == 1]
		q_grid_norm = (2 * np.pi * self.grid.dr**2 * self.grid.dz) 
		q_raw_norm = (cst.q_e * self.n0 )
		scale_q = 1.0/(q_grid_norm * q_raw_norm)
		scale_p = 1/(0.511e6)


		dataset = P.copy() # modify a copy of P
		# print(op(dataset.t),op(dataset.x),op(dataset.y), op(dataset.z),'shift')
		# print
		# print('charge nc:' , 1e9 * np.sum(dataset.weight))

		# dataset.x = dataset.x - op(dataset.x)-54e-6 # adjust beam x
		dataset.x = dataset.x - op(dataset.x) # adjust beam x
		dataset.y = dataset.y - op(dataset.y) # adjust beam y
		if(z_select):
			dataset.z = (dataset.z - op(dataset.z))
		else:
			dataset.z = cst.c * (op(dataset.t) - dataset.t) # calculate z and center beam at high current region

		x, y, z = self.kp * dataset.x, self.kp * dataset.y, self.kp * dataset.z
		px, py, pz = dataset['px']*scale_p, dataset['py']*scale_p, dataset['pz']*scale_p

		# print(len(z),'numparts')
		# print(np.std(x),np.std(y))
		# px = 0 * px
		# py = 0 * py
		# print(np.average(px),np.average(py), np.average(pz))
		# print(np.std(px),np.std(py), np.std(pz))
		q = dataset.weight *scale_q
		hf = h5py.File(directory + '/' + qpic_file_out, 'w')
		hf.create_dataset('x1', data=x)
		hf.create_dataset('x2', data=y)
		hf.create_dataset('x3', data=z)
		# px *= 0
		# py *= 0
		hf.create_dataset('p1', data=px)
		hf.create_dataset('p2', data=py)
		hf.create_dataset('p3', data=pz)
		hf.create_dataset('q',data=-np.abs(q))
		hf.close()
		print(np.min(z),np.max(z),'min/max z')
		self.species_list.append(picmi.Species( particle_type = 'electron', 
			initial_distribution = picmi.FileDistribution(qpic_file_out)))
		self.if_beam.append(True)
		self.layouts.append(picmi.FileLayout(grid = self.grid))


	def add_h5_file_bunch(self, h5_file_in, h5_file_out, op = np.median, directory = '.'):
		assert self.grid is not None, Exception("Must initialize grid before adding bunch")
		
		## simulation paramters ##
		q_grid_norm = (2 * np.pi * self.grid.dr**2 * self.grid.dz) 
		q_raw_norm = (cst.q_e * self.n0 )
		scale_q = 1.0/(q_grid_norm * q_raw_norm)
		scale_p = 1/(0.511e6)


		hf = h5py.File(h5_file_in, 'r')
		x,y,z = hf['x1'][:],hf['x2'][:],hf['x3'][:]
		px, py, pz = hf['p1'][:],hf['p2'][:],hf['p3'][:]
		q = hf['q'][:]
		hf.close()
		x *= self.kp
		y *= self.kp
		z *= self.kp
		q *=scale_q
		# print(np.std(z))
		print(np.std(x),np.std(y))
		# px = 0 * px
		# py = 0 * py
		# print(np.average(px),np.average(py), np.average(pz))
		# print(np.std(px),np.std(py), np.std(pz))
		
		hf = h5py.File(directory + '/' + h5_file_out, 'w')
		hf.create_dataset('x1', data=x)
		hf.create_dataset('x2', data=y)
		hf.create_dataset('x3', data=-z)
		hf.create_dataset('p1', data=px)
		hf.create_dataset('p2', data=py)
		hf.create_dataset('p3', data=pz)
		hf.create_dataset('q',data=-np.abs(q))
		hf.close()

		self.species_list.append(picmi.Species( particle_type = 'electron', 
			initial_distribution = picmi.FileDistribution(h5_file_out)))
		self.if_beam.append(True)
		self.layouts.append(picmi.FileLayout(grid = self.grid))
	"""
	Add tri-Gaussian electron bunch

	Parameters
	----------
	charge: float
	Total charge [C]

	bunch_rms_size: vector of length 3 of floats
		RMS bunch size along (x,y,z) [m]

	bunch_centroid_position: vector of length 3 of floats, default = [0, 0, 0]
		Bunch centroid position (x,y,z) [m]
	
	bunch_centroid_velocity: vector of length 3 of floats, default = [0, 0, 19569.47]
		RMS velocity in units of p/mc (unitless)

	bunch_rms_velocity: vector of length 3 of floats, default = [0, 0, 0]
		RMS velocity in units of sigma_p/mc (unitless)

	n_macroparticles: integer, default = 1e6
		Number of macroparticles representing the beam

	alpha: float, default = 0 
		Twiss parameter alpha determines phase-space tilt

	piecewise_s, piecewise_fs: arrays of floats, default = None
		Longitudinal piecewise beam profile

	
	
	"""
	def add_gaussian_electron_bunch(self, charge, bunch_rms_size, 
		bunch_centroid_position = [0, 0 ,0], bunch_centroid_velocity = [0, 0, 19569.47],
		 bunch_rms_velocity = [0, 0 ,0], n_macroparticles = 1e6, alpha = 0, piecewise_fs = None, piecewise_s = None):

		n_physical_particles = abs(int(charge/cst.q_e))

		# alpha parameter not implemented yet
		# transform = np.sqrt(1 + alpha**2)
		# b_rms_size = np.array(bunch_rms_size)
		# b_rms_velocity = np.array(bunch_rms_velocity)
		# for i in range(2):
		# 	b_rms_size[i] *= transform
		# 	b_rms_velocity[i] /= transform

		
		if(piecewise_fs is not None and piecewise_s is not None):
			beam_dict = { picmi.codename + '_alpha' : [alpha,alpha],picmi.codename + '_piecewise_s' : piecewise_s, picmi.codename + '_piecewise_fs' : piecewise_fs}
		else:
			beam_dict = { picmi.codename + '_alpha' : [alpha,alpha] }
		dist = picmi.GaussianBunchDistribution(
			n_physical_particles = n_physical_particles,
			rms_bunch_size       = bunch_rms_size,
			rms_velocity         = [cst.c * x for x in bunch_rms_velocity],
			centroid_position    = bunch_centroid_position,
			centroid_velocity    = [cst.c * x for x in bunch_centroid_velocity], **beam_dict )
		
		self.species_list.append(picmi.Species( particle_type = 'electron', initial_distribution = dist))
		layout_dict = {}
		# self.layouts.append(picmi.GriddedLayout(
		# 			grid = self.grid,
		# 			n_macroparticle_per_cell = ppc, 
		# 			**layout_dict))
		self.layouts.append(picmi.PseudoRandomLayout(
						grid = self.grid,
						n_macroparticles = n_macroparticles,
						**layout_dict))
		self.if_beam.append(True)


	"""
	Add uniform pre-ionized plasma

	Parameters
	----------
	number_density: float
	Plasma electron number density [m^-3]

	ppc: list of 2 integers, default = [2, 2]
		ppc along x and y

	"""
	def add_uniform_plasma(self, number_density = 0, ppc = [2, 2], rmax = None):
		if(self.grid is None):
			print("Warning: Initialize grid before adding plasma")
			return
		if(rmax is None):
			self.species_list.append(picmi.Species(particle_type = 'electron', 
				initial_distribution = picmi.UniformDistribution(density = number_density) ))
		else:
			str_expr = 'if(sqrt(x^2 + y^2)  <' + str(rmax) + ',' + str(number_density)+', 0)'
			self.species_list.append(picmi.Species(particle_type = 'electron', 
				initial_distribution = picmi.AnalyticDistribution(density_expression = str_expr) ))

		layout_dict = {}
		self.layouts.append(picmi.GriddedLayout(
					grid = self.grid,
					n_macroparticle_per_cell = ppc, 
					**layout_dict))
		self.if_beam.append(False)


	"""
	Add uniform neutral gas (e.g. Li)

	Parameters
	----------
	number_density: float
		Number density of gas [m^-3].

	particle_type: string
		A string specifying an atom (e.g. Li, Ar...) as defined in
		the openPMD 2 species type extension, openPMD-standard/EXT_SpeciesType.md

	max_level: integer, optional 
		Specifies maximum ionization level.

	ppc: list of 2 integers, default = [4, 4]
		ppc along x and y. ppc(1) * ppc(2) = total ionized macroelectrons 
		per cell (default is 16)

		
	"""
	def add_uniform_neutral_gas(self, number_density = 0, particle_type = 'Li', max_level = None, ppc = [4, 4]):
		print("Warning: Ionization not yet included in QUICKPIC PICMI implemented")
		return



	"""
	Add longitudinal neutral gas (e.g. Li)

	Parameters
	----------
	z: array of floats
		Longitudinal position of neutral gas profile [m].

	nz: array of floats
		Number density of neutral gas profile [m^-3].

	n0_factor: float
		Normalizing density factor [m^-3].

	particle_type: string
		A string specifying an atom (e.g. Li, Ar...) as defined in
		the openPMD 2 species type extension, openPMD-standard/EXT_SpeciesType.md

	max_level: integer, optional 
		Specifies maximum ionization level.

	ppc: list of 2 integers, default = [4, 4]
		ppc along x and y. ppc(1) * ppc(2) = total ionized macroelectrons 
		per cell (default is 16)

		
	"""
	def add_longitudinal_neutral_gas_profile(self, z, nz,  particle_type = 'Li', max_level = None, ppc = [4, 4]):
		print("Warning: Ionization not yet included in QUICKPIC PICMI implemented")
		return




	"""
	Add longitudinal pre-ionized plasma

	Parameters
	----------
	number_density: float
	Plasma electron number density [m^-3]

	ppc: list of 2 integers, default = [2, 2]
		ppc along x and y


	"""
	def add_longitudinal_plasma(self, z, nz, ppc = [2, 2], upper_bound = None):
		if(self.grid is None):
			print("Warning: Initialize grid before adding plasma")
			return
		self.species_list.append(picmi.Species(particle_type = 'electron', 
			initial_distribution = picmi.PiecewiseDistribution(density = self.n0, piecewise_s = z, piecewise_fs = nz) ))

		layout_dict = {  }
		self.layouts.append(picmi.GriddedLayout(
					grid = self.grid,
					n_macroparticle_per_cell = ppc, 
					**layout_dict))
		self.if_beam.append(False)


	"""
	Adds Raw Particle Diagnostic for beam dumps

	Parameters
	----------

	period: integer, default = 1
		Frequency of data dumps (1 dumps every timestep)

	period: integer, default = 1
		Sampling frequency of particles (1 dumps every particle, 2 dumps every other part)
		
	"""
	def add_particle_diagnostics(self, period = 1, psample = 1):
		part_diag_dict = { picmi.codename + '_sample' : 1}
		beam_list = []
		for i in range(len(self.species_list)):
			if(self.if_beam[i]):
				beam_list.append(self.species_list[i])
		self.part_diags.append(picmi.ParticleDiagnostic(period = period,
                             species = beam_list,
                              **part_diag_dict))

	"""
	Adds Field Diagnostic to data dumps

	Parameters
	----------
	data_list: list of strings
		Field Data to dump (e.g. ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'psi'])

	period: integer, default = 1
		Frequency of data dumps (1 dumps every timestep)

	"""
	def add_field_diagnostics(self, data_list = [], period = 1, slice_ind = None, slice_coords = 'xz'):
		field_diag_dict = {}
		if(slice_ind is not None):
			field_diag_dict[picmi.codename + '_slice'] = [slice_coords,slice_ind]
		self.field_diags.append(picmi.FieldDiagnostic(data_list = data_list,
	                                   grid = self.grid,
	                                   period = period,
										**field_diag_dict))


	"""
	Constructs simulation input file and runs QPAD

	Parameters
	----------
	dt: float
		Time step of simulation [s].

	tmax: float
		Maximum time of simulation [s].

	nodes: list of 2 integers, default = [1, 1]
		mpi procs along r and z
		
	"""
	def run_simulation(self,dt, tmax, nodes = [1, 1], ndump_restart= 0, sim_dir = '.', path_to_qpic = '.'):
		sim_dict = { picmi.codename + '_nodes' : nodes, picmi.codename + '_n0' : self.n0}
		if(ndump_restart != 0):
			sim_dict[picmi.codename + '_ndump_restart'] = ndump_restart
			sim_dict[picmi.codename + '_dump_restart'] = True
		self.simulation = picmi.Simulation(solver = self.solver, verbose = 1,
			time_step_size = dt, max_time = tmax, **sim_dict)

		for i in range(len(self.species_list)):
			self.simulation.add_species(species = self.species_list[i], layout = self.layouts[i])

		for i in range(len(self.field_diags)):
			self.simulation.add_diagnostic(self.field_diags[i])

		for i in range(len(self.part_diags)):
			self.simulation.add_diagnostic(self.part_diags[i])

		self.simulation.write_input_file(sim_dir+ '/qpinput.json')
		# todo add mpi script
		# subprocess ....
		env = dict(os.environ)
		# env['LD_LIBRARY_PATH'] ='/sdf/group/facet/codes/qpad_libs_openmpi/json-fortran/build/lib:'  + env['LD_LIBRARY_PATH']
		procs = np.prod(nodes)
		f = open(sim_dir + '/output.txt', "w")
		f2 = open(sim_dir + '/stderr.txt', "w")
		subprocess.run(["srun", "-n", str(procs), "-c", str(2), "--cpu_bind=cores", path_to_qpic + "/qpic.e"],stdout =f,stderr =f2, cwd = sim_dir, env=env)
		f.close()
		f2.close()

		



	






		

