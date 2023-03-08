# Copyright 2022-2023 Thamine Dalichaouch, Frank Tsung
# QuickPIC extension of PICMI standard

import picmistandard
import numpy as np
import math
import json 
from json import encoder
from itertools import cycle


encoder.FLOAT_REPR = lambda o: format(o, '.4f')

codename = 'QuickPIC'
picmistandard.register_codename(codename)


class constants:
	c = 299792458.
	ep_0 = 8.8541878128e-12
	q_e = 1.602176634e-19
	m_e = 9.1093837015e-31
	m_p = 1.67262192369e-27

picmistandard.register_constants(constants)

# Species Class
class Species(picmistandard.PICMI_Species):
	"""
	QuickPIC-Specific Parameters
	
	### Beam-specific parameters ####

	QuickPIC_beam_evolution: boolean, optional
		Toggles beam evolution

	QuickPIC_quiet_start: boolean, optional
		If turned on, a set of image particles will be added to suppress the statistic noise. 
	
	QuickPIC_np: integer(3), optional
		Number of beam particles distributed along each direction. The product is the total no of particles.
	
	### Plasma-specific parameters ####

	QuickPIC_ppc: integer(2), optional
		Number of macroparticles per cell in a xi-slice.


	"""
	# initialization 
	def init(self, kw):
		part_types = {'electron': [-constants.q_e, constants.m_e] ,\
		'positron': [constants.q_e, constants.m_e],\
		'proton': [constants.q_e, constants.m_p],\
		'anti-proton' : [-constants.q_e, constants.m_p]}

		if(self.particle_type in part_types):
			if(self.charge is None): self.charge = part_types[self.particle_type][0]
			if(self.mass is None): self.mass = part_types[self.particle_type][1]


		# Handle optional args for beams 
		self.beam_evolution = kw.pop('QuickPIC_beam_evolution', True)
		self.quiet_start = kw.pop('QuickPIC_quiet_start', True)
		


		# set profile type
		if(isinstance(self.initial_distribution, GaussianBunchDistribution)):
			self.profile_type = 'beam'
		elif(isinstance(self.initial_distribution, UniformDistribution)):
			self.profile_type = 'species'
		else:
			print('Warning: Only Uniform and Gaussian distributions are currently supported.')


	def normalize_units(self):
		self.initial_distribution.normalize_units()

	def fill_dict(self, keyvals):
		if(self.profile_type == 'beam'):
			keyvals['evolution'] = self.beam_evolution
			keyvals['quiet_start'] = self.quiet_start
		self.initial_distribution.fill_dict(keyvals)


		


class GaussianBunchDistribution(picmistandard.PICMI_GaussianBunchDistribution):
	"""
	QuickPIC-Specific Parameters
	
	### Plasma-specific parameters ####

	self.profile: integer
		Specifies profile-type of uniform plasma.
		profile = 0 (uniform plasma or piecewise linear function in z), 12 (piecewise linear along r and z)
		Profiles are multiplicative f(r,z) = f(r)  * f(z)

	self.s, self.r: float array
		Specifies longitudinal coordinates of piecewise profile in z=s and r.

	self.fs, self.fz: float array
		Species normalized densities along coordinates specified by self.fs and self.fz.

	"""
	def init(self, kw):
		self.profile = 0

	def normalize_units(self,species, density_norm):
		# get charge, peak density in unnormalized units
		part_charge = species.charge
		total_charge = part_charge * self.n_physical_particles
		if(species.density_scale is not None):
			total_charge *= species.density_scale
		
		peak_density = total_charge/(part_charge * np.prod(self.rms_bunch_size) * (2 * np.pi)**1.5)


		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep_0 * constants.m_e) ) 
		k_pe = w_pe/constants.c


		# QuickPIC takes the spot size in normalized units (k_pe sigma), uth is divergence (sigma_{gamma * beta}), and ufl is fluid velocity (gamma* beta) )

		# normalized spot sizes
		for i in range(3):
			self.rms_bunch_size[i] *= k_pe 
			self.centroid_position[i] *= k_pe 
			self.rms_velocity[i] /= constants.c 
			self.centroid_velocity[i] /= constants.c

		self.gamma = self.centroid_velocity[2]

		# normalized charge, mass, density
		self.q = species.charge/constants.q_e
		self.m = species.mass/constants.m_e
		self.norm_density = peak_density/density_norm

	def fill_dict(self,keyvals):
		keyvals['profile'] = self.profile
		keyvals['q'] = self.q
		keyvals['m'] = self.m
		keyvals['peak_density'] = self.norm_density
		keyvals['gamma'] = self.gamma
		keyvals['center'] = self.centroid_position
		keyvals['centroid_x'] = [0.0, 0.0, 0.0]
		keyvals['centroid_y'] = [0.0, 0.0, 0.0]
		# QuickPIC coordinate in xi = ct-z
		keyvals['center'][2] *= -1
		keyvals['sigma'] = self.rms_bunch_size
		keyvals['sigma_v'] = self.rms_velocity


class UniformDistribution(picmistandard.PICMI_UniformDistribution):
	"""
	QuickPIC-Specific Parameters
	
	### Plasma-specific parameters ####

	self.profile: integer
		Specifies profile-type of uniform plasma.
		profile = 0 (uniform plasma or piecewise linear function in z), 12 (piecewise linear along r and z)
		Profiles are multiplicative f(r,z) = f(r)  * f(z)

	self.z, self.r: array
		Specifies longitudinal coordinates of piecewise profile in z and r.

	self.fz, self.fr: array
		Species normalized densities along coordinates specified by self.fz and self.fr.

	QuickPIC_r_min, QuickPIC_r_max: float, optional
		Radial range (i.e. QuickPIC_r_min <= r <= QuickPIC_r_max) for particles in UniformDistribution. Only required when specifying transverse lower_bounds or upper_bounds.
 
	"""
	def init(self,kw):
		# default profile for uniform plasmas
		self.profile = 0
		self.longitudinal_profile = 'uniform'
		self.s, self.r, self.fs, self.fr = None, None, None, None
		self.z, self.r = None, None

		# Handle optional args
		self.r_min = kw.pop('QuickPIC_r_min', None)
		self.r_max = kw.pop('QuickPIC_r_max', None)

		# if range-bound along z
		if(self.upper_bound[2] is not None and self.lower_bound[2] is not None):
			self.longitudinal_profile = 'piecewise_linear'
			z_low, z_high, z_len = self.lower_bound[2], self.upper_bound[2], np.abs(self.upper_bound[2] - self.lower_bound[2])
			self.z = [z_low - z_len* 1.e-6, z_low, z_high, z_high + z_len * 1.e-6 ]
			self.fz = [0., 1., 1., 0.]

		# check if range bound in transverse direction
		transverse_flag = any(ele is not None for ele in self.lower_bound[:2]) or any(ele is not None for ele in self.upper_bound[:2])
		if(transverse_flag):
			print('Warning: QuickPIC only supports radially symmetric profiles f(r).')
			assert self.r_max != None and self.r_min != None, Exception('Please pass additional parameters QuickPIC_r_max, QuickPIC_r_max into UniformDistribution')
			assert self.r_max != self.r_min and self.r_max > 0 and self.r_min >= 0, Exception('Invalid range for QuickPIC_r_min = ' + str(self.r_min) + ', QuickPIC_r_max = ' + str(self.r_max))
			self.profile = 12
			if(self.r_min == 0):
				self.r = [self.r_min, self.r_max, (1-1.e-6) * self.r_max ]
				self.fr = [1., 1., 0]
			else:
				self.r = [(1-1.e-6) * self.r_min, self.r_min, self.r_max, (1+1.e-6) * self.r_max ]
				self.fr = [0., 1., 1., 0]


	def normalize_units(self,species, density_norm):
		# normalize plasma density
		self.norm_density = self.density/density_norm

		# normalized charge, mass, density
		self.q = species.charge/constants.q_e
		self.m = species.mass/constants.m_e

		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep_0 * constants.m_e) ) 
		k_pe = w_pe/constants.c

		#normalize coordinates 
		if(self.z is not None):
			for i in range(len(self.z)):
				self.z[i] *= k_pe 
		if(self.r is not None):
			for i in range(len(self.r)):
				self.r[i] *= k_pe

		if(np.any(self.rms_velocity != 0.0) or np.any(self.directed_velocity != 0.0)):
			print('Warning: QuickPIC does not support rms velocity or directed velocity for Uniform Distributions.')

	def fill_dict(self,keyvals):
		keyvals['profile'] = self.profile
		keyvals['q'] = self.q
		keyvals['m'] = self.m
		keyvals['density'] = self.norm_density
		keyvals['longitudinal_profile'] = self.longitudinal_profile
		if(self.longitudinal_profile == 'piecewise_linear'):
			keyvals['piecewise_s'] = self.z
			keyvals['piecewise_density'] = self.fz
		if(self.profile == 12):
			keyvals['piecewise_radial_density'] = self.fr
			keyvals['piecewise_r'] = self.r


class AnalyticDistribution(picmistandard.PICMI_AnalyticDistribution):
	"""
	QuickPIC-Specific Parameters
	
	### Plasma-specific parameters ####

	self.profile: integer
		Specifies profile-type of uniform plasma.
		profile = 0 (uniform plasma or piecewise linear function in z), 12 (piecewise linear along r and z)
		Profiles are multiplicative f(r,z) = f(r)  * f(z)

	self.s, self.r: array
		Specifies longitudinal coordinates of piecewise profile in z=s and r.

	self.fs, self.fz: array
		Species normalized densities along coordinates specified by self.fs and self.fz.

	QuickPIC_r_min, QuickPIC_r_max: float, optional
		Radial range (i.e. QuickPIC_r_min <= r <= QuickPIC_r_max) for particles in UniformDistribution. Only required when specifying transverse lower_bounds or upper_bounds.
 
	### TO BE IMPLEMENTED
	"""

	def init(self,kw):
		# default profile for uniform plasmas
		self.profile = 0

		self.s, self.r, self.fs, self.fr = None, None, None, None

		# Handle optional args
		self.r_min = kw.pop('QuickPIC_r_min', None)
		self.r_max = kw.pop('QuickPIC_r_max', None)

		Exception('AnalyticDistribution has not yet been implemented')


	def normalize_units(self,species, density_norm):
		# normalize plasma density
		self.norm_density = self.density/density_norm

		# normalized charge, mass, density
		self.q = species.charge/constants.q_e
		self.m = species.mass/constants.m_e

		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep_0 * constants.m_e) ) 
		k_pe = w_pe/constants.c

		#normalize coordinates 
		if(self.z is not None):
			for i in range(len(self.z)):
				self.z[i] *= k_pe 
		if(self.r is not None):
			for i in range(len(self.r)):
				self.r[i] *= k_pe

		if(np.any(self.rms_velocity != 0.0) or np.any(self.directed_velocity != 0.0)):
			print('Warning: QuickPIC does not support rms velocity or directed velocity for Uniform Distributions. These parameters will be ignored.')

class ElectromagneticSolver(picmistandard.PICMI_ElectromagneticSolver):
	"""
	QuickPIC-Specific Parameters
	
	QuickPIC_maximum_iterations: integer
		Number of iterations for predictor corrector solver.
	"""
	def init(self, kw):
		self.maximum_iterations = kw.pop('QuickPIC_maximum_iterations', None)
		if(self.maximum_iterations == None):
			print('Defaulting to n_iterations = 1 for predictor corrector')
			self.maximum_iterations = 1
	def fill_dict(self,keyvals):
		keyvals['iter'] = self.maximum_iterations
		
class ElectromagneticSolver(picmistandard.PICMI_ElectromagneticSolver):
	def init(self, kw):
		self.maximum_iterations = kw.pop('QuickPIC_maximum_iterations', None)
		if(self.maximum_iterations == None):
			print('Defaulting to n_iterations = 1 for predictor corrector')
			self.maximum_iterations = 1
	def fill_dict(self,keyvals):
		keyvals['iter'] = self.maximum_iterations
		
		





class Cartesian3DGrid(picmistandard.PICMI_Cartesian3DGrid):
	def init(self, kw):
		dims = 3
		# first check if grid cells are power of two
		assert all( self.power_of_two_check(n) for n in self.number_of_cells), Exception('Number_of_cells must be a power of two in each direction.')

		# extract log2 exponents of each number_of_cells dim
		self.indx, self.indy, self.indz = math.frexp(self.number_of_cells[0])[1] - 1, math.frexp(self.number_of_cells[1])[1] - 1, math.frexp(self.number_of_cells[2])[1] - 1

		# second check to make sure window moving forward at c (window speed doesn't actually matter for QuickPIC)
		assert self.moving_window_velocity == [0, 0, constants.c]

		for i in range(dims):
			assert self.lower_boundary_conditions[i] == self.upper_boundary_conditions[i] 
			if(i < 2):
				assert self.lower_boundary_conditions[i] == 'dirichlet', Exception('QuickPIC only supports conductive boundaries (dirichlet).')

		self.boundary = 'conducting'
		self.x = [self.lower_bound[0], self.upper_bound[0]]
		self.y = [self.lower_bound[1], self.upper_bound[1]]
		self.z = [self.lower_bound[2], self.upper_bound[2]]

	def power_of_two_check(self,n):
		return (n & (n-1) == 0) and n != 0

	def normalize_units(self, density_norm):
		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep_0 * constants.m_e) ) 
		k_pe = w_pe/constants.c

		#normalize coordinates 
		for i in range(2):
			self.x[i] *= k_pe 
			self.y[i] *= k_pe
			self.z[i] *= k_pe

	def fill_dict(self,keyvals):
		keyvals['indx'], keyvals['indy'], keyvals['indz'] = self.indx, self.indy, self.indz
		box = {}
		box['x'], box['y'] = self.x, self.y
		# quickpic 3D is in xi not z (multiply by -1 + reverse z coordinate)
		box['z'] = [-self.z[1], -self.z[0]]
		keyvals['box'] = box
		keyvals['boundary'] = self.boundary

		
		

		
		

## Throw Errors if trying to use 2D cartesian and cylindrical grids with QuickPIC
class Cartesian2DGrid(picmistandard.PICMI_Cartesian2DGrid):
	def init(self, kw):
		Exception('Quickpic does not support this feature. Please specify a 3D Cartesian Grid.')

class Cartesian2DGrid(picmistandard.PICMI_Cartesian2DGrid):
	def init(self, kw):
		Exception('Quickpic does not support this feature. Please specify a 3D Cartesian Grid.')

class CylindricalGrid(picmistandard.PICMI_CylindricalGrid):
	def init(self, kw):
		Exception('Quickpic does not support this feature. Please specify a 3D Cartesian Grid.')

class PseudoRandomLayout(picmistandard.PICMI_PseudoRandomLayout):
	"""
	QuickPIC-Specific Parameters
	
	QuickPIC_np_per_dimension: integer array, optional
		Number of iterations for predictor corrector solver.

	QuickPIC_npmax: integer, optional
		Particle buffer size per MPI partition.
	"""

	def init(self,kw):
		# n_macroparticles is required.
		assert self.n_macroparticles is not None, Exception('n_macroparticles must be specified when using PseudoRandomLayout with QuickPIC')
		self.np = kw.pop('QuickPIC_np_per_dimension', None)
		if(self.np is None):
			print('Warning: QuickPIC_np_per_dimension was not specified.')
			np_per_dim = int(np.cbrt(self.n_macroparticles))
			self.np = [np_per_dim] * 3
			print('Warning: Casting n_macroparticles = ' + str(self.n_macroparticles) + ' to np_per_dimension = [' + \
				str(np_per_dim) + ',' +str(np_per_dim) + ',' + str(np_per_dim) + ']' )

		self.npmax = kw.pop('QuickPIC_npmax', 10**6)
	def fill_dict(self, keyvals):
		keyvals['np'] = self.np
		keyvals['npmax'] = self.npmax
				


class GriddedLayout(picmistandard.PICMI_GriddedLayout):
	"""
	QuickPIC-Specific Parameters

	QuickPIC_npmax: integer, optional
		Particle buffer size per MPI partition.
	"""
	def init(self,kw):
		self.npmax = kw.pop('QuickPIC_npmax', 10**6)
		assert len(self.n_macroparticle_per_cell) !=2, print('Warning: QuickPIC only supports 2-dimensions for n_macroparticle_per_cell')

	def fill_dict(self,keyvals):
		keyvals['npmax'] = self.npmax
		keyvals['ppc'] = self.n_macroparticle_per_cell[:2]



class Simulation(picmistandard.PICMI_Simulation):
	"""
	QuickPIC-Specific Parameters

	QuickPIC_n0: float, optional
		Plasma density [m^3] to normalize units.

	QuickPIC_nodes: int(2), optional
		MPI-node configuration

	QuickPIC_read_restart: boolean, optional
		Toggle to read from restart files.
	
	QuickPIC_restart_timestep: integer, optional
		Specifies timestep if read_restart = True.

	"""
	def init(self,kw):
		# set verbose default
		if(self.verbose is None):
			self.verbose = 0
		assert self.time_step_size is not None, Exception('QuickPIC requires a time step size for the 3D loop.')
		if(self.max_time is None):
			self.max_time = self.max_steps * self.time_step_size
		if(self.particle_shape not in  ['linear']):
			print('Warning: Defaulting to linear particle shapes.')
			self.particle_shape = 'linear'

		self.nodes = kw.pop('QuickPIC_nodes', [1, 1])


		### QuickPIC differentiates between beams and plasmas (species)
		self.if_beam = []

		# check if normalized density is specified
		self.n0 = kw.pop('QuickPIC_n0', None)

		# normalize simulation time
		if(self.n0 is not None):
			self.normalize_simulation()

		# check to read from restart files
		self.read_restart = kw.pop('QuickPIC_read_restart', False)
		self.restart_timestep = kw.pop('QuickPIC_restart_timestep', -1)
		if(self.read_restart):
			assert self.restart_timestep != -1, Exception('Please specify QuickPIC_restart_timestep')

		# check if dumping restart files
		self.dump_restart = kw.pop('QuickPIC_dump_restart', False)
		self.ndump_restart = kw.pop('QuickPIC_ndump_restart', -1)
		if(self.dump_restart):
			assert self.ndump_restart != -1, Exception('Please specify QuickPIC_ndump_restart')


	def normalize_simulation(self):
		w_pe = np.sqrt(constants.q_e**2.0 * self.n0/(constants.ep_0 * constants.m_e) ) 
		self.max_time *= w_pe
		self.time_step_size *= w_pe
		self.solver.grid.normalize_units(self.n0)
	
	def add_species(self, species, layout, initialize_self_field = None):
		picmistandard.PICMI_Simulation.add_species( self, species, layout,
									  initialize_self_field )
		if(self.n0 is not None):
			species.initial_distribution.normalize_units(species, self.n0)

		# handle checks for beams
		self.if_beam.append(species.profile_type == 'beam')
			


	def fill_dict(self, keyvals):
		# fill grid and mpi params
		keyvals['nodes'] = self.nodes
		self.solver.grid.fill_dict(keyvals)

		# fill simulation time and dt
		keyvals['time'] = self.max_time
		keyvals['dt'] = self.time_step_size


		if(self.n0 is not None):
			keyvals['n0'] = self.n0 * 1.e-6 # in density in cm^{-3}
		keyvals['nbeams'] = int(np.sum(self.if_beam))
		keyvals['nspecies'] = len(self.if_beam) - keyvals['nbeams']
		self.solver.fill_dict(keyvals)
		keyvals['dump_restart'] = self.dump_restart
		if(self.dump_restart):
			keyvals['ndump_restart'] = self.ndump_restart
		keyvals['read_restart'] = self.read_restart
		if(self.read_restart):
			keyvals['restart_timestep'] = self.restart_timestep
		keyvals['verbose'] = self.verbose

	def write_input_file(self,file_name):
		total_dict = {}

		# simulation object handled
		sim_dict = {}
		self.fill_dict(sim_dict)

		# beam objects 
		beam_dicts = []

		# species objects
		species_dicts = [] 

		# field object
		field_dict = {}
		# iterate over species handle beams first
		for i in range(len(self.species)):
			spec = self.species[i]
			temp_dict = {}
			self.layouts[i].fill_dict(temp_dict)
			self.species[i].fill_dict(temp_dict)

			# fill in source term diagnostics
			diags_srcs = []
			for j in range(len(self.diagnostics)):
				diag = self.diagnostics[j]
				if(isinstance(diag,ParticleDiagnostic) and spec not in diag.species):
					continue
				temp_dict2 = {}
				self.diagnostics[j].fill_dict_src(temp_dict2)
				diags_srcs.append(temp_dict2)
			temp_dict['diag'] = diags_srcs
			if(self.if_beam[i]):
				beam_dicts.append(temp_dict)
			else:
				species_dicts.append(temp_dict)


		diags_flds = []
		for i in range(len(self.diagnostics)):
			diag = self.diagnostics[i]
			temp_dict = {}
			if(isinstance(diag,ParticleDiagnostic)):
				continue
			self.diagnostics[i].fill_dict_fld(temp_dict)
			diags_flds.append(temp_dict)

		field_dict['diag'] = diags_flds

		total_dict['simulation'] = sim_dict
		total_dict['beam'] = beam_dicts
		total_dict['species'] = species_dicts
		total_dict['field'] = field_dict
		with open(file_name, 'w') as file:
			json.dump(total_dict, file, indent =4)

	def step(self, nsteps = 1):
		Exception('The simulation step feature is not yet supported. Please call write_input_file() to construct the input deck.')

class FieldDiagnostic(picmistandard.PICMI_FieldDiagnostic):
	"""
	QuickPIC-Specific Parameters

	QuickPIC_slice: array, optional
		Specifies plane and index of third coordinate to dump (e.g., ["yz", 256])
	"""
	def init(self,kw):
		assert self.write_dir != '.', Exception("Write directory feature not yet supported.")
		assert self.period > 0, Exception("Diagnostic period is not valid")
		self.field_list = []
		self.source_list = []
		if('E' in self.data_list):
			self.field_list += ['ex','ey','ez']
		if('B' in self.data_list):
			self.field_list += ['bx','by','bz']
		if('rho' in self.data_list):
			self.source_list += ['charge']
		if('J' in self.data_list):
			self.source_list += ['jx','jy','jz']


		# need to add to PICMI standard
		if('psi' in self.data_list):
			self.field_list += ['psi']

		self.slice = kw.pop('QuickPIC_slice', None) 

	def fill_dict_fld(self,keyvals):
		keyvals['name'] = self.field_list
		keyvals['ndump'] = self.period
		if(self.slice):
			keyvals['slice'] = [self.slice]

	def fill_dict_src(self,keyvals):
		keyvals['name'] = self.source_list
		keyvals['ndump'] = self.period
		if(self.slice):
			keyvals['slice'] = [self.slice]

## to be implemented	
class ParticleDiagnostic(picmistandard.PICMI_ParticleDiagnostic):
	"""
	QuickPIC-Specific Parameters

	QuickPIC_sample: integer, optional
		Dumps every nth particle.
	"""
	def init(self,kw):
		assert self.write_dir != '.', Exception("Write directory feature not yet supported.")
		assert self.period > 0, Exception("Diagnostic period is not valid")
		print('Warning: Particle diagnostic reporting momentum, position and charge data')
		self.sample = kw.pop('QuickPIC_sample', None) 

	def fill_dict_fld(self,keyvals):
		pass

	def fill_dict_src(self,keyvals):
		keyvals['name'] = ["raw"]
		keyvals['ndump'] = self.period
		keyvals['sample'] = self.sample


