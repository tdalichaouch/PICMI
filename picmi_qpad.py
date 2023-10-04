# Copyright 2022-2023 Thamine Dalichaouch, Frank Tsung
# QPAD extension of PICMI standard

import picmistandard
import numpy as np
import re
import math
import json 
from json import encoder
from itertools import cycle
import periodictable


encoder.FLOAT_REPR = lambda o: format(o, '.4f')

codename = 'QPAD'
picmistandard.register_codename(codename)


class constants:
	c = 299792458.
	ep0 = 8.8541878128e-12
	mu0 = 4 * np.pi * 1e-7
	q_e = 1.602176634e-19
	m_e = 9.1093837015e-31
	m_p = 1.67262192369e-27

picmistandard.register_constants(constants)

# Species Class
class Species(picmistandard.PICMI_Species):
	"""
	QPAD-Specific Parameters
	
	### Beam-specific parameters ####

	QPAD_beam_evolution: boolean, optional
		Toggles beam evolution

	QPAD_quiet_start: boolean, optional
		If turned on, a set of image particles will be added to suppress the statistic noise. 
	
	QPAD_np: integer(3), optional
		Number of beam particles distributed along each direction. The product is the total no of particles.
	
	### Plasma-specific parameters ####

	QPAD_ppc: integer(2), optional
		Number of macroparticles per cell in a xi-slice.


	"""
	# initialization 
	def init(self, kw):
		part_types = {'electron': [-constants.q_e, constants.m_e] ,\
		'positron': [constants.q_e, constants.m_e],\
		'proton': [constants.q_e, constants.m_p],\
		'anti-proton' : [-constants.q_e, constants.m_p]}

		if(self.particle_type in part_types):
			if(self.charge is None): 
				self.charge = part_types[self.particle_type][0]
			if(self.mass is None): 
				self.mass = part_types[self.particle_type][1]
		else:
			self.charge = self.charge_state * constants.q_e
			m = re.match(r'(?P<iso>#[\d+])*(?P<sym>[A-Za-z]+)', self.particle_type)
			element = periodictable.elements.symbol(m['sym'])
			if(m['iso'] is not None):
				element = element[m['iso'][1:]]
			if(self.charge_state is not None):
				assert self.charge_state <= element.number, Exception('%s charge state not valid'%self.particle_type)
				try:
					element = element.ion[self.charge_state]
				except ValueError:
					# Note that not all valid charge states are defined in elements,
					# so this value error can be ignored.
					pass
			self.element = element
			if self.mass is None:
				self.mass = element.mass*periodictable.constants.atomic_mass_constant


		# Handle optional args for beams 
		self.beam_evolution = kw.pop(codename + '_beam_evolution', True)
		self.quiet_start = kw.pop(codename + '_quiet_start', True)
		

		# set profile type
		if(isinstance(self.initial_distribution, GaussianBunchDistribution)):
			self.profile_type = 'beam'
			self.geometry = 'cartesian'
		elif(isinstance(self.initial_distribution, UniformDistribution)):
			self.profile_type = 'species'
			self.push_type = 'standard'
		elif(isinstance(self.initial_distribution, AnalyticDistribution)):
			self.profile_type = 'species'
			self.push_type = 'standard'
		else:
			print('Warning: Only Uniform and Gaussian distributions are currently supported.')


	def normalize_units(self):
		# normalized charge, mass, density
		self.q = self.charge/constants.q_e
		self.m = self.mass/constants.m_e



	def fill_dict(self, keyvals, if_lasers):
		if(self.profile_type == 'beam'):
			keyvals['evolution'] = self.beam_evolution
			keyvals['quiet_start'] = self.quiet_start
			keyvals['geometry'] = self.geometry
		else:
			if(if_lasers):
				keyvals['push_type'] = self.push_type + '_pgc'
			else:
				keyvals['push_type'] = self.push_type
		keyvals['q'] = self.q
		keyvals['m'] = self.m

		if(self.density_scale is not None):
			keyvals['density'] = self.initial_distribution.norm_density *self.density_scale
		else:
			keyvals['density'] = self.initial_distribution.norm_density
		self.initial_distribution.fill_dict(keyvals)

	def activate_field_ionization(self,model,product_species):
		raise Exception('Ionization not yet supported in QPAD')


picmistandard.PICMI_MultiSpecies.Species_class = Species
class MultiSpecies(picmistandard.PICMI_MultiSpecies):
	def init(self, kw):
		return
		# for species in self.species_instances_list:
		# 	print(species.name)



class GaussianBunchDistribution(picmistandard.PICMI_GaussianBunchDistribution):
	"""
	QPAD-Specific Parameters
	
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
		self.profile = ['gaussian', 'gaussian', 'gaussian']

	def normalize_units(self,species, density_norm):
		# get charge, peak density in unnormalized units
		part_charge = species.charge
		total_charge = part_charge * self.n_physical_particles
		if(species.density_scale is not None):
			total_charge *= species.density_scale
		
		peak_density = total_charge/(part_charge * np.prod(self.rms_bunch_size) * (2 * np.pi)**1.5)


		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c


		# QPAD takes the spot size in normalized units (k_pe sigma), uth is divergence (sigma_{gamma * beta}), and ufl is fluid velocity (gamma* beta) )

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

		self.tot_charge = total_charge/(-constants.q_e * density_norm * k_pe**-3)

	def fill_dict(self,keyvals):
		keyvals['profile'] = self.profile
		keyvals['gamma'] = self.gamma
		keyvals['gauss_center'] = self.centroid_position
		keyvals['total_charge'] = self.tot_charge
		# QPAD coordinate in xi = ct-z
		keyvals['gauss_center'][2] *= -1
		if(self.tot_charge is not None):
			keyvals['total_charge'] = self.tot_charge
		keyvals['gauss_sigma'] = self.rms_bunch_size
		keyvals['uth'] = self.rms_velocity
		for j in range(3):
			keyvals['range' + str(j+1)] = [-5 * self.rms_bunch_size[j] +self.centroid_position[j],\
			 5 * self.rms_bunch_size[j] + self.centroid_position[j]]


class UniformDistribution(picmistandard.PICMI_UniformDistribution):
	"""
	QPAD-Specific Parameters
	
	### Plasma-specific parameters ####

	self.profile: integer
		Specifies profile-type of uniform plasma.
		profile = 0 (uniform plasma or piecewise linear function in z), 12 (piecewise linear along r and z)
		Profiles are multiplicative f(r,z) = f(r)  * f(z)

	self.z, self.r: array
		Specifies longitudinal coordinates of piecewise profile in z and r.

	self.fz, self.fr: array
		Species normalized densities along coordinates specified by self.fz and self.fr.

	QPAD_r_min, QPAD_r_max: float, optional
		Radial range (i.e. QPAD_r_min <= r <= QPAD_r_max) for particles in UniformDistribution. Only required when specifying transverse lower_bounds or upper_bounds.
 
	"""
	def init(self,kw):
		# default profile for uniform plasmas
		self.profile = ['uniform', 'uniform']
		self.s, self.r, self.fs, self.fr = None, None, None, None
		self.z, self.r = None, None

		# Handle optional args
		self.r_min = kw.pop(codename + '_r_min', None)
		self.r_max = kw.pop(codename + '_r_max', None)
		# if range-bound along z
		if(self.upper_bound[2] is not None and self.lower_bound[2] is not None):
			self.profile[1] = 'piecewise-linear'
			z_low, z_high, z_len = self.lower_bound[2], self.upper_bound[2], np.abs(self.upper_bound[2] - self.lower_bound[2])
			self.z = [z_low - z_len* 1.e-6, z_low, z_high, z_high + z_len * 1.e-6 ]
			self.fz = [0., 1., 1., 0.]

		# check if range bound in transverse direction
		transverse_flag = any(ele is not None for ele in self.lower_bound[:2]) or any(ele is not None for ele in self.upper_bound[:2])
		if(transverse_flag):
			print('Warning: ' + codename + ' does not support range bound along r.')


	def normalize_units(self,species, density_norm):
		# normalize plasma density
		self.norm_density = self.density/density_norm

		# normalized charge, mass, density
		# self.q = species.charge/constants.q_e
		# self.m = species.mass/constants.m_e

		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c

		#normalize coordinates 
		if(self.z is not None):
			for i in range(len(self.z)):
				self.z[i] *= k_pe 
		if(self.r is not None):
			for i in range(len(self.r)):
				self.r[i] *= k_pe

		if(np.any(self.directed_velocity != 0.0)):
			print('Warning: ' + codename + ' does not support directed velocity for Uniform Distributions.')

	def fill_dict(self,keyvals):
		keyvals['profile'] = self.profile
		keyvals['q'] = self.q
		keyvals['m'] = self.m
		keyvals['uth'] = self.rms_velocity
		if(self.profile[1] == 'piecewise-linear'):
			keyvals['piecewise_s'] = self.z
			keyvals['piecewise_fs'] = self.fz



class AnalyticDistribution(picmistandard.PICMI_AnalyticDistribution):
	"""
	QuickPIC-Specific Parameters
	
	### Plasma-specific parameters ####

	self.profile: integer
		Specifies profile-type of uniform plasma.
		profile = 13 (analytic functions x, y, z)
		Profiles are multiplicative f(r,z) = f(r)  * f(z)

	"""
	def init(self,kw):
		# default profile for uniform plasmas
		self.profile = ['analytic', 'analytic']
		self.math_func = self.density_expression
		if(np.any(self.momentum_expressions == None)):
			print('Warning: QuickPIC does not support momentum expressions for Analytic Distributions.')


	def normalize_units(self,species, density_norm):

		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c
		self.math_func = '{}'.format(self.math_func).replace('x', '(' + str(1.0/k_pe) +'* x)')
		self.math_func = '{}'.format(self.math_func).replace('y', '(' + str(1.0/k_pe) +'* y)')
		self.math_func = '{}'.format(self.math_func).replace('z', '(' + str(1.0/k_pe) +'* z)')
		self.math_func =  self.math_func + '/' + str(density_norm) 
		self.norm_density = 1.0

		for i in range(3):
			self.rms_velocity[i] /= constants.c 

		if(np.any(self.directed_velocity != 0.0)):
			print('Warning: ' + codename + ' does not support directed velocity for Analytic Distributions.')


	def fill_dict(self,keyvals):
		keyvals['profile'] = self.profile
		keyvals['uth'] = self.rms_velocity
		keyvals['math_func'] = self.math_func


class ParticleListDistribution(picmistandard.PICMI_ParticleListDistribution):
	def init(self,kw):
		raise Exception('Particle list distributions not yet supported in QPAD')


# constant, analytic, or mirror fields not yet supported in QuickPIC 
class ConstantAppliedField(picmistandard.PICMI_ConstantAppliedField):
	def init(self,kw):
		raise Exception("Constant applied fields are not yet supported in QPAD")

class AnalyticAppliedField(picmistandard.PICMI_AnalyticAppliedField):
	def init(self,kw):
		raise Exception("Analytic applied fields are not yet supported in QPAD")

class Mirror(picmistandard.PICMI_Mirror):
	def init(self,kw):
		raise Exception("Mirrors are not yet supported in QPAD")


class ElectromagneticSolver(picmistandard.PICMI_ElectromagneticSolver):
	"""
	QPAD-Specific Parameters
	
	QPAD_maximum_iterations: integer
		Number of iterations for predictor corrector solver.
	"""
	def init(self, kw):
		self.maximum_iterations = kw.pop(codename + '_maximum_iterations', None)
		if(self.maximum_iterations == None):
			print('Defaulting to n_iterations = 1 for predictor corrector')
			self.maximum_iterations = 1
	def fill_dict(self,keyvals):
		keyvals['iter_max'] = self.maximum_iterations
		keyvals['iter_reltol'] = 1e-3
		keyvals['iter_abstol'] = 1e-3
		
		
class ElectrostaticSolver(picmistandard.PICMI_ElectrostaticSolver):
	def init(self, kw):
		raise Exception('This feature is not supported. Please use the Electromagnetic solver.')





		
		

		
		

## Throw Errors if trying to use 1D/2D/3D cartesian grids with QPAD
class Cartesian1DGrid(picmistandard.PICMI_Cartesian1DGrid):
	def init(self, kw):
		raise Exception(codename + ' does not support this feature. Please specify a Cylindrical Grid.')

class Cartesian2DGrid(picmistandard.PICMI_Cartesian2DGrid):
	def init(self, kw):
		raise Exception(codename + ' does not support this feature. Please specify a Cylindrical Grid.')

class Cartesian3DGrid(picmistandard.PICMI_Cartesian3DGrid):
	def init(self, kw):
		raise Exception(codename + ' does not support this feature. Please specify a Cylindrical Grid.')


class CylindricalGrid(picmistandard.PICMI_CylindricalGrid):
	def init(self, kw):
		dims = 2

		# second check to make sure window moving forward at c (window speed doesn't actually matter for QPAD)
		assert self.moving_window_velocity == [0, 0, constants.c]

		# check for open boundaries at r_max
		assert self.upper_boundary_conditions[0] == 'open', Exception('QPAD supports open boundaries in r-direction.')
		
		# check for open boundaries along z	
		assert self.lower_boundary_conditions[1] == 'open' and self.upper_boundary_conditions[1] =='open', \
		Exception('QPAD supports open boundaries in z-direction.')


		self.boundary = 'open'
		self.r = [self.lower_bound[0], self.upper_bound[0]]
		self.z = [self.lower_bound[1], self.upper_bound[1]]

	def power_of_two_check(self,n):
		return (n & (n-1) == 0) and n != 0

	def normalize_units(self, density_norm):
		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c

		#normalize coordinates 
		for i in range(2):
			self.r[i] *= k_pe
			self.z[i] *= k_pe

	def fill_dict(self,keyvals):
		keyvals['grid'] = self.number_of_cells
		keyvals['max_mode'] = self.n_azimuthal_modes
		box = {}
		box['r'] = self.r
		# QPAD 3D is in xi not z (multiply by -1 + reverse z coordinate)
		box['z'] = [-self.z[1], -self.z[0]]
		keyvals['box'] = box
		keyvals['field_boundary'] = self.boundary

class PseudoRandomLayout(picmistandard.PICMI_PseudoRandomLayout):
	"""
	QPAD-Specific Parameters
	
	QPAD_np_per_dimension: integer array, optional
		Part per dim in each direction. (for beams only)

	QPAD_npmax: integer, optional
		Particle buffer size per MPI partition.

	QPAD_num_theta: integer, optional
		Number of particles in azimuthal direction. Defaults to 8 * n_azimuthal_modes.
	"""

	def init(self,kw):
		# n_macroparticles is required.
		assert self.n_macroparticles is not None, Exception('n_macroparticles must be specified when using PseudoRandomLayout with QPAD')
		self.profile_type = 'random'


	def fill_dict(self, keyvals,profile_type):
		keyvals['npmax'] = self.n_macroparticles * 2 
		keyvals['total_num'] = self.n_macroparticles
		keyvals['profile_type'] = self.profile_type
		if(profile_type == 'beam'):
			if(self.n_macroparticles_per_cell is not None):
				keyvals['ppc'] = self.n_macroparticles_per_cell
		elif(profile_type == 'species'):
			raise Exception('PseudoRandomLayout not compatible with non-beam species')
				


class GriddedLayout(picmistandard.PICMI_GriddedLayout):
	"""
	QPAD-Specific Parameters

	QPAD_npmax: integer, optional
		Particle buffer size per MPI partition.

	QPAD_num_theta: integer, optional
		Number of particles in azimuthal direction. Defaults to 8 * n_azimuthal_modes.
	"""
	def init(self,kw):
		self.npmax = kw.pop(codename + '_npmax', 2*10**6)
		assert len(self.n_macroparticle_per_cell) !=2, print('Warning: '+ codename + ' only supports 2-dimensions for n_macroparticle_per_cell')
		# setting profile type to standard
		self.profile_type = 'standard'
		self.num_theta = kw.pop(codename + '_num_theta', 1)
		if(self.num_theta * self.n_macroparticle_per_cell[1] < 8 * self.grid.n_azimuthal_modes):
			self.num_theta = int((8 * self.grid.n_azimuthal_modes)/self.n_macroparticle_per_cell[1])
			print('Warning: total azimthal ppc increased to ' + str(self.num_theta * self.n_macroparticle_per_cell[1]))

	def fill_dict(self,keyvals,profile_type):
		keyvals['npmax'] = self.npmax
		# keyvals['profile_type'] = self.profile_type
		if(profile_type == 'beam'):
			keyvals['ppc'] = self.n_macroparticle_per_cell
		elif(profile_type == 'species'):
			keyvals['ppc'] = self.n_macroparticle_per_cell[:2]
		keyvals['num_theta'] = self.num_theta



class Simulation(picmistandard.PICMI_Simulation):
	"""
	QPAD-Specific Parameters

	QPAD_n0: float, optional
		Plasma density [m^3] to normalize units.

	QPAD_nodes: int(2), optional
		MPI-node configuration

	QPAD_interpolation: str, optional
		Interpolation order (linear for QPAD).

	QPAD_read_restart: boolean, optional
		Toggle to read from restart files.
	
	QPAD_restart_timestep: integer, optional
		Specifies timestep if read_restart = True.

	QPAD_random_seed: integer, optional
		No of seeds for pseudo-random numbers. Defaults to 10.

	QPAD_interpolation: str, optional
		Interpolation order (e.g. linear).

	QPAD_algorithm: str, optional
		Type of algorithm (standard, pgc, etc). Defaults to standard.

	QPAD_timings: bool, optional
		Toggle to report timings. Turned off by default.

	"""
	def init(self,kw):
		# set verbose default
		if(self.verbose is None):
			self.verbose = 0
		assert self.time_step_size is not None, Exception('QPAD requires a time step size for the 3D loop.')
		if(self.max_time is None):
			self.max_time = self.max_steps * self.time_step_size
		if(self.particle_shape not in  ['linear']):
			print('Warning: Defaulting to linear particle shapes.')
			self.particle_shape = 'linear'

		self.nodes = kw.pop(codename + '_nodes', [1, 1])


		### QPAD differentiates between beams and plasmas (species)
		self.if_beam = []

		# check if normalized density is specified
		self.n0 = kw.pop(codename + '_n0', None)

		# set interpolation order
		self.interpolation = kw.pop(codename + '_interpolation', 'linear')

		# set number of seeds for pseudo-random numbers
		self.random_seed = kw.pop(codename + '_random_seed', 10)

		# set algorithm type (default is standard)
		self.algorithm = kw.pop(codename + '_algorithm', 'standard')

		# set timings (default is true)
		self.if_timing = kw.pop(codename + '_timings', False)

		# normalize simulation time
		if(self.n0 is not None):
			self.normalize_simulation()

		# check to read from restart files
		self.read_restart = kw.pop(codename + '_read_restart', False)
		self.restart_timestep = kw.pop(codename + '_restart_timestep', -1)
		if(self.read_restart):
			assert self.restart_timestep != -1, Exception('Please specify ' + codename + '_restart_timestep')

		# check if dumping restart files
		self.dump_restart = kw.pop(codename + '_dump_restart', False)
		self.ndump_restart = kw.pop(codename + '_ndump_restart', -1)
		if(self.dump_restart):
			assert self.ndump_restart != -1, Exception('Please specify' + codename + '_ndump_restart')


	def normalize_simulation(self):
		w_pe = np.sqrt(constants.q_e**2.0 * self.n0/(constants.ep0 * constants.m_e) ) 
		self.max_time *= w_pe
		self.time_step_size *= w_pe
		self.solver.grid.normalize_units(self.n0)
	
	def add_species(self, species, layout, initialize_self_field = None):
		if(isinstance(species, MultiSpecies)):
			for spec in species.species_instances_list:
				picmistandard.PICMI_Simulation.add_species( self, spec, layout,
										  initialize_self_field )

				# handle checks for beams
				self.if_beam.append(spec.profile_type == 'beam')
				spec.normalize_units()
			if(self.n0 is not None):
					species.initial_distribution.normalize_units(spec, self.n0)

		else:
			picmistandard.PICMI_Simulation.add_species( self, species, layout,
										  initialize_self_field )
			if(self.n0 is not None):
				species.initial_distribution.normalize_units(species, self.n0)
				species.normalize_units()
			# handle checks for beams
			self.if_beam.append(species.profile_type == 'beam')
	def add_laser(self, laser, injection_method):
		picmistandard.PICMI_Simulation.add_laser(self, laser, injection_method)
		if(injection_method is not None):
			print('Antenna is not supported in QPAD. Initializating laser in box at t=0.')
			laser.focal_position[2] -= injection_method.position[2]
		if(self.n0 is not None):
			laser.normalize_units(self.n0)


			


	def fill_dict(self, keyvals):
		# fill grid and mpi params
		keyvals['nodes'] = self.nodes
		self.solver.grid.fill_dict(keyvals)

		# fill simulation time and dt
		keyvals['time'] = self.max_time
		keyvals['dt'] = self.time_step_size
		keyvals['interpolation'] = self.interpolation


		if(self.n0 is not None):
			keyvals['n0'] = self.n0 * 1.e-6 # in density in cm^{-3}
		keyvals['nbeams'] = int(np.sum(self.if_beam))
		keyvals['nspecies'] = len(self.if_beam) - keyvals['nbeams']
		# TODO add neutrals and laser support
		keyvals['nneutrals'] = 0
		keyvals['nlasers'] = len(self.lasers)
		self.solver.fill_dict(keyvals)
		keyvals['dump_restart'] = self.dump_restart
		if(self.dump_restart):
			keyvals['ndump_restart'] = self.ndump_restart
		keyvals['read_restart'] = self.read_restart
		if(self.read_restart):
			keyvals['restart_timestep'] = self.restart_timestep
		keyvals['verbose'] = self.verbose
		keyvals['if_timing'] = self.if_timing
		keyvals['random_seed'] = self.random_seed
		keyvals['algorithm'] = self.algorithm

	def write_input_file(self,file_name):
		total_dict = {}

		# simulation object handled
		sim_dict = {}
		self.fill_dict(sim_dict)

		# beam objects 
		beam_dicts = []

		# species objects
		species_dicts = [] 

		# lasers objects
		laser_dicts = []

		# field object
		field_dict = {}
		# iterate over species handle beams first
		for i in range(len(self.species)):
			spec = self.species[i]
			temp_dict = {}
			self.layouts[i].fill_dict(temp_dict,spec.profile_type)
			self.species[i].fill_dict(temp_dict, len(self.lasers) > 0)

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

		for i in range(len(self.lasers)):
			laser = self.lasers[i]
			temp_dict = {}
			self.lasers[i].fill_dict(temp_dict)
			laser_dicts.append(temp_dict)
			self.lasers[i].fill_dict_fld(temp_dict,self.diagnostics)

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
		total_dict['laser'] = laser_dicts
		total_dict['field'] = field_dict
		with open(file_name, 'w') as file:
			json.dump(total_dict, file, indent =4)

	def step(self, nsteps = 1):
		raise Exception('The simulation step feature is not yet supported for ' + codename + '. Please call write_input_file() to construct the input deck.')

class FieldDiagnostic(picmistandard.PICMI_FieldDiagnostic):
	"""
	QPAD-Specific Parameters

	"""
	def init(self,kw):
		assert self.write_dir != '.', Exception("Write directory feature not yet supported.")
		assert self.period > 0, Exception("Diagnostic period is not valid")
		self.field_list = []
		self.source_list = []
		if('E' in self.data_list):
			self.field_list += ['er_cyl_m','ephi_cyl_m','ez_cyl_m']
		if('B' in self.data_list):
			self.field_list += ['br_cyl_m','bphi_cyl_m','bz_cyl_m']
		if('rho' in self.data_list):
			self.source_list += ['charge_cyl_m']
		if('J' in self.data_list):
			self.source_list += ['jr_cyl_m','jphi_cyl_m','jz_cyl_m']


		# need to add to PICMI standard
		if('psi' in self.data_list):
			self.field_list += ['psi_cyl_m']


	def fill_dict_fld(self,keyvals):
		keyvals['name'] = self.field_list
		keyvals['ndump'] = self.period

	def fill_dict_src(self,keyvals):
		keyvals['name'] = self.source_list
		keyvals['ndump'] = self.period


# QuickPIC does not support electrostatic and boosted frame diagnostic 
class ElectrostaticFieldDiagnostic(picmistandard.PICMI_ElectrostaticFieldDiagnostic):
	def init(self,kw):
		raise Exception("Electrostatic field diagnostic not supported in QPAD")

class LabFrameParticleDiagnostic(picmistandard.PICMI_LabFrameParticleDiagnostic):
	def init(self,kw):
		raise Exception("Boosted frame diagnostics not support in QPAD")

class LabFrameFieldDiagnostic(picmistandard.PICMI_LabFrameFieldDiagnostic):
	def init(self,kw):
		raise Exception("Boosted frame diagnostics not support in QPAD")


## to be implemented	
class ParticleDiagnostic(picmistandard.PICMI_ParticleDiagnostic):
	"""
	QPAD-Specific Parameters

	QPAD_sample: integer, optional
		Dumps every nth particle.
	"""
	def init(self,kw):
		assert self.write_dir != '.', Exception("Write directory feature not yet supported.")
		assert self.period > 0, Exception("Diagnostic period is not valid")
		print('Warning: Particle diagnostic reporting momentum, position and charge data')
		self.sample = kw.pop(codename + '_sample', 1) 

	def fill_dict_fld(self,keyvals):
		pass

	def fill_dict_src(self,keyvals):
		keyvals['name'] = ["raw"]
		keyvals['ndump'] = self.period
		keyvals['psample'] = self.sample


class GaussianLaser(picmistandard.PICMI_GaussianLaser):
	def init(self, kw):
		self.profile = ['gaussian', 'polynomial']
		self.iteration = 3

	def normalize_units(self, density_norm):
		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c


		self.k0 = self.k0/k_pe
		self.waist = k_pe * self.waist
		self.duration = w_pe * self.duration 
		for i in range(3):
			self.focal_position[i] *= k_pe 
			self.centroid_position[i] *= k_pe


	def fill_dict(self,keyvals):
		keyvals['profile'] = self.profile
		keyvals['a0'] = self.a0
		keyvals['k0'] = self.k0
		keyvals['w0'] = self.waist
		keyvals['iteration'] = self.iteration
		keyvals['focal_distance'] = self.focal_position[2]
		keyvals['t_rise'] = self.duration * 1.5275
		keyvals['t_fall'] = self.duration * 1.5275
		keyvals['t_flat'] = 0
		keyvals['lon_center'] = -self.centroid_position[2]

	def fill_dict_fld(self, keyvals,diagnostics):
		tt = []
		for diag in diagnostics:
			temp_dict = {}
			if(diag.data_list is not None  and 'E' in diag.data_list):
				temp_dict['name'] = ['a_cyl_m']
				temp_dict['ndump'] = diag.period
				tt.append(temp_dict)
				break

		keyvals['diag'] = tt

		
class LaserAntenna(picmistandard.PICMI_LaserAntenna):
	def init(self, kw):
		return

class BinomialSmoother(picmistandard.PICMI_BinomialSmoother):
	def init(self, kw):
		print("Warning: QPAD has no BinomialSmoother. Skipping feature.")