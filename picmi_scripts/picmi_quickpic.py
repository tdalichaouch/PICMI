# Copyright 2022-2023 Thamine Dalichaouch, Frank Tsung
# QuickPIC extension of PICMI standard

import picmistandard
from pydantic import Field, PrivateAttr
import numpy as np
import re
import math
import json
from json import encoder
from itertools import cycle
import periodictable
from decimal import Decimal

# PICMI 0.35+ with pydantic classes: https://github.com/picmi-standard/picmi/pull/133
if tuple(map(int, re.findall(r'\d+', picmistandard.__version__)[:3])) < (0, 35, 0):
	raise ImportError('picmistandard>=0.35.0 required, found ' + picmistandard.__version__ +
		' in ' + picmistandard.__file__)

encoder.FLOAT_REPR = lambda o: format(o, '.4f')

codename = 'QuickPIC'
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
	beam_evolution: bool = Field(default=True, alias=codename + '_beam_evolution',
		description='Toggles beam evolution')
	quiet_start: bool = Field(default=True, alias=codename + '_quiet_start',
		description='Adds image particles to suppress the statistic noise')

	_element: object = PrivateAttr(default=None)
	_profile_type: str | None = PrivateAttr(default=None)
	_push_type: str | None = PrivateAttr(default=None)
	_q: float | None = PrivateAttr(default=None)
	_m: float | None = PrivateAttr(default=None)

	# initialization
	def model_post_init(self, context):
		super().model_post_init(context)
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
			self._element = element
			if self.mass is None:
				self.mass = element.mass*periodictable.constants.atomic_mass_constant


		# set profile type
		if(isinstance(self.initial_distribution, GaussianBunchDistribution)):
			self._profile_type = 'beam'
		elif(isinstance(self.initial_distribution, UniformDistribution)):
			self._profile_type = 'species'
			self._push_type = 'robust'
		elif(isinstance(self.initial_distribution, AnalyticDistribution)):
			self._profile_type = 'species'
			self._push_type = 'robust'
		elif(isinstance(self.initial_distribution, PiecewiseDistribution)):
			self._profile_type = 'species'
			self._push_type = 'robust'
		else:
			print('Warning: Only Uniform and Gaussian distributions are currently supported.')


	def normalize_units(self):
		# normalized charge, mass, density
		self._q = self.charge/constants.q_e
		self._m = self.mass/constants.m_e

	def fill_dict(self, keyvals):
		if(self._profile_type == 'beam'):
			keyvals['evolution'] = self.beam_evolution
			keyvals['quiet_start'] = self.quiet_start
		else:
			keyvals['push_type'] = self._push_type
		keyvals['q'] = self._q
		keyvals['m'] = self._m

		q_scale = np.abs(self.charge/constants.q_e)
		if(not isinstance(self.initial_distribution, GaussianBunchDistribution)):
			if(self.density_scale is not None):
				keyvals['density'] = self.initial_distribution._norm_density *self.density_scale * q_scale
			else:
				keyvals['density'] = self.initial_distribution._norm_density * q_scale
		self.initial_distribution.fill_dict(keyvals)

	def activate_field_ionization(self,model,product_species):
		raise Exception('Ionization not yet supported in QuickPIC Open-source')


picmistandard.PICMI_MultiSpecies.Species_class = Species
class MultiSpecies(picmistandard.PICMI_MultiSpecies):
	pass


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
	alpha: float | list[float] = Field(default=0, alias=codename + '_alpha',
		description='Twiss alpha of the bunch')
	piecewise_s: list[float] | None = Field(default=None, alias=codename + '_piecewise_s',
		description='Longitudinal coordinates of a piecewise-linear bunch profile [m]')
	piecewise_fs: list[float] | None = Field(default=None, alias=codename + '_piecewise_fs',
		description='Relative densities at piecewise_s')

	_profile: int = PrivateAttr(default=0)
	_if_piecewise: bool = PrivateAttr(default=False)
	_gamma: float | None = PrivateAttr(default=None)
	_norm_density: float | None = PrivateAttr(default=None)

	def model_post_init(self, context):
		super().model_post_init(context)
		self._profile = 0
		self._if_piecewise = False
		if(self.piecewise_fs is not None and self.piecewise_s is not None):
			self._if_piecewise = True
			self._profile = 1

	def normalize_units(self,species, density_norm):
		# get charge, peak density in unnormalized units
		part_charge = species.charge
		total_charge = part_charge * self.n_physical_particles
		if(species.density_scale):
			total_charge *= species.density_scale
		
		peak_density = total_charge/(part_charge * np.prod(self.rms_bunch_size) * (2 * np.pi)**1.5)


		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c


		# QuickPIC takes the spot size in normalized units (k_pe sigma), uth is divergence (sigma_{gamma * beta}), and ufl is fluid velocity (gamma* beta) )

		# normalized spot sizes
		for i in range(3):
			self.rms_bunch_size[i] *= k_pe 
			self.centroid_position[i] *= k_pe 
			self.rms_velocity[i] /= constants.c 
			self.centroid_velocity[i] /= constants.c

		if(self._if_piecewise):
			self.piecewise_s = [i * k_pe for i in self.piecewise_s]

		self._gamma = self.centroid_velocity[2]

		self._norm_density = peak_density/density_norm

	def fill_dict(self,keyvals):
		keyvals['profile'] = self._profile
		keyvals['peak_density'] = self._norm_density
		keyvals['gamma'] = self._gamma
		keyvals['center'] = [0,0,self.centroid_position[2]]
		keyvals['centroid_x'] = [0.0, 0.0, -self.centroid_position[0]]
		keyvals['centroid_y'] = [0.0, 0.0, -self.centroid_position[1]]
		keyvals['quiet_start'] = True
		# QuickPIC coordinate in xi = ct-z
		keyvals['center'][2] *= -1
		keyvals['sigma'] = self.rms_bunch_size
		keyvals['sigma_v'] = self.rms_velocity

		if(self._if_piecewise):
			keyvals['piecewise_fz'] = self.piecewise_fs
			keyvals['piecewise_z'] = self.piecewise_s


class PiecewiseDistribution(picmistandard.PICMI_DistributionExtension):
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
	density: float = Field(description='Physical number density [m^-3]')
	lower_bound: list[float | None] = Field(default_factory=lambda: [None, None, None],
		description='Lower bound of the distribution [m]')
	upper_bound: list[float | None] = Field(default_factory=lambda: [None, None, None],
		description='Upper bound of the distribution [m]')
	rms_velocity: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0],
		description='Thermal velocity spread [m/s]')
	directed_velocity: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0],
		description='Directed, average, proper velocity [m/s]')
	fill_in: bool | None = Field(default=None,
		description='Flags whether to fill in the empty spaced opened up when the grid moves')
	piecewise_s: list[float] = Field(default_factory=lambda: [0.0],
		description='Longitudinal coordinates of the piecewise-linear profile [m]')
	piecewise_fs: list[float] = Field(default_factory=lambda: [1.0],
		description='Densities at piecewise_s [m^-3]')

	# default profile for uniform plasmas
	_profile: int = PrivateAttr(default=0)
	_density_expression: str | None = PrivateAttr(default=None)
	_dens: float | None = PrivateAttr(default=None)
	_norm_density: float | None = PrivateAttr(default=None)

	def __init__(self, density, **kw):
		# keep density as the (optional) positional argument of the pre-pydantic signature
		super().__init__(density=density, **kw)


	def normalize_units(self,species, density_norm):

		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c
		

		for i in range(3):
			if(self.lower_bound[i] is not None):
				self.lower_bound[i] *= k_pe
			if(self.upper_bound[i] is not None):
				self.upper_bound[i] *= k_pe

		self.piecewise_s = [i * k_pe for i in self.piecewise_s]
		self.piecewise_fs = [i/density_norm for i in self.piecewise_fs]
		self._density_expression =  str(self.density/density_norm)
		self._dens = self.density/density_norm
		self._norm_density = 1.0

		if(np.any(self.rms_velocity != 0.0) or np.any(self.directed_velocity != 0.0)):
			print('Warning: QuickPIC does not support rms velocity or directed velocity for Piecewise Distributions.')

	def fill_dict(self,keyvals):
		keyvals['profile'] = self._profile
		keyvals['density'] = self._dens
		keyvals['longitudinal_profile'] = 'piecewise'
		keyvals['piecewise_s'] = self.piecewise_s
		keyvals['piecewise_density'] = self.piecewise_fs


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
	# default profile for uniform plasmas
	_profile: int = PrivateAttr(default=13)
	_density_expression: str | None = PrivateAttr(default=None)
	_norm_density: float | None = PrivateAttr(default=None)


	def normalize_units(self,species, density_norm):

		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c
		

		for i in range(3):
			if(self.lower_bound[i] is not None):
				self.lower_bound[i] *= k_pe
			if(self.upper_bound[i] is not None):
				self.upper_bound[i] *= k_pe

		self._density_expression =  str(self.density/density_norm)
		self._norm_density = 1.0

		if(np.any(self.rms_velocity != 0.0) or np.any(self.directed_velocity != 0.0)):
			print('Warning: QuickPIC does not support rms velocity or directed velocity for Analytic Distributions.')

	def fill_dict(self,keyvals):
		back_str,front_str = construct_bounds(self.lower_bound,self.upper_bound)
		keyvals['profile'] = self._profile
		keyvals['math_func'] = front_str + self._density_expression + back_str


class AnalyticDistribution(picmistandard.PICMI_AnalyticDistribution):
	"""
	QuickPIC-Specific Parameters
	
	### Plasma-specific parameters ####

	self.profile: integer
		Specifies profile-type of uniform plasma.
		profile = 13 (analytic functions x, y, z)
		Profiles are multiplicative f(r,z) = f(r)  * f(z)

	"""
	# default profile for uniform plasmas
	_profile: int = PrivateAttr(default=13)
	_math_func: str | None = PrivateAttr(default=None)
	_norm_density: float | None = PrivateAttr(default=None)

	def model_post_init(self, context):
		super().model_post_init(context)
		self._math_func = self.density_expression
		if(np.any(self.momentum_expressions == None)):
			print('Warning: QuickPIC does not support momentum expressions for Analytic Distributions.')


	def normalize_units(self,species, density_norm):
		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c
		
		self._norm_density = 1.0

		for i in range(3):
			if(self.lower_bound[i] is not None):
				self.lower_bound[i] *= k_pe
			if(self.upper_bound[i] is not None):
				self.upper_bound[i] *= k_pe

		self.density_expression = normalize_math_func(self.density_expression, density_norm)
		self.density_expression =  self.density_expression + '/' + str(density_norm)

		if(np.any(self.rms_velocity != 0.0) or np.any(self.directed_velocity != 0.0)):
			print('Warning: QuickPIC does not support rms velocity or directed velocity for Analytic Distributions.')

	def fill_dict(self,keyvals):
		back_str,front_str = construct_bounds(self.lower_bound,self.upper_bound)
		keyvals['profile'] = self._profile
		keyvals['math_func'] = front_str + self.density_expression + back_str

class ParticleListDistribution(picmistandard.PICMI_ParticleListDistribution):
	def model_post_init(self, context):
		raise Exception('Particle list distributions not yet supported in open-source QuickPIC')


# constant, analytic, or mirror fields not yet supported in QuickPIC 
class ConstantAppliedField(picmistandard.PICMI_ConstantAppliedField):
	def model_post_init(self, context):
		raise Exception("Constant applied fields are not yet supported in QuickPIC Open-source")

class AnalyticAppliedField(picmistandard.PICMI_AnalyticAppliedField):
	def model_post_init(self, context):
		raise Exception("Analytic applied fields are not yet supported in QuickPIC Open-source")

class Mirror(picmistandard.PICMI_Mirror):
	def model_post_init(self, context):
		raise Exception("Mirrors are not yet supported in QuickPIC Open-source")



class BinomialSmoother(picmistandard.PICMI_BinomialSmoother):
	# also accept a single value for all axes, as before the pydantic standard
	n_pass: int | list[int] | None = Field(default=None,
		description='Number of passes along each axis (a single integer applies to all axes)')
	compensation: bool | list[bool] | None = Field(default=None,
		description='Flags whether to apply compensation along each axis (a single flag applies to all axes)')

	def model_post_init(self, context):
		super().model_post_init(context)
		print("Warning: QuickPIC has no BinomialSmoother. Skipping feature.")

class ElectromagneticSolver(picmistandard.PICMI_ElectromagneticSolver):
	"""
	QuickPIC-Specific Parameters
	
	QuickPIC_maximum_iterations: integer
		Number of iterations for predictor corrector solver.
	"""
	maximum_iterations: int | None = Field(default=None, alias=codename + '_maximum_iterations',
		description='Number of iterations for predictor corrector solver')

	def model_post_init(self, context):
		super().model_post_init(context)
		if(self.maximum_iterations == None):
			print('Defaulting to n_iterations = 1 for predictor corrector')
			self.maximum_iterations = 1
	def fill_dict(self,keyvals):
		keyvals['iter'] = self.maximum_iterations
		
class ElectrostaticSolver(picmistandard.PICMI_ElectrostaticSolver):
	def model_post_init(self, context):
		raise Exception('This feature is not supported. Please use the Electromagnetic solver.')

class Cartesian3DGrid(picmistandard.PICMI_Cartesian3DGrid):
	_indx: int | None = PrivateAttr(default=None)
	_indy: int | None = PrivateAttr(default=None)
	_indz: int | None = PrivateAttr(default=None)
	_boundary: str | None = PrivateAttr(default=None)
	_x: list | None = PrivateAttr(default=None)
	_y: list | None = PrivateAttr(default=None)
	_z: list | None = PrivateAttr(default=None)

	def __init__(self, **kw):
		super().__init__(**kw)
		# after the validation, which resolves the vector forms (e.g., number_of_cells from nx, ny, nz)
		self._code_init()

	def _code_init(self):
		dims = 3
		# first check if grid cells are power of two
		assert all( self.power_of_two_check(n) for n in self.number_of_cells), Exception('Number_of_cells must be a power of two in each direction.')

		# extract log2 exponents of each number_of_cells dim
		self._indx, self._indy, self._indz = math.frexp(self.number_of_cells[0])[1] - 1, \
			math.frexp(self.number_of_cells[1])[1] - 1, math.frexp(self.number_of_cells[2])[1] - 1

		# second check to make sure window moving forward at c (window speed doesn't actually matter for QuickPIC)
		assert self.moving_window_velocity == [0, 0, constants.c]

		for i in range(dims):
			assert self.lower_boundary_conditions[i] == self.upper_boundary_conditions[i] 
			if(i < 2):
				if(self.lower_boundary_conditions[i] == 'dirichlet'):
					print('QuickPIC defaults to conductive boundaries (dirichlet).')

		self._boundary = 'conducting'
		self._x = [self.lower_bound[0], self.upper_bound[0]]
		self._y = [self.lower_bound[1], self.upper_bound[1]]
		self._z = [self.lower_bound[2], self.upper_bound[2]]

	def power_of_two_check(self,n):
		return (n & (n-1) == 0) and n != 0

	def normalize_units(self, density_norm):
		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) )
		k_pe = w_pe/constants.c

		#normalize coordinates
		for i in range(2):
			self._x[i] *= k_pe
			self._y[i] *= k_pe
			self._z[i] *= k_pe

	def fill_dict(self,keyvals):
		keyvals['indx'], keyvals['indy'], keyvals['indz'] = self._indx, self._indy, self._indz
		box = {}
		box['x'], box['y'] = self._x, self._y
		# quickpic 3D is in xi not z (multiply by -1 + reverse z coordinate)
		box['z'] = [-self._z[1], -self._z[0]]
		keyvals['box'] = box
		keyvals['boundary'] = self._boundary

		
		

		
		

## QuickPIC is a 3D code. Throw Errors if trying to use 1D, 2D cartesian and cylindrical grids with QuickPIC
class Cartesian1DGrid(picmistandard.PICMI_Cartesian1DGrid):
	def model_post_init(self, context):
		raise Exception('Quickpic does not support 1D Cartesian grids. Please specify a 3D Cartesian Grid.')

class Cartesian2DGrid(picmistandard.PICMI_Cartesian2DGrid):
	def model_post_init(self, context):
		raise Exception('Quickpic does not support 2D cartesian grids. Please specify a 3D Cartesian Grid.')

class CylindricalGrid(picmistandard.PICMI_CylindricalGrid):
	def model_post_init(self, context):
		raise Exception('Quickpic does not support cylindrical grids. Please specify a 3D Cartesian Grid.')

class PseudoRandomLayout(picmistandard.PICMI_PseudoRandomLayout):
	"""
	QuickPIC-Specific Parameters
	
	QuickPIC_np_per_dimension: integer array, optional
		Part per dim in each direction. (for beams only)

	QuickPIC_npmax: integer, optional
		Particle buffer size per MPI partition.
	"""
	np_per_dimension: list[int] | None = Field(default=None, alias=codename + '_np_per_dimension',
		description='Number of beam particles along each direction (defaults to cbrt(n_macroparticles))')
	npmax: int | None = Field(default=None, alias=codename + '_npmax',
		description='Particle buffer size per MPI partition (defaults to twice the number of particles)')

	def model_post_init(self, context):
		super().model_post_init(context)
		# n_macroparticles is required.
		assert self.n_macroparticles is not None, Exception('n_macroparticles must be specified when using PseudoRandomLayout with QuickPIC')
		print(self.n_macroparticles, 'macros')
		if(self.np_per_dimension is None):
			print('Warning: QuickPIC_np_per_dimension was not specified.')
			np_per_dim = int(np.cbrt(self.n_macroparticles))
			self.np_per_dimension = [np_per_dim] * 3
			print('Warning: Casting n_macroparticles = ' + str(self.n_macroparticles) + ' to np_per_dimension = [' + \
				str(np_per_dim) + ',' +str(np_per_dim) + ',' + str(np_per_dim) + ']' )

		if(self.npmax is None):
			self.npmax = int(2 * np.prod(self.np_per_dimension))
	def fill_dict(self, keyvals):
		keyvals['np'] = self.np_per_dimension
		keyvals['npmax'] = self.npmax
				


class GriddedLayout(picmistandard.PICMI_GriddedLayout):
	"""
	QuickPIC-Specific Parameters

	QuickPIC_npmax: integer, optional
		Particle buffer size per MPI partition.
	"""
	npmax: int = Field(default=10**6, alias=codename + '_npmax',
		description='Particle buffer size per MPI partition')

	def model_post_init(self, context):
		super().model_post_init(context)
		print(self.n_macroparticle_per_cell,'macroparts')
		# assert len(self.n_macroparticle_per_cell) !=2, print('Warning: QuickPIC only supports 2-dimensions for n_macroparticle_per_cell')

	def fill_dict(self,keyvals):
		keyvals['npmax'] = self.npmax
		keyvals['ppc'] = self.n_macroparticle_per_cell[:2]



class Simulation(picmistandard.PICMI_Simulation):
	"""
	QuickPIC-Specific Parameters

	QuickPIC_n0: float, optional
		Plasma density [m^3] to normalize units.

	QuickPIC_read_restart: boolean, optional
		Toggle to read from restart files.
	
	QuickPIC_restart_timestep: integer, optional
		Specifies timestep if read_restart = True.

	"""
	cpu_split: list[int] = Field(default_factory=lambda: [1, 1], alias=codename + '_nodes',
		description='MPI-node configuration')
	n0: float | None = Field(default=None, alias=codename + '_n0',
		description='Plasma density [m^-3] to normalize units')
	read_restart: bool = Field(default=False, alias=codename + '_read_restart',
		description='Toggle to read from restart files')
	restart_timestep: int = Field(default=-1, alias=codename + '_restart_timestep',
		description='Timestep to restart from if read_restart = True')
	dump_restart: bool = Field(default=False, alias=codename + '_dump_restart',
		description='Toggle to dump restart files')
	ndump_restart: int = Field(default=-1, alias=codename + '_ndump_restart',
		description='Restart dump period if dump_restart = True')

	### QuickPIC differentiates between beams and plasmas (species)
	_if_beam: list = PrivateAttr(default_factory=list)

	def model_post_init(self, context):
		super().model_post_init(context)
		# set verbose default
		if(self.verbose is None):
			self.verbose = 0
		assert self.time_step_size is not None, Exception('QuickPIC requires a time step size for the 3D loop.')

		if(self.particle_shape != 'linear' ):
			print('Warning: Defaulting to linear particle shapes.')
			self.particle_shape = 'linear'

		# normalize simulation time
		if(self.n0 is not None):
			self.normalize_simulation()

		# check to read from restart files
		if(self.read_restart):
			assert self.restart_timestep != -1, Exception('Please specify QuickPIC_restart_timestep')

		# check if dumping restart files
		if(self.dump_restart):
			assert self.ndump_restart != -1, Exception('Please specify QuickPIC_ndump_restart')


	def normalize_simulation(self):
		w_pe = np.sqrt(constants.q_e**2.0 * self.n0/(constants.ep0 * constants.m_e) ) 
		if(self.max_time is not None):
			self.max_time *= w_pe
		self.time_step_size *= w_pe
		self.solver.grid.normalize_units(self.n0)
	
	def add_species(self, species, layout, initialize_self_field = None):
		if(isinstance(species, MultiSpecies)):
			for spec in species.species_instances_list:
				picmistandard.PICMI_Simulation.add_species( self, spec, layout,
										  initialize_self_field )

				# handle checks for beams
				self._if_beam.append(spec._profile_type == 'beam')
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
			self._if_beam.append(species._profile_type == 'beam')

	def add_laser(self,laser, injection_method):
		raise Exception('Laser modules are not available in open-source QuickPIC')
			


	def fill_dict(self, keyvals):
		if(self.max_time is None):
			self.max_time = self.max_steps * self.time_step_size
		# fill grid and mpi params
		keyvals['nodes'] = self.cpu_split
		self.solver.grid.fill_dict(keyvals)

		# fill simulation time and dt
		keyvals['time'] = self.max_time
		keyvals['dt'] = self.time_step_size


		if(self.n0 is not None):
			keyvals['n0'] = self.n0 * 1.e-6 # in density in cm^{-3}
		keyvals['nbeams'] = int(np.sum(self._if_beam))
		keyvals['nspecies'] = len(self._if_beam) - keyvals['nbeams']
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
			if(self._if_beam[i]):
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
		raise Exception('The simulation step feature is not yet supported for QuickPIC. Please call write_input_file() to construct the input deck.')
		print('gothere')

class FieldDiagnostic(picmistandard.PICMI_FieldDiagnostic):
	"""
	QuickPIC-Specific Parameters

	QuickPIC_slice: array, optional
		Specifies plane and index of third coordinate to dump (e.g., ["yz", 256])
	"""
	slice: list | None = Field(default=None, alias=codename + '_slice',
		description='Plane and index of third coordinate to dump (e.g., ["yz", 256])')

	_field_list: list = PrivateAttr(default_factory=list)
	_source_list: list = PrivateAttr(default_factory=list)

	def model_post_init(self, context):
		super().model_post_init(context)
		self._field_list = []
		self._source_list = []
		if('E' in self.data_list):
			self._field_list += ['ex','ey','ez']
		if('B' in self.data_list):
			self._field_list += ['bx','by','bz']
		if('rho' in self.data_list):
			self._source_list += ['charge']
		if('J' in self.data_list):
			self._source_list += ['jx','jy','jz']


		if('Ex' in self.data_list):
			self._field_list.append('ex')
		if('Ey' in self.data_list):
			self._field_list.append('ey')
		if('Ez' in self.data_list):
			self._field_list.append('ez')

		if('Bx' in self.data_list):
			self._field_list.append('bx')
		if('By' in self.data_list):
			self._field_list.append('by')
		if('Bz' in self.data_list):
			self._field_list.append('bz')

		if('Jx' in self.data_list):
			self._source_list.append('jx')
		if('Jy' in self.data_list):
			self._source_list.append('jy')
		if('Jz' in self.data_list):
			self._source_list.append('jz')


		# TODO: add to PICMI standard
		if('psi' in self.data_list):
			self._field_list += ['psi']


	def fill_dict_fld(self,keyvals):
		keyvals['name'] = self._field_list
		keyvals['ndump'] = self.period
		if(self.slice):
			keyvals['slice'] = [self.slice]

	def fill_dict_src(self,keyvals):
		keyvals['name'] = self._source_list
		keyvals['ndump'] = self.period
		if(self.slice):
			keyvals['slice'] = [self.slice]


# QuickPIC does not support electrostatic and boosted frame diagnostic 
class ElectrostaticFieldDiagnostic(picmistandard.PICMI_ElectrostaticFieldDiagnostic):
	def model_post_init(self, context):
		raise Exception("Electrostatic field diagnostic not supported in QuickPIC")

class LabFrameParticleDiagnostic(picmistandard.PICMI_LabFrameParticleDiagnostic):
	def model_post_init(self, context):
		raise Exception("Boosted frame diagnostics not support in QuickPIC")

class LabFrameFieldDiagnostic(picmistandard.PICMI_LabFrameFieldDiagnostic):
	def model_post_init(self, context):
		raise Exception("Boosted frame diagnostics not support in QuickPIC")


	
class ParticleDiagnostic(picmistandard.PICMI_ParticleDiagnostic):
	"""
	QuickPIC-Specific Parameters

	QuickPIC_sample: integer, optional
		Dumps every nth particle.
	"""
	sample: int = Field(default=1, alias=codename + '_sample',
		description='Dumps every nth particle')

	def model_post_init(self, context):
		super().model_post_init(context)
		print('Warning: Particle diagnostic reporting momentum, position and charge data')
		if(self.write_dir and self.write_dir != '.'):
			print('Warning: ParticleDiagnostic write_dir set to "."') 
		if(self.step_min):
			print('Warning: ParticleDiagnostic step_min set to 0')
		if(self.step_max):
			print('Warning: ParticleDiagnostic step_max set to no limit')

	def fill_dict_fld(self,keyvals):
		pass

	def fill_dict_src(self,keyvals):
		keyvals['name'] = ["raw"]
		keyvals['ndump'] = self.period
		keyvals['sample'] = self.sample


def normalize_math_func(math_func, density_norm):
	w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
	k_pe = w_pe/constants.c

	## handle overlap with funcs and variable names
	funcs = ['exp','max', 'tan', 'sqrt','not','int','rect','step']
	funcs2 = ['e1p','ma1', '1an', 'sqr1', 'no1', 'in1', 'rec1', 's1ep']
	math_func2 = math_func[:]
	for i in range(len(funcs)):
		key1, key2 = funcs[i], funcs2[i]
		math_func2 = '{}'.format(math_func2).replace(key1,key2)


	math_func2 = '{}'.format(math_func2).replace('x', '(' + format_decimal(1.0/k_pe) +'* x)')
	math_func2 = '{}'.format(math_func2).replace('y', '(' + format_decimal(1.0/k_pe) +'* y)')
	math_func2 = '{}'.format(math_func2).replace('z', '(' + format_decimal(1.0/k_pe) +'* z)')
	math_func2 = '{}'.format(math_func2).replace('t', '(' + format_decimal(1.0/w_pe) +'* t)')
	for i in range(len(funcs)):
		key1, key2 = funcs2[i], funcs[i]
		math_func2 = '{}'.format(math_func2).replace(key1,key2)
	
	## handle overlap with funcs and variable names
	return math_func2

def format_decimal(decimal):
	str_out= '%.6e' % Decimal(str(decimal))
	return str_out

def construct_bounds(lower_bound,upper_bound):
	front_str =''
	back_str = ''
	coords = ['x','y','z']
	for i in range(3):
		if(upper_bound[i] is not None):
			if(len(front_str) > 0):
				front_str = front_str + ' && ' + coords[i] + '<= (' + format_decimal(upper_bound[i]) + ') ' 
			else:
				front_str = front_str + coords[i] + '<= (' + format_decimal(upper_bound[i]) + ') ' 

		if(lower_bound[i] is not None):
			if(len(front_str) > 0):
				front_str = front_str + ' && ' + coords[i] + '>= (' + format_decimal(lower_bound[i]) + ') ' 
			else:
				front_str = front_str + coords[i] + '>= (' + format_decimal(lower_bound[i]) + ') ' 
			

	front_str = 'if(' + front_str + ','
	back_str = ', 0 )'
	return back_str,front_str

