# Copyright 2022-2023 Thamine Dalichaouch, Frank Tsung
# QPAD FACET extension of PICMI standard

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
import importlib

# PICMI 0.35+ with pydantic classes: https://github.com/picmi-standard/picmi/pull/133
if tuple(map(int, re.findall(r'\d+', picmistandard.__version__)[:3])) < (0, 35, 0):
	raise ImportError('picmistandard>=0.35.0 required, found ' + picmistandard.__version__ +
		' in ' + picmistandard.__file__)

importlib.reload(picmistandard)
encoder.FLOAT_REPR = lambda o: format(o, '.4f')

codename = 'QPAD'
picmistandard.register_codename(codename)


def to_scientific_notation(value, nprec = 7):
	return float(f"{value:.{nprec}e}")


class constants:
	c = 299792458.
	ep0 = 8.8541878128e-12
	mu0 = 4 * np.pi * 1e-7
	q_e = 1.602176634e-19
	m_e = 9.1093837015e-31
	m_p = 1.67262192369e-27

picmistandard.register_constants(constants)
# Species Class
class Neutral(picmistandard.PICMI_Species):
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
	ion_max: int | None = Field(default=None, alias=codename + '_ion_max',
		description='Maximum ionization level (defaults to the atomic number)')

	_element: int | None = PrivateAttr(default=None)
	_profile_type: str | None = PrivateAttr(default=None)
	_push_type: str | None = PrivateAttr(default=None)
	_q: float | None = PrivateAttr(default=None)
	_m: float | None = PrivateAttr(default=None)

	# initialization 
	def model_post_init(self, context):
		super().model_post_init(context)
		
		# self.charge = self.charge_state * constants.q_e
		self.charge = -constants.q_e
		self.mass = constants.m_e
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
		self._element = element.number
		if(self.ion_max is None):
			self.ion_max = self._element

		# set profile type
		
		if(isinstance(self.initial_distribution, UniformDistribution)):
			self._profile_type = 'neutral'
			self._push_type = 'robust'
		elif(isinstance(self.initial_distribution, AnalyticDistribution)):
			self._profile_type = 'neutral'
			self._push_type = 'robust'
		elif(isinstance(self.initial_distribution, PiecewiseDistribution)):
			self._profile_type = 'neutral'
			self._push_type = 'robust'
		else:
			print('Warning: Only Uniform, Analytic, and Piecewise distributions are currently supported.')


	def normalize_units(self):
		# normalized charge, mass, density
		self._q = self.charge/constants.q_e
		self._m = self.mass/constants.m_e



	def fill_dict(self, keyvals, if_lasers):
		if(if_lasers):
			keyvals['push_type'] = self._push_type + '_pgc'
		else:
			keyvals['push_type'] = self._push_type
		keyvals['q'] = self._q
		keyvals['m'] = self._m
		keyvals['element'] = self._element
		keyvals['ion_max'] = self.ion_max
		q_scale = np.abs(self.charge/constants.q_e)
		if(not isinstance(self.initial_distribution, FileDistribution)):
			if(self.density_scale is not None):
				keyvals['density'] = self.initial_distribution._norm_density *self.density_scale * q_scale
			else:
				keyvals['density'] = self.initial_distribution._norm_density * q_scale
			keyvals['density'] = to_scientific_notation(keyvals['density'])
		self.initial_distribution.fill_dict(keyvals)

	def activate_field_ionization(self,model,product_species):
		return


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
	beam_evolution: bool = Field(default=True, alias=codename + '_beam_evolution',
		description='Toggles beam evolution')
	quiet_start: bool = Field(default=True, alias=codename + '_quiet_start',
		description='Adds image particles to suppress the statistic noise')

	_element: object = PrivateAttr(default=None)
	_profile_type: str | None = PrivateAttr(default=None)
	_push_type: str | None = PrivateAttr(default=None)
	_geometry: str | None = PrivateAttr(default=None)
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

		# print(self.charge)

		# set profile type
		if(isinstance(self.initial_distribution, GaussianBunchDistribution)):
			self._profile_type = 'beam'
			self._geometry = 'cartesian'
			self._push_type = 'reduced'
		elif(isinstance(self.initial_distribution, FileDistribution)):
			self._profile_type = 'beam'
			self._push_type = 'reduced'
			self._geometry = 'cartesian'
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



	def fill_dict(self, keyvals, if_lasers):
		if(self._profile_type == 'beam'):
			keyvals['evolution'] = self.beam_evolution
			keyvals['quiet_start'] = self.quiet_start
			keyvals['geometry'] = self._geometry
		else:
			if(if_lasers):
				keyvals['push_type'] = self._push_type + '_pgc'
			else:
				keyvals['push_type'] = self._push_type
		keyvals['q'] = self._q
		keyvals['m'] = self._m

		q_scale = np.abs(self.charge/constants.q_e)
		if(not isinstance(self.initial_distribution, FileDistribution)):
			if(self.density_scale is not None):
				keyvals['density'] = self.initial_distribution._norm_density *self.density_scale * q_scale
			else:
				keyvals['density'] = self.initial_distribution._norm_density * q_scale
			keyvals['density'] = to_scientific_notation(keyvals['density'])
		self.initial_distribution.fill_dict(keyvals)

	def activate_field_ionization(self,model,product_species):
		return


picmistandard.PICMI_MultiSpecies.Species_class = Species
class MultiSpecies(picmistandard.PICMI_MultiSpecies):
	pass



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
	piecewise_s: list[float] | None = Field(default=None, alias=codename + '_piecewise_s',
		description='Longitudinal coordinates of a piecewise-linear bunch profile [m]')
	piecewise_fs: list[float] | None = Field(default=None, alias=codename + '_piecewise_fs',
		description='Relative densities at piecewise_s')
	alpha: float | list[float] | None = Field(default=None, alias=codename + '_alpha',
		description='Twiss alpha of the bunch')

	_profile: list | None = PrivateAttr(default=None)
	_if_piecewise: bool = PrivateAttr(default=False)
	_gamma: float | None = PrivateAttr(default=None)
	_q: float | None = PrivateAttr(default=None)
	_m: float | None = PrivateAttr(default=None)
	_norm_density: float | None = PrivateAttr(default=None)
	_tot_charge: float | None = PrivateAttr(default=None)

	def model_post_init(self, context):
		super().model_post_init(context)
		self._profile = ['gaussian', 'gaussian', 'gaussian']
		self._if_piecewise = False
		if(self.piecewise_fs is not None and self.piecewise_s is not None):
			self._if_piecewise = True
			# print('piecewise fs/s', self.piecewise_fs)
			self._profile = ['gaussian', 'gaussian', 'piecewise-linear']

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

		if(self._if_piecewise):
			self.piecewise_s = [i * k_pe for i in self.piecewise_s]
		self._gamma = self.centroid_velocity[2]
		self.centroid_position[2] *= -1
		# normalized charge, mass, density
		self._q = species.charge/constants.q_e
		self._m = species.mass/constants.m_e
		self._norm_density = peak_density/density_norm

		self._tot_charge = total_charge/(-constants.q_e * density_norm * k_pe**-3)

	def fill_dict(self,keyvals):
		keyvals['profile'] = self._profile
		keyvals['gamma'] = to_scientific_notation(self._gamma)
		# keyvals['gauss_center'] = [to_scientific_notation(i) for i in self.centroid_position]
		centroid_ = [0, 0, self.centroid_position[2]]
		keyvals['gauss_center'] = [to_scientific_notation(i) for i in centroid_]
		keyvals['perp_offset_x'] = [0, to_scientific_notation(self.centroid_position[0]), 0]
		keyvals['perp_offset_y'] = [0, to_scientific_notation(self.centroid_position[1]), 0]
		keyvals['push_type'] = 'reduced'
		if(self.alpha is not None):
			keyvals['alpha'] = self.alpha
		# keyvals['total_charge'] = self.tot_charge
		# QPAD coordinate in xi = ct-z

		if(self._if_piecewise):
			keyvals['piecewise_fx3'] = self.piecewise_fs
			keyvals['piecewise_x3'] = self.piecewise_s
			rms_size_ = [to_scientific_notation(i) for i in self.rms_bunch_size]
			rms_size_[2] = "none"
			keyvals['gauss_sigma'] = rms_size_

			for j in range(2):
				keyvals['range' + str(j+1)] = [to_scientific_notation(-4 * self.rms_bunch_size[j] +centroid_[j]),\
				 to_scientific_notation(4 * self.rms_bunch_size[j] + centroid_[j])]
			keyvals['range' + str(3)] = [to_scientific_notation(np.min(self.piecewise_s)), to_scientific_notation(np.max(self.piecewise_s))]
		else:
			keyvals['gauss_sigma'] = [to_scientific_notation(i) for i in self.rms_bunch_size]
			for j in range(3):
				keyvals['range' + str(j+1)] = [to_scientific_notation(-4 * self.rms_bunch_size[j] +centroid_[j]),\
				 to_scientific_notation(4 * self.rms_bunch_size[j] + centroid_[j])]
		
		# if(self.tot_charge is not None):
		# 	keyvals['total_charge'] = self.tot_charge
		# keyvals['gauss_sigma'] = [to_scientific_notation(i) for i in self.rms_bunch_size]

		keyvals['uth'] = [to_scientific_notation(i) for i in self.rms_velocity]
		

class FileDistribution(picmistandard.PICMI_DistributionExtension):
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
	filename: str | None = Field(default=None, description='Beam file (HDF5) in QPAD units')
	beam_center: list[float] = Field(default_factory=lambda: [0, 0 ,0],
		description='Position of the beam center [m]')
	file_center: list[float] = Field(default_factory=lambda: [0, 0, 0],
		description='Position of the beam center in the file [m]')
	has_spin: bool = Field(default=False, description='Whether the file contains spin data')
	npmax: int = Field(default=2*10**6, alias=codename + '_npmax',
		description='Particle buffer size per MPI partition')

	def __init__(self, filename = None, **kw):
		# keep filename as the (optional) positional argument of the pre-pydantic signature
		super().__init__(filename=filename, **kw)
		


	def normalize_units(self,species, density_norm):
		# normalized charge, mass, density
		# self.q = species.charge/constants.q_e
		# self.m = species.mass/constants.m_e

		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c

		self.file_center= [k_pe * i for i in self.file_center]
		self.beam_center= [k_pe * i for i in self.beam_center]



	def fill_dict(self,keyvals):
		keyvals['filename'] = self.filename
		keyvals['beam_center'] = self.beam_center
		keyvals['file_center'] = self.file_center
		keyvals['push_type'] = 'boris'
		keyvals['npmax'] = self.npmax

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
	# default profile for uniform plasmas
	_profile: list = PrivateAttr(default_factory=lambda: ['uniform', 'uniform'])
	_norm_density: float | None = PrivateAttr(default=None)
		


	def normalize_units(self,species, density_norm):
		# normalize plasma density
		self._norm_density = self.density/density_norm

		# normalized charge, mass, density
		# self.q = species.charge/constants.q_e
		# self.m = species.mass/constants.m_e

		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c

		# self.density_expression =  str(self.norm_density)
		# self.norm_density = 1.0

		for i in range(3):
			if(self.lower_bound[i] is not None):
				self.lower_bound[i] *= k_pe
			if(self.upper_bound[i] is not None):
				self.upper_bound[i] *= k_pe

		for i in range(3):
			self.rms_velocity[i] /= constants.c 

		# if(np.any(self.directed_velocity != 0.0)):
		# 	print('Warning: ' + codename + ' does not support directed velocity for Analytic Distributions.')

	def fill_dict(self,keyvals):
		back_str,front_str = construct_bounds(self.lower_bound,self.upper_bound)
		keyvals['profile'] = self._profile
		keyvals['uth'] = [to_scientific_notation(i) for i in self.rms_velocity]
		keyvals['density'] = to_scientific_notation(self._norm_density)


class PiecewiseDistribution(picmistandard.PICMI_DistributionExtension):
	"""
	QPAD-Specific Parameters
	
	### Plasma-specific parameters ####

	self.profile: integer
		Specifies profile-type of uniform plasma.
		profile = 0 (uniform plasma or piecewise linear function in z), 12 (piecewise linear along r and z)
		Profiles are multiplicative f(r,z) = f(r)  * f(z)

	

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

	_profile: list = PrivateAttr(default_factory=lambda: ['uniform', 'piecewise-linear'])
	_norm_density: float | None = PrivateAttr(default=None)

	def __init__(self, density, **kw):
		# keep density as the (optional) positional argument of the pre-pydantic signature
		super().__init__(density=density, **kw)
		


	def normalize_units(self,species, density_norm):
		# normalize plasma density
		self._norm_density = self.density/density_norm

		# normalized charge, mass, density
		# self.q = species.charge/constants.q_e
		# self.m = species.mass/constants.m_e

		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c

		self.piecewise_s = [k_pe * i for i in self.piecewise_s]
		self.piecewise_fs = [i/density_norm for i in self.piecewise_fs]
		for i in range(3):
			if(self.lower_bound[i] is not None):
				self.lower_bound[i] *= k_pe
			if(self.upper_bound[i] is not None):
				self.upper_bound[i] *= k_pe

		for i in range(3):
			self.rms_velocity[i] /= constants.c 

		# if(np.any(self.directed_velocity != 0.0)):
		# 	print('Warning: ' + codename + ' does not support directed velocity for Analytic Distributions.')


	def fill_dict(self,keyvals):
		keyvals['profile'] = self._profile
		keyvals['uth'] = [to_scientific_notation(i) for i in self.rms_velocity]
		keyvals['density'] = to_scientific_notation(self._norm_density)
		keyvals['piecewise_s'] = [to_scientific_notation(i) for i in self.piecewise_s]
		keyvals['piecewise_fs'] = [to_scientific_notation(i) for i in self.piecewise_fs]





class AnalyticDistribution(picmistandard.PICMI_AnalyticDistribution):
	"""
	QPAD-Specific Parameters
	
	### Plasma-specific parameters ####

	self.profile: integer
		Specifies profile-type of uniform plasma.
		profile = 13 (analytic functions x, y, z)
		Profiles are multiplicative f(r,z) = f(r)  * f(z)

	"""
	# default profile for uniform plasmas
	_profile: list = PrivateAttr(default_factory=lambda: ['analytic', 'analytic'])
	_norm_density: float | None = PrivateAttr(default=None)

	def model_post_init(self, context):
		super().model_post_init(context)
		if(np.any(self.momentum_expressions == None)):
			print('Warning: QPAD does not support momentum expressions for Analytic Distributions.')


	def normalize_units(self,species, density_norm):

		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c

		for i in range(3):
			if(self.lower_bound[i] is not None):
				self.lower_bound[i] *= k_pe
			if(self.upper_bound[i] is not None):
				self.upper_bound[i] *= k_pe

		self.density_expression = normalize_math_func(self.density_expression, density_norm)
		self.density_expression =  self.density_expression + '/' + str(density_norm)
		self._norm_density = 1.0

		for i in range(3):
			self.rms_velocity[i] /= constants.c 

		# if(np.any(self.directed_velocity != 0.0)):
		# 	print('Warning: ' + codename + ' does not support directed velocity for Analytic Distributions.')


	def fill_dict(self,keyvals):
		# if(self.lower_bound is not None)
		back_str,front_str = construct_bounds(self.lower_bound,self.upper_bound)
		keyvals['profile'] = self._profile
		keyvals['uth'] = [to_scientific_notation(i) for i in self.rms_velocity]
		# keyvals['math_func'] = front_str + self.density_expression + back_str
		keyvals['math_func'] =  self.density_expression


class ParticleListDistribution(picmistandard.PICMI_ParticleListDistribution):
	def model_post_init(self, context):
		raise Exception('Particle list distributions not yet supported in QPAD')


# constant, analytic, or mirror fields not yet supported in QPAD
class ConstantAppliedField(picmistandard.PICMI_ConstantAppliedField):
	def model_post_init(self, context):
		raise Exception("Constant applied fields are not yet supported in QPAD")

class AnalyticAppliedField(picmistandard.PICMI_AnalyticAppliedField):
	def model_post_init(self, context):
		raise Exception("Analytic applied fields are not yet supported in QPAD")

class Mirror(picmistandard.PICMI_Mirror):
	def model_post_init(self, context):
		raise Exception("Mirrors are not yet supported in QPAD")


class ElectromagneticSolver(picmistandard.PICMI_ElectromagneticSolver):
	"""
	QPAD-Specific Parameters
	
	QPAD_maximum_iterations: integer
		Number of iterations for predictor corrector solver.
	"""
	maximum_iterations: int | None = Field(default=None, alias=codename + '_maximum_iterations',
		description='Number of iterations for predictor corrector solver')

	def model_post_init(self, context):
		super().model_post_init(context)
		if(self.maximum_iterations == None):
			print('Defaulting to n_iterations = 10 for predictor corrector')
			self.maximum_iterations = 10
	def fill_dict(self,keyvals):
		keyvals['iter_max'] = self.maximum_iterations
		keyvals['iter_reltol'] = 1e-3
		keyvals['iter_abstol'] = 1e-3
		keyvals['relax_fac'] = to_scientific_notation(1e-3 * (self.grid.dr/0.02)**2)

		
		
class ElectrostaticSolver(picmistandard.PICMI_ElectrostaticSolver):
	def model_post_init(self, context):
		raise Exception('This feature is not supported. Please use the Electromagnetic solver.')

## Throw Errors if trying to use 1D/2D/3D cartesian grids with QPAD
class Cartesian1DGrid(picmistandard.PICMI_Cartesian1DGrid):
	def model_post_init(self, context):
		raise Exception(codename + ' does not support this feature. Please specify a Cylindrical Grid.')

class Cartesian2DGrid(picmistandard.PICMI_Cartesian2DGrid):
	def model_post_init(self, context):
		raise Exception(codename + ' does not support this feature. Please specify a Cylindrical Grid.')

class Cartesian3DGrid(picmistandard.PICMI_Cartesian3DGrid):
	def model_post_init(self, context):
		raise Exception(codename + ' does not support this feature. Please specify a Cylindrical Grid.')


class CylindricalGrid(picmistandard.PICMI_CylindricalGrid):
	_dr: float | None = PrivateAttr(default=None)
	_dz: float | None = PrivateAttr(default=None)
	_boundary: str | None = PrivateAttr(default=None)
	_r: list | None = PrivateAttr(default=None)
	_z: list | None = PrivateAttr(default=None)

	def __init__(self, **kw):
		super().__init__(**kw)
		# after the validation, which resolves the vector forms (e.g., number_of_cells from nr, nz)
		self._code_init()

	@property
	def dr(self):
		"""Cell size along r (normalized by normalize_units)"""
		return self._dr

	@property
	def dz(self):
		"""Cell size along z (normalized by normalize_units)"""
		return self._dz

	def _code_init(self):
		dims = 2

		# second check to make sure window moving forward at c (window speed doesn't actually matter for QPAD)

		# check for open boundaries at r_max
		if(self.upper_boundary_conditions[0] != 'open' or self.lower_boundary_conditions[1] != 'open' or self.upper_boundary_conditions[1] !='open'): 
			print('QPAD Defaulting to open boundaries in r and z-directions.')

		self._dr = np.abs(self.upper_bound[0]- self.lower_bound[0])/self.number_of_cells[0]
		self._dz = np.abs(self.upper_bound[1]- self.lower_bound[1])/self.number_of_cells[1]
		self._boundary = 'open'
		self._r = [self.lower_bound[0], self.upper_bound[0]]
		self._z = [self.lower_bound[1], self.upper_bound[1]]

	def power_of_two_check(self,n):
		return (n & (n-1) == 0) and n != 0

	def normalize_units(self, density_norm):
		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c

		#normalize coordinates 
		for i in range(2):
			self._r[i] *= k_pe
			self._z[i] *= k_pe
		self._dr *= k_pe
		self._dz *= k_pe

	def fill_dict(self,keyvals):
		keyvals['grid'] = self.number_of_cells
		keyvals['max_mode'] = self.n_azimuthal_modes
		box = {}
		# box['r'] = self.r
		box['r'] = [to_scientific_notation(i) for i in self._r]
		# QPAD 3D is in xi not z (multiply by -1 + reverse z coordinate)
		box['z'] = [to_scientific_notation(i) for i in [-self._z[1], -self._z[0]]]
		keyvals['box'] = box
		keyvals['field_boundary'] = self._boundary


class FileLayout(picmistandard.PICMI_LayoutExtension):
	"""
	QPAD-Specific Parameters
	
	QPAD_np_per_dimension: integer array, optional
		Part per dim in each direction. (for beams only)

	QPAD_npmax: integer, optional
		Particle buffer size per MPI partition.

	QPAD_num_theta: integer, optional
		Number of particles in azimuthal direction. Defaults to 8 * n_azimuthal_modes.
	"""

	grid: picmistandard.PICMI_AnyGrid | None = Field(default=None,
		description='Grid object specifying the grid to follow')

	_profile_type: str = PrivateAttr(default='file')


	def fill_dict(self, keyvals,profile_type):
		keyvals['profile_type'] = self._profile_type


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

	_profile_type: str = PrivateAttr(default='random')

	def model_post_init(self, context):
		super().model_post_init(context)
		# n_macroparticles is required.
		assert self.n_macroparticles is not None, Exception('n_macroparticles must be specified when using PseudoRandomLayout with QPAD')


	def fill_dict(self, keyvals,profile_type):
		keyvals['npmax'] = self.n_macroparticles * 2 
		keyvals['total_num'] = self.n_macroparticles
		keyvals['profile_type'] = self._profile_type
		keyvals['random_theta'] = False
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
	npmax: int = Field(default=2*10**6, alias=codename + '_npmax',
		description='Particle buffer size per MPI partition')
	num_theta: int = Field(default=1, alias=codename + '_num_theta',
		description='Number of particles in azimuthal direction')

	# setting profile type to standard
	_profile_type: str = PrivateAttr(default='standard')

	def model_post_init(self, context):
		super().model_post_init(context)
		# assert len(self.n_macroparticle_per_cell) !=2, print('Warning: '+ codename + ' only supports 2-dimensions for n_macroparticle_per_cell')
		# if(self.num_theta * self.n_macroparticle_per_cell[1] < 8 * self.grid.n_azimuthal_modes):
		# 	self.num_theta = int((8 * self.grid.n_azimuthal_modes)/self.n_macroparticle_per_cell[1])
		# 	print('Warning: total azimthal ppc increased to ' + str(self.num_theta * self.n_macroparticle_per_cell[1]))

	def fill_dict(self,keyvals,profile_type):
		keyvals['npmax'] = self.npmax
		# keyvals['profile_type'] = self.profile_type
		if(profile_type == 'beam'):
			keyvals['ppc'] = self.n_macroparticle_per_cell
		elif(profile_type == 'species' or profile_type == 'neutral'):
			keyvals['ppc'] = self.n_macroparticle_per_cell[:2]
		keyvals['num_theta'] = self.num_theta
		keyvals['random_theta'] = False



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
	cpu_split: list[int] = Field(default_factory=lambda: [1, 1], alias=codename + '_nodes',
		description='MPI-node configuration')
	n0: float | None = Field(default=None, alias=codename + '_n0',
		description='Plasma density [m^-3] to normalize units')
	random_seed: int = Field(default=10, alias=codename + '_random_seed',
		description='Number of seeds for pseudo-random numbers')
	algorithm: str = Field(default='standard', alias=codename + '_algorithm',
		description='Type of algorithm (standard, pgc, etc.)')
	if_timing: bool = Field(default=False, alias=codename + '_timings',
		description='Toggle to report timings')
	read_restart: bool = Field(default=False, alias=codename + '_read_restart',
		description='Toggle to read from restart files')
	restart_timestep: int = Field(default=-1, alias=codename + '_restart_timestep',
		description='Timestep to restart from if read_restart = True')
	dump_restart: bool = Field(default=False, alias=codename + '_dump_restart',
		description='Toggle to dump restart files')
	ndump_restart: int = Field(default=-1, alias=codename + '_ndump_restart',
		description='Restart dump period if dump_restart = True')

	### QPAD differentiates between beams, neutrals, and plasmas (species)
	_if_beam: list = PrivateAttr(default_factory=list)
	# no of neutrals
	_if_neutral: list = PrivateAttr(default_factory=list)
	# no of species
	_if_species: list = PrivateAttr(default_factory=list)

	def model_post_init(self, context):
		super().model_post_init(context)
		# set verbose default
		if(self.verbose is None):
			self.verbose = 0
		assert self.time_step_size is not None, Exception('QPAD requires a time step size for the 3D loop.')

		if(self.particle_shape not in  ['linear']):
			print('Warning: Defaulting to linear particle shapes.')
			self.particle_shape = 'linear'

		# normalize simulation time
		if(self.n0 is not None):
			self.normalize_simulation()

		# check to read from restart files
		if(self.read_restart):
			assert self.restart_timestep != -1, Exception('Please specify ' + codename + '_restart_timestep')

		# check if dumping restart files
		if(self.dump_restart):
			assert self.ndump_restart != -1, Exception('Please specify' + codename + '_ndump_restart')


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
				self._if_neutral.append(spec._profile_type == 'neutral')
				self._if_species.append(spec._profile_type == 'species')
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
			self._if_neutral.append(species._profile_type == 'neutral')
			self._if_species.append(species._profile_type == 'species')
	def add_laser(self, laser, injection_method):
		picmistandard.PICMI_Simulation.add_laser(self, laser, injection_method)
		if(injection_method is not None):
			print('Antenna is not supported in QPAD. Initializating laser in box at t=0.')
			laser.focal_position[2] -= injection_method.position[2]
		if(self.n0 is not None):
			laser.normalize_units(self.n0)


			


	def fill_dict(self, keyvals):
		if(self.max_time is None):
			self.max_time = self.max_steps * self.time_step_size

		# fill grid and mpi params
		keyvals['nodes'] = self.cpu_split
		self.solver.grid.fill_dict(keyvals)

		# fill simulation time and dt
		keyvals['time'] = to_scientific_notation(self.max_time)
		keyvals['dt'] = to_scientific_notation(self.time_step_size)
		keyvals['interpolation'] = self.particle_shape


		if(self.n0 is not None):
			keyvals['n0'] = to_scientific_notation(self.n0 * 1.e-6) # in density in cm^{-3}
		keyvals['nbeams'] = int(np.sum(self._if_beam))
		keyvals['nspecies'] = int(np.sum(self._if_species))
		keyvals['nneutrals'] = int(np.sum(self._if_neutral))
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

		# neutral objects
		neutral_dicts = []

		# lasers objects
		laser_dicts = []

		# field object
		field_dict = {}
		# iterate over species handle beams first
		for i in range(len(self.species)):
			spec = self.species[i]
			temp_dict = {}
			self.layouts[i].fill_dict(temp_dict,spec._profile_type)
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
			if(self._if_beam[i]):
				beam_dicts.append(temp_dict)
			elif(self._if_neutral[i]):
				neutral_dicts.append(temp_dict)
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
		if(len(beam_dicts) > 0):
			total_dict['beam'] = beam_dicts
		if(len(species_dicts) > 0):
			total_dict['species'] = species_dicts
		if(len(neutral_dicts) > 0):
			total_dict['neutrals'] = neutral_dicts
		if(len(laser_dicts) > 0):
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
	_field_list: list = PrivateAttr(default_factory=list)
	_source_list: list = PrivateAttr(default_factory=list)

	def model_post_init(self, context):
		super().model_post_init(context)
		assert self.write_dir != '.', Exception("Write directory feature not yet supported.")
		assert self.period > 0, Exception("Diagnostic period is not valid")
		self._field_list = []
		self._source_list = []
		if('E' in self.data_list):
			self._field_list += ['er_cyl_m','ephi_cyl_m','ez_cyl_m']
		if('B' in self.data_list):
			self._field_list += ['br_cyl_m','bphi_cyl_m','bz_cyl_m']
		if('rho' in self.data_list):
			self._source_list += ['charge_cyl_m']
		if('J' in self.data_list):
			self._source_list += ['jr_cyl_m','jphi_cyl_m','jz_cyl_m']


		if('Ex' in self.data_list or 'Er' in self.data_list):
			self._field_list.append('er_cyl_m')
		if('Ey' in self.data_list or 'Ephi' in self.data_list):
			self._field_list.append('ephi_cyl_m')
		if('Ez' in self.data_list or 'Ez' in self.data_list):
			self._field_list.append('ez_cyl_m')

		if('Bx' in self.data_list or 'Br' in self.data_list):
			self._field_list.append('br_cyl_m')
		if('By' in self.data_list or 'Bphi' in self.data_list):
			self._field_list.append('bphi_cyl_m')
		if('Bz' in self.data_list or 'Bz' in self.data_list):
			self._field_list.append('bz_cyl_m')

		if('Jx' in self.data_list or 'Jr' in self.data_list):
			self._source_list.append('jr_cyl_m')
		if('Jy' in self.data_list or 'Jphi' in self.data_list):
			self._source_list.append('jphi_cyl_m')
		if('Jz' in self.data_list or 'Jz' in self.data_list):
			self._source_list.append('jz_cyl_m')

		# need to add to PICMI standard
		if('psi' in self.data_list):
			self._field_list += ['psi_cyl_m']


	def fill_dict_fld(self,keyvals):
		keyvals['name'] = self._field_list
		keyvals['ndump'] = self.period

	def fill_dict_src(self,keyvals):
		keyvals['name'] = self._source_list
		keyvals['ndump'] = self.period


# QPAD does not support electrostatic and boosted frame diagnostic 
class ElectrostaticFieldDiagnostic(picmistandard.PICMI_ElectrostaticFieldDiagnostic):
	def model_post_init(self, context):
		raise Exception("Electrostatic field diagnostic not supported in QPAD")

class LabFrameParticleDiagnostic(picmistandard.PICMI_LabFrameParticleDiagnostic):
	def model_post_init(self, context):
		raise Exception("Boosted frame diagnostics not support in QPAD")

class LabFrameFieldDiagnostic(picmistandard.PICMI_LabFrameFieldDiagnostic):
	def model_post_init(self, context):
		raise Exception("Boosted frame diagnostics not support in QPAD")


## to be implemented	
class ParticleDiagnostic(picmistandard.PICMI_ParticleDiagnostic):
	"""
	QPAD-Specific Parameters

	QPAD_sample: integer, optional
		Dumps every nth particle.
	"""
	sample: int = Field(default=1, alias=codename + '_sample',
		description='Dumps every nth particle')

	def model_post_init(self, context):
		super().model_post_init(context)
		assert self.write_dir != '.', Exception("Write directory feature not yet supported.")
		assert self.period > 0, Exception("Diagnostic period is not valid")
		print('Warning: Particle diagnostic reporting momentum, position and charge data')

	def fill_dict_fld(self,keyvals):
		pass

	def fill_dict_src(self,keyvals):
		keyvals['name'] = ["raw"]
		keyvals['ndump'] = self.period
		keyvals['psample'] = self.sample


class GaussianLaser(picmistandard.PICMI_GaussianLaser):
	_profile: list = PrivateAttr(default_factory=lambda: ['gaussian', 'polynomial'])
	_iteration: int = PrivateAttr(default=3)
	# laser wavenumber, normalized by normalize_units (the standard's k0 is derived from the wavelength)
	_k0: float | None = PrivateAttr(default=None)

	def model_post_init(self, context):
		super().model_post_init(context)
		self._k0 = self.k0

	def normalize_units(self, density_norm):
		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c


		self._k0 = self._k0/k_pe
		self.waist = k_pe * self.waist
		self.duration = w_pe * self.duration 
		for i in range(3):
			self.focal_position[i] *= k_pe 
			self.centroid_position[i] *= k_pe


	def fill_dict(self,keyvals):
		keyvals['profile'] = self._profile
		keyvals['a0'] = self.a0
		keyvals['k0'] = self._k0
		keyvals['w0'] = self.waist
		keyvals['iteration'] = self._iteration
		keyvals['focal_distance'] = self.focal_position[2]
		keyvals['t_rise'] = self.duration * 1.5275
		keyvals['t_fall'] = self.duration * 1.5275
		keyvals['t_flat'] = 0
		keyvals['lon_center'] = -self.centroid_position[2]

	def fill_dict_fld(self, keyvals,diagnostics):
		tt = []
		for diag in diagnostics:
			temp_dict = {}
			if(diag.data_list is not None):
				if('E' in diag.data_list or 'Ex' in diag.data_list or 'Ey' in diag.data_list or 'Ez' in diag.data_list):
					temp_dict['name'] = ['a_cyl_m']
					temp_dict['ndump'] = diag.period
					tt.append(temp_dict)
					break

		keyvals['diag'] = tt

		
class LaserAntenna(picmistandard.PICMI_LaserAntenna):
	pass

class BinomialSmoother(picmistandard.PICMI_BinomialSmoother):
	# also accept a single value for all axes, as before the pydantic standard
	n_pass: int | list[int] | None = Field(default=None,
		description='Number of passes along each axis (a single integer applies to all axes)')
	compensation: bool | list[bool] | None = Field(default=None,
		description='Flags whether to apply compensation along each axis (a single flag applies to all axes)')

	def model_post_init(self, context):
		super().model_post_init(context)
		print("Warning: QPAD has no BinomialSmoother. Skipping feature.")

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
