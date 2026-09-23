# Copyright 2022-2023 Thamine Dalichaouch, Frank Tsung
# OSIRIS extension of PICMI standard

import picmistandard
from pydantic import Field, PrivateAttr
from typing import ClassVar
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

codename = 'OSIRIS'
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
class Species(picmistandard.PICMI_Species):
	"""
	"""
	beam_evolution: bool = Field(default=True, alias=codename + '_beam_evolution',
		description='Toggles beam evolution (free streaming if False)')

	_element: object = PrivateAttr(default=None)
	_profile_type: str | None = PrivateAttr(default=None)
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
		elif(isinstance(self.initial_distribution, OpenPMDFileDistribution)):
			self._profile_type = 'beam'
		elif(isinstance(self.initial_distribution, UniformDistribution)):
			self._profile_type = 'species'
		elif(isinstance(self.initial_distribution, AnalyticDistribution)):
			self._profile_type = 'species'
		elif(isinstance(self.initial_distribution, PiecewiseDistribution)):
			self._profile_type = 'species'
		else:
			print('Warning: Only Uniform and Gaussian distributions are currently supported.')


	def normalize_units(self):
		# normalized charge, mass, density
		self._q = self.charge/constants.q_e
		self._m = self.mass/constants.m_e



	def fill_dict(self, keyvals,init_self_field):
		keyvals['name'] = self.name
		# if(self._profile_type == 'beam' or init_self_field):
		keyvals['free_stream'] = not self.beam_evolution
		# 	keyvals['init_fields'] = True
		# if(isinstance(self.initial_distribution, OpenPMDFileDistribution)):
		# 	keyvals['init_type'] = 'file'


		keyvals['rqm'] = self._m/self._q


	def activate_field_ionization(self,model,product_species):
		raise Exception('Ionization not yet supported.')


# Species Class
class Neutral(picmistandard.PICMI_Species):
	"""
	"""
	multi_max: int | None = Field(default=None, alias=codename + '_multi_max',
		description='Maximum ionization level (defaults to the atomic number)')
	multi_min: int = Field(default=0, alias=codename + '_multi_min',
		description='Minimum ionization level')
	beam_evolution: bool = Field(default=True, alias=codename + '_beam_evolution',
		description='Toggles beam evolution')

	_element: int | None = PrivateAttr(default=None)
	_profile_type: str | None = PrivateAttr(default=None)
	_q: float | None = PrivateAttr(default=None)
	_m: float | None = PrivateAttr(default=None)

	# initialization 
	def model_post_init(self, context):
		super().model_post_init(context)
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
		if(self.multi_max is None):
			self.multi_max = self._element

		# set profile type
		
		if(isinstance(self.initial_distribution, UniformDistribution)):
			self._profile_type = 'neutral'
		elif(isinstance(self.initial_distribution, AnalyticDistribution)):
			self._profile_type = 'neutral'
		elif(isinstance(self.initial_distribution, PiecewiseDistribution)):
			self._profile_type = 'neutral'
		else:
			print('Warning: Only Uniform, Analytic, and Piecewise distributions are currently supported.')


	def normalize_units(self):
		# normalized charge, mass, density
		self._q = self.charge/constants.q_e
		self._m = self.mass/constants.m_e



	def fill_neutral_dict(self, keyvals,init_self_field):
		keyvals['neutral_gas'] = self.particle_type
		# if(self._profile_type == 'beam' or init_self_field):
		keyvals['multi_min'] = self.multi_min
		keyvals['multi_max'] = self.multi_max
		

	def fill_dict(self, keyvals,init_self_field):
		keyvals['name'] = self.name
		# if(self._profile_type == 'beam' or init_self_field):
		# keyvals['free_stream'] = not self.beam_evolution
		# 	keyvals['init_fields'] = True
		# if(isinstance(self.initial_distribution, OpenPMDFileDistribution)):
		# 	keyvals['init_type'] = 'file'
		keyvals['rqm'] = self._m/self._q




picmistandard.PICMI_MultiSpecies.Species_class = Species
class MultiSpecies(picmistandard.PICMI_MultiSpecies):
	pass



class GaussianBunchDistribution(picmistandard.PICMI_GaussianBunchDistribution):
	"""

	"""
	_q: float | None = PrivateAttr(default=None)
	_m: float | None = PrivateAttr(default=None)
	_norm_density: float | None = PrivateAttr(default=None)
	_tot_charge: float | None = PrivateAttr(default=None)

	def model_post_init(self, context):
		super().model_post_init(context)
		rotate_list(self.rms_velocity)
		rotate_list(self.centroid_velocity)
		rotate_list(self.centroid_position)
		rotate_list(self.rms_bunch_size)
		rotate_list(self.velocity_divergence)

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


		# spot size in normalized units (k_pe sigma), uth is divergence (sigma_{gamma * beta}), and ufl is fluid velocity (gamma* beta) )

		# normalized spot sizes
		for i in range(3):
			self.rms_bunch_size[i] *= k_pe 
			self.centroid_position[i] *= k_pe 
			self.rms_velocity[i] /= constants.c 
			self.centroid_velocity[i] /= constants.c

		# normalized charge, mass, density
		self._q = species.charge/constants.q_e
		self._m = species.mass/constants.m_e
		self._norm_density = peak_density/density_norm

		self._tot_charge = total_charge/(-constants.q_e * density_norm * k_pe**-3)

	def fill_dict(self,species,grid):
		n_x_dim = grid.n_x_dim
		udist_dict, profile_dict = {},{}

		udist_dict['uth'] = self.rms_velocity
		udist_dict['ufl'] = self.centroid_velocity
		udist_dict['use_classical_uadd'] = True

		udist_dict['n_accelerate'] = 512
		
		profile_dict['profile_type'] = ["gaussian"] * n_x_dim
		profile_dict['gauss_center'] = [to_scientific_notation(i) for i in self.centroid_position[:n_x_dim]]
		profile_dict['gauss_sigma'] = [to_scientific_notation(i) for i in self.rms_bunch_size[:n_x_dim]]

		if(isinstance(grid,CylindricalGrid)):
			profile_dict['aspect_ratio'] = to_scientific_notation(self.rms_bunch_size[2]/self.rms_bunch_size[1])

		q_scale = np.abs(species.charge/constants.q_e)
		if(species.density_scale is not None):
			profile_dict['density'] = to_scientific_notation(self._norm_density * species.density_scale * q_scale)
		else:
			profile_dict['density'] = to_scientific_notation(self._norm_density * q_scale)

		for j in range(n_x_dim):
			profile_dict['gauss_range(:,' + str(j+1) + ')'] = [to_scientific_notation(-4 * self.rms_bunch_size[j] +self.centroid_position[j]),\
			 to_scientific_notation(4 * self.rms_bunch_size[j] + self.centroid_position[j])]
		return udist_dict,profile_dict


class AnalyticDistribution(picmistandard.PICMI_AnalyticDistribution):
	"""
	OSIRIS-Specific Parameters
	
	### Plasma-specific parameters ####

	"""
	_norm_density: float | None = PrivateAttr(default=None)
	_q: float | None = PrivateAttr(default=None)
	_m: float | None = PrivateAttr(default=None)

	def model_post_init(self, context):
		super().model_post_init(context)
		# default profile for uniform plasmas
		
		rotate_list(self.momentum_expressions)
		rotate_list(self.lower_bound)
		rotate_list(self.upper_bound)
		rotate_list(self.rms_velocity)
		rotate_list(self.directed_velocity)
		assert self.fill_in is not None, Exception('OSIRIS defaults to fill_in = True.')
		assert np.any(self.momentum_expressions is not None), Exception('OSIRIS does not yet support fluid momentum expressions for (gamma * V)')


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
		self._q = species.charge/constants.q_e
		self._m = species.mass/constants.m_e
		


		for i in range(3):
			self.rms_velocity[i] /= constants.c 

		# if(np.any(self.directed_velocity != 0.0)):
		# 	print('Warning: ' + codename + ' does not support directed velocity for Analytic Distributions.')


	def fill_dict(self,species,grid):
		back_str,front_str = construct_bounds(self.lower_bound,self.upper_bound,grid.n_x_dim)
		
		q_scale = np.abs(species.charge/constants.q_e)
		udist_dict, profile_dict = {}, {}
		if(species.density_scale is not None):
			profile_dict['density'] = to_scientific_notation(species.density_scale * q_scale)
		else:
			profile_dict['density'] = to_scientific_notation(q_scale)
		profile_dict['profile_type'] = "math func" 
		profile_dict['math_func_expr'] = front_str + self.density_expression + back_str

		udist_dict['uth'] = [to_scientific_notation(i) for i in self.rms_velocity]
		udist_dict['ufl'] = [to_scientific_notation(i) for i in self.directed_velocity]
		udist_dict['use_classical_uadd'] = True
		
		return udist_dict, profile_dict


class OpenPMDFileDistribution(picmistandard.PICMI_DistributionExtension):
	"""
	OSIRIS-Specific Parameters
	
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
	filename: str | None = Field(default=None, description='Beam file (HDF5) in OSIRIS units')
	beam_center: list[float] = Field(default_factory=lambda: [0, 0 ,0],
		description='Position of the beam center [m]')
	file_center: list[float] = Field(default_factory=lambda: [0, 0, 0],
		description='Position of the beam center in the file [m]')

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



	def fill_dict(self,species,grid):
		udist_dict, profile_dict = {}, {}
		udist_dict['n_accelerate'] = 512
		udist_dict['use_classical_uadd'] = True
		udist_dict['use_particle_uacc'] = True
		profile_dict['file_name'] = self.filename
		return udist_dict, profile_dict
		# keyvals['beam_center'] = self.beam_center
		# keyvals['file_center'] = self.file_center


class UniformDistribution(picmistandard.PICMI_UniformDistribution):
	_norm_density: float | None = PrivateAttr(default=None)

	def model_post_init(self, context):
		super().model_post_init(context)
		# default profile for uniform plasmas
		rotate_list(self.lower_bound)
		rotate_list(self.upper_bound)
		rotate_list(self.rms_velocity)
		rotate_list(self.directed_velocity)
		if(not self.fill_in):
			print('OSIRIS defaults to fill_in = True.')




	def normalize_units(self,species, density_norm):
		# normalize plasma density
		self._norm_density = self.density/density_norm

		# normalized charge, mass, density
		# self.q = species.charge/constants.q_e
		# self.m = species.mass/constants.m_e

		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c

		for i in range(3):
			if(self.lower_bound[i] is not None):
				self.lower_bound[i] *= k_pe
			if(self.upper_bound[i] is not None):
				self.upper_bound[i] *= k_pe

		# if(np.any(self.directed_velocity != 0.0)):
		# 	print('Warning: ' + codename + ' does not support directed velocity for Uniform Distributions.')

	def fill_dict(self,species,grid):
		back_str,front_str = construct_bounds(self.lower_bound,self.upper_bound,grid.n_x_dim)
		
		q_scale = np.abs(species.charge/constants.q_e)
		udist_dict, profile_dict = {}, {}
		if(species.density_scale is not None):
			profile_dict['density'] = to_scientific_notation(species.density_scale * q_scale)
		else:
			profile_dict['density'] = to_scientific_notation(q_scale)
		profile_dict['profile_type'] = "math func" 
		profile_dict['math_func_expr'] = front_str + str(self._norm_density)+ back_str
		udist_dict['uth'] = [to_scientific_notation(i) for i in self.rms_velocity]
		udist_dict['ufl'] = [to_scientific_notation(i) for i in self.directed_velocity]
		udist_dict['use_classical_uadd'] = True
		
		return udist_dict, profile_dict

		
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

	_profile: list | None = PrivateAttr(default=None)
	_norm_density: float | None = PrivateAttr(default=None)

	def __init__(self, density, **kw):
		# keep density as the (optional) positional argument of the pre-pydantic signature
		super().__init__(density=density, **kw)

	def model_post_init(self, context):
		super().model_post_init(context)
		self._profile = ['uniform','piecewise-linear']
		rotate_list(self._profile)
		rotate_list(self.directed_velocity)
		rotate_list(self.rms_velocity)
		rotate_list(self.upper_bound)
		rotate_list(self.lower_bound)
		


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


	def fill_dict(self,species,grid):

		q_scale = np.abs(species.charge/constants.q_e)
		udist_dict, profile_dict = {}, {}
		if(species.density_scale is not None):
			profile_dict['density'] = species.density_scale * q_scale
		else:
			profile_dict['density'] = q_scale

		profile_dict['profile_type'] = self._profile 
		num_x = len(self.piecewise_s)
		profile_dict['num_x'] = num_x
		profile_dict['fx(1:' + str(num_x) + ',1)'] = [to_scientific_notation(i) for i in self.piecewise_fs]
		profile_dict['x(1:' + str(num_x) + ',1)'] = [to_scientific_notation(i) for i in self.piecewise_s]
		udist_dict['uth'] = [to_scientific_notation(i) for i in self.rms_velocity]
		udist_dict['ufl'] = [to_scientific_notation(i) for i in self.directed_velocity]
		udist_dict['use_classical_uadd'] = True
		return udist_dict, profile_dict






class ParticleListDistribution(picmistandard.PICMI_ParticleListDistribution):
	def model_post_init(self, context):
		raise Exception('Particle list distributions not yet supported in OSIRIS')


# constant, analytic, or mirror fields not yet supported in QuickPIC 
class ConstantAppliedField(picmistandard.PICMI_ConstantAppliedField):
	# normalized field values, as expressions
	_Ex_expression: str | None = PrivateAttr(default=None)
	_Ey_expression: str | None = PrivateAttr(default=None)
	_Ez_expression: str | None = PrivateAttr(default=None)
	_Bx_expression: str | None = PrivateAttr(default=None)
	_By_expression: str | None = PrivateAttr(default=None)
	_Bz_expression: str | None = PrivateAttr(default=None)

	def model_post_init(self, context):
		super().model_post_init(context)
		rotate_list(self.lower_bound)
		rotate_list(self.upper_bound)
	def normalize_units(self,density_norm):
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c

		E_norm = constants.m_e * constants.c**2/ constants.q_e  * k_pe
		B_norm = E_norm/ constants.c

		for i in range(3):
			self.lower_bound[i] *= k_pe
			self.upper_bound[i] *= k_pe


		E_list = [self.Ex, self.Ey, self.Ez]
		B_list = [self.Bx, self.By, self.Bz]
		for i in range(3):
			expr = E_list[i]
			if(expr is not None):
				E_list[i] = format_decimal(E_list[i]/E_norm)

			expr = B_list[i]
			if(expr is not None):
				B_list[i] = format_decimal(B_list[i]/B_norm)

		
		self._Ex_expression = E_list[0]
		self._Ey_expression = E_list[1]
		self._Ez_expression = E_list[2]
		self._Bx_expression = B_list[0]
		self._By_expression = B_list[1]
		self._Bz_expression = B_list[2]


	def fill_dict(self,keyvals,grid):
		keyvals["ext_fld"] = "static"
		E_list = [self._Ez_expression, self._Ex_expression, self._Ey_expression]
		B_list = [self._Bz_expression, self._Bx_expression, self._By_expression]

		back_str,front_str = construct_bounds(self.lower_bound,self.upper_bound,grid.n_x_dim)

		if(isinstance(grid,CylindricalGrid)):
			print('Warning: Applied fields are assumed to be in cylindrical coordinates (r, phi, z)')
			for i in range(len(E_list)):
				expr = E_list[i]
				if(expr is not None):
					keyvals['ext_e_re_mfunc(0,' + str(i+1) + ')'] = front_str + expr + back_str
					keyvals['type_ext_e(' + str(i+1) + ')'] = "math func"
				expr = B_list[i]
				if(expr is not None):
					keyvals['ext_b_re_mfunc(0,' + str(i+1) + ')'] = front_str + expr + back_str
					keyvals['type_ext_b(' + str(i+1) + ')'] = "math func"
		else:
			for i in range(len(E_list)):
				expr = E_list[i]
				if(expr is not None):
					keyvals['ext_e_mfunc(' + str(i+1) + ')'] = front_str + expr + back_str
					keyvals['type_ext_e(' + str(i+1) + ')'] = "math func"
				expr = B_list[i]
				if(expr is not None):
					keyvals['ext_b_mfunc(' + str(i+1) + ')'] = front_str + expr + back_str
					keyvals['type_ext_b(' + str(i+1) + ')'] = "math func"



class AnalyticAppliedField(picmistandard.PICMI_AnalyticAppliedField):
	def model_post_init(self, context):
		super().model_post_init(context)
		rotate_list(self.lower_bound)
		rotate_list(self.upper_bound)
	def normalize_units(self,density_norm):
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c

		E_norm = constants.m_e * constants.c**2/ constants.q_e  * k_pe
		B_norm = E_norm/ constants.c

		for i in range(3):
			self.lower_bound[i] *= k_pe
			self.upper_bound[i] *= k_pe

		E_list = [self.Ex_expression, self.Ey_expression, self.Ez_expression]
		B_list = [self.Bx_expression, self.By_expression, self.Bz_expression]
		for i in range(3):
			expr = E_list[i]
			if(expr is not None):
				expr = normalize_math_func(expr,density_norm)
				expr =  expr + '/' + format_decimal(E_norm)
				E_list[i] = expr

			expr = B_list[i]
			if(expr is not None):
				normalize_math_func(expr,density_norm)
				expr = expr + '/' + format_decimal(B_norm)
				B_list[i] = expr

		self.Ex_expression = E_list[0]
		self.Ey_expression = E_list[1]
		self.Ez_expression = E_list[2]
		self.Bx_expression = B_list[0]
		self.By_expression = B_list[1]
		self.Bz_expression = B_list[2]


	def fill_dict(self,keyvals,grid):
		keyvals["ext_fld"] = "dynamic"
		E_list = [self.Ez_expression, self.Ex_expression, self.Ey_expression]
		B_list = [self.Bz_expression, self.Bx_expression, self.By_expression]
		back_str,front_str = construct_bounds(self.lower_bound,self.upper_bound,grid.n_x_dim)

		if(isinstance(grid,CylindricalGrid)):
			print('Warning: Applied fields are assumed to be in cylindrical coordinates (r, phi, z)')
			for i in range(len(E_list)):
				expr = E_list[i]
				if(expr is not None):
					keyvals['ext_e_re_mfunc(0,' + str(i+1) + ')'] = front_str + expr + back_str
					keyvals['type_ext_e(' + str(i+1) + ')'] = "math func"
				expr = B_list[i]
				if(expr is not None):
					keyvals['ext_b_re_mfunc(0,' + str(i+1) + ')'] = front_str + expr + back_str
					keyvals['type_ext_b(' + str(i+1) + ')'] = "math func"
		else:
			for i in range(len(E_list)):
				expr = E_list[i]
				if(expr is not None):
					keyvals['ext_e_mfunc(' + str(i+1) + ')'] = front_str + expr + back_str
					keyvals['type_ext_e(' + str(i+1) + ')'] = "math func"
				expr = B_list[i]
				if(expr is not None):
					keyvals['ext_b_mfunc(' + str(i+1) + ')'] = front_str + expr + back_str
					keyvals['type_ext_b(' + str(i+1) + ')'] = "math func"





		
		

class Mirror(picmistandard.PICMI_Mirror):
	def model_post_init(self, context):
		raise Exception("Mirrors are not yet supported in OSIRIS")


class ElectromagneticSolver(picmistandard.PICMI_ElectromagneticSolver):
	solver_method: str | None = Field(default=None, alias=codename + '_method',
		description='OSIRIS field solver (e.g. fei), overrides method')

	# OSIRIS name of the field solver
	_method: str | None = PrivateAttr(default=None)

	def model_post_init(self, context):
		super().model_post_init(context)
		# override if supplemented
		self._method = self.solver_method if self.solver_method is not None else self.method
		if(self._method is not None):
			self._method = self._method.lower()
		if(self._method == 'ckc'):
			self._method = 'ck'

	def get_info(self):
		el_mag_fld_dict = {}
		el_mag_fld_dict['solver'] = self._method
		# write_dict_fortran(file,el_mag_fld_dict,'el_mag_fld')

		emf_bound_dict = {}
		for i in range(len(self.grid.lower_boundary_conditions)):
			if(self.grid.lower_boundary_conditions[i] is not None and self.grid.upper_boundary_conditions[i] is not None):
				emf_bound_dict['type(:,' + str(i+1) + ')'] = [self.grid.lower_boundary_conditions[i], self.grid.upper_boundary_conditions[i]]
		el_mag_fld_solver_dict = {}
		if(self._method == 'fei'):
			el_mag_fld_solver_dict['type'] = 'xu'
			el_mag_fld_solver_dict['solver_ord'] = 2
			el_mag_fld_solver_dict['n_coef'] = 16
			el_mag_fld_solver_dict['weight_w'] = 0.3
			el_mag_fld_solver_dict['weight_n'] = 10

			el_mag_fld_solver_dict['filter_current'] = True
			el_mag_fld_solver_dict['correct_current'] = True
			el_mag_fld_solver_dict['filter_limit'] = 0.6
			el_mag_fld_solver_dict['filter_width'] = 0.1
			el_mag_fld_solver_dict['n_damp_cell'] = 15
			
		# write_dict_fortran(file,el_mag_fld_solver_dict,'emf_solver')
		return el_mag_fld_dict, emf_bound_dict,el_mag_fld_solver_dict
	def fill_src_smoother_dict(self,smooth_dict):
		smooth_dict['type'] = ['binomial'] * self.grid.n_x_dim
		smooth_dict['order'] = [self.source_smoother.n_pass] * self.grid.n_x_dim


		
class ElectrostaticSolver(picmistandard.PICMI_ElectrostaticSolver):
	def model_post_init(self, context):
		raise Exception('This feature is not yet supported. Please use the Electromagnetic solver.')





		
	
		
		

## Grids
class Cartesian1DGrid(picmistandard.PICMI_Cartesian1DGrid):
	n_x_dim: ClassVar[int] = 1

	_if_periodic: list | None = PrivateAttr(default=None)
	_lower_particle_bounds: list | None = PrivateAttr(default=None)
	_upper_particle_bounds: list | None = PrivateAttr(default=None)
	_if_move: list | None = PrivateAttr(default=None)
	_coordinates: str | None = PrivateAttr(default=None)

	def __init__(self, **kw):
		super().__init__(**kw)
		# after the validation, which resolves the vector forms (e.g., number_of_cells from nx, ...)
		self._code_init()

	def _code_init(self):

		
		self._if_periodic = [ele == 'periodic' for ele in self.lower_boundary_conditions]
		self._lower_particle_bounds, self._upper_particle_bounds = check_grid_bounds(self.lower_boundary_conditions,
			self.upper_boundary_conditions)


		self._if_move = [ele != 0 for ele in self.moving_window_velocity]

		if(self.lower_boundary_conditions[0] == 'open'):
			self.lower_boundary_conditions[0] = 'lindman'
		if(self.upper_boundary_conditions[0] == 'open'):
			self.upper_boundary_conditions[0] = 'lindman'

		self._coordinates = 'cartesian'

	def normalize_units(self, density_norm):
		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c

		#normalize coordinates 
		for i in range(self.n_x_dim):
			self.lower_bound[i] *= k_pe
			self.upper_bound[i] *= k_pe
		
	def fill_dict(self,keyvals):
		keyvals['nx_p'] = self.number_of_cells[:self.n_x_dim]
		keyvals['coordinates'] = self._coordinates

	def get_info(self):
		part_bound_dict = {}
		for i in range(self.n_x_dim):
			if(self._lower_particle_bounds[i] is not None and self._upper_particle_bounds[i] is not None):
				part_bound_dict['type(:,' + str(i+1) + ')'] = [self._lower_particle_bounds[i],
					self._upper_particle_bounds[i]]
		return part_bound_dict



class Cartesian2DGrid(picmistandard.PICMI_Cartesian2DGrid):
	n_x_dim: ClassVar[int] = 2

	_if_periodic: list | None = PrivateAttr(default=None)
	_lower_particle_bounds: list | None = PrivateAttr(default=None)
	_upper_particle_bounds: list | None = PrivateAttr(default=None)
	_if_move: list | None = PrivateAttr(default=None)
	_coordinates: str | None = PrivateAttr(default=None)

	def __init__(self, **kw):
		super().__init__(**kw)
		# after the validation, which resolves the vector forms (e.g., number_of_cells from nx, ...)
		self._code_init()

	def _code_init(self):

		
		self._if_periodic = [ele == 'periodic' for ele in self.lower_boundary_conditions]
		self._lower_particle_bounds, self._upper_particle_bounds = check_grid_bounds(self.lower_boundary_conditions,
			self.upper_boundary_conditions)


		self._if_move = [ele != 0 for ele in self.moving_window_velocity]
		rotate_list(self.lower_boundary_conditions)
		rotate_list(self.upper_boundary_conditions)
		rotate_list(self._lower_particle_bounds)
		rotate_list(self._upper_particle_bounds)
		rotate_list(self._if_move)
		rotate_list(self._if_periodic)
		rotate_list(self.lower_bound)
		rotate_list(self.upper_bound)
		rotate_list(self.number_of_cells)

		if(self.lower_boundary_conditions[0] == 'open'):
			self.lower_boundary_conditions[0] = 'lindman'
		if(self.upper_boundary_conditions[0] == 'open'):
			self.upper_boundary_conditions[0] = 'lindman'

		self._coordinates = 'cartesian'

	def normalize_units(self, density_norm):
		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c

		#normalize coordinates 
		for i in range(self.n_x_dim):
			self.lower_bound[i] *= k_pe
			self.upper_bound[i] *= k_pe
		
	def fill_dict(self,keyvals):
		keyvals['nx_p'] = self.number_of_cells[:self.n_x_dim]
		keyvals['coordinates'] = self._coordinates

	def get_info(self):
		part_bound_dict = {}
		for i in range(self.n_x_dim):
			if(self._lower_particle_bounds[i] is not None and self._upper_particle_bounds[i] is not None):
				part_bound_dict['type(:,' + str(i+1) + ')'] = [self._lower_particle_bounds[i],
					self._upper_particle_bounds[i]]
		return part_bound_dict



class Cartesian3DGrid(picmistandard.PICMI_Cartesian3DGrid):
	n_x_dim: ClassVar[int] = 3

	_if_periodic: list | None = PrivateAttr(default=None)
	_lower_particle_bounds: list | None = PrivateAttr(default=None)
	_upper_particle_bounds: list | None = PrivateAttr(default=None)
	_if_move: list | None = PrivateAttr(default=None)
	_coordinates: str | None = PrivateAttr(default=None)

	def __init__(self, **kw):
		super().__init__(**kw)
		# after the validation, which resolves the vector forms (e.g., number_of_cells from nx, ...)
		self._code_init()

	def _code_init(self):

		
		self._if_periodic = [ele == 'periodic' for ele in self.lower_boundary_conditions]
		self._lower_particle_bounds, self._upper_particle_bounds = check_grid_bounds(self.lower_boundary_conditions,
			self.upper_boundary_conditions)


		self._if_move = [ele != 0 for ele in self.moving_window_velocity]
		rotate_list(self.lower_boundary_conditions)
		rotate_list(self.upper_boundary_conditions)
		rotate_list(self._lower_particle_bounds)
		rotate_list(self._upper_particle_bounds)
		rotate_list(self._if_move)
		rotate_list(self._if_periodic)
		rotate_list(self.lower_bound)
		rotate_list(self.upper_bound)
		rotate_list(self.number_of_cells)

		if(self.lower_boundary_conditions[0] == 'open'):
			self.lower_boundary_conditions[0] = 'lindman'
		if(self.upper_boundary_conditions[0] == 'open'):
			self.upper_boundary_conditions[0] = 'lindman'

		self._coordinates = 'cartesian'

	def normalize_units(self, density_norm):
		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c

		#normalize coordinates 
		for i in range(self.n_x_dim):
			self.lower_bound[i] *= k_pe
			self.upper_bound[i] *= k_pe
		
	def fill_dict(self,keyvals):
		keyvals['nx_p'] = self.number_of_cells[:self.n_x_dim]
		keyvals['coordinates'] = self._coordinates

	def get_info(self):
		part_bound_dict = {}
		for i in range(self.n_x_dim):
			if(self._lower_particle_bounds[i] is not None and self._upper_particle_bounds[i] is not None):
				part_bound_dict['type(:,' + str(i+1) + ')'] = [self._lower_particle_bounds[i],
					self._upper_particle_bounds[i]]
		return part_bound_dict


class CylindricalGrid(picmistandard.PICMI_CylindricalGrid):
	n_x_dim: ClassVar[int] = 2

	_if_periodic: list | None = PrivateAttr(default=None)
	_lower_particle_bounds: list | None = PrivateAttr(default=None)
	_upper_particle_bounds: list | None = PrivateAttr(default=None)
	_if_move: list | None = PrivateAttr(default=None)
	_coordinates: str | None = PrivateAttr(default=None)
	_dr: float | None = PrivateAttr(default=None)
	_dz: float | None = PrivateAttr(default=None)

	def __init__(self, **kw):
		super().__init__(**kw)
		# after the validation, which resolves the vector forms (e.g., number_of_cells from nx, ...)
		self._code_init()

	def _code_init(self):
		
		self._if_periodic = [ele == 'periodic' for ele in self.lower_boundary_conditions]
		self._lower_particle_bounds, self._upper_particle_bounds = check_grid_bounds(self.lower_boundary_conditions,
			self.upper_boundary_conditions)
		self._lower_particle_bounds[0] = 'axial'
		self.lower_boundary_conditions[0] = 'axial' 


		self._dr = np.abs(self.upper_bound[0]- self.lower_bound[0])/self.number_of_cells[0]
		self._dz = np.abs(self.upper_bound[1]- self.lower_bound[1])/self.number_of_cells[1]

		self._if_move = [ele != 0 for ele in self.moving_window_velocity]
		rotate_list(self.lower_boundary_conditions)
		rotate_list(self.upper_boundary_conditions)
		rotate_list(self._lower_particle_bounds)
		rotate_list(self._upper_particle_bounds)
		rotate_list(self._if_move)
		rotate_list(self._if_periodic)
		rotate_list(self.lower_bound)
		rotate_list(self.upper_bound)
		rotate_list(self.number_of_cells)

		

		if(self.lower_boundary_conditions[0] == 'open'):
			self.lower_boundary_conditions[0] = 'lindman'
		if(self.upper_boundary_conditions[0] == 'open'):
			self.upper_boundary_conditions[0] = 'lindman'

		self._coordinates = 'cylindrical'

	@property
	def dr(self):
		"""Cell size along r"""
		return self._dr

	@property
	def dz(self):
		"""Cell size along z"""
		return self._dz

	def normalize_units(self, density_norm):
		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c

		#normalize coordinates 
		for i in range(self.n_x_dim):
			self.lower_bound[i] *= k_pe
			self.upper_bound[i] *= k_pe
		
	def fill_dict(self,keyvals):
		keyvals['nx_p'] = self.number_of_cells
		keyvals['coordinates'] = self._coordinates
		keyvals['n_cyl_modes'] = self.n_azimuthal_modes

	def get_info(self):
		part_bound_dict = {}
		for i in range(self.n_x_dim):
			if(self._lower_particle_bounds[i] is not None and self._upper_particle_bounds[i] is not None):
				part_bound_dict['type(:,' + str(i+1) + ')'] = [self._lower_particle_bounds[i],
					self._upper_particle_bounds[i]]
		return part_bound_dict


class FileLayout(picmistandard.PICMI_LayoutExtension):
	"""
	OSIRIS-Specific Parameters
	
	OSIRIS_np_per_dimension: integer array, optional
		Part per dim in each direction. (for beams only)

	OSIRIS_npmax: integer, optional
		Particle buffer size per MPI partition.

	OSIRIS_num_theta: integer, optional
		Number of particles in azimuthal direction. Defaults to 8 * n_azimuthal_modes.
	"""

	grid: picmistandard.PICMI_AnyGrid | None = Field(default=None,
		description='Grid object specifying the grid to follow')

	_profile_type: str = PrivateAttr(default='file')



	def fill_dict(self, keyvals,grid):
		keyvals['init_type'] = self._profile_type

		grid_t = grid
		if(isinstance(grid_t,CylindricalGrid)):
			keyvals['num_par_x'] = [1,1]
			keyvals['num_par_theta'] = 1

		else:
			keyvals['num_par_x'] = self.n_macroparticles_per_cell
		

class PseudoRandomLayout(picmistandard.PICMI_PseudoRandomLayout):
	"""
	OSIRIS-Specific Parameters

	"""
	# OSIRIS takes the number of particles per cell along each axis
	n_macroparticles_per_cell: int | list[int] | None = Field(default=None,
		description='Number of macroparticles to load per cell (along each axis)')

	def model_post_init(self, context):
		super().model_post_init(context)
		# n_macroparticles is required.
		assert self.n_macroparticles_per_cell is not None, Exception('n_macroparticles_per_cell must be specified when using OSIRIS')
		rotate_list(self.n_macroparticles_per_cell)
		print('PseudoRandomLayout defaults to GriddedLayout')

	def fill_dict(self, keyvals, grid):
		if(self.grid is not None):
			grid_t = self.grid
		else:
			grid_t = grid

		if(isinstance(grid_t,CylindricalGrid)):
			num_par_theta  = self.n_macroparticles_per_cell[2]
			if(num_par_theta < 8 * grid_t.n_azimuthal_modes):
				num_par_theta = 8 * grid_t.n_azimuthal_modes
				print('Warning: total azimuthal ppc increased to ' + str(num_par_theta))
			keyvals['num_par_theta'] = num_par_theta
			keyvals['num_par_x'] = self.n_macroparticles_per_cell[:2]

		else:
			keyvals['num_par_x'] = self.n_macroparticles_per_cell

				


class GriddedLayout(picmistandard.PICMI_GriddedLayout):
	"""
	OSIRIS-Specific Parameters

	"""
	def model_post_init(self, context):
		super().model_post_init(context)
		rotate_list(self.n_macroparticle_per_cell)

	def fill_dict(self,keyvals,grid):
		# keyvals['profile_type'] = self.profile_type
		if(self.grid is not None):
			grid_t = self.grid
		else:
			grid_t = grid

		if(isinstance(grid_t,CylindricalGrid)):
			num_par_theta  = self.n_macroparticle_per_cell[2]
			if(num_par_theta < 8 * grid_t.n_azimuthal_modes):
				num_par_theta = 8 * grid_t.n_azimuthal_modes
				print('Warning: total azimthal ppc increased to ' + str(num_par_theta))
			keyvals['num_par_theta'] = num_par_theta
			keyvals['num_par_x'] = self.n_macroparticle_per_cell[:2]

		else:
			keyvals['num_par_x'] = self.n_macroparticle_per_cell



class Simulation(picmistandard.PICMI_Simulation):
	"""
	OSIRIS-Specific Parameters

	OSIRIS_n0: float, optional
		Plasma density [m^3] to normalize units.

	OSIRIS_read_restart: boolean, optional
		Toggle to read from restart files.
	
	OSIRIS_restart_timestep: integer, optional
		Specifies timestep if read_restart = True.


	"""
	cpu_split: list[int] = Field(default_factory=lambda: [1,1,1], alias=codename + '_nodes',
		description='MPI-node configuration')
	n0: float | None = Field(default=None, alias=codename + '_n0',
		description='Plasma density [m^-3] to normalize units')
	random_seed: int = Field(default=10, alias=codename + '_random_seed',
		description='Number of seeds for pseudo-random numbers')
	ndump_sim: int = Field(default=1, alias=codename + '_ndump_sim',
		description='Base dump period of the diagnostics')
	if_timing: bool = Field(default=False, alias=codename + '_timings',
		description='Toggle to report timings')
	read_restart: bool = Field(default=False, alias=codename + '_read_restart',
		description='Toggle to read from restart files')
	ndump_restart: int = Field(default=0, alias=codename + '_ndump_restart',
		description='Restart dump period (0: no restart files)')

	### OSIRIS beam flag
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
		assert self.time_step_size is not None, Exception('OSIRIS requires a time step size.')
		if(self.particle_shape == 'NGP' or self.particle_shape == None):
			print('OSIRIS does not support NGP or None. Defaulting to linear')
			self.particle_shape = 'linear'

		rotate_list(self.cpu_split)

		# normalize simulation time
		# print('n0', self.n0)
		if(self.n0 is not None):
			self.normalize_simulation()
		

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


	def add_applied_field(self, applied_field):
		picmistandard.PICMI_Simulation.add_applied_field( self, applied_field )
		if(self.n0 is not None):
			applied_field.normalize_units(self.n0)

	def add_laser(self, laser, injection_method):
		picmistandard.PICMI_Simulation.add_laser(self, laser, injection_method)
		if(self.n0 is not None):
			laser.normalize_units(self.n0)
			if(injection_method is not None):
				injection_method.normalize_units(self.n0)


			


	def sim_fill_dict(self, file):
		if(self.max_time is None):
			self.max_time = self.max_steps * self.time_step_size
		## fill out simulation info
		sim_dict = {}
		cylin_flag = isinstance(self.solver.grid,CylindricalGrid)
		if(self.n0 is not None):
			sim_dict['n0'] = self.n0 * 1.e-6 # in density in cm^{-3}
		if(cylin_flag):
			sim_dict['algorithm'] = 'quasi-3D'
		write_dict_fortran(file,sim_dict, 'simulation')


		## fill out node config
		node_dict = {}
		node_dict['node_number'] = self.cpu_split[:self.solver.grid.n_x_dim]
		node_dict['if_periodic'] = self.solver.grid._if_periodic
		write_dict_fortran(file,node_dict, 'node_conf')

		grid_dict = {}
		self.solver.grid.fill_dict(grid_dict)
		write_dict_fortran(file,grid_dict, 'grid')

		time_step_dict = {}
		time_step_dict['dt'] = self.time_step_size
		time_step_dict['ndump'] = self.ndump_sim
		write_dict_fortran(file,time_step_dict,'time_step')

		restart_dict = {}
		restart_dict['ndump_fac'] = self.ndump_restart
		restart_dict['if_restart'] = self.read_restart
		restart_dict['if_remold'] = True
		write_dict_fortran(file,restart_dict, 'restart')

		space_dict = {}
		space_dict['xmin'] = self.solver.grid.lower_bound
		space_dict['xmax'] = self.solver.grid.upper_bound
		space_dict['if_move'] = self.solver.grid._if_move
		write_dict_fortran(file,space_dict, 'space')

		time_dict = {}
		time_dict['tmin'] = 0
		time_dict['tmax'] = self.max_time
		write_dict_fortran(file,time_dict,'time')

		el_mag_fld_dict, emf_bound_dict, el_mag_fld_solver_dict = self.solver.get_info()

		for applied_field in self.applied_fields:
			applied_field.fill_dict(el_mag_fld_dict,self.solver.grid)
		write_dict_fortran(file, el_mag_fld_dict, 'el_mag_fld')
		write_dict_fortran(file,emf_bound_dict, 'emf_bound')
		write_dict_fortran(file,el_mag_fld_solver_dict, 'emf_solver')

		diag_emf_dict = {}
		for i in range(len(self.diagnostics)):
			diag = self.diagnostics[i]
			if(isinstance(diag,ParticleDiagnostic)):
				continue
			self.diagnostics[i].fill_dict_fld(diag_emf_dict, cylin_flag)
		write_dict_fortran(file,diag_emf_dict, 'diag_emf')

		particles_dict = {}
		# particles_dict['num_species'] = len(self.species)
		particles_dict['num_species'] = int(np.sum(self._if_species))+ int(np.sum(self._if_beam))
		particles_dict['num_neutral'] = int(np.sum(self._if_neutral)) 
		particles_dict['interpolation'] = self.particle_shape
		write_dict_fortran(file,particles_dict, 'particles')

		for i in range(len(self.species)):
			spec = self.species[i]
			temp_dict = {}
			if(self._if_neutral[i]):
				self.species[i].fill_neutral_dict(temp_dict,self.initialize_self_fields[i])
				write_dict_fortran(file,temp_dict,'neutral')

				udist_dict, profile_dict = self.species[i].initial_distribution.fill_dict(self.species[i],self.solver.grid)
				write_dict_fortran(file,profile_dict,'profile')

				diag_neutral = {}
				for j in range(len(self.diagnostics)):
					diag = self.diagnostics[j]
					if(isinstance(diag,ParticleDiagnostic) and spec not in diag.species):
						continue
					self.diagnostics[j].fill_dict_neut(diag_neutral,cylin_flag)
				write_dict_fortran(file,diag_neutral,'diag_neutral')
				
				temp_dict = {}
				self.species[i].fill_dict(temp_dict,self.initialize_self_fields[i])
				self.layouts[i].fill_dict(temp_dict,self.solver.grid)
				write_dict_fortran(file,temp_dict,'species')

				spe_bound_dict = self.solver.grid.get_info()
				write_dict_fortran(file,spe_bound_dict,'spe_bound')
				# fill in source term diagnostics
				# diags_srcs = []
				diag_species_dict = {}
				for j in range(len(self.diagnostics)):
					diag = self.diagnostics[j]
					if(isinstance(diag,ParticleDiagnostic) and spec not in diag.species):
						continue
					temp_dict2 = {}
					self.diagnostics[j].fill_dict_src(diag_species_dict,cylin_flag)
				write_dict_fortran(file,diag_species_dict,'diag_species')
			else:
				self.species[i].fill_dict(temp_dict,self.initialize_self_fields[i])
				self.layouts[i].fill_dict(temp_dict,self.solver.grid)

				write_dict_fortran(file,temp_dict,'species')

				udist_dict, profile_dict = self.species[i].initial_distribution.fill_dict(self.species[i],self.solver.grid)

				write_dict_fortran(file,udist_dict,'udist')
				write_dict_fortran(file,profile_dict,'profile')

				spe_bound_dict = self.solver.grid.get_info()
				write_dict_fortran(file,spe_bound_dict,'spe_bound')
				# fill in source term diagnostics
				# diags_srcs = []
				diag_species_dict = {}
				for j in range(len(self.diagnostics)):
					diag = self.diagnostics[j]
					if(isinstance(diag,ParticleDiagnostic) and spec not in diag.species):
						continue
					temp_dict2 = {}
					self.diagnostics[j].fill_dict_src(diag_species_dict,cylin_flag)
				write_dict_fortran(file,diag_species_dict,'diag_species')
		
		n_antennas = 0
		## write out regular lasers first
		for i in range(len(self.lasers)):
			if(self.laser_injection_methods[i] is not None):
				continue
			laser = self.lasers[i]
			zpulse_dict = {}
			self.lasers[i].fill_dict(zpulse_dict,self.solver.grid)
			write_dict_fortran(file,zpulse_dict,'zpulse')

		# write out antennas
		for i in range(len(self.lasers)):
			if(self.laser_injection_methods[i] is not None):
				laser = self.lasers[i]
				antenna_dict = {}
				self.lasers[i].fill_antenna_dict(antenna_dict,self.solver.grid,self.laser_injection_methods[i])
				write_dict_fortran(file,antenna_dict,'zpulse_mov_wall')
			

		

		current_dict = {}
		write_dict_fortran(file,current_dict,'current')
		smooth_dict = {}
		if(self.solver.source_smoother is not None):
			self.solver.fill_src_smoother_dict(smooth_dict)
		else:
			smooth_dict = { "type" : ["5pass", "none"]}
		write_dict_fortran(file,smooth_dict,'smooth')

		diag_current_dict = {}
		for i in range(len(self.species)):
			spec = self.species[i]
			# fill in source term diagnostics
			# diags_srcs = []
			for j in range(len(self.diagnostics)):
				diag = self.diagnostics[j]
				if(not isinstance(diag,FieldDiagnostic)):
					continue
				self.diagnostics[j].fill_dict_src2(diag_current_dict,cylin_flag)
		write_dict_fortran(file,diag_current_dict,'diag_current')

		

		## fill out grid info

		# fill grid and mpi params
		# self.solver.grid.fill_dict(keyvals)
		# self.solver.grid
		# fill simulation time and dt
		# keyvals['time'] = self.max_time
		# keyvals['dt'] = self.time_step_size
		# keyvals['interpolation'] = self.interpolation


		
		# keyvals['nbeams'] = int(np.sum(self._if_beam))
		# keyvals['nspecies'] = len(self._if_beam) - keyvals['nbeams']
		# # TODO add neutrals and laser support
		# keyvals['nneutrals'] = 0
		# keyvals['nlasers'] = len(self.lasers)
		# self.solver.fill_dict(keyvals)
		# keyvals['dump_restart'] = self.dump_restart
		# if(self.dump_restart):
		# 	keyvals['ndump_restart'] = self.ndump_restart
		# keyvals['read_restart'] = self.read_restart
		# if(self.read_restart):
		# 	keyvals['restart_timestep'] = self.restart_timestep
		# keyvals['verbose'] = self.verbose
		# keyvals['if_timing'] = self.if_timing
		# keyvals['random_seed'] = self.random_seed
		# keyvals['algorithm'] = self.algorithm

	def write_input_file(self,file_name):
		total_dict = {}

		# simulation object handled

		f = open(file_name, "w")
		self.sim_fill_dict(f)
		f.close()

	def step(self, nsteps = 1):
		raise Exception('The simulation step feature is not yet supported for ' + codename + '. Please call write_input_file() to construct the input deck.')

class FieldDiagnostic(picmistandard.PICMI_FieldDiagnostic):
	"""

	"""
	_field_list: list = PrivateAttr(default_factory=list)
	_source_list: list = PrivateAttr(default_factory=list)
	_source_list2: list = PrivateAttr(default_factory=list)
	_neutral_list: list = PrivateAttr(default_factory=list)

	def model_post_init(self, context):
		super().model_post_init(context)
		assert self.write_dir != '.', Exception("Write directory feature not yet supported.")
		assert self.period > 0, Exception("Diagnostic period is not valid")
		self._field_list = []
		self._source_list = []
		self._source_list2 = []
		self._neutral_list = []
		if('E' in self.data_list):
			self._field_list += ['e1','e2','e3']
		if('B' in self.data_list):
			self._field_list += ['b1','b2','b3']
		if('rho' in self.data_list):
			self._source_list += ['charge']
			self._neutral_list += ['ion_charge']
		# if('ion' in self.data_list):
			
		if('J' in self.data_list):
			self._source_list2 += ['j1','j2','j3']

		if('Ex' in self.data_list or 'Er' in self.data_list):
			self._field_list.append('e2')
		if('Ey' in self.data_list or 'Ephi' in self.data_list):
			self._field_list.append('e3')
		if('Ez' in self.data_list):
			self._field_list.append('e1')

		if('Bx' in self.data_list or 'Br' in self.data_list):
			self._field_list.append('b2')
		if('By' in self.data_list or 'Bphi' in self.data_list):
			self._field_list.append('b3')
		if('Bz' in self.data_list):
			self._field_list.append('b1')

		if('Jx' in self.data_list or 'Jr' in self.data_list):
			self._source_list2.append('j2')
		if('Jy' in self.data_list or 'Jphi' in self.data_list):
			self._source_list2.append('j3')
		if('Jz' in self.data_list):
			self._source_list2.append('j1')



		# need to add to PICMI standard
		if('psi' in self.data_list):
			self._field_list += ['psi']


	def fill_dict_fld(self,keyvals,cylin_flag):
		if(cylin_flag):
			fld_list = [ele + '_cyl_m' for ele in self._field_list]
		else:
			fld_list = [ele for ele in self._field_list]
		keyvals['reports'] = fld_list
		keyvals['ndump_fac'] = self.period
	
	def fill_dict_neut(self,keyvals,cylin_flag):
		neut_list = [ele for ele in self._neutral_list]
		keyvals['reports'] = neut_list
		keyvals['ndump_fac'] = self.period

	def fill_dict_src(self,keyvals,cylin_flag):
		if(cylin_flag):
			src_list = [ele + '_cyl_m' for ele in self._source_list]
		else:
			src_list = [ele for ele in self._source_list]
		if('reports' in keyvals):
			keyvals['reports'] = src_list
			keyvals['ndump_fac'] = min(keyvals['ndump_fac'],self.period)
		else:
			keyvals['reports'] = src_list
			keyvals['ndump_fac'] = self.period

	def fill_dict_src2(self,keyvals,cylin_flag):
		if(cylin_flag):
			src_list = [ele + '_cyl_m' for ele in self._source_list2]
		else:
			src_list = [ele for ele in self._source_list2]
		if('reports' in keyvals):
			keyvals['reports'] = src_list
			keyvals['ndump_fac'] = min(keyvals['ndump_fac'],self.period)
		else:
			keyvals['reports'] = src_list
			keyvals['ndump_fac'] = self.period


# QuickPIC does not support electrostatic and boosted frame diagnostic 
class ElectrostaticFieldDiagnostic(picmistandard.PICMI_ElectrostaticFieldDiagnostic):
	def model_post_init(self, context):
		raise Exception("Electrostatic field diagnostic not supported in OSIRIS")

class LabFrameParticleDiagnostic(picmistandard.PICMI_LabFrameParticleDiagnostic):
	def model_post_init(self, context):
		raise Exception("Boosted frame diagnostics not support in OSIRIS")

class LabFrameFieldDiagnostic(picmistandard.PICMI_LabFrameFieldDiagnostic):
	def model_post_init(self, context):
		raise Exception("Boosted frame diagnostics not yet supported")


## to be implemented	
class ParticleDiagnostic(picmistandard.PICMI_ParticleDiagnostic):
	"""
	OSIRIS-Specific Parameters

	OSIRIS_sample: integer, optional
		Dumps every nth particle.
	"""
	sample: int = Field(default=1, alias=codename + '_sample',
		description='Dumps every nth particle')
	raw_gamma_limit: float | None = Field(default=None, alias=codename + '_raw_gamma_limit',
		description='Only dump particles above this Lorentz factor')

	def model_post_init(self, context):
		super().model_post_init(context)
		assert self.write_dir != '.', Exception("Write directory feature not yet supported.")
		assert self.period > 0, Exception("Diagnostic period is not valid")
		print('Warning: Particle diagnostic reporting momentum, position and charge data')

	def fill_dict_fld(self,keyvals):
		pass

	def fill_dict_neut(self,keyvals,cylin_flag):
		pass

	def fill_dict_src(self,keyvals,cylin_flag):
		if('ndump_fac_raw' in keyvals):
			keyvals['ndump_fac_raw'] = min(keyvals['ndump_fac_raw'],self.period)
			keyvals['raw_fraction'] = min(keyvals['raw_fraction'], self.sample)
		else:
			keyvals['ndump_fac_raw'] = self.period
			keyvals['raw_fraction'] = self.sample
		if(self.raw_gamma_limit is not None):
			keyvals['raw_gamma_limit'] = self.raw_gamma_limit
		


class GaussianLaser(picmistandard.PICMI_GaussianLaser):
	# laser wavenumber, normalized by normalize_units (the standard's k0 is derived from the wavelength)
	_k0: float | None = PrivateAttr(default=None)

	def model_post_init(self, context):
		super().model_post_init(context)
		self._k0 = self.k0
		rotate_list(self.focal_position)
		rotate_list(self.centroid_position)
		rotate_list(self.propagation_direction)
		rotate_list(self.polarization_direction)

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


	def fill_dict(self,keyvals,grid):
		keyvals['lon_type'] = "gaussian"
		keyvals['lon_duration'] = self.duration * 2 * np.log(2.0)
		keyvals['lon_x0'] = self.centroid_position[0]
		keyvals['lon_range'] = 4 * self.duration 
		if(self.propagation_direction[0] > 0):
			keyvals['propagation'] = 'forward'
		else:
			keyvals['propagation'] = 'backward'
		keyvals['a0'] = self.a0
		keyvals['omega0'] = self._k0
		if(grid.n_x_dim > 2):
			keyvals['per_w0'] = [self.waist, self.waist]
		else:
			keyvals['per_w0'] = self.waist
		keyvals['per_type'] =  "gaussian"
		keyvals['per_center'] = self.centroid_position[1:]
		keyvals['pol'] = np.arctan2(self.polarization_direction[1], self.polarization_direction[0])/ np.pi * 180.0
		keyvals['per_focus'] = self.focal_position[0]
		

	def fill_antenna_dict(self,keyvals,grid, injection_method):
		keyvals['tenv_type'] = "gaussian"
		keyvals['xi0'] = injection_method.position[0]
		keyvals['wall_pos'] = injection_method.position[0]
		keyvals['tenv_duration'] = self.duration * 2 * np.log(2.0)
		keyvals['tenv_launch_duration'] = 2 * np.abs(injection_method.position[0] - self.centroid_position[0])
		keyvals['tenv_range'] = keyvals['tenv_launch_duration']

		if(self.propagation_direction[0] > 0):
			keyvals['propagation'] = 'forward'
			keyvals['wall_vel'] = 0
		else:
			keyvals['propagation'] = 'backward'
			keyvals['wall_vel'] = 0.0
		keyvals['a0'] = self.a0
		keyvals['omega0'] = self._k0
		if(grid.n_x_dim > 2):
			keyvals['per_w0'] = [self.waist, self.waist]
		else:
			keyvals['per_w0'] = self.waist
		keyvals['per_type'] =  "gaussian"
		keyvals['per_center'] = self.centroid_position[1:]
		keyvals['pol'] = np.arctan2(self.polarization_direction[1], self.polarization_direction[0])/ np.pi * 180.0
		keyvals['per_focus'] = self.focal_position[0]
		
		keyvals['ncells'] = 11

		
class LaserAntenna(picmistandard.PICMI_LaserAntenna):
	def model_post_init(self, context):
		super().model_post_init(context)
		self.position = [self.position[2],self.position[1],self.position[0]]
		self.normal_vector = [self.normal_vector[2],self.normal_vector[1],self.normal_vector[0]]
		assert np.any(self.position[1:] != 0), Exception("OSIRIS PICMI currently only supports Antenna injection along z")
		assert np.any(self.normal_vector[1:] != 0), Exception("OSIRIS PICMI currently only supports Antenna injection along z")
		return
	def normalize_units(self, density_norm):
		# normalize quantities to plasma density and skin depths
		w_pe = np.sqrt(constants.q_e**2 * density_norm/(constants.ep0 * constants.m_e) ) 
		k_pe = w_pe/constants.c
		for i in range(3):
			self.position[i] = k_pe * self.position[i]

class BinomialSmoother(picmistandard.PICMI_BinomialSmoother):
	# also accept a single value for all axes, as before the pydantic standard
	n_pass: int | list[int] | None = Field(default=None,
		description='Number of passes along each axis (a single integer applies to all axes)')
	compensation: bool | list[bool] | None = Field(default=None,
		description='Flags whether to apply compensation along each axis (a single flag applies to all axes)')

	def model_post_init(self, context):
		super().model_post_init(context)
		vars_ignored = ''
		if(self.compensation is not None):
			vars_ignored += 'compensation '
		if(self.stride is not None):
			vars_ignored += 'stride '
		if(self.alpha is not None):
			vars_ignored += 'alpha '
		if(vars_ignored != ''):
			print('Ignoring parameters in BinomialSmoother: ' + vars_ignored)


def fortran_format(val):
	if(isinstance(val,bool)):
		if(val):
			val = ".true."
		else:
			val = ".false."
	elif(isinstance(val,str)):
		val = "'" + val + "'"
	else:
		val = str(val)
	return val

def write_dict_fortran(file, diction, name):
	file.write(name + "\n")
	file.write('{ \n')
	for key in diction:
		val = diction[key]
		out = ''
		if(isinstance(val,list)):
			for item in val:
				out = out + fortran_format(item) + ','
		else:
			out = fortran_format(val) + ','
		file.write('\t' + str(key) + ' = ' + out + '\n')
	file.write('} \n \n')

def rotate_list(lst):
	temp = lst[-1]
	lst[1:] = lst[:-1]
	lst[0] = temp

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


	math_func2 = '{}'.format(math_func2).replace('x', '(' + format_decimal(1.0/k_pe) +'* x2)')
	math_func2 = '{}'.format(math_func2).replace('y', '(' + format_decimal(1.0/k_pe) +'* x3)')
	math_func2 = '{}'.format(math_func2).replace('z', '(' + format_decimal(1.0/k_pe) +'* x1)')
	math_func2 = '{}'.format(math_func2).replace('t', '(' + format_decimal(1.0/w_pe) +'* t)')
	for i in range(len(funcs)):
		key1, key2 = funcs2[i], funcs[i]
		math_func2 = '{}'.format(math_func2).replace(key1,key2)
	
	## handle overlap with funcs and variable names
	return math_func2
	
def format_decimal(decimal):
	str_out= '%.6e' % Decimal(str(decimal))
	return str_out

def check_grid_bounds(lower_boundary_conditions, upper_boundary_conditions):
	grid_flag = ['dirichlet', 'neumann', 'periodic','open']
	grid_bound = ['pec', 'pmc',None, 'open']
	part_bound = ['specular', 'specular','periodic','open']
	lower_part_bounds = []
	upper_part_bounds = []
	for i in range(len(lower_boundary_conditions)):
		lower_part_bounds.append(None)
		upper_part_bounds.append(None)
		for j in range(len(grid_flag)):
			if(lower_boundary_conditions[i] == grid_flag[j]):
				lower_boundary_conditions[i] = grid_bound[j]
				lower_part_bounds[i] = part_bound[j]
			if(upper_boundary_conditions[i] == grid_flag[j]):
				upper_boundary_conditions[i] = grid_bound[j]
				upper_part_bounds[i] = part_bound[j]
	return lower_part_bounds, upper_part_bounds

def construct_bounds(lower_bound,upper_bound,n_x_dim):
	front_str =''
	back_str = ''
	coords = ['x1','x2','x3']
	for i in range(n_x_dim):
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



