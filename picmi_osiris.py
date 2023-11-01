# Copyright 2022-2023 Thamine Dalichaouch, Frank Tsung
# OSIRIS extension of PICMI standard

import picmistandard
import numpy as np
import re
import math
import json 
from json import encoder
from itertools import cycle
import periodictable
from decimal import Decimal

encoder.FLOAT_REPR = lambda o: format(o, '.4f')

codename = 'OSIRIS'
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
		elif(isinstance(self.initial_distribution, UniformDistribution)):
			self.profile_type = 'species'
		elif(isinstance(self.initial_distribution, AnalyticDistribution)):
			self.profile_type = 'species'
		else:
			print('Warning: Only Uniform and Gaussian distributions are currently supported.')


	def normalize_units(self):
		# normalized charge, mass, density
		self.q = self.charge/constants.q_e
		self.m = self.mass/constants.m_e



	def fill_dict(self, keyvals,init_self_field):
		keyvals['name'] = self.name
		if(self.profile_type == 'beam' or init_self_field):
			keyvals['free_stream'] = not self.beam_evolution
			keyvals['init_fields'] = True

		keyvals['rqm'] = self.m/self.q


	def activate_field_ionization(self,model,product_species):
		raise Exception('Ionization not yet supported.')


picmistandard.PICMI_MultiSpecies.Species_class = Species
class MultiSpecies(picmistandard.PICMI_MultiSpecies):
	def init(self, kw):
		return
		# for species in self.species_instances_list:
		# 	print(species.name)



class GaussianBunchDistribution(picmistandard.PICMI_GaussianBunchDistribution):
	"""

	"""
	def init(self, kw):
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
		self.q = species.charge/constants.q_e
		self.m = species.mass/constants.m_e
		self.norm_density = peak_density/density_norm

		self.tot_charge = total_charge/(-constants.q_e * density_norm * k_pe**-3)

	def fill_dict(self,species,grid):
		n_x_dim = grid.n_x_dim
		udist_dict, profile_dict = {},{}

		udist_dict['uth'] = self.rms_velocity
		udist_dict['ufl'] = self.centroid_velocity
		udist_dict['use_classical_uadd'] = True


		profile_dict['profile_type'] = ["gaussian"] * n_x_dim
		profile_dict['gauss_center'] = self.centroid_position[:n_x_dim]
		profile_dict['gauss_sigma'] = self.rms_bunch_size[:n_x_dim]

		if(isinstance(grid,CylindricalGrid)):
			profile_dict['aspect_ratio'] = self.rms_bunch_size[2]/self.rms_bunch_size[1]

		q_scale = np.abs(species.charge/constants.q_e)
		if(species.density_scale is not None):
			profile_dict['density'] = self.norm_density * species.density_scale * q_scale
		else:
			profile_dict['density'] = self.norm_density * q_scale

		for j in range(n_x_dim):
			profile_dict['gauss_range(:,' + str(j+1) + ')'] = [-4 * self.rms_bunch_size[j] +self.centroid_position[j],\
			 4 * self.rms_bunch_size[j] + self.centroid_position[j]]
		return udist_dict,profile_dict


class AnalyticDistribution(picmistandard.PICMI_AnalyticDistribution):
	"""
	OSIRIS-Specific Parameters
	
	### Plasma-specific parameters ####

	"""
	def init(self,kw):
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
		self.norm_density = 1.0
		self.q = species.charge/constants.q_e
		self.m = species.mass/constants.m_e
		


		for i in range(3):
			self.rms_velocity[i] /= constants.c 

		if(np.any(self.directed_velocity != 0.0)):
			print('Warning: ' + codename + ' does not support directed velocity for Analytic Distributions.')


	def fill_dict(self,species,grid):
		back_str,front_str = construct_bounds(self.lower_bound,self.upper_bound,grid.n_x_dim)
		
		q_scale = np.abs(species.charge/constants.q_e)
		udist_dict, profile_dict = {}, {}
		if(species.density_scale is not None):
			profile_dict['density'] = species.density_scale * q_scale
		else:
			profile_dict['density'] = q_scale
		profile_dict['profile_type'] = "math func" 
		profile_dict['math_func_expr'] = front_str + self.density_expression + back_str

		udist_dict['uth'] = self.rms_velocity
		udist_dict['ufl'] = self.directed_velocity
		udist_dict['use_classical_uadd'] = True
		
		return udist_dict, profile_dict



class UniformDistribution(picmistandard.PICMI_UniformDistribution):
	def init(self,kw):
		# default profile for uniform plasmas
		rotate_list(self.lower_bound)
		rotate_list(self.upper_bound)
		rotate_list(self.rms_velocity)
		rotate_list(self.directed_velocity)
		if(not self.fill_in):
			print('OSIRIS defaults to fill_in = True.')




	def normalize_units(self,species, density_norm):
		# normalize plasma density
		self.norm_density = self.density/density_norm

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

		if(np.any(self.directed_velocity != 0.0)):
			print('Warning: ' + codename + ' does not support directed velocity for Uniform Distributions.')

	def fill_dict(self,species,grid):
		back_str,front_str = construct_bounds(self.lower_bound,self.upper_bound,grid.n_x_dim)
		
		q_scale = np.abs(species.charge/constants.q_e)
		udist_dict, profile_dict = {}, {}
		if(species.density_scale is not None):
			profile_dict['density'] = species.density_scale * q_scale
		else:
			profile_dict['density'] = q_scale
		profile_dict['profile_type'] = "math func" 
		profile_dict['math_func_expr'] = front_str + str(self.norm_density)+ back_str
		udist_dict['uth'] = self.rms_velocity
		udist_dict['ufl'] = self.directed_velocity
		udist_dict['use_classical_uadd'] = True
		
		return udist_dict, profile_dict
		






class ParticleListDistribution(picmistandard.PICMI_ParticleListDistribution):
	def init(self,kw):
		raise Exception('Particle list distributions not yet supported in OSIRIS')


# constant, analytic, or mirror fields not yet supported in QuickPIC 
class ConstantAppliedField(picmistandard.PICMI_ConstantAppliedField):
	def init(self,kw):
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

		
		self.Ex_expression = E_list[0]
		self.Ey_expression = E_list[1]
		self.Ez_expression = E_list[2]
		self.Bx_expression = B_list[0]
		self.By_expression = B_list[1]
		self.Bz_expression = B_list[2]


	def fill_dict(self,keyvals,grid):
		keyvals["ext_fld"] = "static"
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



class AnalyticAppliedField(picmistandard.PICMI_AnalyticAppliedField):
	def init(self,kw):
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
	def init(self,kw):
		raise Exception("Mirrors are not yet supported in OSIRIS")


class ElectromagneticSolver(picmistandard.PICMI_ElectromagneticSolver):
	def init(self, kw):
		self.method = self.method.lower()
		if(self.method == 'ckc'):
			self.method = 'ck'

	def get_info(self):
		el_mag_fld_dict = {}
		el_mag_fld_dict['solver'] = self.method
		# write_dict_fortran(file,el_mag_fld_dict,'el_mag_fld')

		emf_bound_dict = {}
		for i in range(len(self.grid.lower_boundary_conditions)):
			if(self.grid.lower_boundary_conditions[i] is not None and self.grid.upper_boundary_conditions[i] is not None):
				emf_bound_dict['type(:,' + str(i+1) + ')'] = [self.grid.lower_boundary_conditions[i], self.grid.upper_boundary_conditions[i]]
		el_mag_fld_solver_dict = {}
		if(self.method == 'fei'):
			el_mag_fld_solver_dict['type'] = 'xu'
			el_mag_fld_solver_dict['solver_ord'] = 2
			el_mag_fld_solver_dict['n_coef'] = 8
			el_mag_fld_solver_dict['weight_w'] = 0.3
			el_mag_fld_solver_dict['weight_n'] = 10
		# write_dict_fortran(file,el_mag_fld_solver_dict,'emf_solver')
		return el_mag_fld_dict, emf_bound_dict,el_mag_fld_solver_dict
	def fill_src_smoother_dict(self,smooth_dict):
		smooth_dict['type'] = ['binomial'] * self.grid.n_x_dim
		smooth_dict['order'] = [self.source_smoother.n_pass] * self.grid.n_x_dim


		
class ElectrostaticSolver(picmistandard.PICMI_ElectrostaticSolver):
	def init(self, kw):
		raise Exception('This feature is not yet supported. Please use the Electromagnetic solver.')





		
	
		
		

## Grids
class Cartesian1DGrid(picmistandard.PICMI_Cartesian1DGrid):
	def init(self, kw):

		self.n_x_dim = 1
		
		self.if_periodic = [ele == 'periodic' for ele in self.lower_boundary_conditions]
		self.lower_particle_bounds, self.upper_particle_bounds = check_grid_bounds(self.lower_boundary_conditions,self.upper_boundary_conditions)


		self.if_move = [ele != 0 for ele in self.moving_window_velocity]

		if(self.lower_boundary_conditions[0] == 'open'):
			self.lower_boundary_conditions[0] = 'lindman'
		if(self.upper_boundary_conditions[0] == 'open'):
			self.upper_boundary_conditions[0] = 'lindman'

		self.coordinates = 'cartesian'

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
		keyvals['coordinates'] = self.coordinates

	def get_info(self):
		part_bound_dict = {}
		for i in range(self.n_x_dim):
			if(self.lower_particle_bounds[i] is not None and self.upper_particle_bounds[i] is not None):
				part_bound_dict['type(:,' + str(i+1) + ')'] = [self.lower_particle_bounds[i], self.upper_particle_bounds[i]]
		return part_bound_dict



class Cartesian2DGrid(picmistandard.PICMI_Cartesian2DGrid):
	def init(self, kw):

		self.n_x_dim = 2
		
		self.if_periodic = [ele == 'periodic' for ele in self.lower_boundary_conditions]
		self.lower_particle_bounds, self.upper_particle_bounds = check_grid_bounds(self.lower_boundary_conditions,self.upper_boundary_conditions)


		self.if_move = [ele != 0 for ele in self.moving_window_velocity]
		rotate_list(self.lower_boundary_conditions)
		rotate_list(self.upper_boundary_conditions)
		rotate_list(self.lower_particle_bounds)
		rotate_list(self.upper_particle_bounds)
		rotate_list(self.if_move)
		rotate_list(self.if_periodic)
		rotate_list(self.lower_bound)
		rotate_list(self.upper_bound)
		rotate_list(self.number_of_cells)

		if(self.lower_boundary_conditions[0] == 'open'):
			self.lower_boundary_conditions[0] = 'lindman'
		if(self.upper_boundary_conditions[0] == 'open'):
			self.upper_boundary_conditions[0] = 'lindman'

		self.coordinates = 'cartesian'

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
		keyvals['coordinates'] = self.coordinates

	def get_info(self):
		part_bound_dict = {}
		for i in range(self.n_x_dim):
			if(self.lower_particle_bounds[i] is not None and self.upper_particle_bounds[i] is not None):
				part_bound_dict['type(:,' + str(i+1) + ')'] = [self.lower_particle_bounds[i], self.upper_particle_bounds[i]]
		return part_bound_dict



class Cartesian3DGrid(picmistandard.PICMI_Cartesian3DGrid):
	def init(self, kw):

		self.n_x_dim = 3
		
		self.if_periodic = [ele == 'periodic' for ele in self.lower_boundary_conditions]
		self.lower_particle_bounds, self.upper_particle_bounds = check_grid_bounds(self.lower_boundary_conditions,self.upper_boundary_conditions)


		self.if_move = [ele != 0 for ele in self.moving_window_velocity]
		rotate_list(self.lower_boundary_conditions)
		rotate_list(self.upper_boundary_conditions)
		rotate_list(self.lower_particle_bounds)
		rotate_list(self.upper_particle_bounds)
		rotate_list(self.if_move)
		rotate_list(self.if_periodic)
		rotate_list(self.lower_bound)
		rotate_list(self.upper_bound)
		rotate_list(self.number_of_cells)

		if(self.lower_boundary_conditions[0] == 'open'):
			self.lower_boundary_conditions[0] = 'lindman'
		if(self.upper_boundary_conditions[0] == 'open'):
			self.upper_boundary_conditions[0] = 'lindman'

		self.coordinates = 'cartesian'

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
		keyvals['coordinates'] = self.coordinates

	def get_info(self):
		part_bound_dict = {}
		for i in range(self.n_x_dim):
			if(self.lower_particle_bounds[i] is not None and self.upper_particle_bounds[i] is not None):
				part_bound_dict['type(:,' + str(i+1) + ')'] = [self.lower_particle_bounds[i], self.upper_particle_bounds[i]]
		return part_bound_dict


class CylindricalGrid(picmistandard.PICMI_CylindricalGrid):
	def init(self, kw):

		self.n_x_dim = 2
		
		self.if_periodic = [ele == 'periodic' for ele in self.lower_boundary_conditions]
		self.lower_particle_bounds, self.upper_particle_bounds = check_grid_bounds(self.lower_boundary_conditions,self.upper_boundary_conditions)
		self.lower_particle_bounds[0] = 'axial'
		self.lower_boundary_conditions[0] = 'axial' 


		self.if_move = [ele != 0 for ele in self.moving_window_velocity]
		rotate_list(self.lower_boundary_conditions)
		rotate_list(self.upper_boundary_conditions)
		rotate_list(self.lower_particle_bounds)
		rotate_list(self.upper_particle_bounds)
		rotate_list(self.if_move)
		rotate_list(self.if_periodic)
		rotate_list(self.lower_bound)
		rotate_list(self.upper_bound)
		rotate_list(self.number_of_cells)

		if(self.lower_boundary_conditions[0] == 'open'):
			self.lower_boundary_conditions[0] = 'lindman'
		if(self.upper_boundary_conditions[0] == 'open'):
			self.upper_boundary_conditions[0] = 'lindman'

		self.coordinates = 'cylindrical'

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
		keyvals['coordinates'] = self.coordinates
		keyvals['n_cyl_modes'] = self.n_azimuthal_modes

	def get_info(self):
		part_bound_dict = {}
		for i in range(self.n_x_dim):
			if(self.lower_particle_bounds[i] is not None and self.upper_particle_bounds[i] is not None):
				part_bound_dict['type(:,' + str(i+1) + ')'] = [self.lower_particle_bounds[i], self.upper_particle_bounds[i]]
		return part_bound_dict



		

class PseudoRandomLayout(picmistandard.PICMI_PseudoRandomLayout):
	"""
	OSIRIS-Specific Parameters

	"""
	def init(self,kw):
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
	def init(self,kw):
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
	def init(self,kw):
		# set verbose default
		if(self.verbose is None):
			self.verbose = 0
		assert self.time_step_size is not None, Exception('OSIRIS requires a time step size.')
		if(self.particle_shape == 'NGP' or self.particle_shape == None):
			print('OSIRIS does not support NGP or None. Defaulting to linear')
			self.particle_shape = 'linear'

		if(self.cpu_split is None):
			self.cpu_split = [1,1,1]

		rotate_list(self.cpu_split)


		### OSIRIS beam flag
		self.if_beam = []

		# check if normalized density is specified
		self.n0 = kw.pop(codename + '_n0', None)
		# set number of seeds for pseudo-random numbers
		self.random_seed = kw.pop(codename + '_random_seed', 10)

		# set timings (default is true)
		self.if_timing = kw.pop(codename + '_timings', False)

		# normalize simulation time
		# print('n0', self.n0)
		if(self.n0 is not None):
			self.normalize_simulation()
		

		# check to read from restart files
		self.read_restart = kw.pop(codename + '_read_restart', False)

		# check if dumping restart files
		self.ndump_restart = kw.pop(codename + '_ndump_restart', 0)
		

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


	def add_applied_field(self, applied_field):
		picmistandard.PICMI_Simulation.add_applied_field( self, applied_field )
		if(self.n0 is not None):
			applied_field.normalize_units(self.n0)

	def add_laser(self, laser, injection_method):
		picmistandard.PICMI_Simulation.add_laser(self, laser, injection_method)
		if(self.n0 is not None):
			laser.normalize_units(self.n0)


			


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
		node_dict['if_periodic'] = self.solver.grid.if_periodic
		write_dict_fortran(file,node_dict, 'node_conf')

		grid_dict = {}
		self.solver.grid.fill_dict(grid_dict)
		write_dict_fortran(file,grid_dict, 'grid')

		time_step_dict = {}
		time_step_dict['dt'] = self.time_step_size
		time_step_dict['ndump'] = 1
		write_dict_fortran(file,time_step_dict,'time_step')

		restart_dict = {}
		restart_dict['ndump_fac'] = self.ndump_restart
		restart_dict['if_restart'] = self.read_restart
		restart_dict['if_remold'] = False
		write_dict_fortran(file,restart_dict, 'restart')

		space_dict = {}
		space_dict['xmin'] = self.solver.grid.lower_bound
		space_dict['xmax'] = self.solver.grid.upper_bound
		space_dict['if_move'] = self.solver.grid.if_move
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
		particles_dict['num_species'] = len(self.species)
		particles_dict['interpolation'] = self.particle_shape
		write_dict_fortran(file,particles_dict, 'particles')

		for i in range(len(self.species)):
			spec = self.species[i]
			temp_dict = {}
			# 
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


		
		# keyvals['nbeams'] = int(np.sum(self.if_beam))
		# keyvals['nspecies'] = len(self.if_beam) - keyvals['nbeams']
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
	def init(self,kw):
		assert self.write_dir != '.', Exception("Write directory feature not yet supported.")
		assert self.period > 0, Exception("Diagnostic period is not valid")
		self.field_list = []
		self.source_list = []
		self.source_list2 = []
		if('E' in self.data_list):
			self.field_list += ['e1','e2','e3']
		if('B' in self.data_list):
			self.field_list += ['b1','b2','b3']
		if('rho' in self.data_list):
			self.source_list += ['charge']
		if('J' in self.data_list):
			self.source_list2 += ['j1','j2','j3']

		if('Ex' in self.data_list):
			self.field_list.append('e2')
		if('Ey' in self.data_list):
			self.field_list.append('e3')
		if('Ez' in self.data_list):
			self.field_list.append('e1')

		if('Bx' in self.data_list):
			self.field_list.append('b2')
		if('By' in self.data_list):
			self.field_list.append('b3')
		if('Bz' in self.data_list):
			self.field_list.append('b1')

		if('Jx' in self.data_list):
			self.source_list2.append('j2')
		if('Jy' in self.data_list):
			self.source_list2.append('j3')
		if('Jz' in self.data_list):
			self.source_list2.append('j1')



		# need to add to PICMI standard
		if('psi' in self.data_list):
			self.field_list += ['psi']


	def fill_dict_fld(self,keyvals,cylin_flag):
		if(cylin_flag):
			fld_list = [ele + '_cyl_m' for ele in self.field_list]
		else:
			fld_list = [ele for ele in self.field_list]
		keyvals['reports'] = fld_list
		keyvals['ndump_fac'] = self.period

	def fill_dict_src(self,keyvals,cylin_flag):
		if(cylin_flag):
			src_list = [ele + '_cyl_m' for ele in self.source_list]
		else:
			src_list = [ele for ele in self.source_list]
		if('reports' in keyvals):
			keyvals['reports'] = src_list
			keyvals['ndump_fac'] = min(keyvals['ndump_fac'],self.period)
		else:
			keyvals['reports'] = src_list
			keyvals['ndump_fac'] = self.period

	def fill_dict_src2(self,keyvals,cylin_flag):
		if(cylin_flag):
			src_list = [ele + '_cyl_m' for ele in self.source_list2]
		else:
			src_list = [ele for ele in self.source_list2]
		if('reports' in keyvals):
			keyvals['reports'] = src_list
			keyvals['ndump_fac'] = min(keyvals['ndump_fac'],self.period)
		else:
			keyvals['reports'] = src_list
			keyvals['ndump_fac'] = self.period


# QuickPIC does not support electrostatic and boosted frame diagnostic 
class ElectrostaticFieldDiagnostic(picmistandard.PICMI_ElectrostaticFieldDiagnostic):
	def init(self,kw):
		raise Exception("Electrostatic field diagnostic not supported in OSIRIS")

class LabFrameParticleDiagnostic(picmistandard.PICMI_LabFrameParticleDiagnostic):
	def init(self,kw):
		raise Exception("Boosted frame diagnostics not support in OSIRIS")

class LabFrameFieldDiagnostic(picmistandard.PICMI_LabFrameFieldDiagnostic):
	def init(self,kw):
		raise Exception("Boosted frame diagnostics not yet supported")


## to be implemented	
class ParticleDiagnostic(picmistandard.PICMI_ParticleDiagnostic):
	"""
	OSIRIS-Specific Parameters

	OSIRIS_sample: integer, optional
		Dumps every nth particle.
	"""
	def init(self,kw):
		assert self.write_dir != '.', Exception("Write directory feature not yet supported.")
		assert self.period > 0, Exception("Diagnostic period is not valid")
		print('Warning: Particle diagnostic reporting momentum, position and charge data')
		self.sample = kw.pop(codename + '_sample', 1) 

	def fill_dict_fld(self,keyvals):
		pass

	def fill_dict_src(self,keyvals,cylin_flag):
		if('ndump_fac_raw' in keyvals):
			keyvals['ndump_fac_raw'] = min(keyvals['ndump_fac_raw'],self.period)
			keyvals['raw_fraction'] = min(keyvals['raw_fraction'], self.sample)
		else:
			keyvals['ndump_fac_raw'] = self.period
			keyvals['raw_fraction'] = self.sample
		


class GaussianLaser(picmistandard.PICMI_GaussianLaser):
	def init(self, kw):
		rotate_list(self.focal_position)
		rotate_list(self.centroid_position)
		rotate_list(self.propagation_direction)
		rotate_list(self.polarization_direction)

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


	def fill_dict(self,keyvals,grid):
		keyvals['lon_type'] = "polynomial"
		if(self.propagation_direction[0] > 0):
			keyvals['propagation'] = 'forward'
		else:
			keyvals['propagation'] = 'backward'
		keyvals['a0'] = self.a0
		keyvals['omega0'] = self.k0
		if(grid.n_x_dim > 2):
			keyvals['per_w0'] = [self.waist, self.waist]
		else:
			keyvals['per_w0'] = self.waist
		keyvals['per_type'] =  "gaussian"
		keyvals['per_center'] = self.centroid_position[1:]
		keyvals['pol'] = np.arctan2(self.polarization_direction[1], self.polarization_direction[0])/ np.pi * 180.0
		keyvals['per_focus'] = self.focal_position[0]
		keyvals['lon_rise'] = self.duration * 1.5275
		keyvals['lon_fall'] = self.duration * 1.5275
		keyvals['lon_flat'] = 0
		keyvals['lon_start'] = self.centroid_position[0] + keyvals['lon_rise']

	def fill_antenna_dict(self,keyvals,grid, injection_method):
		keyvals['tenv_type'] = "polynomial"
		if(self.propagation_direction[0] > 0):
			keyvals['propagation'] = 'forward'
			keyvals['wall_vel'] = -0.99
		else:
			keyvals['propagation'] = 'backward'
			keyvals['wall_vel'] = 0.99
		keyvals['a0'] = self.a0
		keyvals['omega0'] = self.k0
		if(grid.n_x_dim > 2):
			keyvals['per_w0'] = [self.waist, self.waist]
		else:
			keyvals['per_w0'] = self.waist
		keyvals['per_type'] =  "gaussian"
		keyvals['per_center'] = self.centroid_position[1:]
		keyvals['pol'] = np.arctan2(self.polarization_direction[1], self.polarization_direction[0])/ np.pi * 180.0
		keyvals['per_focus'] = self.focal_position[0]
		keyvals['tenv_rise'] = self.duration * 1.5275
		keyvals['tenv_fall'] = self.duration * 1.5275
		keyvals['wall_pos'] = self.centroid_position[0] + keyvals['tenv_rise']
		keyvals['xi0'] = keyvals['wall_pos']
		keyvals['ncells'] = 11

		
class LaserAntenna(picmistandard.PICMI_LaserAntenna):
	def init(self, kw):
		return

class BinomialSmoother(picmistandard.PICMI_BinomialSmoother):
	def init(self, kw):
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


