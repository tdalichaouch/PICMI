# Basic two bunch PICMI input file


# import picmi_qpic
import picmi_qpic as picmi
# from quickpic import picmi
import numpy as np

cst = picmi.constants



plasma_density = 1.e17 * 1e6
w_pe = np.sqrt(cst.q_e**2 * plasma_density/(cst.ep_0 * cst.m_e))
k_pe = w_pe/cst.c
plasma_min     = [None,  None, 5.0/k_pe]
plasma_max     = [None,  None,  105.0/k_pe]

nx = 512
ny = 512
nz = 512
xmax = 7.5/k_pe
xmin = -xmax
ymax = 7.5/k_pe
ymin = -ymax
zmin = -7.5/k_pe
zmax = 2.5/k_pe


# --- drive bunch 
bunch_rms_size_dr            = [0.331/k_pe, 0.331/k_pe, 0.48/k_pe] # 5.56 um, 5.56 um, 8.07 um
bunch_physical_particles_dr  = int(16 * plasma_density * np.prod(bunch_rms_size_dr) * (2 * np.pi)**1.5) # Q = 1 nc
bunch_rms_velocity_dr        = [10.*cst.c,10.*cst.c,0.0]
bunch_centroid_position_dr   = [0.,0., 0.0]
bunch_centroid_velocity_dr   = [0.,0.,20000.*cst.c]

# --- trailing bunch 
bunch_rms_size_tr            = [0.1026/k_pe, 0.1026/k_pe, 0.24/k_pe] # 1.72 um, 1.72 um, 4.03 um
bunch_physical_particles_tr  = int(100 * plasma_density * np.prod(bunch_rms_size_tr) * (2 * np.pi)**1.5) # Q = 0.3 nC
bunch_rms_velocity_tr        = [10.*cst.c,10.*cst.c,0.0]
bunch_centroid_position_tr   = [0.,0.,-5.55/k_pe]
bunch_centroid_velocity_tr   = [0.,0.,20000.*cst.c]

moving_window_velocity = [0., 0., cst.c]

## QuickPIC needs a quasistatic solver...

em_solver_method = 'Yee'
geometry = '3D'
n_macroparticle_per_cell = [2, 2, 2]

if picmi.codename == 'QPAD':
	geometry = 'Quasi-3D'




if(geometry == '3D'):
	grid = picmi.Cartesian3DGrid(
			number_of_cells           = [nx, ny, nz],
			lower_bound               = [xmin, ymin, zmin],
			upper_bound               = [xmax, ymax, zmax],
			lower_boundary_conditions = ['dirichlet', 'dirichlet', 'open'],
			upper_boundary_conditions = ['dirichlet', 'dirichlet', 'open'],
			moving_window_velocity    = moving_window_velocity)
elif(geometry == 'Quasi-3D'):
	grid = picmi.CylindricalGrid(
			number_of_cells           = [nx//2, nz],
			lower_bound               = [0. , zmin],
			upper_bound               = [xmax, zmax],
			lower_boundary_conditions = ['open', 'open'],
			upper_boundary_conditions = ['open', 'open'],
			n_azimuthal_modes = 1,
			moving_window_velocity    = [0,moving_window_velocity[-1]])



beam_dist_dr = picmi.GaussianBunchDistribution(
			n_physical_particles = bunch_physical_particles_dr,
			rms_bunch_size       = bunch_rms_size_dr,
			rms_velocity         = bunch_rms_velocity_dr,
			centroid_position    = bunch_centroid_position_dr,
			centroid_velocity    = bunch_centroid_velocity_dr )

beam_dist_tr = picmi.GaussianBunchDistribution(
			n_physical_particles = bunch_physical_particles_tr,
			rms_bunch_size       = bunch_rms_size_tr,
			rms_velocity         = bunch_rms_velocity_tr,
			centroid_position    = bunch_centroid_position_tr,
			centroid_velocity    = bunch_centroid_velocity_tr )

drive_beam = picmi.Species( particle_type        = 'electron',
					  name                 = 'drive',
					  initial_distribution = beam_dist_dr)

trailing_beam = picmi.Species( particle_type        = 'electron',
					  name                 = 'trailing',
					  initial_distribution = beam_dist_tr)

solver = picmi.ElectromagneticSolver( grid            = grid,
									  cfl             = 1.,
									  method          = em_solver_method)

plasma_dist = picmi.UniformDistribution(
			density = plasma_density,
			lower_bound = plasma_min,
			upper_bound = plasma_max,
			QuickPIC_r_max = 20.e-6,
			QuickPIC_r_min = 0.0 )

plasma = picmi.Species(particle_type = 'electron', 
						name = 'plasma',
						initial_distribution = plasma_dist)


beam_layout = picmi.PseudoRandomLayout(
					grid = grid,
					n_macroparticles = 128**3)


### Particle diagnostics for each species
field_diag = picmi.FieldDiagnostic(data_list = ['E','rho','psi'],
                                   grid = grid,
                                   period = 1,
                                   QuickPIC_slice = ['yz',257])

part_diag = picmi.ParticleDiagnostic(period = 1,
                                     species = [drive_beam, trailing_beam],
                                     QuickPIC_sample = 20)

plasma_layout = picmi.GriddedLayout(
					grid = grid,
					n_macroparticle_per_cell = n_macroparticle_per_cell)



## modify time-step for OSIRIS/WarpX codes
dt = 10.0/w_pe
tmax = 100.0/w_pe
sim = picmi.Simulation(solver = solver, verbose = 1, QuickPIC_n0 = plasma_density,\
	QuickPIC_nodes = [128, 1], time_step_size = dt, max_time =tmax)


sim.add_species(species = drive_beam, layout = beam_layout)
sim.add_species(species = trailing_beam, layout = beam_layout)
sim.add_species(species = plasma, layout = plasma_layout)
sim.add_diagnostic(field_diag)
sim.add_diagnostic(part_diag)

run_python_simulation = False
max_steps = int(tmax/dt)
if run_python_simulation:
	sim.step(max_steps)
else:
	sim.write_input_file('qpinput_qpic.json')

