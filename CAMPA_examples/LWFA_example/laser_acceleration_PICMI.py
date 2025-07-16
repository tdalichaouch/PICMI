# This should be the only line that needs to be changed for different codes
# e.g. `from pywarpx import picmi`
#      `from fbpic import picmi`
#      `from warp import picmi`
#from fbpic import picmi
# from pywarpx import picmi
# import picmi_osiris as picmi
import os,sys
sys.path.append(os.path.abspath('../../picmi_scripts'))

import picmi_osiris as picmi
import numpy as np

# Create alias fror constants
cst = picmi.constants

# Run parameters - can be in separate file
# ========================================

# Physics parameters
# ------------------

# --- laser
import math
laser_polarization   = math.pi/2 # Polarization angle (in rad)
laser_a0             = 4.        # Normalized potential vector
laser_wavelength     = 8e-07     # Wavelength of the laser (in meters)
laser_waist          = 5e-06     # Waist of the laser (in meters)
laser_duration       = 15e-15    # Duration of the laser (in seconds)
laser_injection_loc  = 9.e-6     # Position of injection (in meters, along z)
laser_focal_distance = 100.e-6   # Focal distance from the injection (in meters)
laser_t_peak         = 30.e-15   # The time at which the laser reaches its peak
                                 # at the antenna injection location (in seconds)
# --- plasma
plasma_density = 4.2 * 1e17 * 1e6
plasma_density_expression = "1.e23*(1+tanh((z-20.e-6)/10.e-6))/2."
plasma_min     = [-20.e-6, -20.e-6,  0.0e-6]
plasma_max     = [ 20.e-6,  20.e-6,  1.e-3]

# --- electron bunch
bunch_physical_particles  = 1.e8
bunch_rms_size            = [1.e-6, 1.e-6, 1.e-6]
bunch_rms_velocity        = [0.,0.,10.*cst.c]
bunch_centroid_position   = [0.,0.,-22.e-6]
bunch_centroid_velocity   = [0.,0.,1000.*cst.c]

# Numerics parameters
# -------------------

# --- geometry and solver
em_solver_method = 'CKC'
geometry = '3D'


# Note that code-specific changes can be introduced with `picmi.codename`

# --- Nb time steps
max_steps = 1000

# --- grid
nx = 64
ny = 64
nz = 480
xmin = 1.5*plasma_min[0]
xmax = 1.5*plasma_max[0]
ymin = 1.5*plasma_min[1]
ymax = 1.5*plasma_max[1]
zmin = -38.e-6
zmax = 10.e-6
moving_window_velocity = [0., 0., cst.c]
n_macroparticle_per_cell = [2, 2, 2]

# --- geometry and solver
em_solver_method = 'CKC'  # Cole-Karkkainen-Cowan stencil
geometry = '3D'

# Note that code-specific changes can be introduced with `picmi.codename`
if picmi.codename == 'fbpic':
    em_solver_method = 'PSATD' # Pseudo-Spectral Analytical Time Domain solver
    geometry = 'RZ'
    n_macroparticle_per_cell = [2, 4, 2]
if picmi.codename == 'QPAD':
    em_solver_method = 'Yee' # yee solver
    geometry = 'RZ'
    n_macroparticle_per_cell = [2, 4, 2]

if picmi.codename == 'OSIRIS' or picmi.codename == 'QuickPIC':
    em_solver_method = 'Yee' # yee solver -- to do add fei solver
    geometry = '3D'
    n_macroparticle_per_cell = [2, 4, 2]
    # number of particle per cell in the r, theta, z direction respectively

if(geometry == '3D'):
    cpu_split = [2,2,4]
else:
    cpu_split = [4,2]
# Physics part - can be in separate file
# ======================================

# Physics components
# ------------------

# --- laser
laser = picmi.GaussianLaser(
    wavelength             = laser_wavelength,
    waist                  = laser_waist,
    duration               = laser_duration,
    focal_position         = [0., 0., laser_focal_distance + laser_injection_loc],
    centroid_position      = [0., 0., laser_injection_loc - cst.c*laser_t_peak],
    polarization_direction = [math.cos(laser_polarization), math.sin(laser_polarization), 0.],
    propagation_direction  = [0,0,1],
    a0                     = laser_a0)

# --- plasma
plasma_dist = picmi.AnalyticDistribution(
                density_expression = plasma_density_expression,
                lower_bound        = plasma_min,
                upper_bound        = plasma_max,
                fill_in            = True)
plasma = picmi.MultiSpecies(
                particle_types = ['He', 'Ar', 'electron'],
                names          = ['He+', 'Argon', 'e-'],
                charge_states  = [1, 5, None],
                proportions    = [0.2, 0.8, 0.2 + 5*0.8],
                initial_distribution=plasma_dist)


# --- electron bunch
beam_dist = picmi.GaussianBunchDistribution(
            n_physical_particles = bunch_physical_particles,
            rms_bunch_size       = bunch_rms_size,
            rms_velocity         = bunch_rms_velocity,
            centroid_position    = bunch_centroid_position,
            centroid_velocity    = bunch_centroid_velocity )
beam = picmi.Species( particle_type        = 'electron',
                      name                 = 'beam',
                      initial_distribution = beam_dist)

# Numerics components
# -------------------

if geometry == '3D':
    grid = picmi.Cartesian3DGrid(
        number_of_cells           = [nx, ny, nz],
        lower_bound               = [xmin, ymin, zmin],
        upper_bound               = [xmax, ymax, zmax],
        lower_boundary_conditions = ['periodic', 'periodic', 'open'],
        upper_boundary_conditions = ['periodic', 'periodic', 'open'],
        moving_window_velocity    = moving_window_velocity,
        warpx_max_grid_size       = 32)
        # Note that code-specific arguments use the code name as a prefix.
elif geometry == 'RZ':
    # In the following lists:
    # - the first element corresponds to the radial direction
    # - the second element corresponds to the longitudinal direction
    grid = picmi.CylindricalGrid(
        number_of_cells           = [nx//2, nz],
        lower_bound               = [0., zmin],
        upper_bound               = [xmax, zmax],
        lower_boundary_conditions = [ None, 'open'],
        upper_boundary_conditions = ['open', 'open'],
        n_azimuthal_modes         = 2,
        moving_window_velocity    = moving_window_velocity,
        warpx_max_grid_size       = 32)

smoother = picmi.BinomialSmoother( n_pass = 1, compensation = False )
solver = picmi.ElectromagneticSolver( grid            = grid,
                                      cfl             = 1.,
                                      method          = em_solver_method,
                                     source_smoother = smoother)



# Default Warpx Simulation/diagnostic params
dt = 3.335640952e-16
ndump = 100

if(picmi.codename == 'OSIRIS'):
    dt = 0.5 * dt
    ndump = 200
    max_steps = max_steps * 2

if(picmi.codename =='QPAD' or picmi.codename == 'QuickPIC'):
    dt = 100 * dt
    ndump = 1
    max_steps = max_steps/100


# Diagnostics
# -----------
field_diag = picmi.FieldDiagnostic(grid = grid,
                                    period = ndump,
                                    data_list=['rho','Ey','Ez','Jz'],
                                    warpx_format="openpmd",
                                    warpx_openpmd_backend="h5")
#                                    warpx_plot_raw_fields = 1,
#                                    warpx_plot_raw_fields_guards = 1,
#                                    warpx_plot_finepatch = 1,
#                                    warpx_plot_crsepatch = 1)
part_diag = picmi.ParticleDiagnostic(period = ndump,
                                      species = [beam],
                                    warpx_format="openpmd",
                                    warpx_openpmd_backend="h5")

# Simulation setup
# -----------------

# MPI Split
sim_dict ={}
if(picmi.codename == 'QPAD' or picmi.codename == 'OSIRIS' or picmi.codename == 'QuickPIC'):
    sim_dict[picmi.codename + '_n0'] = plasma_density
    sim_dict[picmi.codename + '_nodes'] = cpu_split

# Initialize the simulation object
# Note that the time step size is obtained from the solver
sim = picmi.Simulation(solver = solver, verbose = 1,
                    time_step_size = dt,  **sim_dict)

# Inject the laser through an antenna
antenna = picmi.LaserAntenna(
                position = (0, 0, 9.e-6),
                normal_vector = (0, 0, 1.))
sim.add_laser(laser, injection_method = antenna)

# Add the plasma: continuously injected by the moving window
plasma_layout = picmi.GriddedLayout(
                    grid = grid,
                    n_macroparticle_per_cell = n_macroparticle_per_cell)
sim.add_species(species=plasma, layout=plasma_layout)

# Add the beam
if(picmi.codename == 'OSIRIS'):
    beam_layout = picmi.PseudoRandomLayout(
                n_macroparticles_per_cell = n_macroparticle_per_cell,
                seed = 0)
else:
    beam_layout = picmi.PseudoRandomLayout(
                n_macroparticles = 1e5,
                seed = 0)

sim.add_species(species=beam, layout=beam_layout, 
                initialize_self_field=False)

# Add the diagnostics
sim.add_diagnostic(field_diag)
sim.add_diagnostic(part_diag)

# Picmi input script
# ==================
if picmi.codename in ['fbpic', 'warpx']:
    # `sim.step` will run the code, controlling it from Python
    sim.write_input_file(file_name='input_script')
    sim.step(max_steps)
else:
    # `write_inputs` will create an input file that can be used to run
    # with the compiled version.
    sim.set_max_step(max_steps)
    sim.write_input_file(file_name=picmi.codename + '_input_script')
