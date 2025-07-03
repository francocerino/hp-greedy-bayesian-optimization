import argparse
import numpy as np
import h5py
import gwsurrogate
import time

# Argument parsing
parser = argparse.ArgumentParser(description="Generate Gravitational Waves")
parser.add_argument("-d", "--dimension", type=int, default=1, choices=[1, 2, 3],
                    help="Dimension of parameter space")
parser.add_argument("-q", "--massratio", type=int, default=100,
                    help="Number of q (mass ratio) samples")
parser.add_argument("-x", "--spinz", type=int, default=10,
                    help="Number of spin z samples")
args = parser.parse_args()

# Load surrogate model
sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')

# Time and frequency settings
dt = 0.1
times = np.linspace(-2750, 100, 28501)
f_low = 5e-3

# Sampling grid
qs = np.linspace(1, 8, args.massratio)
chis_z = np.linspace(-0.8, 0.8, args.spinz)

# Generate data
gws = []
parameters = []

print(f"\nGravitational Wave generation started!\n")
start_time = time.time()

if args.dimension == 1:
    chi_z = 0
    chi = [0, 0, chi_z]
    for q in qs:
        t, h, dyn = sur(q, chi, chi, times=times, mode_list=[(2, 2)], f_low=f_low)
        gws.append(h[(2, 2)])
        parameters.append([q, chi_z, chi_z])
    suffix = f"q{args.massratio}"
elif args.dimension == 2:
    for q in qs:
        for chi_z in chis_z:
            chi = [0, 0, chi_z]
            t, h, dyn = sur(q, chi, chi, times=times, mode_list=[(2, 2)], f_low=f_low)
            gws.append(h[(2, 2)])
            parameters.append([q, chi_z, chi_z])
    suffix = f"q{args.massratio}_chi{args.spinz}_total{args.massratio * args.spinz}"
elif args.dimension == 3:
    for q in qs:
        for chi_z1 in chis_z:
            for chi_z2 in chis_z:
                chi1 = [0, 0, chi_z1]
                chi2 = [0, 0, chi_z2]
                t, h, dyn = sur(q, chi1, chi2, times=times, mode_list=[(2, 2)], f_low=f_low)
                gws.append(h[(2, 2)])
                parameters.append([q, chi_z1, chi_z2])
    suffix = f"q{args.massratio}_chi{args.spinz}_total{args.massratio * args.spinz**2}"
else:
    raise ValueError("Dimension must be 1, 2 or 3.")

end_time = time.time()
elapsed_time = end_time - start_time

# Convert to arrays
gws = np.asarray(gws)              # shape: (N_samples, N_timepoints)
parameters = np.asarray(parameters)  # shape: (N_samples, 3)

# Save in HDF5
filename = f"gw_{args.dimension}d_{suffix}.h5"
with h5py.File(filename, 'w') as f:
    f.create_dataset('waveforms/h22', data=gws)
    f.create_dataset('parameters', data=parameters)
    f.create_dataset('times', data=times)

    f.attrs['dimension'] = args.dimension
    f.attrs['massratio_samples'] = args.massratio
    f.attrs['spinz_samples'] = args.spinz
    f.attrs['sampling_rate'] = 1 / dt
    f.attrs['f_low'] = f_low
    f.attrs['description'] = "GW waveforms generated using NRHybSur3dq8"

print(f"\nSaved {gws.shape[0]} waveforms to {filename}")
print(f"Total generation time: {elapsed_time:.2f} seconds")
