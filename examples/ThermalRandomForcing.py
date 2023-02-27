import os

import FrictionQPotSpringBlock.Line1d as model
import h5py
import numpy as np
import prrng
import tqdm

try:
    import matplotlib.pyplot as plt

    plot = True
except ImportError:
    plot = False

N = 1000
initstate = np.arange(N)
initseq = np.zeros(N)
chunk = prrng.pcg32_tensor_cumsum_1_1(
    shape=[1500],
    initstate=initstate,
    initseq=initseq,
    distribution=prrng.distribution.random,
    parameters=[2.0],
    align=prrng.alignment(buffer=5, margin=50, min_margin=25, strict=False),
)
chunk -= 50

# generate sequence of random forces and define start and end increment at which they are applied
# (fixed internal "dinc" that is randomly shifted per particle by maximum "dinc")

f = prrng.pcg32_array(initstate, initseq).normal([11000], mu=0.0, sigma=0.05)

delta_inc = 100
offset = (delta_inc * prrng.pcg32_array(initstate, initseq).random([1])).astype(int)
start_inc = delta_inc * np.tile(np.arange(f.shape[1] + 1), (N, 1))
start_inc += offset
start_inc[:, 0] = 0

# define system
# initialise by minimising energy athermally

system = model.SystemThermalRandomForcing(
    m=1.0,
    eta=2.0 * np.sqrt(3.0) / 10.0,
    mu=1.0,
    k_neighbours=1.0,
    k_frame=1.0 / N,
    dt=0.1,
    chunk=chunk,
)

system.minimise()
system.inc = 0
system.setRandomForceSequence(f=f, start_inc=start_inc)

# apply load at small finite rate "delta_gamma", write output every "dinc" increments

nout = 500
dinc = 1000
delta_gamma = 5e-2
ret_x_frame = np.empty([nout], dtype=float)
ret_f_frame = np.empty([nout], dtype=float)
ret_t_insta = np.empty([nout], dtype=float)

for iout in tqdm.tqdm(range(nout)):

    system.flowSteps(dinc, delta_gamma)

    ret_x_frame[iout] = system.x_frame
    ret_f_frame[iout] = np.mean(system.f_frame)
    ret_t_insta[iout] = system.temperature()

with h5py.File(os.path.join(os.path.dirname(__file__), "ThermalRandomForcing.h5")) as file:
    assert np.allclose(ret_x_frame, file["x_frame"][...])
    assert np.allclose(ret_f_frame, file["f_frame"][...])
    assert np.allclose(ret_t_insta, file["t_insta"][...] * 0.5) # correcting definition in v0.22.0

if plot:
    fig, ax = plt.subplots()
    ax.plot(ret_x_frame, ret_f_frame)
    plt.show()
