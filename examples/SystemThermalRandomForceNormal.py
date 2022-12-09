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

# prepare by minimising athermal

system = model.System(
    m=1.0,
    eta=2.0 * np.sqrt(3.0) / 10.0,
    mu=1.0,
    k_neighbours=1.0,
    k_frame=1.0 / N,
    dt=0.1,
    chunk=chunk,
)

system.minimise()
x = np.copy(system.x)

# define thermal system

system = model.SystemThermalRandomForceNormal(
    m=1.0,
    eta=2.0 * np.sqrt(3.0) / 10.0,
    mu=1.0,
    k_neighbours=1.0,
    k_frame=1.0 / N,
    dt=0.1,
    chunk=chunk,
    mean=0.0,
    stddev=0.05,
    seed=0,
    dinc_init=prrng.pcg32(0).randint([N], 100),
    dinc=100 * np.ones(N, dtype=int),
)
system.x = x

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

root = os.path.dirname(__file__)
with h5py.File(os.path.join(root, "SystemThermalRandomForceNormal.h5"), "w") as file:
    file["x_frame"] = ret_x_frame
    file["f_frame"] = ret_f_frame
    file["t_insta"] = ret_t_insta

root = os.path.dirname(__file__)
with h5py.File(os.path.join(root, "SystemThermalRandomForceNormal.h5")) as file:
    assert np.allclose(ret_x_frame, file["x_frame"][...])
    assert np.allclose(ret_f_frame, file["f_frame"][...])
    assert np.allclose(ret_t_insta, file["t_insta"][...])

if plot:
    fig, ax = plt.subplots()
    ax.plot(ret_x_frame, ret_f_frame)
    plt.show()
