import pathlib

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

# prepare by minimising athermal

system = model.System_Cuspy_Laplace(
    m=1.0,
    eta=2.0 * np.sqrt(3.0) / 10.0,
    mu=1.0,
    k_interactions=1.0,
    k_frame=1.0 / N,
    dt=0.1,
    shape=[N],
    seed=0,
    distribution="random",
    parameters=[2.0],
    offset=-50,
)

system.minimise()
u = np.copy(system.u)

# define thermal system

system = model.System_Cuspy_Laplace_RandomForcing(
    m=1.0,
    eta=2.0 * np.sqrt(3.0) / 10.0,
    mu=1.0,
    k_interactions=1.0,
    k_frame=1.0 / N,
    dt=0.1,
    mean=0.0,
    stddev=0.05,
    seed_forcing=0,
    dinc_init=prrng.pcg32(0).randint([N], 100),
    dinc=100 * np.ones(N, dtype=int),
    shape=[N],
    seed=0,
    distribution="random",
    parameters=[2.0],
    offset=-50,
)
system.u = u

# apply load at small finite rate "delta_gamma", write output every "dinc" increments

nout = 500
dinc = 1000
delta_gamma = 5e-2
ret_u_frame = np.empty([nout], dtype=float)
ret_f_frame = np.empty([nout], dtype=float)
ret_t_insta = np.empty([nout], dtype=float)

for iout in tqdm.tqdm(range(nout)):
    system.flowSteps(dinc, delta_gamma)

    ret_u_frame[iout] = system.u_frame
    ret_f_frame[iout] = np.mean(system.f_frame)
    ret_t_insta[iout] = system.temperature

base = pathlib.Path(__file__)
with h5py.File(base.parent / (base.stem + ".h5")) as file:
    assert np.allclose(ret_u_frame, file["x_frame"][...])
    assert np.allclose(ret_f_frame, file["f_frame"][...])
    assert np.allclose(ret_t_insta, file["t_insta"][...])

if plot:
    fig, ax = plt.subplots()
    ax.plot(ret_u_frame, ret_f_frame)
    plt.show()
