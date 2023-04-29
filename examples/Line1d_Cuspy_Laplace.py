import pathlib

import FrictionQPotSpringBlock.Line1d as model
import h5py
import numpy as np
import tqdm

try:
    import matplotlib.pyplot as plt

    plot = True
except ImportError:
    plot = False

N = 1000
xdelta = 1e-3

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

nstep = 1000
ret_u_frame = np.empty([nstep], dtype=float)
ret_f_frame = np.empty([nstep], dtype=float)
ret_S = np.empty([nstep], dtype=int)

pbar = tqdm.tqdm(total=nstep)

for step in range(nstep):
    # Extract output data.
    i_n = np.copy(system.chunk.index_at_align)

    # Apply event-driven protocol.
    if step == 0:
        system.u_frame = 0.0  # initial quench
    else:
        system.eventDrivenStep(xdelta, step % 2 == 0)  # normal event driven step

    # Minimise energy.
    if step % 2 == 0:
        inc_n = system.inc
        ret = system.minimise()
        assert ret == 0
        pbar.n = step
        pbar.set_description(f"step = {step:4d}, niter = {system.inc - inc_n:8d}")
        pbar.refresh()

    # Extract output data.
    ret_u_frame[step] = system.u_frame
    ret_f_frame[step] = np.mean(system.f_frame)
    ret_S[step] = np.sum(system.chunk.index_at_align - i_n)

base = pathlib.Path(__file__)
with h5py.File(base.parent / (base.stem + ".h5")) as file:
    print(ret_S - file["S"][...])
    assert np.all(ret_S == file["S"][...])
    assert np.allclose(ret_u_frame, file["x_frame"][...])
    assert np.allclose(ret_f_frame, file["f_frame"][...])

if plot:
    fig, ax = plt.subplots()
    ax.plot(ret_u_frame, ret_f_frame)
    plt.show()
