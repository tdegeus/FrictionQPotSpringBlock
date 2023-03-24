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
chunk = prrng.pcg32_tensor_cumsum_1_1(
    shape=[1500],
    initstate=np.arange(N),
    initseq=np.zeros(N),
    distribution=prrng.distribution.random,
    parameters=[2.0],
    align=prrng.alignment(buffer=5, margin=50, min_margin=25, strict=False),
)
chunk -= 50
xdelta = 1e-3

system = model.System_Cuspy_LongRange(
    m=1.0,
    eta=2.0 * np.sqrt(3.0) / 10.0,
    mu=1.0,
    k_interactions=1.0,
    alpha=1,
    k_frame=1.0 / N,
    dt=0.1,
    chunk=chunk,
)

nstep = 200
ret_u_frame = np.empty([nstep], dtype=float)
ret_f_frame = np.empty([nstep], dtype=float)
ret_S = np.empty([nstep], dtype=int)

pbar = tqdm.tqdm(total=nstep)

for step in range(nstep):
    # Extract output data.
    i_n = np.copy(chunk.index_at_align)

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
    ret_S[step] = np.sum(chunk.index_at_align - i_n)

base = pathlib.Path(__file__)
with h5py.File(base.parent / (base.stem + ".h5")) as file:
    assert np.allclose(ret_u_frame, file["x_frame"][...])
    assert np.allclose(ret_f_frame, file["f_frame"][...])
    assert np.all(ret_S == file["S"][...])

if plot:
    fig, axes = plt.subplots(ncols=2, figsize=(2 * 8, 6))
    axes[0].plot(ret_u_frame, ret_f_frame)
    axes[1].plot(system.u)
    plt.show()