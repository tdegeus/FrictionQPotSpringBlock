import os

import FrictionQPotSpringBlock.Line1d as model
import h5py
import numpy as np
import prrng
import QPot
import tqdm

try:
    import matplotlib.pyplot as plt

    plot = True
except ImportError:
    plot = False

N = 1000

initstate = np.arange(N)
initseq = np.zeros(N)
generators = prrng.pcg32_array(initstate, initseq)

nchunk = 1500
nbuffer = 200
state = generators.state()
istart = np.zeros(N, dtype=np.int64)
istate = np.zeros(N, dtype=np.int64)

y = 2.0 * generators.random([nchunk])
y = np.cumsum(y, 1)
y -= 50.0

istate += y.shape[1]
state = generators.state()

xdelta = 1e-3

system = model.System(
    m=1.0,
    eta=2.0 * np.sqrt(3.0) / 10.0,
    mu=1.0,
    k_neighbours=1.0,
    k_frame=1.0 / N,
    dt=0.1,
    x_yield=y,
)

ninc = 1000
ret_x_frame = np.empty([ninc], dtype=float)
ret_f_frame = np.empty([ninc], dtype=float)
ret_S = np.empty([ninc], dtype=int)

pbar = tqdm.tqdm(total=ninc)

for inc in range(ninc):

    # Update chunk.
    if np.any(system.i > nchunk - nbuffer):
        shift = system.i - nbuffer + 1
        advance = np.where(shift < 0, shift + istart - istate, nchunk + istart - istate)
        generators.advance(advance)
        istate += advance
        n = np.max(np.abs(shift)) + 1
        dy = 2.0 * generators.random([n])
        state = generators.state()
        istart += shift
        istate += n
        system.y = QPot.cumsum_chunk(system.y, dy, shift)

    # Extract output data.
    i_n = np.copy(system.i)

    # Apply event-driven protocol.
    if inc == 0:
        system.x_frame = 0.0  # initial quench
    else:
        system.eventDrivenStep(xdelta, inc % 2 == 0)  # normal event driven step

    # Minimise energy.
    if inc % 2 == 0:
        niter = system.minimise(nmargin=5)
        assert niter > 0
        pbar.n = inc
        pbar.set_description(f"inc = {inc:4d}, niter = {niter:8d}")
        pbar.refresh()

    # Extract output data.
    ret_x_frame[inc] = system.x_frame
    ret_f_frame[inc] = np.mean(system.f_frame)
    ret_S[inc] = np.sum(system.i - i_n)

with h5py.File(os.path.join(os.path.dirname(__file__), "QuasiStatic.h5")) as file:
    assert np.allclose(ret_x_frame, file["x_frame"][...])
    assert np.allclose(ret_f_frame, file["f_frame"][...])
    assert np.all(ret_S == file["S"][...])

if plot:
    fig, ax = plt.subplots()
    ax.plot(ret_x_frame, ret_f_frame)
    plt.show()
