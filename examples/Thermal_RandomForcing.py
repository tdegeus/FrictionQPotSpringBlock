import FrictionQPotSpringBlock.Line1d as model
import matplotlib.pyplot as plt
import numpy as np
import prrng
import tqdm

N = 1000

initstate = np.arange(N)
initseq = np.zeros(N)
generators = prrng.pcg32_array(initstate, initseq)

y = 2.0 * generators.random([20000])
y = np.cumsum(y, 1)
y -= 50.0

force_generators = prrng.pcg32_array(initstate, initseq)
f = force_generators.normal([11000], mu=0.0, sigma=0.05)

dinc = 100
inc_generators = prrng.pcg32_array(initstate, initseq)
start_inc = dinc * np.tile(np.arange(f.shape[1] + 1), (N, 1))
start_inc += np.floor(dinc * inc_generators.random([1])).astype(start_inc.dtype)
start_inc[:, 0] = 0

system = model.SystemThermalRandomForcing(
    m=1.0,
    eta=2.0 * np.sqrt(3.0) / 10.0,
    mu=1.0,
    k_neighbours=1.0,
    k_frame=1.0 / N,
    dt=0.1,
    x_yield=y,
)

system.minimise()
system.set_inc(0)

system.setRandomForceSequence(f=f, start_inc=start_inc)

ninc = 1000
ret_x_frame = np.empty([ninc], dtype=float)
ret_f_frame = np.empty([ninc], dtype=float)

for inc in tqdm.tqdm(range(ninc)):

    system.flowSteps(1000, 1e-2)

    ret_x_frame[inc] = system.x_frame()
    ret_f_frame[inc] = np.mean(system.f_frame())

fig, ax = plt.subplots()
ax.plot(ret_x_frame, ret_f_frame)
plt.show()
