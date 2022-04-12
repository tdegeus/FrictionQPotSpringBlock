import FrictionQPotSpringBlock.Line1d as model
import matplotlib.pyplot as plt
import numpy as np
import prrng
import tqdm

N = 1000

initstate = np.arange(N)
initseq = np.zeros(N)

# generate yield positions

generators = prrng.pcg32_array(initstate, initseq)

y = 2.0 * generators.random([20000])
y = np.cumsum(y, 1)
y -= 50.0

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
    x_yield=y,
)

system.minimise()
system.set_inc(0)
system.setRandomForceSequence(f=f, start_inc=start_inc)

# apply load at small finite rate "delta_gamma", write output every "dinc" increments

nout = 1000
dinc = 1000
delta_gamma = 1e-2
ret_x_frame = np.empty([nout], dtype=float)
ret_f_frame = np.empty([nout], dtype=float)

for iout in tqdm.tqdm(range(nout)):

    system.flowSteps(dinc, delta_gamma)

    ret_x_frame[iout] = system.x_frame()
    ret_f_frame[iout] = np.mean(system.f_frame())

# plot output

fig, ax = plt.subplots()
ax.plot(ret_x_frame, ret_f_frame)
plt.show()
