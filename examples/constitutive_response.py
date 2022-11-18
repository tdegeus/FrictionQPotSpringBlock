import FrictionQPotSpringBlock.Line1d as model
import matplotlib.pyplot as plt
import numpy as np
import prrng

N = 1
chunk = prrng.pcg32_tensor_cumsum_1_1(
    shape=[100],
    initstate=np.arange(N),
    initseq=np.zeros(N),
    distribution=prrng.distribution.random,
    parameters=[2.0],
    align=prrng.alignment(buffer=5, margin=50, min_margin=25, strict=False),
)
chunk -= 5

fig, ax = plt.subplots()

# Cusp

system = model.System(
    m=1,
    eta=1,
    mu=1,
    k_neighbours=1,
    k_frame=1,
    dt=1,
    chunk=chunk,
)

x = np.linspace(0, 10, 1000)
f = np.zeros(x.size)

for i in range(x.size):
    system.x = np.array([x[i]])
    f[i] = -system.f_potential[0]

ax.plot(x, f, c="k")

# Smooth

system = model.SystemSmooth(
    m=1,
    eta=1,
    mu=1,
    k_neighbours=1,
    k_frame=1,
    dt=1,
    chunk=chunk,
)

x = np.linspace(0, 10, 1000)
f = np.zeros(x.size)

for i in range(x.size):
    system.x = np.array([x[i]])
    f[i] = -system.f_potential[0]

ax.plot(x, f, c="b")

# SemiSmooth

system = model.SystemSemiSmooth(
    m=1,
    eta=1,
    mu=1,
    kappa=1,
    k_neighbours=1,
    k_frame=1,
    dt=1,
    chunk=chunk,
)

x = np.linspace(0, 10, 1000)
f = np.zeros(x.size)

for i in range(x.size):
    system.x = np.array([x[i]])
    f[i] = -system.f_potential[0]

ax.plot(x, f, c="r")

# annotations

x = np.linspace(0, 10, 1000)
ax.plot(x, np.zeros_like(x), c="k", ls="--")

plt.show()
