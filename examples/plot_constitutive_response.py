import FrictionQPotSpringBlock.Line1d as model
import matplotlib.pyplot as plt
import numpy as np
import prrng

generator = prrng.pcg32()
y = np.cumsum(generator.random([15])) - 5

fig, ax = plt.subplots()

# Cusp

system = model.System(
    m=1,
    eta=1,
    mu=1,
    k_neighbours=1,
    k_frame=1,
    dt=1,
    x_yield=y.reshape(1, -1),
)

x = np.linspace(y[0], y[-1], 1000)
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
    x_yield=y.reshape(1, -1),
)

x = np.linspace(y[0], y[-1], 1000)
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
    x_yield=y.reshape(1, -1),
)

x = np.linspace(y[0], y[-1], 1000)
f = np.zeros(x.size)

for i in range(x.size):
    system.x = np.array([x[i]])
    f[i] = -system.f_potential[0]

ax.plot(x, f, c="r")

# annotations

x = np.linspace(y[0], y[-1], 2)
ax.plot(x, np.zeros_like(x), c="k", ls="--")

plt.show()
