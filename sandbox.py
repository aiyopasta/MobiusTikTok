import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def C(t_):
    t_ = 2 * np.pi * np.asarray(t_)
    u = t_ / (2 * np.pi)
    h = 100 * (0.5 * u ** 2 - 1.5 * u ** 3 + 1.5 * u ** 4 - 0.5 * u ** 5)
    h += 100 * (0.5 * u ** 3 - u ** 4 + 0.5 * u ** 5)
    h1 = 0.32 * h
    return np.array([np.cbrt(np.sin(t_)), h1])


def get_parameterization(xscale, yscale, N=1000):
    t_values = np.linspace(0, 1, N + 1)  # t in [0, 1]
    segment_lengths = np.sqrt(np.sum(np.diff(C(t_values), axis=1) ** 2, axis=0))
    S = np.sum(segment_lengths)
    normalized_cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths))) / S

    x_interp = interp1d(normalized_cumulative_lengths, C(t_values)[0, :], kind='linear')
    y_interp = interp1d(normalized_cumulative_lengths, C(t_values)[1, :], kind='linear')

    def get_point(s):
        # Ensure s is in [0, 1]
        s = np.clip(s, 0, 1)
        return np.array([xscale * x_interp(s), yscale * y_interp(s)])

    return get_point


# Example usage
get_point = get_parameterization(837.5, 628.125, N=200)

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))
s_values = np.linspace(0, 1, 50)  # s in [0, 1]
points = np.array([get_point(s) for s in s_values]).T
ax.plot(points[0, :], points[1, :], 'bo-')
ax.set_aspect('equal', 'box')
plt.title("Arclength Parametrization of Curve C(t)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
