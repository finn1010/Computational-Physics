#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 17:01:03 2022

@author: finn
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim  # for animating the graphs
from tqdm import tqdm

# define constants
G = 6.6743015E-11
M_E = 5.9722E24  # mass of Earth
M_M = 7.342E22  # mass of the Moon
m = 1.5E6  # roughly the mass of Falcon Heavy
r_E = 6.3781E6  # radius of Earth
r_M = 1.7374E6  # radius of Moon
r = 384.403E6  # distance between Earth & Moon


def gravitational_comp(x: float, y: float, _dir: str, fn: str):
    """
    Evaluate f3 (dv_x/dt) and f4 (dv_y/dt).
    Parameters
    ----------
        x : float
            x-component of position.
        y : float
            y-component of position.
        _dir : str
            Direction to evaluate in.
        fn : str
            Specifies which equation of motion to use (earth or moon).
    Returns
    -------
        acceleration : float
            Acceleration of the body at the current time.
    """
    if fn == 'earth':
        if _dir == 'x':
            return -G*M_E*x/((np.square(x) + np.square(y))**(3/2))
        elif _dir == 'y':
            return -G*M_E*y/((np.square(x) + np.square(y))**(3/2))

    # use modified equation that takes into account separation of the Moon and
    # Earth (hence why we use `y-r` instead of just `y`)
    elif fn == 'moon':
        if _dir == 'x':
            return (-G*M_E*x/(np.square(x) + np.square(y))**(3/2)) - (G*M_M*x/((np.square(x) + np.square(y-r))**(3/2)))
        elif _dir == 'y':
            return (-G*M_E*y/(np.square(x) + np.square(y))**(3/2)) - (G*M_M*(y-r)/((np.square(x) + np.square(y-r))**(3/2)))


def calculate_energies(x: float, y: float, v_x: float, v_y: float, fn: str = None):
    """
    Calculate gravitational and kinetic components of the energy of the rocket.
    Parameters
    ----------
        x : float
            x-component of position.
        y : float
            y-comnponent of position.
        v_x : float
            Tangential component of velocity.
        v_y : float
            Radial component of velocity.
        fn : str
            Define whether an extra term is needed for GPE (used in the moon
            slingshot).[Optional]
    Returns
    -------
        E_k : float
            Kinetic component of total energy.
        E_g : float
            Gravitation component of total energy.
        E_t : float
            Total energy of the rocket.
    """
    E_k = (1/2) * m * (np.square(v_x) + np.square(v_y))
    if fn == 'moon':
        E_g = (-G*M_E*m)/(np.sqrt(np.square(x) + np.square(y))) + \
            (-G*M_M*m)/(np.sqrt(np.square(x) + np.square(y)))
    else:
        E_g = (-G*M_E*m)/(np.sqrt(np.square(x) + np.square(y)))
    E_t = E_k + E_g

    return E_k, E_g, E_t


def stepping_equations(x, y, v_x, v_y, t, delta_t: float, k1, k2, k3, k4):
    """
    Calculate x, y, v_x & v_y for each time-step.
    Parameters
    ----------
        x: array_like
            An array of values for the x-component of position
        y: array_like
            An array of values for the y-component of position
        v_x: array_like
            An array of values for the x-component of velocity
        v_y: array_like
            An array of values for the y-component of velocity
        delta_t: float
            The timestep to iterate over.
        n: int
            The number of data points to iterate over.
        k1: array_like
            An array of values for k1
        k2: array_like
            An array of values for k2
        k3: array_like
            An array of values for k3
        k4: array_like
            An array of values for k4
    Returns
    -------
        x: array_like
            An array of values for the x-component of position
        y: array_like
            An array of values for the y-component of position
        v_x: array_like
            An array of values for the x-component of velocity
        v_y: array_like
            An array of values for the y-component of velocity
        t: array_like
            An array of time values.
    """
    x += delta_t/6 * \
        (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    y += delta_t/6 * \
        (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    v_x += delta_t/6 * \
        (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    v_y += delta_t/6 * \
        (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
    t += delta_t
    return x, y, v_x, v_y, t


def define_k(x, y, v_x, v_y, delta_t: float, orbit: str = None):
    """
    Create and handle all the values required for k1 through k4.
    Reference: k1[x, y, vx, vy]
    Parameters
    ----------
        x: array_like
            Values for the x-component of position.
        y: array_like
            Values for the y-component of position.
        v_x: array_like
            Values for the x-component of velocity.
        v_y: array_like
            Values for the y-component of velocity.
        delta_t: float
            Timestep to iterate over.
        orbit : str
            Change to using formula for moon slingshot.
    Returns
    -------
        k1: array_like
            Values for k1
        k2: array_like
            Values for k2
        k3: array_like
            Values for k3
        k4: array_like
            Values for k4
    """
    # use equations for dual body orbits
    if orbit == 'moon':
        k1x = v_x
        k1y = v_y
        k1vx = gravitational_comp(x, y, 'x', 'moon')
        k1vy = gravitational_comp(x, y, 'y', 'moon')

        k2x = v_x + delta_t*k1vx/2
        k2y = v_y + delta_t*k1vy/2
        k2vx = gravitational_comp(
            x + delta_t*k1x/2, y + delta_t*k1y/2, 'x', 'moon')
        k2vy = gravitational_comp(
            x + delta_t*k1x/2, y + delta_t*k1y/2, 'y', 'moon')

        k3x = v_x + delta_t*k2vx/2
        k3y = v_y + delta_t*k2vy/2
        k3vx = gravitational_comp(
            x + delta_t*k2x/2, y + delta_t*k2y/2, 'x', 'moon')
        k3vy = gravitational_comp(
            x + delta_t*k2x/2, y + delta_t*k2y/2, 'y', 'moon')

        k4x = v_x + delta_t*k3vx/2
        k4y = v_y + delta_t*k3vy/2
        k4vx = gravitational_comp(
            x + delta_t*k3x/2, y + delta_t*k3y/2, 'x', 'moon')
        k4vy = gravitational_comp(
            x + delta_t*k3x/2, y + delta_t*k3y/2, 'y', 'moon')
    # this should be used for single body orbits
    else:
        k1x = v_x
        k1y = v_y
        k1vx = gravitational_comp(x, y, 'x', 'earth')
        k1vy = gravitational_comp(x, y, 'y', 'earth')

        k2x = v_x + delta_t*k1vx/2
        k2y = v_y + delta_t*k1vy/2
        k2vx = gravitational_comp(
            x + delta_t*k1x/2, y + delta_t*k1y/2, 'x', 'earth')
        k2vy = gravitational_comp(
            x + delta_t*k1x/2, y + delta_t*k1y/2, 'y', 'earth')

        k3x = v_x + delta_t*k2vx/2
        k3y = v_y + delta_t*k2vy/2
        k3vx = gravitational_comp(
            x + delta_t*k2x/2, y + delta_t*k2y/2, 'x', 'earth')
        k3vy = gravitational_comp(
            x + delta_t*k2x/2, y + delta_t*k2y/2, 'y', 'earth')

        k4x = v_x + delta_t*k3vx/2
        k4y = v_y + delta_t*k3vy/2
        k4vx = gravitational_comp(
            x + delta_t*k3x/2, y + delta_t*k3y/2, 'x', 'earth')
        k4vy = gravitational_comp(
            x + delta_t*k3x/2, y + delta_t*k3y/2, 'y', 'earth')

    return k1x, k1y, k1vx, k1vy, k2x, k2y, k2vx, k2vy, k3x, k3y, k3vx, k3vy, k4x, k4y, k4vx, k4vy


def _custom_values():
    """
    Allow the user to specify custom values for the orbits.
    Returns
    -------
        x: float
            Initial horizontal component of position.
        y: float
            Initial vertical component of position.
        v_x: float
            Initial horizontal component velocity.
        v_y: float
            Initial vertical component velocity.
        T: float
            Time period to simulate an orbit for.
        delta_t: float
            The time period step(dt). Defaults to 2.0 s.
    """
    print('Please enter the values you wish to use. Leave blank for '
          'the default. (Scientific notation is accepted e.g. 10E6)')

    # loop each input until valid value
    while True:
        try:
            T = float(input(
                "Please enter a value for T (time period to simulate over) in "
                "seconds: ") or 60000)
            if (T <= 0):
                raise ValueError(
                    'T must be larger than 0. Please try again.')
        # not a float
        except ValueError:
            print('Invalid input. Please enter a float.')
            continue
        else:
            break

    while True:
        try:
            delta_t = float(
                input("Please input a value for \u0394t in seconds: ") or 2)
            if (delta_t <= 0):
                raise ValueError(
                    '\u0394t must be larger than 0. Please try again.')
        except ValueError:
            print('Invalid input. Please enter a float.')
            continue
        else:
            break

    # set n to nearest integer
    n = round(T/delta_t)

    while True:
        try:
            x = float(
                input("Please enter a value for the initial tangential "
                      "position: ") or 7E6)
            y = float(
                input("Please enter a value for the initial radial "
                      "position: ") or 0)
            if (np.sqrt(np.square(x) + np.square(y)) <= r_E) and y < r/2:
                raise ValueError('Oops! That position is smaller than the '
                                 'radius of the Earth. Please enter a larger '
                                 'value for x or y.')
            elif (np.sqrt(np.square(x) + np.square(y-r)) <= r_M) and y > r/2:
                raise ValueError('Oops! That position is smaller than the '
                                 'radius of the Moon. Please enter a different'
                                 ' value for x or y.')
        # not a valid position value
        except ValueError:
            print('Invalid input. Please try again.')
            continue
        else:
            break

    while True:
        try:
            v_x = float(
                input("Please enter a value for the initial value of v_x: ") or 7500)
        except ValueError:
            print('Invalid value entered for v_x. Please try again.')
            continue
        else:
            break

    while True:
        try:
            v_y = float(
                input("Please enter a value for the initial value of v_y: ") or 0)
        except ValueError:
            print('Invalid value entered for v_y. Please try again.')
            continue
        else:
            break

    # create arrays for positional variables
    ax = np.zeros(n)
    ay = np.zeros(n)
    av_x = np.zeros(n)
    av_y = np.zeros(n)

    # assign custom values to 0th index
    ax[0] = x
    ay[0] = y
    av_x[0] = v_x
    av_y[0] = v_y

    return ax, ay, av_x, av_y, delta_t, n, T


def plot_graphs(x, y, t, E_t, E_g, E_k):
    """
    Handle plotting & animating the graphs.
    Parameters
    ----------
        x : array_like
            Position values for the x-component of motion.
        y : array_like
            Position values for the y-component of motion.
        t : array_like
            Array of time values.
        E_t : array_like
            Total energy values.
        E_g : array_like
            Gravitational potential energy values.
        E_k : array_like
            Kinetic energy values.
    """
    fig, ax = plt.subplots()

    # create positional graph
    ax.plot(x, y, color='r', label='Rocket Path')
    ax.set_xlabel('Relative distance in x direction (m)')
    ax.set_ylabel('Relative distance in y direction (m)')
    # create a green dot to represent the Earth
    ax.add_patch(plt.Circle((0, 0), 6.37E6, color='green', label='Earth'))
    # create a grey dot to represent the Moon
    if user_input == 'c':
        ax.add_patch(plt.Circle((0, r), 1.74E6, color='grey', label='Moon'))
    # attempt to set axes to equal sizes
    ax.set_aspect('equal', 'box')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
              ncol=2, fancybox=True, shadow=True)

    rocket, = ax.plot([], [], 'ro')

    def init():
        rocket.set_data([], [])
        return rocket,

    def update(i):
        rocket.set_data(x[i-1], y[i-1])
        return rocket,

    rocket_animation = anim.FuncAnimation(
        fig, update, interval=0.05, blit=True, init_func=init, repeat=True)

    plt.show()

    # create energy graph
    plt.plot(t, E_g, label='Gravitational Energy')
    plt.plot(t, E_k, label='Kinetic Energy')
    plt.plot(t, E_t, label='Total Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.legend(ncol=2, fancybox=True, shadow=True)
    plt.show()


user_input = '0'
while user_input != 'q':
    user_input = input(
        '\nChoose an option:'
        '\na: circular orbit around the Earth,'
        '\nb: elliptical orbit around the Earth,'
        '\nc: rocket slingshot around the Moon,'
        '\n or q to quit.'
    ).lower()

    if user_input == 'a':
        print('Simulating a circular orbit around the Earth using a 4th order '
              'Runge-Kutta method')
        # loop until valid input
        while True:
            try:
                # allows the user to change the paramters used in the
                # calculation
                custom_values = str(input(
                    "Would you like to specify the values used in the "
                    "simulation? (y/n) ")).lower()
                if custom_values == 'y' or custom_values == 'n':
                    break
                else:
                    raise ValueError()
            except ValueError:
                # not a string
                print("Invalid input. Please enter 'y' or 'n'.")
                continue
        if custom_values == 'y':
            x, y, v_x, v_y, delta_t, n, T = _custom_values()
            t = np.zeros(n)
            print('Using custom values.\nT = {}s\n\u0394t = {}s\nx = {}m\ny ='
                  ' {}m\nv_x = {:.2f}m/s\nv_y = {:.2f}m/s'.format(
                      T, delta_t, x[0], y[0], v_x[0], v_y[0]))
        elif custom_values == 'n':
            delta_t = 2
            n = 30000
            # create arrays for positional variables
            x = np.zeros(n)
            y = np.zeros(n)
            v_x = np.zeros(n)
            v_y = np.zeros(n)
            t = np.zeros(n)

            x[0] = 10000000
            y[0] = 0
            v_x[0] = 0
            v_y[0] = np.sqrt(G*M_E/x[0])
            print(
                'Using default values.\nT = 60000s\n\u0394t = 2s\nx = 10E6m\nv'
                ' = {:.2f}m/s'.format(v_y[0]))

        # create arrays for rk4
        k1 = np.zeros(shape=(n, 4))
        k2 = np.zeros(shape=(n, 4))
        k3 = np.zeros(shape=(n, 4))
        k4 = np.zeros(shape=(n, 4))

        # create arrays for energies
        E_t = np.zeros(n)
        E_g = np.zeros(n)
        E_k = np.zeros(n)

        # iterates through the rk4 values and creates a progress bar
        for i in tqdm(range(0, n-1), desc='Simulating...', mininterval=0.25, bar_format='{desc} {percentage:3.0f}%|{bar}| Iteration {n_fmt} of {total_fmt} [Estimated time remaining: {remaining}, {rate_fmt}]'):
            k1[i, 0], k1[i, 1], k1[i, 2], k1[i, 3], k2[i, 0], k2[i, 1], k2[i, 2], k2[i, 3], k3[i, 0], k3[i, 1], k3[i,
                                                                                                                   2], k3[i, 3], k4[i, 0], k4[i, 1], k4[i, 2], k4[i, 3] = define_k(x[i], y[i], v_x[i], v_y[i], delta_t)
            # i have no idea why the linter is breaking the line in two in
            # such a weird way? but it works so POG

            x[i+1], y[i+1], v_x[i+1], v_y[i+1], t[i+1] = stepping_equations(
                x[i], y[i], v_x[i], v_y[i], t[i], delta_t, k1[i, ], k2[i, ], k3[i, ], k4[i, ])

            E_k[i], E_g[i], E_t[i] = calculate_energies(
                x[i], y[i], v_x[i], v_y[i])

            # break fn if we hit the Earth
            if (np.sqrt(np.square(x[i] + np.square(y[i])) <= r_E)):
                break

        plot_graphs(x, y, t, E_t, E_g, E_k)

    elif user_input == 'b':
        print('Simulating an elliptical orbit around the Earth.')
        # loop until valid input
        while True:
            try:
                # allows the user to change the paramters used in the
                # calculation
                custom_values = str(input(
                    "Would you like to specify the values used in the "
                    "simulation? (y/n) ")).lower()
                if custom_values == 'y' or custom_values == 'n':
                    # valid option
                    break
                else:
                    raise ValueError()
            except ValueError:
                # not a string
                print("Invalid input. Please enter 'y' or 'n'.")
                continue
        if custom_values == 'y':
            x, y, v_x, v_y, delta_t, n, T = _custom_values()
            t = np.zeros(n)
            print('Using custom values.\nT={}s\n\u0394t={}s\nx={}m'
                  '\ny={}m\nv_x={: .2f}m/s\nv_y={: .2f}m/s'.format(
                      T, delta_t, x[0], y[0], v_x[0], v_y[0]))

        elif custom_values == 'n':
            delta_t = 2
            n = 30000
            # create arrays for positional variables
            x = np.zeros(n)
            y = np.zeros(n)
            v_x = np.zeros(n)
            v_y = np.zeros(n)
            t = np.zeros(n)

            x[0] = 10000000
            y[0] = 0
            v_x[0] = 0
            v_y[0] = 7500
            print(
                'Using default values.\nT = 60000s\n\u0394t = 2s\nx = 10E6m\nv'
                ' = {:.2f}m/s'.format(v_y[0]))

        # create arrays for rk4
        k1 = np.zeros(shape=(n, 4))
        k2 = np.zeros(shape=(n, 4))
        k3 = np.zeros(shape=(n, 4))
        k4 = np.zeros(shape=(n, 4))

        # create arrays for energies
        E_t = np.zeros(n)
        E_g = np.zeros(n)
        E_k = np.zeros(n)

        # iterates through the rk4 values and creates a progress bar
        for i in tqdm(range(0, n-1), desc='Simulating...', mininterval=0.25, bar_format='{desc} {percentage:3.0f}%|{bar}| Iteration {n_fmt} of {total_fmt} [Estimated time remaining: {remaining}, {rate_fmt}]'):
            k1[i, 0], k1[i, 1], k1[i, 2], k1[i, 3], k2[i, 0], k2[i, 1], k2[i, 2], k2[i, 3], k3[i, 0], k3[i, 1], k3[i,
                                                                                                                   2], k3[i, 3], k4[i, 0], k4[i, 1], k4[i, 2], k4[i, 3] = define_k(x[i], y[i], v_x[i], v_y[i], delta_t)
            # i have no idea why the linter is breaking the line in two in
            # such a weird way? but it works so POG

            x[i+1], y[i+1], v_x[i+1], v_y[i+1], t[i+1] = stepping_equations(
                x[i], y[i], v_x[i], v_y[i], t[i], delta_t, k1[i, ], k2[i, ], k3[i, ], k4[i, ])

            E_k[i], E_g[i], E_t[i] = calculate_energies(
                x[i], y[i], v_x[i], v_y[i])

            # break fn if we hit the Earth
            if (np.sqrt(np.square(x[i]) + np.square(y[i])) <= r_E):
                break

        plot_graphs(x, y, t, E_t, E_g, E_k)

    elif user_input == 'c':
        print('Simulating a slingshot orbit around the moon.')
        # loop until valid input
        while True:
            try:
                # allows the user to change the paramters used in the
                # calculation
                custom_values = str(input(
                    "Would you like to specify the values used in the simulation?"
                    " (y/n) ")).lower()
                if custom_values == 'y' or custom_values == 'n':
                    break
                else:
                    raise ValueError()
            except ValueError:
                # not a string
                print("Invalid input. Please enter 'y' or 'n'.")
                continue
        if custom_values == 'y':
            x, y, v_x, v_y, delta_t, n, T = _custom_values()
            t = np.zeros(n)
            print('Using custom values.\nT = {}s\n\u0394t = {}s\nx = {}m\ny = '
                  '{}m\nv_x = {:.2f}m/s\nv_y = {:.2f}m/s'.format(
                      T, delta_t, x[0], y[0], v_x[0], v_y[0]))

        elif custom_values == 'n':
            delta_t = 2
            n = 500000
            # create arrays for positional variables
            x = np.zeros(n)
            y = np.zeros(n)
            v_x = np.zeros(n)
            v_y = np.zeros(n)
            t = np.zeros(n)

            x[0] = 0
            y[0] = -7E6
            v_x[0] = 10591
            v_y[0] = 0
            print(
                'Using default values.\nT = 1000000s\n\u0394t = 2s\nInitial '
                'Orbit Height = {:.0f}m\nv = {:.2f}m/s'.format(y[0], v_x[0]))

        # create arrays for rk4
        k1 = np.zeros(shape=(n, 4))
        k2 = np.zeros(shape=(n, 4))
        k3 = np.zeros(shape=(n, 4))
        k4 = np.zeros(shape=(n, 4))

        # create arrays for energies
        # E_ge is contrib from Earth, E_gm is contrib from moon
        E_t = np.zeros(n)
        E_g = np.zeros(n)
        E_k = np.zeros(n)

        # iterates through the rk4 values and creates a progress bar
        for i in tqdm(range(0, n-1), desc='Simulating...', mininterval=0.25, bar_format='{desc} {percentage:3.0f}%|{bar}| Iteration {n_fmt} of {total_fmt} [Estimated time remaining: {remaining}, {rate_fmt}]'):
            k1[i, 0], k1[i, 1], k1[i, 2], k1[i, 3], k2[i, 0], k2[i, 1], k2[i, 2], k2[i, 3], k3[i, 0], k3[i, 1], k3[i,
                                                                                                                   2], k3[i, 3], k4[i, 0], k4[i, 1], k4[i, 2], k4[i, 3] = define_k(x[i], y[i], v_x[i], v_y[i], delta_t, 'moon')
            # i have no idea why the linter is breaking the line in two in
            # such a weird way? but it works so POG

            x[i+1], y[i+1], v_x[i+1], v_y[i+1], t[i+1] = stepping_equations(
                x[i], y[i], v_x[i], v_y[i], t[i], delta_t, k1[i, ], k2[i, ], k3[i, ], k4[i, ])

            E_k[i], E_g[i], E_t[i] = calculate_energies(
                x[i], y[i], v_x[i], v_y[i], 'moon')

            # break fn if we hit the Earth or Moon
            if (np.sqrt(np.square(x[i]) + np.square(y[i])) <= r_E) and y[i] < r/2:
                print("\nCrashed into the Earth!")
                break
            elif (abs(np.sqrt(np.square(x[i]) + np.square(y[i] - r))) <= (r_M)) and y[i] > r/2:
                print("\nCrashed into the Moon!")
                break

        plot_graphs(x, y, t, E_t, E_g, E_k)
        
        
        
        
        #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 22:09:27 2021
@author: bertiethorpe
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

np.seterr(divide='ignore', invalid='ignore')

G = 6.67430e-11
M_earth = 5.9742e+24
M_moon = 7.36e+22
moon_average_orbit = 3.844e+8
m_starship = 1.4e+6
r_earth = 6.3781e+6
r_moon = 1.7374e+6

h=100
T=2000000
N = 2000

pos_x = np.zeros(N)
pos_y = np.zeros(N)
vel_x = np.zeros(N)
vel_y = np.zeros(N)


pos_I = [r_earth + 6e+7, 0]
vel_I = [0, 10000]

x0 = r_earth + 6e+7
y0 = 0
vx0 = 0
vy0 = 1000

def GravityX(pos_x, pos_y):

    return ((-G * M_earth * pos_x/((pos_x**2) + (pos_y**2))**(3/2)) + (-G * M_moon * pos_x/((pos_x**2) + ((pos_y - moon_average_orbit)**2))**(3/2)))

def GravityY(pos_x, pos_y):

    return ((-G * M_earth * pos_y/((pos_x**2) + (pos_y**2))**(3/2)) + (-G * M_moon * (pos_y - moon_average_orbit)/((pos_x**2) + ((pos_y - moon_average_orbit)**2))**(3/2)))

def RK4(pos_x, pos_y, vel_x, vel_y):

    for i in np.arange(N):

        k1_x = vel_x
        k1_y = vel_y
        k1_vx = GravityX(pos_x, pos_y)
        k1_vy = GravityY(pos_x, pos_y)

        k2_x = vel_x + (h * k1_vx)/2
        k2_y = vel_y + (h * k1_vy)/2
        k2_vx = GravityX(pos_x + (h * k1_x)/2, pos_y + (h * k1_y)/2)
        k2_vy = GravityY(pos_x + (h * k1_x)/2, pos_y + (h * k1_y)/2)

        k3_x = vel_x + (h * k2_vx)/2
        k3_y = vel_y + (h * k2_vy)/2
        k3_vx = GravityX(pos_x + (h * k2_x)/2, pos_y + (h * k2_y)/2)
        k3_vy = GravityY(pos_x + (h * k2_x)/2, pos_y + (h * k2_y)/2)

        k4_x = vel_x + (h * k3_vx)
        k4_y = vel_y + (h * k3_vy)
        k4_vx = GravityX(pos_x + (h * k3_x), pos_y + (h * k3_y))
        k4_vy = GravityY(pos_x + (h * k3_x), pos_y + (h * k3_y))

        pos_x = pos_x + h/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        pos_y = pos_y + h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        vel_x = vel_x + h/6 * (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx)
        vel_y = vel_y + h/6 * (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy)

    return pos_x, pos_y, vel_x, vel_y


pos_x, pos_y, vel_x, vel_y = RK4(pos_x, pos_y, vel_x, vel_y) 

fig, ax = plt.subplots()
ax.plot(pos_x, pos_y)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class FreeReturnTrajectory:
    
    def __init__(self, orbital_mass, primary_mass, secondary_mass, secondary_mass_orbit, initial_position, initial_velocity, totaltime, step):
        
        self.m = orbital_mass 
        self.M1 = primary_mass
        self.M2 = secondary_mass
        self.M2_orbit = secondary_mass_orbit
        self.pos_I = initial_position
        self.vel_I = initial_velocity
        self.h = step 
        self.G = 6.67408e-11 
        self.T = totaltime 
    
    def gravityX(self, pos_x, pos_y):
        """function that returns the calculated x-component of the orbiting mass's velocity affected by the central mass,
        using Newtons law of gravitation."""
        
        return ((-self.G * self.M1 * pos_x/((pos_x)**2 + (pos_y)**2)**(3/2)) - (self.G * self.M2 * pos_x/((pos_x**2) + ((pos_y - self.M2_orbit)**2))**(3/2)))
    
    def gravityY(self, pos_x, pos_y):
        """function that returns the calculated y-component of the orbiting mass's velocity affected by the central mass,
        using Newtons law of gravitation."""
        
        return ((-self.G * self.M1 * pos_y/((pos_x**2) + (pos_y**2))**(3/2)) - (self.G * self.M2 * (pos_y - self.M2_orbit)/((pos_x**2) + ((pos_y - self.M2_orbit)**2))**(3/2)))
    
    def energy(self, pos_x, pos_y, vel_x, vel_y):
        """function that returns the total, potential, and kinetic energy of the orbital with time."""
        
        E_k = (1/2) * self.m * ((vel_x)**2 + (vel_y)**2)
        
        E_g = (-self.G * self.M1 * self.m)/(np.sqrt((pos_x)**2 + (pos_y)**2)) + (-self.G * self.M2 * self.m)/(np.sqrt((pos_x)**2 + (pos_y)**2))
        
        E_t = E_k + E_g
        
        return E_k, E_g, E_t
    
    def k_pos_vel(self, pos_x, pos_y, vel_x, vel_y):
        """function that returns multiple arrays of Runge-Kutta 4th Order values for position and velocity of the orbiting mass
        in x and y components. These arrays are later used with the thping functions for velocity and position to determine
        how the orbiting mass moves around the central mass."""
        
        k1_x = vel_x
        k1_y = vel_y
        k1_vx = self.gravityX(pos_x, pos_y)
        k1_vy = self.gravityY(pos_x, pos_y)
        
        k2_x = vel_x + (self.h * k1_vx) / 2
        k2_y = vel_y + (self.h * k1_vy) / 2
        k2_vx = self.gravityX(pos_x + (self.h * k1_x) / 2, pos_y + (self.h * k1_y) / 2)
        k2_vy = self.gravityY(pos_x + (self.h * k1_x) / 2, pos_y + (self.h * k1_y) / 2)
      
        k3_x = vel_x + (self.h * k2_vx) / 2
        k3_y = vel_y + (self.h * k2_vy) / 2
        k3_vx = self.gravityX(pos_x + (self.h * k2_x) / 2, pos_y + (self.h * k2_y) / 2)
        k3_vy = self.gravityY(pos_x + (self.h * k2_x) / 2, pos_y + (self.h * k2_y) / 2) 
       
        k4_x = vel_x + self.h * k3_vx
        k4_y = vel_y + self.h * k3_vy
        k4_vx = self.gravityX(pos_x + self.h * k3_x, pos_y + self.h * k3_y)
        k4_vy = self.gravityY(pos_x + self.h * k3_x, pos_y + self.h * k3_y)
        
        return [k1_x, k2_x, k3_x, k4_x], [k1_y, k2_y, k3_y, k4_y], [k1_vx, k2_vx, k3_vx, k4_vx], [k1_vy, k2_vy, k3_vy, k4_vy]
    
    def thping_position(self, pos_x, pos_y, kpos_xlist, kpos_ylist):
        """function that returns the x and y component of the orbiting mass's position for the next time h."""
        
        pos_x +=  (self.h / 6) * (kpos_xlist[0] + 2 * kpos_xlist[1] + 2 * kpos_xlist[2] + kpos_xlist[3])
        pos_y += (self.h / 6) * (kpos_ylist[0] + 2 * kpos_ylist[1] + 2 * kpos_ylist[2] + kpos_ylist[3])
        
        return pos_x, pos_y
    
    def thping_velocity(self, vel_x, vel_y, kvel_xlist, kvel_ylist):
        """function that returns the x and y component of the orbiting mass's velocity for the next time h."""
        
        vel_x += (self.h / 6) * (kvel_xlist[0] + 2 * kvel_xlist[1] + 2 * kvel_xlist[2] + kvel_xlist[3])
        vel_y += (self.h / 6) * (kvel_ylist[0] + 2 * kvel_ylist[1] + 2 * kvel_ylist[2] + kvel_ylist[3])
       
        return vel_x, vel_y
    
    def time(self, t0 = 0):
        """function that returns an array of times between the initial time and final time with selected time h."""
        
        t = np.linspace(t0, self.T, int(self.T / self.h) +1 )
        
        return t
    
    def simulate_orbit(self):
        """function that returns the x and y components of the orbiting mass's position and velocity throughout the entire simulation
        as arrays used to animate the simulation or used in matplotlib.pyplot to plot a graph of the mass's orbit."""
        
        pos_x, pos_y = self.pos_I[0], self.pos_I[1]
        vel_x, vel_y = self.vel_I[0], self.vel_I[1]
        time_list = self.time()
        rx, ry = np.zeros(len(time_list)), np.zeros(len(time_list))
        rdotx, rdoty = np.zeros(len(time_list)), np.zeros(len(time_list))
        
        for i in np.arange(0, len(time_list)):
            kpos_xlist, kpos_ylist, kvel_xlist, kvel_ylist = self.k_pos_vel(pos_x, pos_y, vel_x, vel_y)
            pos_x, pos_y = self.thping_position(pos_x, pos_y, kpos_xlist, kpos_ylist)
            vel_x, vel_y = self.thping_velocity(vel_x, vel_y, kvel_xlist, kvel_ylist)
            rx[i], ry[i], rdotx[i], rdoty[i] = pos_x, pos_y, vel_x, vel_y
        
        return rx, ry, rdotx, rdoty

day_length = 24 * 60 * 60  
earth_radius = 6.3781e+6
moon_radius = 1.7371e+6
initial_orbit = 5e5 + earth_radius 
earth_mass = 5.972e24 
moon_mass = 7.342e+22
moon_orbit = 3.844e+8
starship_mass = 768 # SpaceX second stage mass
initial_speed = 10.656e3  
multiplier = 20 
        
starship = FreeReturnTrajectory(starship_mass, earth_mass, moon_mass, moon_orbit, [0, -initial_orbit], [initial_speed, 0], (day_length * multiplier), (10))
        
rx, ry, rdotx, rdoty = starship.simulate_orbit()
t = starship.time()
E_k, E_g, E_t = starship.energy(rx, ry, rdotx, rdoty)
        
fig = plt.figure(figsize=(10,7))
gs = fig.add_gridspec(21,2)
        
ax1 = fig.add_subplot(gs[:,1])
p1, = ax1.plot(rx, ry, '--y', linewidth=0.5)
ax1.set_aspect('equal')
ax1.xaxis.set_ticklabels([])
ax1.yaxis.set_ticklabels([])
ax1.set_facecolor('k')
ax1.set_xlim([-2e8,2e8])
ax1.set_ylim([-1e8,5e8])
ax1.add_patch(plt.Circle((0, 0), earth_radius, color='blue'))
ax1.add_patch(plt.Circle((0, moon_orbit), moon_radius, color='0.7'))
        
ax2 = fig.add_subplot(gs[7:19,0])
ax2.set_facecolor('lightgoldenrodyellow')
p2, = ax2.plot(t, E_g, label='Gravitational Energy')
p3, = ax2.plot(t, E_k, label='Kinetic Energy')
p4, = ax2.plot(t, E_t, label='Total Energy')
ax2.set_xlabel('Time (s)', fontsize=9)
ax2.set_ylabel('Energy (J)', fontsize=9)
ax2.legend(ncol=1, fancybox=True, fontsize=6)
        
velocity_slide = plt.axes([0.14,0.81,0.3,0.03], facecolor='k')
orbit_slide = plt.axes([0.14,0.76,0.3,0.03], facecolor='k')
h_slide = plt.axes([0.14,0.71,0.3,0.03], facecolor='k')
        
velocity_factor = Slider(velocity_slide,'Initial Velocity (m/s)',valmin=96,valmax=12e3,valinit=initial_speed,valstep=96,color='y')
velocity_factor.label.set_size(7)
orbit_factor = Slider(orbit_slide,'Initial Orbit (m)',valmin=earth_radius,valmax=8e+6,valinit=initial_orbit,valstep=1000,color='y')
orbit_factor.label.set_size(7)
h_factor = Slider(h_slide,'Time Step (s)',valmin=10,valmax=1000,valinit=100,valstep=10,color='y')
h_factor.label.set_size(7)


    
plt.show()
    
