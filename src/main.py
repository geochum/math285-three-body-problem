import numpy as np
import math
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as ani

GRAVITATIONAL_CONSTANT = 6.67408e-11

def explicit_euler(F, Y0, h, n):
    Y = []
    Y.append(Y0)
    for k in range(n - 1):
        Y.append(Y[k] + h * F(*Y[k]))
    return Y

def adams_bashforth(F, Y0, h, n):
    Y = []
    Y.append(Y0)
    Y.append(Y0 + h * F(*Y0))
    for k in range(n - 2):
        Y.append(Y[k + 1] + h * ((3 / 2) * F(*Y[k + 1]) - (1 / 2) * F(*Y[k])))
    return Y

def runge_kutta(F, Y0, h, n):
    Y = []
    Y.append(Y0)
    for k in range(n - 1):
        mk = F(*Y[k]) # (Forward) Euler
        nk = F(*(Y[k] + mk * h / 2)) # Midpoint slope
        pk = F(*(Y[k] + nk * h / 2)) # Better midpoint slope
        qk = F(*(Y[k] + pk * h)) # Endpoint slope
        Y.append(Y[k] + (h / 6) * (mk + 2 * nk + 2 * pk + qk))
    return Y

def get_configurations():
    """Returns a list of all configuration dictionaries."""
    configs = [
        {
            'name': 'Ovals with flourishes',
            't_min': 0,
            't_max': 8.094721,
            'h': 0.01,
            'r_1': (0.716248295713, 0.384288553041, 0),
            'r_2': (0.086172594591, 1.342795868577, 0),
            'r_3': (0.538777980808, 0.481049882656, 0),
            'v_1': (1.245268230896, 2.444311951777, 0),
            'v_2': (-0.675224323690, -0.962879613630, 0),
            'v_3': (-0.570043907206, -1.481432338147, 0),
        },
        {
            'name': 'Figure 8',
            't_min': 0,
            't_max': 6.324449,
            'h': 0.01,
            'p1': 0.347111,
            'p2': 0.532728,
            'use_symmetric': True,
        },
        {
            'name': 'Dragonfly',
            't_min': 0,
            't_max': 21.270975,
            'h': 0.0001,
            'p1': 0.080584,
            'p2': 0.588836,
            'use_symmetric': True,
        },
        {
            'name': 'Yin-Yang 1b',
            't_min': 0,
            't_max': 10.962563,
            'h': 0.00001,
            'p1': 0.282699,
            'p2': 0.327209,
            'use_symmetric': True,
        },
        {
            'name': 'Yin-Yang 1a',
            't_min': 0,
            't_max': 17.328370,
            'h': 0.0001,
            'p1': 0.513938,
            'p2': 0.304736,
            'use_symmetric': True,
        },
        {
            'name': 'Yarn',
            't_min': 0,
            't_max': 55.501762,
            'h': 0.00001,
            'p1': 0.559064,
            'p2': 0.349192,
            'use_symmetric': True,
        },
        {
            'name': 'GOGGLES',
            't_min': 0,
            't_max': 10.466818,
            'h': 0.0001,
            'p1': 0.083300,
            'p2': 0.127889,
            'use_symmetric': True,
        },
        {
            'name': 'Skinny pineapple',
            't_min': 0,
            't_max': 5.095054,
            'h': 0.000001,
            'r_1': (0.419698802831, 1.190466261252, 0),
            'r_2': (0.076399621771, 0.296331688995, 0),
            'r_3': (0.100310663856, -0.729358656127, 0),
            'v_1': (0.102294566003, 0.687248445943, 0),
            'v_2': (0.148950262064, 0.240179781043, 0),
            'v_3': (-0.251244828060, -0.927428226977, 0),
        },
    ]
    return configs

def run_simulation(config, method_func, method_name):
    """Run a single simulation with given configuration and method."""
    # Extract configuration
    name = config['name']
    t_min = config['t_min']
    t_max = config['t_max']
    h = config['h']
    
    # Set up initial conditions
    if config.get('use_symmetric', False):
        p1, p2 = config['p1'], config['p2']
        r_1_x, r_1_y, r_1_z = -1, 0, 0
        r_2_x, r_2_y, r_2_z = 1, 0, 0
        r_3_x, r_3_y, r_3_z = 0, 0, 0
        v_1_x, v_1_y, v_1_z = p1, p2, 0
        v_2_x, v_2_y, v_2_z = p1, p2, 0
        v_3_x, v_3_y, v_3_z = -2 * p1, -2 * p2, 0
    else:
        r_1_x, r_1_y, r_1_z = config['r_1']
        r_2_x, r_2_y, r_2_z = config['r_2']
        r_3_x, r_3_y, r_3_z = config['r_3']
        v_1_x, v_1_y, v_1_z = config['v_1']
        v_2_x, v_2_y, v_2_z = config['v_2']
        v_3_x, v_3_y, v_3_z = config['v_3']
    
    # Number of iterations
    n = int((t_max - t_min) / h) + 1
    
    # Constants
    G, m_1, m_2, m_3 = 1, 1, 1, 1
    
    # Initial conditions
    U0 = np.array([r_1_x,v_1_x,r_1_y,v_1_y,r_1_z,v_1_z,r_2_x,v_2_x,r_2_y,v_2_y,r_2_z,v_2_z,r_3_x,v_3_x,r_3_y,v_3_y,r_3_z,v_3_z])
    
    # Velocity Equations
    v_1 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_2 * (u_7 - u_1) / ((u_7 - u_1) ** 2 + (u_9 - u_3) ** 2 + (u_11 - u_5) ** 2) ** (3/2) + \
             m_3 * (u_13 - u_1) / ((u_13 - u_1) ** 2 + (u_15 - u_3) ** 2 + (u_17 - u_5) ** 2) ** (3/2))
    v_2 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_2 * (u_9 - u_3) / ((u_7 - u_1) ** 2 + (u_9 - u_3) ** 2 + (u_11 - u_5) ** 2) ** (3/2) + \
             m_3 * (u_15 - u_3) / ((u_13 - u_1) ** 2 + (u_15 - u_3) ** 2 + (u_17 - u_5) ** 2) ** (3/2))
    v_3 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_2 * (u_11 - u_5) / ((u_7 - u_1) ** 2 + (u_9 - u_3) ** 2 + (u_11 - u_5) ** 2) ** (3/2) + \
             m_3 * (u_17 - u_5) / ((u_13 - u_1) ** 2 + (u_15 - u_3) ** 2 + (u_17 - u_5) ** 2) ** (3/2)) 
    v_4 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_1 * (u_1 - u_7) / ((u_1 - u_7) ** 2 + (u_3 - u_9) ** 2 + (u_5 - u_11) ** 2) ** (3/2) + \
             m_3 * (u_13 - u_7) / ((u_13 - u_7) ** 2 + (u_15 - u_9) ** 2 + (u_17 - u_11) ** 2) ** (3/2))
    v_5 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_1 * (u_3 - u_9) / ((u_1 - u_7) ** 2 + (u_3 - u_9) ** 2 + (u_5 - u_11) ** 2) ** (3/2) + \
             m_3 * (u_15 - u_9) / ((u_13 - u_7) ** 2 + (u_15 - u_9) ** 2 + (u_17 - u_11) ** 2) ** (3/2))
    v_6 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_1 * (u_5 - u_11) / ((u_1 - u_7) ** 2 + (u_3 - u_9) ** 2 + (u_5 - u_11) ** 2) ** (3/2) + \
             m_3 * (u_17 - u_11) / ((u_13 - u_7) ** 2 + (u_15 - u_9) ** 2 + (u_17 - u_11) ** 2) ** (3/2))
    v_7 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_1 * (u_1 - u_13) / ((u_1 - u_13) ** 2 + (u_3 - u_15) ** 2 + (u_5 - u_17) ** 2) ** (3/2) + \
             m_2 * (u_7 - u_13) / ((u_7 - u_13) ** 2 + (u_9 - u_15) ** 2 + (u_11 - u_17) ** 2) ** (3/2))
    v_8 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_1 * (u_3 - u_15) / ((u_1 - u_13) ** 2 + (u_3 - u_15) ** 2 + (u_5 - u_17) ** 2) ** (3/2) + \
             m_2 * (u_9 - u_15) / ((u_7 - u_13) ** 2 + (u_9 - u_15) ** 2 + (u_11 - u_17) ** 2) ** (3/2))
    v_9 = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        G * (m_1 * (u_5 - u_17) / ((u_1 - u_13) ** 2 + (u_3 - u_15) ** 2 + (u_5 - u_17) ** 2) ** (3/2) + \
             m_2 * (u_11 - u_17) / ((u_7 - u_13) ** 2 + (u_9 - u_15) ** 2 + (u_11 - u_17) ** 2) ** (3/2))
    
    # Vector Function
    F = lambda u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18: \
        np.array([u_2, v_1(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18), \
                  u_4, v_2(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18), \
                  u_6, v_3(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18), \
                  u_8, v_4(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18), \
                  u_10, v_5(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18), \
                  u_12, v_6(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18), \
                  u_14, v_7(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18), \
                  u_16, v_8(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18), \
                  u_18, v_9(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15, u_16, u_17, u_18)])
    
    # Run numerical method
    U = method_func(F, U0, h, n)
    full_name = '{} - {}'.format(name, method_name)
    
    # Extract positions
    R1, R2, R3 = [], [], []
    for u in U:
        R1.append([u[index] for index in [0, 2, 4]])
        R2.append([u[index] for index in [6, 8, 10]])
        R3.append([u[index] for index in [12, 14, 16]])
    
    X1 = [p[0] for p in R1]
    Y1 = [p[1] for p in R1]
    Z1 = [p[2] for p in R1]
    
    X2 = [p[0] for p in R2]
    Y2 = [p[1] for p in R2]
    Z2 = [p[2] for p in R2]
    
    X3 = [p[0] for p in R3]
    Y3 = [p[1] for p in R3]
    Z3 = [p[2] for p in R3]
    
    # Plot static figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    fig_1 = plt.figure(figsize=(8,8))
    ax_1 = plt.axes(projection='3d')
    plt.suptitle(full_name)
    plt.title('step size: h = {}'.format(h))
    ax_1.plot3D(X1, Y1, Z1, 'blue', label='Mass 1')
    ax_1.plot3D(X2, Y2, Z2, 'red', label='Mass 2')
    ax_1.plot3D(X3, Y3, Z3, 'green', label='Mass 3')
    ax_1.set_xlabel('$X$', fontsize=20)
    ax_1.set_ylabel('$Y$', fontsize=20)
    ax_1.set_zlabel('$Z$', fontsize=20)
    ax_1.legend()
    plt.savefig(os.path.join(output_dir, '{}.png'.format(full_name)), dpi=300)
    plt.close(fig_1)
    
    # Animate
    frames, fps = 60, 10
    
    # Calculate fixed axis limits once before animation
    x_min, x_max = min(X1 + X2 + X3), max(X1 + X2 + X3)
    y_min, y_max = min(Y1 + Y2 + Y3), max(Y1 + Y2 + Y3)
    z_min, z_max = min(Z1 + Z2 + Z3), max(Z1 + Z2 + Z3)
    
    # Fix zlim issue: add small epsilon if min == max (e.g., for 2D problems)
    if z_min == z_max:
        z_min -= 0.1
        z_max += 0.1
    
    fig_2 = plt.figure(figsize=(8,8))
    ax_2 = plt.axes(projection='3d')
    plt.suptitle(full_name)
    plt.title('step size: h = {}'.format(h))
    # Set initial limits and disable auto-scaling
    ax_2.set_xlim([x_min, x_max])
    ax_2.set_ylim([y_min, y_max])
    ax_2.set_zlim([z_min, z_max])
    ax_2.set_autoscale_on(False)
    
    # Store data globally for animation function
    global ax, X1_anim, X2_anim, X3_anim, Y1_anim, Y2_anim, Y3_anim, Z1_anim, Z2_anim, Z3_anim
    global x_min_anim, x_max_anim, y_min_anim, y_max_anim, z_min_anim, z_max_anim, frames_anim
    ax = ax_2
    X1_anim, X2_anim, X3_anim = X1, X2, X3
    Y1_anim, Y2_anim, Y3_anim = Y1, Y2, Y3
    Z1_anim, Z2_anim, Z3_anim = Z1, Z2, Z3
    x_min_anim, x_max_anim = x_min, x_max
    y_min_anim, y_max_anim = y_min, y_max
    z_min_anim, z_max_anim = z_min, z_max
    frames_anim = frames
    
    animator = ani.FuncAnimation(fig_2, buildmebarchart, frames=frames)
    animator.save(os.path.join(output_dir, '{}.gif'.format(full_name)), fps=fps)
    plt.close(fig_2)
    
    print('Generated: {}'.format(full_name))
    
def main():
    """Main function that loops through all configurations and methods."""
    configs = get_configurations()
    methods = [
        (explicit_euler, 'Explicit Euler'),
        (adams_bashforth, 'Adams-Bashforth (2-step)'),
        (runge_kutta, 'Runge-Kutta 4'),
    ]
    
    total = len(configs) * len(methods)
    current = 0
    
    print('Starting generation of {} simulations...'.format(total))
    print('=' * 60)
    
    for config in configs:
        for method_func, method_name in methods:
            current += 1
            print('[{}/{}] Processing: {} - {}'.format(current, total, config['name'], method_name))
            try:
                run_simulation(config, method_func, method_name)
            except Exception as e:
                print('ERROR: Failed to generate {} - {}'.format(config['name'], method_name))
                print('Error message: {}'.format(str(e)))
                print('Continuing with next simulation...')
                print()
    
    print('=' * 60)
    print('Completed! Generated {} simulations.'.format(current))

def buildmebarchart(i=int):
    """Animation function that updates the plot for each frame."""
    global ax, X1_anim, X2_anim, X3_anim, Y1_anim, Y2_anim, Y3_anim, Z1_anim, Z2_anim, Z3_anim
    global x_min_anim, x_max_anim, y_min_anim, y_max_anim, z_min_anim, z_max_anim, frames_anim
    # Remove only line artists instead of clearing everything
    # This preserves axis limits and settings
    for line in ax.lines:
        line.remove()
    k = i * (math.floor(len(X1_anim) / frames_anim))
    if k == 0:
        k = 1  # Ensure at least one point is plotted
    ax.plot3D(X1_anim[:k], Y1_anim[:k], Z1_anim[:k], 'blue', label='Mass 1')
    ax.plot3D(X2_anim[:k], Y2_anim[:k], Z2_anim[:k], 'red', label='Mass 2')
    ax.plot3D(X3_anim[:k], Y3_anim[:k], Z3_anim[:k], 'green', label='Mass 3')
    ax.set_xlabel('$X$', fontsize=20)
    ax.set_ylabel('$Y$', fontsize=20)
    ax.set_zlabel('$Z$', fontsize=20)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))    
    plt.legend(by_label.values(), by_label.keys())
    # Ensure limits remain fixed (they should already be set, but re-enforce)
    ax.set_xlim([x_min_anim, x_max_anim])
    ax.set_ylim([y_min_anim, y_max_anim])
    ax.set_zlim([z_min_anim, z_max_anim])

if __name__ == '__main__':
    main()