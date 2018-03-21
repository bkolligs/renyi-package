import os, numpy, time, math, pylab, datetime, sys
label_size = 36
pylab.rcParams['xtick.labelsize'] = label_size
pylab.rcParams['ytick.labelsize'] = label_size
pylab.rcParams['legend.fontsize'] = label_size
pylab.rc('text', usetex=True)
pylab.rc('font', family='serif')


y_tilde_graph = 1.0
theta_coefficient_graph = 2
graph = 'RMI'
data_folder = 'Theta2'
data_directory = 'QCD Model Aggregate Data/{0}'.format(data_folder)


def ordering(data_name):
    name_split = data_name.split(',')
    return float(name_split[4])


def derivative(x_plot, y_plot, yerr=None, graph='show', g_title=r"Data vs It's Derivative"):

    step = x_plot[1] - x_plot[0]
    y_prime_plot = []
    y_prime_sigma = []

    count = len(x_plot)
    for index in range(count):
        dy = 0
        dy_sigma = 0
        if (count - 1) > index > 0:
            dy = (y_plot[index + 1] - y_plot[index - 1]) / (2 * step)
            if yerr is not None:
                dy_sigma = sqrt((yerr[index + 1] / (2 * step)) ** 2 + (yerr[index - 1] / (2 * step)) ** 2)
                y_prime_sigma.append(dy_sigma)

        y_prime_plot.append(dy)
    if graph == 'show':
        plot(x_plot, y_plot, 'b', label=r"$0^{th}$ order")
        plot(x_plot, y_prime_plot, 'r--', label=r"$1^{st}$ order")
        errorbar(x_plot, y_plot, yerr=yerr, capsize=2, color='b')
        # errorbar(x_plot, y_prime_plot, yerr=y_prime_sigma, capsize=2, color='r')

        title(g_title + " for N = Various")
        x_label, y_label = g_title.split('vs')
        xlabel(x_label)
        ylabel(y_label)
        xlim(0, 2)
        legend()
        show()

    return x_plot, y_plot, y_prime_plot


contents = os.listdir(data_directory)

colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
color_index = 0
for file_name in sorted(contents, key=ordering):
    deconstruction = file_name.split(',')
    prefix = deconstruction[0].split(';')[0]
    date_of_data = deconstruction[0].split(';')[1]
    measurements = int(deconstruction[0].split(';')[2])
    T_min = float(deconstruction[1])
    T_max = float(deconstruction[2])
    T_step = float(deconstruction[3])
    lattice_size = int(deconstruction[4])
    n = int(deconstruction[5].split('=')[1])
    y_tilde = float(deconstruction[6].split('~')[1])
    theta_coefficient = int(deconstruction[7].split('=')[1].split('.')[0])

    if y_tilde == y_tilde_graph and theta_coefficient == theta_coefficient_graph:
        print(file_name)
        data_chunk = numpy.loadtxt('{0}/{1}'.format(data_directory, file_name))

        if graph == 'RMI':
            # This first section graphs RMI
            T_plot = data_chunk[0]
            RMI = data_chunk[13]
            RMI_sigma = data_chunk[14]
            color = colors[color_index]
            color_index += 1
            pylab.plot(T_plot, RMI, color, label='N={0}'.format(lattice_size))
            pylab.errorbar(T_plot, RMI, yerr=RMI_sigma, color=color)

            E_normal = data_chunk[5]
            sigma_normal = data_chunk[6]

            capacitance = derivative(T_plot, E_normal, graph='no')[2]
            trough_cap = min(capacitance)
            T_cap = T_plot[capacitance.index(trough_cap)]
            print(T_cap)

        if graph == 'Energy':

            # This graphs Energy vs T
            T_plot = data_chunk[0]

            E_replica = data_chunk[1]
            sigma_replica = data_chunk[2]

            E_AUB = data_chunk[3]
            sigma_AUB = data_chunk[4]

            E_normal = data_chunk[5]
            sigma_normal = data_chunk[6]

            color = colors[color_index]
            color_index += 1
            pylab.plot(T_plot, E_replica, color, label='N={0}'.format(lattice_size))
            pylab.errorbar(T_plot, E_replica, yerr=sigma_replica, color=color)
            pylab.plot(T_plot, E_AUB, color + '.', label='N={0}'.format(lattice_size))
            pylab.errorbar(T_plot, E_AUB, yerr=sigma_AUB, color=color)
            pylab.plot(T_plot, E_normal, color + '--', label='N={0}'.format(lattice_size))
            pylab.errorbar(T_plot, E_normal, yerr=sigma_normal, color=color)

        if graph == 'Capacitance':
            # This graphs Capacitance
            T_plot = data_chunk[0]

            # E_normal = data_chunk[5]
            # sigma_normal = data_chunk[6]
            #
            # capacitance = derivative(T_plot, E_normal, graph='no')[2]
            # color = colors[color_index]
            # color_index += 1
            # pylab.plot(T_plot, capacitance, color, label='N={}'.format(lattice_size))
            heatcap = data_chunk[9]
            sigma_heatcap = data_chunk[10]
            color = colors[color_index]
            color_index += 1
            pylab.plot(T_plot, heatcap, color, label='N={}'.format(lattice_size))
            #pylab.errorbar(T_plot, heatcap, yerr=sigma_heatcap, color=color)

        if graph == 'Magnetization':
            T_plot = data_chunk[0]
            magnetization = data_chunk[7]
            mag_sigma = data_chunk[8]
            color = colors[color_index]
            color_index += 1
            pylab.plot(T_plot, magnetization, color, label='N={0}'.format(lattice_size))
            pylab.errorbar(T_plot, magnetization, yerr=mag_sigma, color=color)

            susceptibility = derivative(T_plot, magnetization, graph='no')[2]

            trough_cap = min(susceptibility)
            T_cap = T_plot[susceptibility.index(trough_cap)]
            print(T_cap)

        if graph == 'Susceptibility':
            T_plot = data_chunk[0]
            # magnetization = data_chunk[7]
            # mag_sigma = data_chunk[8]
            # color = colors[color_index]
            # color_index += 1
            #
            # suscepibility = derivative(T_plot, magnetization, graph='no')[2]
            # pylab.plot(T_plot, suscepibility, color, label='N={0}'.format(lattice_size))
            #
            # trough_cap = min(suscepibility)
            # T_cap = T_plot[suscepibility.index(trough_cap)]
            # print(T_cap)
            susceptibility = data_chunk[11]
            sigma_susceptibility = data_chunk[12]
            color = colors[color_index]
            color_index += 1
            pylab.plot(T_plot, susceptibility, color, label='N={}'.format(lattice_size))
            pylab.errorbar(T_plot, susceptibility, yerr=sigma_susceptibility, color=color)


if graph == 'RMI':
    pylab.title(r"Finite Size Scaling for $\tilde{y}$=" + "{}".format(y_tilde_graph) + r" and $p$=" + '{}'.format(theta_coefficient_graph), fontsize=label_size)
    pylab.xlabel(r'$T$', fontsize=label_size)
    pylab.ylabel(r'$\frac{I_2}{\ell}$', fontsize=label_size)
    pylab.legend( loc=1)
    pylab.xlim(0, 1.0)
    pylab.show()

    #pylab.savefig(r'C:\Users\~Benjamin~\Documents\Rogers Research\Strong Force XY\RMI vs T Graphs\y~{0}, theta={1}; RMI for N=8,16,24,32,40,48,56.pdf'.format(y_tilde_graph, theta_coefficient_graph,))

if graph == 'Energy':
    pylab.title(r"Energy vs $T$ for $\tilde{y}$=" + "{}".format(y_tilde_graph) + r" and $p$=" + '{}'.format(theta_coefficient_graph), fontsize=label_size)
    pylab.xlabel(r'$T$', fontsize=label_size)
    pylab.ylabel(r'Energy', fontsize=label_size)
    pylab.legend()
    pylab.xlim(0, 4.0)
    pylab.show()

if graph == 'Capacitance':
    pylab.title(r"Finite Size Scaling for $\tilde{y}$=" + "{}".format(
        y_tilde_graph) + r" and $p$=" + '{}'.format(theta_coefficient_graph), fontsize=label_size)
    pylab.xlabel(r'$T$', fontsize=label_size)
    pylab.ylabel(r'$\frac{dE_0}{dT}$', fontsize=label_size)
    pylab.legend()
    pylab.xlim(0, 2.0)
    pylab.show()

if graph == 'Magnetization':
    pylab.title(r"Magnetization for $\tilde{y}$=" + "{}".format(
        y_tilde_graph) + r" and $p$=" + '{}'.format(theta_coefficient_graph), fontsize=label_size)
    pylab.xlabel(r'$T$', fontsize=label_size)
    pylab.ylabel(r'$\mathcal{M}$', fontsize=label_size)
    pylab.legend()
    pylab.xlim(0, 2.0)
    pylab.show()

if graph == 'Susceptibility':
    pylab.title(r"Susceptibility for $\tilde{y}$=" + "{}".format(
        y_tilde_graph) + r" and $p$=" + '{}'.format(theta_coefficient_graph), fontsize=label_size)
    pylab.xlabel(r'$T$', fontsize=label_size)
    pylab.ylabel(r'$\frac{d\mathcal{M}}{dT}$', fontsize=label_size)
    pylab.xlim(0.0, 2.0)
    pylab.legend()
    pylab.show()

