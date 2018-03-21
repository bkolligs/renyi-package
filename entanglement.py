import time, numpy, datetime, mpmath, pylab


# Calculates the RMI for the XY model and for any model that uses normal thermodynamics (Ising)
def RMI_XY(Data, T_max, T_step, N_global, E_measurements, save_data='no', graph='no', path=' '):
    global date
    t1 = time.time()
    T_plot = Data[0]
    # Gathers the replica data
    E_replica = Data[1]
    sigma_replica = Data[2]
    # Gathers the normal data
    E_A_U_B = Data[3]
    sigma_A_U_B = Data[4]

    E_normal = Data[5]
    sigma_normal = Data[6]
    # Calculating RMI for each T
    print('Working on Renyi Mutual Information...')
    count = len(E_A_U_B)

    RMI_plot = []
    RMI_sigma_plot = []
    deltaT = T_step
    # Calculates the RMI and the sigma for each RMI(T)
    for i in range(count):
        RMI = 0.0
        sigma_sigma_i = 0.0
        for j in range(i, count):
            term_j = deltaT * (2 * (E_replica[j]) - (E_A_U_B[j]) - 2 * E_normal[j]) / ((T_plot[j]) ** 2)
            RMI += term_j
            # Now to propagate the error from each E measurement...
            sigma_sigma_j = ((2 * deltaT) / ((T_plot[j] ** 2) * N_global * 2)) ** 2 * (sigma_replica[j] ** 2) + (deltaT / ((
                T_plot[j] ** 2) * N_global)) ** 2 * (sigma_A_U_B[j] ** 2) + ((2 * deltaT) / ((T_plot[j] ** 2) * N_global * 2)) ** 2 * (sigma_normal[j] ** 2)
            sigma_sigma_i += sigma_sigma_j
        sigma_i = sqrt(sigma_sigma_i)
        RMI /= 2 * N_global
        RMI_plot.append(RMI)
        RMI_sigma_plot.append(sigma_i)
        if i % 100 == 0:
            print('Calculating RMI for T =', i * T_step)

    if graph == 'yes':
        pylab.plot(T_plot, RMI_plot, 'b', linewidth=3)
        pylab.errorbar(T_plot, RMI_plot, yerr=RMI_sigma_plot, capsize=2, ecolor='r')
        pylab.title(r'RMI vs $T$; $T_{step}$' + ' = {0};'.format(T_step) + ' $T_{max}$' + ' = {0} '.format(T_max), fontsize=16)
        pylab.xlabel(r'$T$', fontsize=16)
        pylab.ylabel(r'$\frac{I_2(T)}{\ell}$', fontsize=16)
        pylab.xlim(0.7, 2)
        # ylim(0, 0.5)
        t_elapse = (time.time() - t1) / 60
        print('Done in {0:.3f} minutes'.format(t_elapse))
        pylab.show()

    if save_data == 'yes':
        RMI_data = numpy.array([RMI_plot, RMI_sigma_plot])
        Data = numpy.array(Data)
        Data_txt = numpy.vstack((Data, RMI_data))
        t_elapse = (time.time() - t1) / 60
        numpy.savetxt('{5}/RMI XY; {0}; {1}, {2}, {3}, {4}.txt'.format(date, E_measurements, T_max, T_step, N_global, path), Data_txt, header='This RMI Data was calculated by "Finite Size Scaling Ising.py", independent from the Monte Carlo Run. This calculation took {0:.3f} minutes and was recorded on {1}'.format(t_elapse, datetime.datetime.today()))

    return T_plot, RMI_plot, RMI_sigma_plot


# RMI calculation for inverted thermodynamics
def RMI_QCD(Data, T_max, T_step, N_global, E_measurements, save_data='no', graph='no', path=' '):
    date = datetime.date.today()
    t1 = time.time()
    T_plot = Data[0]
    # Gathers the replica data
    E_replica = Data[1]
    sigma_replica = Data[2]
    # Gathers the normal data
    E_A_U_B = Data[3]
    sigma_A_U_B = Data[4]

    E_normal = Data[5]
    sigma_normal = Data[6]
    # Calculating RMI for each T
    print('Working on Renyi Mutual Information...')
    count = len(E_A_U_B)

    RMI_plot = []
    RMI_sigma_plot = []
    deltaT = T_step
    # Calculates the RMI and the sigma for each RMI(T)
    for i in range(count):
        RMI = 0.0
        sigma_sigma_i = 0.0
        for j in range(0, i):
            term_j = deltaT * (2 * (E_replica[j]) - (E_A_U_B[j]) - 2 * E_normal[j])
            RMI += term_j
            # Now to propagate the error from each E measurement...
            sigma_sigma_j = ((2 * deltaT) / (N_global * 2)) ** 2 * (sigma_replica[j] ** 2) + (deltaT / (N_global)) ** 2 * (sigma_A_U_B[j] ** 2) + ((2 * deltaT) / ( N_global * 2)) ** 2 * (sigma_normal[j] ** 2)
            sigma_sigma_i += sigma_sigma_j
        sigma_i = mpmath.sqrt(sigma_sigma_i)
        RMI /= 2 * N_global
        RMI_plot.append(RMI)
        RMI_sigma_plot.append(sigma_i)
        if i % 100 == 0:
            print('Calculating RMI for T =', i * T_step)

    if graph == 'yes':
        pylab.plot(T_plot, RMI_plot, 'b', linewidth=2)
        pylab.errorbar(T_plot, RMI_plot, yerr=RMI_sigma_plot, capsize=2, ecolor='r')
        pylab.title(r'RMI vs $T$; $T_{step}$' + ' = {0};'.format(T_step) + ' $T_{max}$' + ' = {0} '.format(T_max), fontsize=16)
        pylab.xlabel(r'$T$', fontsize=16)
        pylab.ylabel(r'$\frac{I_2(T)}{\ell}$', fontsize=16)
        pylab.xlim(0.0, 2)
        # ylim(0, 0.5)
        t_elapse = (time.time() - t1) / 60
        print('Done in {0:.3f} minutes'.format(t_elapse))
        pylab.show()

    if save_data == 'yes':
        RMI_data = numpy.array([RMI_plot, RMI_sigma_plot])
        Data = numpy.array(Data)
        Data_txt = numpy.vstack((Data, RMI_data))
        t_elapse = (time.time() - t1) / 60
        numpy.savetxt('({5}/RMI QCD; {0}; {1}, {2}, {3}, {4}.txt'.format(date, E_measurements, T_max, T_step, N_global, path), Data_txt, header='This RMI Data was calculated by "Finite Size Scaling Ising.py", independent from the Monte Carlo Run. This calculation took {0:.3f} minutes and was recorded on {1}'.format(t_elapse, datetime.datetime.today()))

    return T_plot, RMI_plot, RMI_sigma_plot


# A simple way to graph RMI or energy when using data files.
def I_T_plot(Data, y_tilde, model='QCD', color='b', graph='RMI', e_bars='yes'):
    global label_size
    T_plot = Data[0]

    E_replica = Data[1]
    sigma_replica = Data[2]
    # Gathers the normal data
    E_A_U_B = Data[3]
    sigma_A_U_B = Data[4]

    E_normal = Data[5]
    sigma_normal = Data[6]

    if model == 'QCD':
        RMI = Data[13]
        RMI_sigma = Data[14]
    elif model == 'XY':
        RMI = Data[9]
        RMI_sigma = Data[10]

    if graph == 'RMI':
        pylab.plot(T_plot, RMI, color, label=r'$\tilde{y}$=' + '{}'.format(y_tilde))
        if e_bars == 'yes':
            pylab.errorbar(T_plot, RMI, yerr=RMI_sigma, capsize=2, color=color)
        pylab.title(r"Renyi Mutual Information vs Temperature for $p=4$", fontsize=label_size)
        pylab.xlim(0.0, 6)
        #xlim(0, 10)
        pylab.ylabel(r"$\frac{I_2(T)}{\ell}$", fontsize=label_size)
        pylab.xlabel('T', fontsize=label_size)
        pylab.show()
    if graph == 'Energy':
        pylab.plot(T_plot, E_replica, 'b', label='Replica QCD-XY')
        pylab.plot(T_plot, E_A_U_B, 'r', label='A U B')
        pylab.plot(T_plot, E_normal, 'g', label='Normal QCD-XY')
        if e_bars == 'yes':
            pylab.errorbar(T_plot, E_replica, yerr=sigma_replica, capsize=2, color='b')
            pylab.errorbar(T_plot, E_A_U_B, yerr=sigma_A_U_B, capsize=2, color='r')
            pylab.errorbar(T_plot, E_normal, yerr=sigma_normal, capsize=2, color='g')
        pylab.title("Energy of the Three Models", fontsize=label_size)
        pylab.xlabel(r"$T$", fontsize=label_size)
        pylab.ylabel("Energy", fontsize=label_size)
        pylab.xlim(0, 10)
        pylab.legend()
        pylab.show()


# Renyi Entropy for normal thermodynamics
def EE_calc(Data, T_step, N, n, color=None):
    T_plot = Data[0]
    E_replica = Data[1]
    sigma_replica = Data[2]
    E_normal = Data[5]
    count = len(T_plot)
    deltaT = T_step

    S_plot = []
    integrand = []
    S_sigma = []
    for i in range(count):
        S_A = 0.0
        sigma_sigma_i = 0.0
        term_integrand = deltaT * ((E_replica[i]) - (n * E_normal[i])) / (T_plot[i] ** 2)
        integrand.append(term_integrand)
        for j in range(i, count):
            term_j = deltaT * ((E_replica[j]) - (n * E_normal[j])) / (T_plot[j] ** 2)
            S_A += term_j
            # error propagation:
            sigma_sigma_j = ((2 * deltaT) / ((T_plot[j] ** 2) * N * 2)) ** 2 * (sigma_replica[j] ** 2) # this is wrong
            sigma_sigma_i += sigma_sigma_j
        sigma_i = sqrt(sigma_sigma_i)
        S_A /= 2 * N
        S_plot.append(S_A)
        S_sigma.append(sigma_i)
        if i % 100 == 0:
            print("N={0}, n={1}, Working on EE for T=".format(N, n), i * T_step)

    if color is not None:
        pylab.plot(T_plot, S_plot, color, label='N={0}, n={1}'.format(N, n))
    else:
        pylab.plot(T_plot, S_plot, label='N={0}, n={1}'.format(N, n))
    pylab.title("Renyi Entropy of N={0} and n={1}".format(N, n), fontsize=16)
    pylab.xlabel("T", fontsize=16)
    pylab.ylabel("Renyi Entropy", fontsize=16)
    #show()


# Renyi Entropy for the inverted thermodynamics
def EE_QCD(Data, T_step, N, n, y_tilde, p):
    T_plot = Data[0]
    E_replica = Data[1]
    sigma_replica = Data[2]
    E_normal = Data[5]
    count = len(T_plot)
    deltaT = T_step

    S_plot = []
    integrand = []
    S_sigma = []
    for i in range(count):
        S_A = 0.0
        sigma_sigma_i = 0.0
        term_integrand = deltaT * ((E_replica[i]) - (n * E_normal[i]))
        integrand.append(term_integrand)
        for j in range(0, i):
            term_j = deltaT * ((E_replica[j]) - (n * E_normal[j]))
            S_A += term_j
            # error propagation:
            sigma_sigma_j = ((2 * deltaT) / (N * 2)) ** 2 * (sigma_replica[j] ** 2) # this is wrong
            sigma_sigma_i += sigma_sigma_j
        sigma_i = sqrt(sigma_sigma_i)
        S_plot.append(S_A)
        S_sigma.append(sigma_i)

    pylab.plot(T_plot, S_plot, 'b', label=r'EE for n={},'.format(n) + r'$\tilde{y}$' + '={}'.format(y_tilde))
    pylab.title(r"$S(A \cup B)$" + " of N={0} and".format(N) + r" $\tilde{y}$" + " = {0}".format(y_tilde) + r" $p = {0}$".format(p), fontsize=label_size)
    pylab.xlabel(r"$T$", fontsize=label_size)
    pylab.ylabel(r"$S(A \cup B)$", fontsize=label_size)
    pylab.show()

    pylab.plot(T_plot, integrand, 'b', label='EE integrand')
    pylab.title("EE Integrand", fontsize=label_size)
    pylab.xlim(0, 2)
    pylab.xlabel("T", fontsize = label_size)
    pylab.ylabel("Renyi Entropy", fontsize=label_size)
    pylab.legend()
    pylab.show()


# A method of creating arrays that doesn't include the last value, no matter the range. More reliable than numpy.arange
def float_array(T_min, T_max, T_step, check_num='yes'):
    number = int(round((T_max - T_min), 4) / T_step)
    if check_num == 'yes':
        print("Going to try this as num: ", number)
    array = numpy.linspace(T_min, T_max, number, endpoint=False)
    return array


# if you have a latex distribution installed, you can copy past the body of this function to turn latex on your graphs. Or just call the function.
def use_latex(label_size):
    pylab.rcParams['xtick.labelsize'] = label_size
    pylab.rcParams['ytick.labelsize'] = label_size
    pylab.rcParams['legend.fontsize'] = label_size
    pylab.rc('text', usetex=True)
    pylab.rc('font', family='serif')
