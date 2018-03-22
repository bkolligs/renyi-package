import time, numpy, datetime, mpmath, pylab, random
from numpy import ones, arange, sqrt, array, savetxt, vstack, zeros
from math import exp, pi, cos, sin


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


# This is a function that will calculate the Topological renyi entropy from energy data output by the TEE Monte Carlo simulation function.
def TEE_calc(Data, T_step):
    global date, n
    t1 = time.time()
    T_plot = Data[0]
    # Gathers the replica data
    E_shape_1 = Data[1]
    sigma_shape_1 = Data[2]

    # Gathers the normal data
    E_shape_2 = Data[3]
    sigma_shape_2 = Data[4]

    E_shape_3 = Data[5]
    sigma_shape_3 = Data[6]

    E_normal = Data[7]
    sigma_normal = Data[8]
    print("Calculating EE for... ")
    # calculates the Renyi Entropy
    print("Shape 1...")
    shape_1_data = EE_calc(T_plot, E_shape_1, sigma_shape_1, E_normal, sigma_normal, T_step, 2)
    print("Shape 2...")
    shape_2_data = EE_calc(T_plot, E_shape_2, sigma_shape_2, E_normal, sigma_normal, T_step, 2)
    print("Shape 3...")
    shape_3_data = EE_calc(T_plot, E_shape_3, sigma_shape_3, E_normal, sigma_normal, T_step, 2)

    S_shape_1 = shape_1_data[0]
    S_shape_1_sigma = shape_1_data[1]
    S_shape_2 = shape_2_data[0]
    S_shape_2_sigma = shape_2_data[1]
    S_shape_3 = shape_3_data[0]
    S_shape_3_sigma = shape_3_data[1]


    # Calculating TEE for each T
    print('Working on Topological Entanglement Entropy...')
    count = len(T_plot)
    TEE_plot = []
    TEE_sigma_plot = []
    for T in range(count):
        S_i = -S_shape_1[T] + 2 * S_shape_2[T] - S_shape_3[T]
        TEE_plot.append(S_i)
        sigma = numpy.sqrt(S_shape_1_sigma[T]**2 + 4*S_shape_2_sigma[T]**2 + S_shape_3_sigma[T]**2)
        TEE_sigma_plot.append(sigma)
    pylab.plot(T_plot, TEE_plot, 'b')
    pylab.errorbar(T_plot, TEE_plot, yerr=TEE_sigma_plot, ecolor='r')
    pylab.title(r"Topological Entanglement Entropy for $p =$" + '{}'.format(theta))
    pylab.xlim(0, 4)
    pylab.xlabel("T", fontsize = label_size)
    pylab.ylabel("TEE", fontsize=label_size)
    pylab.show()
    # prime = derivative(T_plot, TEE_plot)[1]

    return T_plot, TEE_plot, TEE_sigma_plot


# The inverted thermodynamics monte carlo functions for the three energies
def QCD_E(T):
    # To perform an equilbriium test, set the following variable to yes and plot the last array this function returns
    equilibrium_test = 'no'
    global N_global, E_measurements, tau_after, y_tilde, theta_coefficient
    kappa = 4 * pi
    #J = - (8 * T) / (pi * kappa)
    J = 1
    N = N_global  # The lattice size: NxN
    tau = tau_global  # The correlation time
    # if T > 20:
    #     tau = tau_after
    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -2 * (N * N) - y_tilde * (N * N)  # Initial Value of Energy since all spins start pointed up at \theta_i = 0.0
    m_1 = N*N # Initial value of magnetization
    m_2 = 0
    L = zeros([N, N], float)  # Generates the lattice where each entry is a value of \theta_i

    # JT = J / T  # The parameter J divided by T (temperature) Multiplying dE by this will potentially save time,
    # but if this is done, make sure to multiply dE by T again when doing E += dE/(N*N)

    print("N=", N, "; Normal XY-Model at T=", T)

    expE = 0.0  # Expectation value of E
    expM = 0.0
    E_plot = []
    measurements = []  # List of Measurements
    M_measurements = []
    # Main Monte Carlo cycle
    for x in range(steps + 1):
        i = random.randrange(0, N)
        j = random.randrange(0, N)  # Picks a random starting location

        # Decides an anticipated spin amount
        L_update = random.random() * 2 * pi
        # Calculates change in energy that would occur if this spin was accepted
        dE = 0.0
        # Starts calculating the nearest neighbor sum at location L[ i-1 , j]
        neighbor = i - 1
        if neighbor > -1:
            dE += cos(L_update - L[neighbor, j]) - cos(L[i, j] - L[neighbor, j])  # Checks if the neighbor is within the lattice
        else:
            dE += cos(L_update - L[N - 1, j]) - cos(L[i, j] - L[N - 1, j])  # Periodic boundary conditions
        neighbor = i + 1
        if neighbor < N:
            dE += cos(L_update - L[neighbor, j]) - cos(L[i, j] - L[neighbor, j])
        else:
            dE += cos(L_update - L[0, j]) - cos(L[i, j] - L[0, j])
        neighbor = j - 1
        if neighbor > -1:
            dE += cos(L_update - L[i, neighbor]) - cos(L[i, j] - L[i, neighbor])
        else:
            dE += cos(L_update - L[i, N - 1]) - cos(L[i, j] - L[i, N - 1])
        neighbor = j + 1
        if neighbor < N:
            dE += cos(L_update - L[i, neighbor]) - cos(L[i, j] - L[i, neighbor])
        else:
            dE += cos(L_update - L[i, 0]) - cos(L[i, j] - L[i, 0])
        dE *= -J
        dE += y_tilde * (cos(theta_coefficient * L[i, j]) - cos(theta_coefficient * L_update))

        # Calculates whether L[i,j] rotates
        R = exp(-dE * T)
        if R > 1 or random.random() < R:
            m_1 = m_1 + cos(L_update) - cos(L[i, j])
            m_2 = m_2 + sin(L_update) - sin(L[i, j])
            L[i, j] = L_update
            E += dE  # / (N * N)
        if equilibrium_test == 'yes':
            E_plot.append(E)
        if x != 0 and x % (2 * tau) == 0:
            expE += E
            M = sqrt(m_1**2 + m_2**2)
            expM += M
            # print("at x = ",x,"  ", expE)
            measurements.append(E)  # Adds the measurement to the list
            M_measurements.append(M)
    expE /= BM
    expM /= BM
    # print(expE)

    # The Bootstrap Error Analysis
    resample = BM  # times to repeat re-sampling
    B_i = []  # for the calculation of <B> and sigma
    M_i = []
    for y in range(resample):
        B = 0.0
        M_error = 0.0
        for z in range(int(BM)):
            n = random.randrange(0, BM)
            B += measurements[n]
            M_error += M_measurements[n]
        B /= BM
        M_error /= BM
        B_i.append(B)
        M_i.append(M_error)

    # Now to calculate the Bootstrap sigma
    sigma_sigma = 0.0
    sigma_sigma_M = 0.0
    for w in range(resample):
        sigma_sigma += (B_i[w] - expE) ** 2
        sigma_sigma_M += (M_i[w] - expM) ** 2
    sigma_sigma /= resample
    sigma_sigma_M /= resample
    sigma_bootstrap = sqrt(sigma_sigma)
    sigma_bootstrap_M = sqrt(sigma_sigma_M)

    return [T, expE, sigma_bootstrap, M, sigma_bootstrap_M, E_plot]  # This will create a results matrix which can be plotted


def QCD_A_U_B(T):
    # To perform an equilbriium test, set the following variable to yes and plot the last array this function returns
    equilibrium_test = 'no'
    global N_global, E_measurements, tau_after
    kappa = 4 * pi
    #J = - (8 * T) / (pi * kappa)
    J = 1
    N = N_global  # The lattice size: NxN
    tau = tau_global  # The correlation time
    # if T > 20:
    #     tau = tau_after
    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -4 * (N * N) - 2 * y_tilde * (N * N)  # Initial Value of Energy since all spins start pointed up at \theta_i = 0.0
    L = zeros([N, N], float)  # Generates the lattice where each entry is a value of \theta_i

    # JT = J / T  # The parameter J divided by T (temperature) Multiplying dE by this will potentially save time,
    # but if this is done, make sure to multiply dE by T again when doing E += dE/(N*N)

    print("N=", N, "; A-union-B XY-Model at T=", T)
    E_plot = []
    expE = 0.0  # Expectation value of E
    measurements = []  # List of Measurements
    # Main Monte Carlo cycle
    for x in range(steps + 1):
        i = random.randrange(0, N)
        j = random.randrange(0, N)  # Picks a random starting location

        # Decides an anticipated spin amount
        L_update = random.random() * 2 * pi
        # Calculates change in energy that would occur if this spin was accepted
        dE = 0.0
        # Starts calculating the nearest neighbor sum at location L[ i-1 , j]
        neighbor = i - 1
        if neighbor > -1:
            dE += cos(L_update - L[neighbor, j]) - cos(
                L[i, j] - L[neighbor, j])  # Checks if the neighbor is within the lattice
        else:
            dE += cos(L_update - L[N - 1, j]) - cos(L[i, j] - L[N - 1, j])  # Periodic boundary conditions
        neighbor = i + 1
        if neighbor < N:
            dE += cos(L_update - L[neighbor, j]) - cos(L[i, j] - L[neighbor, j])
        else:
            dE += cos(L_update - L[0, j]) - cos(L[i, j] - L[0, j])
        neighbor = j - 1
        if neighbor > -1:
            dE += cos(L_update - L[i, neighbor]) - cos(L[i, j] - L[i, neighbor])
        else:
            dE += cos(L_update - L[i, N - 1]) - cos(L[i, j] - L[i, N - 1])
        neighbor = j + 1
        if neighbor < N:
            dE += cos(L_update - L[i, neighbor]) - cos(L[i, j] - L[i, neighbor])
        else:
            dE += cos(L_update - L[i, 0]) - cos(L[i, j] - L[i, 0])
        dE *= 2 *  -J
        dE += 2 * y_tilde * (cos(theta_coefficient * L[i, j]) - cos(theta_coefficient * L_update))

        # Calculates whether L[i,j] rotates
        R = exp(-dE * T)
        if R > 1 or random.random() < R:
            L[i, j] = L_update
            E += dE  # / (N * N)
        if equilibrium_test == 'yes':
            E_plot.append(E)
        if x != 0 and x % (2 * tau) == 0:
            expE += E
            # print("at x = ",x,"  ", expE)
            measurements.append(E)  # Adds the measurement to the list
    expE /= BM
    # print(expE)

    # The Bootstrap Error Analysis
    resample = BM  # times to repeat re-sampling
    B_i = []  # for the calculation of <B> and sigma
    for y in range(resample):
        B = 0.0
        for z in range(int(BM)):
            n = random.randrange(0, BM)
            B += measurements[n]
        B /= BM
        B_i.append(B)

    # Now to calculate the Bootstrap sigma
    sigma_sigma = 0.0
    for w in range(resample):
        sigma_sigma += (B_i[w] - expE) ** 2
    sigma_sigma /= resample
    sigma_bootstrap = sqrt(sigma_sigma)

    return [T,  expE, sigma_bootstrap, E_plot]  # This will create a results matrix which can be plotted


def QCD_Replica_E(T):
    # To perform an equilbriium test, set the following variable to yes and plot the last array this function returns
    equilibrium_test = 'no'
    global N_global, E_measurements, tau_global, tau_after
    kappa = 4 * pi
    #J = - (8 * T) / (pi * kappa)
    J = 1
    N = N_global  # The lattice size: NxN
    # A test to make things quicker; higher temperatures equilibrate faster
    tau = tau_global
    # if T > 20:
    #     tau = tau_after
    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -4 * (N * N) - 2 * y_tilde * (N * N) # Initial Value of Energy since all spins start pointed up at \theta_i = 0.0
    boundary = N // 2
    L1 = zeros([N, N], float)  # Lattice 1 where each entry is a value of \theta_i
    L2 = zeros([N, N], float)  # Lattice 2
    A_1 = L1[:, 0:boundary]
    A_2 = L2[:, 0:boundary]
    B_1 = L1[:, boundary: N]
    B_2 = L2[:, boundary: N]

    print("N=", N, "; Replica XY-Model at T=", T)
    E_plot = []
    expE = 0.0  # Expectation value of E
    measurements = []  # List of Measurements
    # Main Monte Carlo cycle
    for x in range(steps + 1):
        i = random.randrange(0, N)
        j = random.randrange(0, N)  # Picks a random starting location

        # Decides an anticipated spin amount
        L1_update = random.random() * 2 * pi
        L2_update = random.random() * 2 * pi

        if j < boundary:
            L1_update = L2_update
        # Calculates change in energy that would occur if this spin was accepted
        dE = 0.0
        # Starts calculating the nearest neighbor sum at location L[ i-1 , j]
        neighbor = i - 1
        if neighbor > -1:
            dE += cos(L1_update - L1[neighbor, j]) - cos(L1[i, j] - L1[neighbor, j])
            dE += cos(L2_update - L2[neighbor, j]) - cos(L2[i, j] - L2[neighbor, j])
        else:
            dE += cos(L1_update - L1[N - 1, j]) - cos(L1[i, j] - L1[N - 1, j])  # Periodic boundary conditions
            dE += cos(L2_update - L2[N - 1, j]) - cos(L2[i, j] - L2[N - 1, j])
        neighbor = i + 1
        if neighbor < N:
            dE += cos(L1_update - L1[neighbor, j]) - cos(L1[i, j] - L1[neighbor, j])
            dE += cos(L2_update - L2[neighbor, j]) - cos(L2[i, j] - L2[neighbor, j])
        else:
            dE += cos(L1_update - L1[0, j]) - cos(L1[i, j] - L1[0, j])
            dE += cos(L2_update - L2[0, j]) - cos(L2[i, j] - L2[0, j])
        neighbor = j - 1
        if neighbor > -1:
            dE += cos(L1_update - L1[i, neighbor]) - cos(L1[i, j] - L1[i, neighbor])
            dE += cos(L2_update - L2[i, neighbor]) - cos(L2[i, j] - L2[i, neighbor])
        else:
            dE += cos(L1_update - L1[i, N - 1]) - cos(L1[i, j] - L1[i, N - 1])
            dE += cos(L2_update - L2[i, N - 1]) - cos(L2[i, j] - L2[i, N - 1])
        neighbor = j + 1
        if neighbor < N:
            dE += cos(L1_update - L1[i, neighbor]) - cos(L1[i, j] - L1[i, neighbor])
            dE += cos(L2_update - L2[i, neighbor]) - cos(L2[i, j] - L2[i, neighbor])
        else:
            dE += cos(L1_update - L1[i, 0]) - cos(L1[i, j] - L1[i, 0])
            dE += cos(L2_update - L2[i, 0]) - cos(L2[i, j] - L2[i, 0])
        dE *= -J
        dE += y_tilde * (cos(theta_coefficient * L1[i, j]) - cos(theta_coefficient * L1_update)) + y_tilde * (cos(theta_coefficient * L2[i, j]) - cos(theta_coefficient * L2_update))

        # Calculates whether L[i,j] rotates
        R = exp(-dE * T)
        if R > 1 or random.random() < R:
            L1[i, j] = L1_update
            L2[i, j] = L2_update
            E += dE  # / (N * N)
        if equilibrium_test == 'yes':
            E_plot.append(E)
        if x != 0 and x % (2 * tau) == 0:
            expE += E
            # print("at x = ",x,"  ", expE)
            measurements.append(E)  # Adds the measurement to the list
    expE /= BM
    # print(expE)

    # The Bootstrap Error Analysis
    resample = BM  # times to repeat re-sampling
    B_i = []  # for the calculation of <B> and sigma
    for y in range(resample):
        B = 0.0
        for z in range(int(BM)):
            n = random.randrange(0, BM)
            B += measurements[n]
        B /= BM
        B_i.append(B)

    # Now to calculate the Bootstrap sigma
    sigma_sigma = 0.0
    for w in range(resample):
        sigma_sigma += (B_i[w] - expE) ** 2
    sigma_sigma /= resample
    sigma_bootstrap = sqrt(sigma_sigma)

    # This is a test to make sure that A_1 and A_2 are indeed being updated the same.
    equivalence_test = 'no'
    if equivalence_test == 'yes':
        matches = 0.0
        for columns in range(0, boundary):
            for rows in range(N):
                if A_1[rows, columns] == A_2[rows, columns]:
                    matches += 1
        if matches == N * boundary:
            print("A_1 and A_2 are the same!")
        else:
            print("We messed up somewhere :(")

    return [T, expE, sigma_bootstrap, E_plot]  # This will create a results matrix which can be plotted


# Ising Model functions
def Ising_E(T):
    # To perform an equilbriium test, set the following variable to yes and plot the last array this function returns
    equilibrium_test = 'no'
    global N_global, E_measurements, tau_global
    J = 1
    N = N_global  # The lattice size: NxN
    tau = tau_global  # The correlation time

    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -2 * (N * N)  # Initial Value of Energy
    L = ones([N, N], int)  # Generates the lattice

    # JT = J / T  # The parameter J divided by T (temperature) Multiplying dE by this will potentially save time,
    # but if this is done, make sure to multiply dE by T again when doing E += dE/(N*N)

    print("N=", N, "; Normal Model at T=", T)
    E_plot = []
    expE = 0.0  # Expectation value of E
    measurements = []  # List of Measurements
    # Main Monte Carlo cycle
    for x in range(steps + 1):
        i = random.randrange(0, N)
        j = random.randrange(0, N)  # Picks a random starting location

        # Calculates change in energy
        dE = 0.0
        # Starts calculating the nearest neighbor sum at location a[ i-1 , j]
        neighbor = i - 1
        if neighbor > -1:
            dE += L[neighbor, j]  # Checks if the neighbor is within the lattice
        else:
            dE += L[N - 1, j]  # Periodic boundary conditions
        neighbor = i + 1
        if neighbor < N:
            dE += L[neighbor, j]
        else:
            dE += L[0, j]
        neighbor = j - 1
        if neighbor > -1:
            dE += L[i, neighbor]
        else:
            dE += L[i, N - 1]
        neighbor = j + 1
        if neighbor < N:
            dE += L[i, neighbor]
        else:
            dE += L[i, 0]
        dE *= J * 2 * L[i, j]
        # print(E)
        # Calculates whether L[i,j] flips
        R = exp(-dE / T)
        if R > 1 or random.random() < R:
            L[i, j] = -L[i, j]
            E += dE  # / (N * N)
        if equilibrium_test == 'yes':
            E_plot.append(E)
        if x != 0 and x % (2 * tau) == 0:
            expE += E
            # print("at x = ",x,"  ", expE)
            measurements.append(E)  # Adds the measurement to the list
    expE /= BM
    # print(expE)

    # The Bootstrap Error Analysis
    resample = BM  # times to repeat re-sampling
    B_i = []  # for the calculation of <B> and sigma
    for y in range(resample):
        B = 0.0
        for z in range(int(BM)):
            n = random.randrange(0, BM)
            B += measurements[n]
        B /= BM
        B_i.append(B)

    # Now to calculate the Bootstrap sigma
    sigma_sigma = 0.0
    for w in range(resample):
        sigma_sigma += (B_i[w] - expE) ** 2
    sigma_sigma /= resample
    sigma_bootstrap = sqrt(sigma_sigma)

    return [T, expE, sigma_bootstrap, BM, E_plot]  # This will create a results matrix which can be plotted


def I_A_U_B(T):
    # To perform an equilbriium test, set the following variable to yes and plot the last array this function returns
    equilibrium_test = 'no'
    global N_global, E_measurements, tau_global
    J = 1
    N = N_global  # The lattice size: NxN
    tau = 2 * tau_global  # The correlation time
    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -4 * (N * N)  # Initial Value of Energy
    L1 = ones([N, N], int)  # Generates the lattice
    # JT = J / T  # The parameter J divided by T (temperature) Multiplying dE by this will potentially save time,
    # but if this is done, make sure to multiply dE by T again when doing E += dE/(N*N)

    print("N=", N, "; A-union-B Model at T=", T)
    expE = 0.0  # Expectation value of E
    measurements = []  # List of Measurements
    E_plot = []
    # Main Monte Carlo cycle
    for x in range(steps + 1):
        i = random.randrange(0, N)
        j = random.randrange(0, N)  # Picks a random starting location

        # Calculates change in energy
        dE = 0.0
        # Starts calculating the nearest neighbor sum at location a[ i-1 , j]
        neighbor = i - 1
        if neighbor > -1:
            dE += L1[neighbor, j]  # Checks if the neighbor is within the lattice
        else:
            dE += L1[N - 1, j]  # Periodic boundary conditions
        neighbor = i + 1
        if neighbor < N:
            dE += L1[neighbor, j]
        else:
            dE += L1[0, j]
        neighbor = j - 1
        if neighbor > -1:
            dE += L1[i, neighbor]
        else:
            dE += L1[i, N - 1]
        neighbor = j + 1
        if neighbor < N:
            dE += L1[i, neighbor]
        else:
            dE += L1[i, 0]
        dE *= 2 * J * 2 * L1[i, j]
        # print(E)
        # Calculates whether L[i,j] flips
        R = exp(-dE / T)
        if R > 1 or random.random() < R:
            L1[i, j] = -L1[i, j]
            E += dE  # / (N * N)
        if equilibrium_test == 'yes':
            E_plot.append(E)
        if x != 0 and x % (2 * tau) == 0:
            expE += E
            # print("at x = ",x,"  ", expE)
            measurements.append(E)  # Adds the measurement to the list
    expE /= BM
    # print(expE)

    # plot(E_plot)
    # The Bootstrap Error Analysis
    resample = BM  # times to repeat re-sampling
    B_i = []  # for the calculation of <B> and sigma
    for y in range(resample):
        B = 0.0
        for z in range(int(BM)):
            n = random.randrange(0, BM)
            B += measurements[n]
        B /= BM
        B_i.append(B)

    # Now to calculate the Bootstrap sigma
    sigma_sigma = 0.0
    for w in range(resample):
        sigma_sigma += (B_i[w] - expE) ** 2
    sigma_sigma /= resample
    sigma_bootstrap = sqrt(sigma_sigma)

    return [T, expE, sigma_bootstrap, BM,  E_plot]  # This will create a results matrix which can be plotted


def I_Replica_E(T):
    # To perform an equilbriium test, set the following variable to yes and plot the last array this function returns
    equilibrium_test = 'no'
    global N_global, tau_global, E_measurements
    J = 1
    JT = J / T  # The parameter J divided by T (temperature)
    N = N_global  # The lattice size: NxN
    tau = 2 * tau_global  # The correlation time
    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -4 * (N * N)  # Initial Value of Energy for each i,j of the "double layered lattice site"
    boundary = N // 2

    L1 = ones([N, N], int)  # Generates the lattices
    L2 = ones([N, N], int)
    A_1 = L1[:, 0:boundary]
    A_2 = L2[:, 0:boundary]
    B_1 = L1[:, boundary: N]
    B_2 = L2[:, boundary: N]

    print("N=", N, "; Replica Model at T=", T)
    E_plot = []
    expE = 0.0  # Expectation value of E
    measurements = []  # List of Measurements
    # Main Monte Carlo cycle
    for x in range(steps + 1):
        # Picks random lattice site
        i = random.randrange(0, N)
        j = random.randrange(0, N)

        # Decides the anticipated flip
        L1_update = L1[i, j]
        L2_update = L2[i, j]

        if random.random() < 0.5:
            L1_update = -L1[i, j]
        if random.random() < 0.5:
            L2_update = -L2[i, j]
        # locates whether the spin is in A or B
        if j < boundary:
            L2_update = L1_update

        # Calculates change in energy based off of the above anticipated flip
        nn1_sum = 0.0  # nearest neighbor for lattice L1
        nn2_sum = 0.0  # nearest neighbor for lattice L2
        # Starts calculating the nearest neighbor sum at location a[ i-1 , j]
        neighbor = i - 1
        if neighbor > -1:
            nn1_sum += L1[neighbor, j]
            nn2_sum += L2[neighbor, j]  # Checks if the neighbor is within the lattice
        else:
            nn1_sum += L1[N - 1, j]
            nn2_sum += L2[N - 1, j]  # Periodic boundary conditions
        neighbor = i + 1
        if neighbor < N:
            nn1_sum += L1[neighbor, j]
            nn2_sum += L2[neighbor, j]
        else:
            nn1_sum += L1[0, j]
            nn2_sum += L2[0, j]
        neighbor = j - 1
        if neighbor > -1:
            nn1_sum += L1[i, neighbor]
            nn2_sum += L2[i, neighbor]
        else:
            nn1_sum += L1[i, N - 1]
            nn2_sum += L2[i, N - 1]
        neighbor = j + 1
        if neighbor < N:
            nn1_sum += L1[i, neighbor]
            nn2_sum += L2[i, neighbor]
        else:
            nn1_sum += L1[i, 0]
            nn2_sum += L2[i, 0]
        dE = -JT * (L1_update - L1[i, j]) * nn1_sum - JT * (L2_update - L2[i, j]) * nn2_sum
        #    print("dE=", dE)
        R = exp(-dE)
        if R > 1 or random.random() < R:
            L1[i, j] = L1_update
            L2[i, j] = L2_update
            E += (dE * T)  # / (N * N)
        if equilibrium_test == 'yes':
            E_plot.append(E)
        if x != 0 and x % (2 * tau) == 0:
            expE += E
            #            print("at x = ", x, "  ", expE)
            measurements.append(E)  # Adds the measurement to the list
    expE /= BM
    #    print(expE)

    # The Bootstrap Error Analysis
    re_sample = BM  # times to repeat re-sampling
    B_i = []  # for the calculation of <B> and sigma
    for y in range(re_sample):
        B = 0.0
        for z in range(int(BM)):
            n = random.randrange(0, BM)
            B += measurements[n]
        B /= BM
        B_i.append(B)

    # Now to calculate the Bootstrap sigma
    sigma_sigma = 0.0
    for w in range(re_sample):
        sigma_sigma += (B_i[w] - expE) ** 2
    sigma_sigma /= re_sample
    sigma_bootstrap = sqrt(sigma_sigma)

    A_test = 'no'
    if A_test == 'yes':
        # tests if the replica's A sections match
        matches = 0.0
        for columns in range(0, boundary):
            for rows in range(N):
                if A_1[rows, columns] == A_2[rows, columns]:
                    matches += 1
        if matches == N * boundary:
            print("A_1 and A_2 are the same!")
        else:
            print("We messed up somewhere :(")

    return [T, expE, sigma_bootstrap, BM, E_plot]  # This will create a results matrix which can be plotted


# XY Model Raw Data functions
def XY_E(T):
    # To perform an equilbriium test, set the following variable to yes and plot the last array this function returns
    equilibrium_test = 'no'
    global N_global, E_measurements
    J = 1
    N = N_global  # The lattice size: NxN
    tau = tau_global  # The correlation time
    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -2 * (N * N)  # Initial Value of Energy since all spins start pointed up at \theta_i = 0.0
    L = zeros([N, N], float)  # Generates the lattice where each entry is a value of \theta_i

    # JT = J / T  # The parameter J divided by T (temperature) Multiplying dE by this will potentially save time,
    # but if this is done, make sure to multiply dE by T again when doing E += dE/(N*N)

    print("N=", N, "; Normal XY-Model at T=", T)
    E_plot = []
    expE = 0.0  # Expectation value of E
    measurements = []  # List of Measurements
    # Main Monte Carlo cycle
    for x in range(steps + 1):
        i = random.randrange(0, N)
        j = random.randrange(0, N)  # Picks a random starting location

        # Decides an anticipated spin amount
        L_update = random.random() * 2 * pi
        # Calculates change in energy that would occur if this spin was accepted
        dE = 0.0
        # Starts calculating the nearest neighbor sum at location L[ i-1 , j]
        neighbor = i - 1
        if neighbor > -1:
            dE += cos(L_update - L[neighbor, j]) - cos(
                L[i, j] - L[neighbor, j])  # Checks if the neighbor is within the lattice
        else:
            dE += cos(L_update - L[N - 1, j]) - cos(L[i, j] - L[N - 1, j])  # Periodic boundary conditions
        neighbor = i + 1
        if neighbor < N:
            dE += cos(L_update - L[neighbor, j]) - cos(L[i, j] - L[neighbor, j])
        else:
            dE += cos(L_update - L[0, j]) - cos(L[i, j] - L[0, j])
        neighbor = j - 1
        if neighbor > -1:
            dE += cos(L_update - L[i, neighbor]) - cos(L[i, j] - L[i, neighbor])
        else:
            dE += cos(L_update - L[i, N - 1]) - cos(L[i, j] - L[i, N - 1])
        neighbor = j + 1
        if neighbor < N:
            dE += cos(L_update - L[i, neighbor]) - cos(L[i, j] - L[i, neighbor])
        else:
            dE += cos(L_update - L[i, 0]) - cos(L[i, j] - L[i, 0])
        dE *= -J
        # Calculates whether L[i,j] rotates
        R = exp(-dE / T)
        if R > 1 or random.random() < R:
            L[i, j] = L_update
            E += dE  # / (N * N)
        if equilibrium_test == 'yes':
            E_plot.append(E)
        if x != 0 and x % (2 * tau) == 0:
            expE += E
            # print("at x = ",x,"  ", expE)
            measurements.append(E)  # Adds the measurement to the list
    expE /= BM
    # print(expE)

    # The Bootstrap Error Analysis
    resample = BM  # times to repeat re-sampling
    B_i = []  # for the calculation of <B> and sigma
    for y in range(resample):
        B = 0.0
        for z in range(int(BM)):
            n = random.randrange(0, BM)
            B += measurements[n]
        B /= BM
        B_i.append(B)

    # Now to calculate the Bootstrap sigma
    sigma_sigma = 0.0
    for w in range(resample):
        sigma_sigma += (B_i[w] - expE) ** 2
    sigma_sigma /= resample
    sigma_bootstrap = sqrt(sigma_sigma)

    return [T, E_plot, expE, sigma_bootstrap]  # This will create a results matrix which can be plotted


def XY_A_U_B(T):
    # To perform an equilbriium test, set the following variable to yes and plot the last array this function returns
    equilibrium_test = 'no'
    global N_global, E_measurements
    J = 1
    N = N_global  # The lattice size: NxN
    tau = tau_global  # The correlation time
    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -4 * (N * N)  # Initial Value of Energy since all spins start pointed up at \theta_i = 0.0
    L = zeros([N, N], float)  # Generates the lattice where each entry is a value of \theta_i

    # JT = J / T  # The parameter J divided by T (temperature) Multiplying dE by this will potentially save time,
    # but if this is done, make sure to multiply dE by T again when doing E += dE/(N*N)

    print("N=", N, "; A-union-B XY-Model at T=", T)
    E_plot = []
    expE = 0.0  # Expectation value of E
    measurements = []  # List of Measurements
    # Main Monte Carlo cycle
    for x in range(steps + 1):
        i = random.randrange(0, N)
        j = random.randrange(0, N)  # Picks a random starting location

        # Decides an anticipated spin amount
        L_update = random.random() * 2 * pi
        # Calculates change in energy that would occur if this spin was accepted
        dE = 0.0
        # Starts calculating the nearest neighbor sum at location L[ i-1 , j]
        neighbor = i - 1
        if neighbor > -1:
            dE += cos(L_update - L[neighbor, j]) - cos(
                L[i, j] - L[neighbor, j])  # Checks if the neighbor is within the lattice
        else:
            dE += cos(L_update - L[N - 1, j]) - cos(L[i, j] - L[N - 1, j])  # Periodic boundary conditions
        neighbor = i + 1
        if neighbor < N:
            dE += cos(L_update - L[neighbor, j]) - cos(L[i, j] - L[neighbor, j])
        else:
            dE += cos(L_update - L[0, j]) - cos(L[i, j] - L[0, j])
        neighbor = j - 1
        if neighbor > -1:
            dE += cos(L_update - L[i, neighbor]) - cos(L[i, j] - L[i, neighbor])
        else:
            dE += cos(L_update - L[i, N - 1]) - cos(L[i, j] - L[i, N - 1])
        neighbor = j + 1
        if neighbor < N:
            dE += cos(L_update - L[i, neighbor]) - cos(L[i, j] - L[i, neighbor])
        else:
            dE += cos(L_update - L[i, 0]) - cos(L[i, j] - L[i, 0])
        dE *= 2 * -J
        # Calculates whether L[i,j] rotates
        R = exp(-dE / T)
        if R > 1 or random.random() < R:
            L[i, j] = L_update
            E += dE  # / (N * N)
        if equilibrium_test == 'yes':
            E_plot.append(E)
        if x != 0 and x % (2 * tau) == 0:
            expE += E
            # print("at x = ",x,"  ", expE)
            measurements.append(E)  # Adds the measurement to the list
    expE /= BM
    # print(expE)

    # The Bootstrap Error Analysis
    resample = BM  # times to repeat re-sampling
    B_i = []  # for the calculation of <B> and sigma
    for y in range(resample):
        B = 0.0
        for z in range(int(BM)):
            n = random.randrange(0, BM)
            B += measurements[n]
        B /= BM
        B_i.append(B)

    # Now to calculate the Bootstrap sigma
    sigma_sigma = 0.0
    for w in range(resample):
        sigma_sigma += (B_i[w] - expE) ** 2
    sigma_sigma /= resample
    sigma_bootstrap = sqrt(sigma_sigma)

    return [T, sigma_bootstrap, E_plot, expE]  # This will create a results matrix which can be plotted


def XY_Replica_E(T):
    # To perform an equilbriium test, set the following variable to yes and plot the last array this function returns
    equilibrium_test = 'no'
    global N_global, E_measurements, tau_global, tau_after
    J = 1
    N = N_global  # The lattice size: NxN
    # A test to make things quicker; higher temperatures equilibrate faster
    tau = tau_global
    if T > 20:
        tau = tau_after
    BM = E_measurements  # Number of independent measurements for the bootstrap analysis
    steps = 2 * tau * BM  # Number of times the program will run
    E = -4 * (N * N)  # Initial Value of Energy since all spins start pointed up at \theta_i = 0.0
    boundary = N // 2
    L1 = zeros([N, N], float)  # Lattice 1 where each entry is a value of \theta_i
    L2 = zeros([N, N], float)  # Lattice 2
    A_1 = L1[:, 0:boundary]
    A_2 = L2[:, 0:boundary]
    B_1 = L1[:, boundary: N]
    B_2 = L2[:, boundary: N]

    print("N=", N, "; Replica XY-Model at T=", T)
    E_plot = []
    expE = 0.0  # Expectation value of E
    measurements = []  # List of Measurements
    # Main Monte Carlo cycle
    for x in range(steps + 1):
        i = random.randrange(0, N)
        j = random.randrange(0, N)  # Picks a random starting location

        # Decides an anticipated spin amount
        L1_update = random.random() * 2 * pi
        L2_update = random.random() * 2 * pi

        if j < boundary:
            L1_update = L2_update
        # Calculates change in energy that would occur if this spin was accepted
        dE = 0.0
        # Starts calculating the nearest neighbor sum at location L[ i-1 , j]
        neighbor = i - 1
        if neighbor > -1:
            dE += cos(L1_update - L1[neighbor, j]) - cos(L1[i, j] - L1[neighbor, j])
            dE += cos(L2_update - L2[neighbor, j]) - cos(L2[i, j] - L2[neighbor, j])
        else:
            dE += cos(L1_update - L1[N - 1, j]) - cos(L1[i, j] - L1[N - 1, j])  # Periodic boundary conditions
            dE += cos(L2_update - L2[N - 1, j]) - cos(L2[i, j] - L2[N - 1, j])
        neighbor = i + 1
        if neighbor < N:
            dE += cos(L1_update - L1[neighbor, j]) - cos(L1[i, j] - L1[neighbor, j])
            dE += cos(L2_update - L2[neighbor, j]) - cos(L2[i, j] - L2[neighbor, j])
        else:
            dE += cos(L1_update - L1[0, j]) - cos(L1[i, j] - L1[0, j])
            dE += cos(L2_update - L2[0, j]) - cos(L2[i, j] - L2[0, j])
        neighbor = j - 1
        if neighbor > -1:
            dE += cos(L1_update - L1[i, neighbor]) - cos(L1[i, j] - L1[i, neighbor])
            dE += cos(L2_update - L2[i, neighbor]) - cos(L2[i, j] - L2[i, neighbor])
        else:
            dE += cos(L1_update - L1[i, N - 1]) - cos(L1[i, j] - L1[i, N - 1])
            dE += cos(L2_update - L2[i, N - 1]) - cos(L2[i, j] - L2[i, N - 1])
        neighbor = j + 1
        if neighbor < N:
            dE += cos(L1_update - L1[i, neighbor]) - cos(L1[i, j] - L1[i, neighbor])
            dE += cos(L2_update - L2[i, neighbor]) - cos(L2[i, j] - L2[i, neighbor])
        else:
            dE += cos(L1_update - L1[i, 0]) - cos(L1[i, j] - L1[i, 0])
            dE += cos(L2_update - L2[i, 0]) - cos(L2[i, j] - L2[i, 0])
        dE *= -J

        # Calculates whether L[i,j] rotates
        R = exp(-dE / T)
        if R > 1 or random.random() < R:
            L1[i, j] = L1_update
            L2[i, j] = L2_update
            E += dE  # / (N * N)
        if equilibrium_test == 'yes':
            E_plot.append(E)
        if x != 0 and x % (2 * tau) == 0:
            expE += E
            # print("at x = ",x,"  ", expE)
            measurements.append(E)  # Adds the measurement to the list
    expE /= BM
    # print(expE)

    # The Bootstrap Error Analysis
    resample = BM  # times to repeat re-sampling
    B_i = []  # for the calculation of <B> and sigma
    for y in range(resample):
        B = 0.0
        for z in range(int(BM)):
            n = random.randrange(0, BM)
            B += measurements[n]
        B /= BM
        B_i.append(B)

    # Now to calculate the Bootstrap sigma
    sigma_sigma = 0.0
    for w in range(resample):
        sigma_sigma += (B_i[w] - expE) ** 2
    sigma_sigma /= resample
    sigma_bootstrap = sqrt(sigma_sigma)

    # This is a test to make sure that A_1 and A_2 are indeed being updated the same.
    equivalence_test = 'no'
    if equivalence_test == 'yes':
        matches = 0.0
        for columns in range(0, boundary):
            for rows in range(N):
                if A_1[rows, columns] == A_2[rows, columns]:
                    matches += 1
        if matches == N * boundary:
            print("A_1 and A_2 are the same!")
        else:
            print("We messed up somewhere :(")

    return [T, expE, sigma_bootstrap, E_plot]  # This will create a results matrix which can be plotted