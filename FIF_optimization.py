"""
 Fast Iterative Filtering python package

 Dependencies : numpy, scipy, numba

 Authors:
    Python version: Emanuele Papini - INAF (emanuele.papini@inaf.it)
    Original matlab version: Antonio Cicone - university of L'Aquila (antonio.cicone@univaq.it)
"""
import os
import numpy as np
from numpy import linalg as LA
from scipy.io import loadmat
from scipy.signal import argrelextrema
from numpy import fft
from numba import jit
import pyfftw

__version__ = '3.2'


# WRAPPER (version unaware. To be called by FIF_optimization.py)
def FIF_run(x, *args, options=None, M=np.array([]), **kwargs):
    if options is None:
        return FIF(x, *args, **kwargs)
    else:
        return FIF(x, options['delta'], options['alpha'], options['NumSteps'], \
                   options['ExtPoints'], options['NIMFs'], options['MaxInner'], Xi=options['Xi'], \
                   M=M, \
                   MonotoneMaskLength=options['MonotoneMaskLength'], verbose=options['verbose'], **kwargs)


def get_mask_v1_1(y, k):
    """
    Rescale the mask y so that its length becomes 2*k+1.
    k could be an integer or not an integer.
    y is the area under the curve for each bar
    """
    n = np.size(y)
    m = (n - 1) // 2
    k = int(k)

    if k <= m:
        indices = np.linspace(0, 2 * m, 2 * k + 1)
        interpolated_y = np.interp(indices, np.arange(n), y)
        return interpolated_y / np.sum(interpolated_y)
    else:
        # For larger values of k, use interpolation as before
        dx = 0.01
        f = y / dx
        dy = m * dx / k
        b = np.interp(np.linspace(0, m, int(np.ceil(k + 1))), np.linspace(0, m, m + 1), f[m:2 * m + 1])
        a = np.concatenate((np.flipud(b[1:]), b)) * dy

        if abs(LA.norm(a, 1) - 1) > 1e-14:
            a = a / LA.norm(a, 1)

        return a

def FFT_with_pyFFTW(signal, threads=4):
    # Create pyFFTW object for FFT
    fft_object = pyfftw.builders.fft(signal, threads=threads)
    return fft_object()

def IFFT_with_pyFFTW(signal, threads=4):
    # Create pyFFTW object for IFFT
    ifft_object = pyfftw.builders.ifft(signal, threads=threads)
    return ifft_object()

def FIF(x, delta, alpha, NumSteps, ExtPoints, NIMFs, MaxInner, Xi=1.6, M=np.array([]), MonotoneMaskLength=True,
        verbose=False, window_file='prefixed_double_filter.mat'):
    f = np.asarray(x)
    N = f.size
    IMF = np.zeros([NIMFs, N])
    Norm1f = np.max(np.abs(f))  # LA.norm(f, np.inf)
    f = f / Norm1f

    ###############################################################
    #                   Iterative Filtering                       #
    ###############################################################
    MM = loadmat(window_file)['MM'].flatten()

    ### Create a signal without zero regions and compute the number of extrema ###
    f_pp = np.delete(f, np.argwhere(abs(f) <= 1e-18))
    maxmins_pp = Maxmins_v3_4(f_pp, mode='wrap')
    maxmins_pp = maxmins_pp[0]
    diffMaxmins_pp = np.diff(maxmins_pp)

    N_pp = len(f_pp)
    k_pp = maxmins_pp.shape[0]
    countIMFs = 0
    stats_list = []

    ### Begin Iterating ###
    while countIMFs < NIMFs and k_pp >= ExtPoints:
        countIMFs += 1
        print('IMF', countIMFs)

        SD = 1
        h = f

        if 'M' not in locals() or np.size(M) < countIMFs:

            if isinstance(alpha, str):

                if alpha == 'ave':
                    m = 2 * np.round(N_pp / k_pp * Xi)
                elif alpha == 'Almost_min':
                    m = 2 * np.min([Xi * np.percentile(diffMaxmins_pp, 30), np.round(N_pp / k_pp * Xi)])
                else:
                    raise Exception('Value of alpha not recognized!\n')

            else:
                m = 2 * np.round(
                    Xi * (np.max(diffMaxmins_pp) * alpha / 100 + np.min(diffMaxmins_pp) * (1 - alpha / 100)))

            if countIMFs > 1:
                if m <= stats['logM'][-1]:
                    if verbose:
                        print('Warning mask length is decreasing at step %1d. ' % countIMFs)
                    if MonotoneMaskLength:
                        m = np.ceil(stats['logM'][-1] * 1.1)
                        if verbose:
                            print(('The old mask length is %1d whereas the new one is forced to be %1d.\n' % (
                                stats['logM'][-1], np.ceil(stats['logM'][-1]) * 1.1)))
                    else:
                        if verbose:
                            print('The old mask length is %1d whereas the new one is %1d.\n' % (stats['logM'][-1], m))
        else:
            m = M[countIMFs - 1]

        inStepN = 0
        if verbose:
            print('\n IMF # %1.0d   -   # Extreme points %5.0d\n' % (countIMFs, k_pp))
            print('\n  step #            SD             Mask length \n\n')

        stats = {'logM': [], 'posF': [], 'valF': [], 'inStepN': [], 'diffMaxmins_pp': []}
        stats['logM'].append(int(m))

        a = get_mask_v1_1(MM, m)  #
        ExtendSig = False

        if N < np.size(a):
            ExtendSig = True
            Nxs = int(np.ceil(np.size(a) / N))
            N_old = N
            if np.mod(Nxs, 2) == 0:
                Nxs = Nxs + 1

            h_n = np.hstack([h] * Nxs)

            h = h_n
            N = Nxs * N

        Nza = N - np.size(a)
        if np.mod(Nza, 2) == 0:
            a = np.concatenate((np.zeros(Nza // 2), a, np.zeros(Nza // 2)))
            l_a = a.size
            fftA = FFT_with_pyFFTW(np.roll(a, a.size // 2))
        else:
            a = np.concatenate((np.zeros((Nza - 1) // 2), a, np.zeros((Nza - 1) // 2 + 1)))
            l_a = a.size
            fftA = FFT_with_pyFFTW(np.roll(a, a.size // 2))
        fftA = np.clip(fftA, 1e-12, 1 - 1e-12)
        fftH = FFT_with_pyFFTW(h)
        fft_h_new = fftH.copy()

        # determine number of iterations necessary for convergence
        compute_n_its = True
        if compute_n_its:
            fh = np.abs(fftH);
            fm = abs(fftA)

            r = lambda inStepN: evaluate_residual(fh, fm, inStepN) - delta

            inStepN = 1
            res = np.inf
            omfm = 1 - fm
            r2 = fh
            r1 = r2 * fm
            if np.any(np.isnan(r1)) or np.any(np.isnan(r2)):
                raise RuntimeError("NaN encountered in residual calculation.")
            while res > 0:
                r2 *= omfm
                r1 *= omfm
                omfm = omfm ** 2
                res = np.sum(r1 ** 2) / np.sum(r2 ** 2) - delta
                inStepN *= 2

            jl = inStepN // 2;
            jr = inStepN
            rm = res;
            jm = jr
            while jr > jl + 1:
                jm = np.floor((jl + jr) / 2)
                rm = r(jm)
                if rm < 0:
                    jr = jm
                else:
                    jl = jm

            inStepN = jm if rm < 0 else jm + 1

        fft_h_old = fft_h_new.copy()
        fft_h_new = (1 - fftA) ** inStepN * fftH

        SD = LA.norm(fft_h_new - fft_h_old) ** 2 / LA.norm(fft_h_old) ** 2

        ############### Generating f_n #############
        if verbose:
            print('    %2.0d      %1.40f          %2.0d\n' % (inStepN, SD, m))

        h = FFT_with_pyFFTW(fft_h_new)
        if ExtendSig:
            N = N_old
            h = h[int(N * (Nxs - 1) / 2):int(N * ((Nxs - 1) / 2 + 1))]

        if inStepN >= MaxInner:
            print('Max # of inner steps reached')

        stats['inStepN'] = inStepN
        h = np.real(h)
        IMF[countIMFs - 1, :] = h
        f = f - h

        #### Create a signal without zero regions and compute the number of extrema ####

        f_pp = np.delete(f, np.argwhere(abs(f) <= 1e-18))
        maxmins_pp = Maxmins_v3_4(f_pp, mode='wrap')[0]
        if maxmins_pp is None:
            break

        diffMaxmins_pp = np.diff(maxmins_pp)
        N_pp = np.size(f_pp)
        k_pp = maxmins_pp.shape[0]

        stats_list.append(stats)

    IMF = IMF[0:countIMFs + 1, :]
    IMF[-1, :] = f[:]

    IMF = IMF * Norm1f  # We scale back to the original values

    return IMF, stats_list


def Maxmins_v3_4(x, mode='wrap'):
    @jit(nopython=True)
    def maxmins_wrap(x, df, N, Maxs, Mins):

        h = 1
        while h < N and np.abs(df[h]) <= tol:
            h = h + 1

        if h == N:
            return None, None

        cmaxs = 0
        cmins = 0
        c = 0
        N_old = N
        df = np.zeros(N + h + 2)
        df[0:N] = x
        df[N + 1:N + h + 2] = x[1:h + 2]
        for i in range(N + h + 1):
            df[i] = df[i + 1] - df[i]
        N = N + h
        # beginfor
        for i in range(h - 1, N - 2):
            if df[i] * df[i + 1] <= tol and df[i] * df[i + 1] >= -tol:
                if df[i] < -tol:
                    last_df = -1
                    posc = i
                elif df[i] > tol:
                    last_df = +1
                    posc = i
                c = c + 1

                if df[i + 1] < -tol:
                    if last_df == 1:
                        cmaxs = cmaxs + 1
                        Maxs[cmaxs] = (posc + (c - 1) // 2 + 1) % N_old
                    c = 0

                if df[i + 1] > tol:
                    if last_df == -1:
                        cmins = cmins + 1
                        Mins[cmins] = (posc + (c - 1) // 2 + 1) % N_old
                    c = 0

            if df[i] * df[i + 1] < -tol:
                if df[i] < -tol and df[i + 1] > tol:
                    cmins = cmins + 1
                    Mins[cmins] = (i + 1) % N_old
                    if Mins[cmins] == 0:
                        Mins[cmins] = 1
                    last_df = -1

                elif df[i] > tol and df[i + 1] < -tol:
                    cmaxs = cmaxs + 1
                    Maxs[cmaxs] = (i + 1) % N_old
                    if Maxs[cmaxs] == 0:
                        Maxs[cmaxs] = 1

                    last_df = +1

        # endfor
        if c > 0:
            if Mins[cmins] == 0: Mins[cmins] = N
            if Mins[cmaxs] == 0: Mins[cmaxs] = N

        return Maxs[0:cmaxs], Mins[0:cmins]

    tol = 1e-15
    N = np.size(x)

    Maxs = np.zeros(N)
    Mins = np.zeros(N)

    df = np.diff(x)

    if mode == 'wrap':
        Maxs, Mins = maxmins_wrap(x, df, N, Maxs, Mins)
        if Maxs is None or Mins is None:
            return None, None, None

        maxmins = np.sort(np.concatenate((Maxs, Mins)))

        if any(Mins == 0): Mins[Mins == 0] = 1
        if any(Maxs == 0): Maxs[Maxs == 0] = 1
        if any(maxmins == 0): maxmins[maxmins == 0] = 1

    return maxmins, Maxs, Mins


def evaluate_residual(fh, fm, j):
    r2 = fh * (1 - fm) ** (j - 1)
    r1 = r2 * fm
    return np.sum(r1 ** 2) / np.sum(r2 ** 2)


def Maxmins_v3_4(x, mode='wrap'):
    @jit(nopython=True)
    def maxmins_wrap(x, df, N, Maxs, Mins):

        h = 1
        while h < N and np.abs(df[h]) <= tol:
            h = h + 1

        if h == N:
            return None, None

        cmaxs = 0
        cmins = 0
        c = 0
        N_old = N
        df = np.zeros(N + h + 2)
        df[0:N] = x
        df[N + 1:N + h + 2] = x[1:h + 2]
        for i in range(N + h + 1):
            df[i] = df[i + 1] - df[i]
        N = N + h
        # beginfor
        for i in range(h - 1, N - 2):
            if df[i] * df[i + 1] <= tol and df[i] * df[i + 1] >= -tol:
                if df[i] < -tol:
                    last_df = -1
                    posc = i
                elif df[i] > tol:
                    last_df = +1
                    posc = i
                c = c + 1

                if df[i + 1] < -tol:
                    if last_df == 1:
                        cmaxs = cmaxs + 1
                        Maxs[cmaxs] = (posc + (c - 1) // 2 + 1) % N_old
                    c = 0

                if df[i + 1] > tol:
                    if last_df == -1:
                        cmins = cmins + 1
                        Mins[cmins] = (posc + (c - 1) // 2 + 1) % N_old
                    c = 0

            if df[i] * df[i + 1] < -tol:
                if df[i] < -tol and df[i + 1] > tol:
                    cmins = cmins + 1
                    Mins[cmins] = (i + 1) % N_old
                    if Mins[cmins] == 0:
                        Mins[cmins] = 1
                    last_df = -1

                elif df[i] > tol and df[i + 1] < -tol:
                    cmaxs = cmaxs + 1
                    Maxs[cmaxs] = (i + 1) % N_old
                    if Maxs[cmaxs] == 0:
                        Maxs[cmaxs] = 1

                    last_df = +1

        # endfor
        if c > 0:
            if Mins[cmins] == 0: Mins[cmins] = N
            if Mins[cmaxs] == 0: Mins[cmaxs] = N

        return Maxs[0:cmaxs], Mins[0:cmins]

    tol = 1e-15
    N = np.size(x)

    Maxs = np.zeros(N)
    Mins = np.zeros(N)

    df = np.diff(x)

    if mode == 'wrap':
        Maxs, Mins = maxmins_wrap(x, df, N, Maxs, Mins)
        if Maxs is None or Mins is None:
            return None, None, None

        maxmins = np.sort(np.concatenate((Maxs, Mins)))

        if any(Mins == 0): Mins[Mins == 0] = 1
        if any(Maxs == 0): Maxs[Maxs == 0] = 1
        if any(maxmins == 0): maxmins[maxmins == 0] = 1

    return maxmins, Maxs, Mins


def evaluate_residual(fh, fm, j):
    r2 = fh * (1 - fm) ** (j - 1)
    r1 = r2 * fm
    return np.sum(r1 ** 2) / np.sum(r2 ** 2)