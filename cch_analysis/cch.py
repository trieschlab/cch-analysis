import numpy as np
from collections.abc import Iterable
import matplotlib.pyplot as plt
from .tools import hollow_kernel, poisson_continuity_correction, histogram

def correlogram(t1, t2=None, bin_size=.001, limit=.02, auto=False,
                density=False):
    """Calculate cross correlation histogram of two spike trains.
    Essentially, this algorithm subtracts each spike time in `t1`
    from all of `t2` and bins the results with np.histogram, though
    several tweaks were made for efficiency.
    Originally authored by Chris Rodger, copied from OpenElectrophy, licenced
    with CeCill-B.
    Parameters
    ---------
    t1 : np.array
        First spiketrain, raw spike times in seconds.
    t2 : np.array
        Second spiketrain, raw spike times in seconds.
    bin_size : float
        Width of each bar in histogram in seconds.
    limit : float
        Positive and negative extent of histogram, in seconds.
    auto : bool
        If True, then returns autocorrelogram of `t1` and in
        this case `t2` can be None. Default is False.
    density : bool
        If True, then returns the probability density function.
    See also
    --------
    :func:`numpy.histogram` : The histogram function in use.
    Returns
    -------
    (count, bins) : tuple
        A tuple containing the bin right edges and the
        count/density of spikes in each bin.
    Note
    ----
    `bins` are relative to `t1`. That is, if `t1` leads `t2`, then
    `count` will peak in a positive time bin.
    Examples
    --------
    >>> t1 = np.arange(0, .5, .1)
    >>> t2 = np.arange(0.1, .6, .1)
    >>> limit = 1
    >>> bin_size = .1
    >>> counts, bins = correlogram(t1=t1, t2=t2, bin_size=bin_size,
    ...                            limit=limit, auto=False)
    """
    if auto: t2 = t1
    # For auto-CCGs, make sure we use the same exact values
    # Otherwise numerical issues may arise when we compensate for zeros later
    if not int(limit * 1e10) % int(bin_size * 1e10) == 0:
        raise ValueError(
            'Time limit {} must be a '.format(limit) +
            'multiple of bin_size {}'.format(bin_size) +
            ' remainder = {}'.format(limit % bin_size))
    # For efficiency, `t1` should be no longer than `t2`
    swap_args = False
    if len(t1) > len(t2):
        swap_args = True
        t1, t2 = t2, t1

    # Sort both arguments (this takes negligible time)
    t1 = np.sort(t1)
    t2 = np.sort(t2)

    # Determine the bin edges for the histogram
    # Later we will rely on the symmetry of `bins` for undoing `swap_args`
    limit = float(limit)

    # The numpy.arange method overshoots slightly the edges i.e. bin_size + epsilon
    # which leads to inclusion of spikes falling on edges.
    bins = np.arange(-limit, limit + bin_size, bin_size)

    # Determine the indexes into `t2` that are relevant for each spike in `t1`
    ii2 = np.searchsorted(t2, t1 - limit)
    jj2 = np.searchsorted(t2, t1 + limit)

    # Concatenate the recentered spike times into a big array
    # We have excluded spikes outside of the histogram range to limit
    # memory use here.
    big = np.concatenate([t2[i:j] - t for t, i, j in zip(t1, ii2, jj2)])

    # Actually do the histogram. Note that calls to np.histogram are
    # expensive because it does not assume sorted data. Therefore we use
    # the local histogram function
    count, bins = histogram(big, bins=bins, density=density)

    if auto:
        # Compensate for the peak at time zero that results in autocorrelations
        # by subtracting the total number of spikes from that bin. Note
        # possible numerical issue here because 0.0 may fall at a bin edge.
        c_temp, bins_temp = np.histogram([0.], bins=bins)
        bin_containing_zero = np.nonzero(c_temp)[0][0]
        count[bin_containing_zero] = 0#-= len(t1)

    # Finally compensate for the swapping of t1 and t2
    if swap_args:
        # Here we rely on being able to simply reverse `counts`. This is only
        # possible because of the way `bins` was defined (bins = -bins[::-1])
        count = count[::-1]

    return count, bins


def fit_latency(pre, post, bin_size=.1e-3, limit=20e-3, init=[5e-4, 5e-4], plot=False):
    '''
    Fit a gaussian PDF to density of CCH
    '''
    from scipy.optimize import leastsq
    import scipy.stats as st
    c, b = correlogram(pre, post, bin_size=bin_size, limit=limit, density=True)
    b = b[1:]
    normpdf  = lambda p, x: st.norm.pdf(x, *p)
    error  = lambda p, x, y: (y - normpdf(p, x))
    (delta_t, sigma), _ = leastsq(error, init, args=(b, c))
    if plot:
        import matplotlib.pyplot as plt
        plt.bar(b, c, width=-bin_size, align='edge')
        y = normpdf((delta_t, sigma), b)
        plt.plot(b, y, 'r--', linewidth=2)
        plt.title('$\Delta t$ {:.3f} $\sigma$ {:.3f}'.format(delta_t, sigma))
        plt.axvspan(delta_t - sigma, delta_t + sigma, alpha=.5, color='cyan')
    return delta_t, sigma


def cch_convolve(cch, width, hollow_fraction):
    '''Convolve a cross correlation histogram (cch) with a hollow kernel as in
    _[1].

    Copied from CINPLA/causal-optoconnectics/
    Parameters
    ----------
    cch : array
        The cross correlation histogram
    width : float
        Width of kernel (std if gaussian)
    hollow_fraction : float
        Fractoin of the central bin to removed.
    Authors
    -------
    Tristan Stoeber, Mikkel Lepperød
    '''
    import scipy.signal as scs
    kernlen = len(cch) - 1
    kernel = hollow_kernel(kernlen, width, hollow_fraction)
    # padd edges
    len_padd = int(kernlen / 2.)
    cch_padded = np.zeros(len(cch) + 2 * len_padd)
    # "firstW/2 bins (excluding the very first bin) are duplicated,
    # reversed in time, and prepended to the cch prior to convolving"
    cch_padded[0:len_padd] = cch[1:len_padd+1][::-1]
    cch_padded[len_padd: - len_padd] = cch
    # # "Likewise, the lastW/2 bins aresymmetrically appended to the cch."
    cch_padded[-len_padd:] = cch[-len_padd-1:-1][::-1]
    # convolve cch with kernel
    result = scs.fftconvolve(cch_padded, kernel, mode='valid')
    assert len(cch) == len(result)
    return result


def cch_significance(t1, t2, bin_size, limit, hollow_fraction, width):
    """
    Parameters
    ---------
    t1 : array
        First spiketrain, raw spike times in seconds.
    t2 : array
        Second spiketrain, raw spike times in seconds.
    bin_size : float
        Width of each bar in histogram in seconds.
    limit : float
        Positive and negative extent of histogram, in seconds.
    kernlen : int
        Length of kernel, must be uneven (kernlen % 2 == 1)
    width : float
        Width of kernel (std if gaussian)
    hollow_fraction : float
        Fraction of the central bin to removed.
    References
    ----------
    Stark, E., & Abeles, M. (2009). Unbiased estimation of precise temporal
    correlations between spike trains. Journal of neuroscience methods, 179(1),
    90-100.
    English et al. 2017, Neuron, Pyramidal Cell-Interneuron Circuit Architecture
    and Dynamics in Hippocampal Networks
    Authors
    -------
    Copied from CINPLA/causal-optoconnectics/
    Tristan Stoeber, Mikkel Lepperød
    """
    cch, bins = correlogram(
        t1, t2, bin_size=bin_size, limit=limit, density=False)
    pfast = np.zeros(cch.shape)
    cch_smooth = cch_convolve(
        cch=cch, width=width, hollow_fraction=hollow_fraction)
    pfast = poisson_continuity_correction(cch, cch_smooth)
    # ppeak describes the probability of obtaining a peak with positive lag
    # of the histogram, that is signficantly larger than the largest peak
    # in the negative lag direction.
    ppeak = np.zeros(cch.shape)
    max_vals = np.zeros(cch.shape)
    cch_half_len = int(np.floor(len(cch) / 2.))
    max_vals[cch_half_len:] = np.max(cch[:cch_half_len])
    max_vals[:cch_half_len] = np.max(cch[cch_half_len:])
    ppeak = poisson_continuity_correction(cch, max_vals)
    return ppeak, pfast, bins, cch, cch_smooth

