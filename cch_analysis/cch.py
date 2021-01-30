import numpy as np
from collections.abc import Iterable
import matplotlib.pyplot as plt

def histogram(val, bins, density=False):
    '''Fast histogram
    Assuming:
        val, bins are sorted
        bins increase monotonically and uniformly
        all(bins[0] <= v <= bins[-1] for v in val)
        Copied from CINPLA/causal-optoconnectics/
    '''
    result = np.zeros(len(bins) - 1).astype(int)
    search = np.searchsorted(bins, val, side='right')
    cnt = np.bincount(search)[1:len(result)]
    result[:len(cnt)] = cnt
    if density:
        db = np.array(np.diff(bins), float)
        return result / db / result.sum(), bins
    return result, bins


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
    limit : float, list
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

    # allow to set start and stop of limit
    if not isinstance(limit, Iterable):
        limit = [-limit, limit]
    
    # For auto-CCGs, make sure we use the same exact values
    # Otherwise numerical issues may arise when we compensate for zeros later
    for lim in limit:
        if not int(lim * 1e10) % int(bin_size * 1e10) == 0:
            raise ValueError(
                'Time limit {} must be a '.format(lim) +
                'multiple of bin_size {}'.format(bin_size) +
                ' remainder = {}'.format(lim % bin_size))
    
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
    limit = [float(lim) for lim in limit]

    # The numpy.arange method overshoots slightly the edges i.e. bin_size + epsilon
    # which leads to inclusion of spikes falling on edges.
    bins = np.arange(limit[0], limit[1] + bin_size, bin_size)

    # Determine the indexes into `t2` that are relevant for each spike in `t1`
    ii2 = np.searchsorted(t2, t1 + limit[0])
    jj2 = np.searchsorted(t2, t1 + limit[1])

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


def fit_latency(pre, post, bin_size=.1e-3, limit=20e-3, init=[5e-4, 5e-4], plot=False, ax=None):
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
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.bar(b, c, width=-bin_size, align='edge')
        y = normpdf((delta_t, sigma), b)
        ax.plot(b, y, 'r--', linewidth=2)
        ax.set_title('$\Delta t$ {:.3f} $\sigma$ {:.3f}'.format(delta_t, sigma))
        ax.axvspan(delta_t - sigma, delta_t + sigma, alpha=.5, color='cyan')
    return delta_t, sigma
