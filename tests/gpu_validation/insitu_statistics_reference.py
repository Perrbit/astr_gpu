"""Independent two-pass quadrature oracle for the bounded periodic TGV gate."""

from pathlib import Path
import numpy as np


def read_statistics(path):
    with Path(path).open('rb') as stream:
        if stream.read(8) != b'ASTRST01':
            raise ValueError('Invalid statistics signature')
        header = np.fromfile(stream, '<i4', 9)
        metadata = np.fromfile(stream, '<f8', 10)
        values = np.fromfile(stream, '<f8').reshape((*header[3:6], 41), order='F')
    if not np.isfinite(values).all() or not np.isfinite(metadata).all():
        raise ValueError('Nonfinite statistics')
    return header, metadata, values


def endpoint_weights(times, window):
    times = np.asarray(times, dtype=np.longdouble)
    weights = np.zeros(len(times), dtype=np.longdouble)
    for i, (t0, t1) in enumerate(zip(times[:-1], times[1:])):
        lo, hi = max(t0, window[0]), min(t1, window[1])
        if hi <= lo:
            continue
        right = ((lo-t0)+(hi-t0))/(2*(t1-t0))
        weights[i] += (hi-lo)*(1-right)
        weights[i+1] += (hi-lo)*right
    return weights


def reference(samples, weights):
    # Center only after forming the complete independent quadrature mean.
    rho = np.asarray(samples[..., 5], dtype=np.longdouble)
    velocity = np.asarray(samples[..., 6:9], dtype=np.longdouble)
    w = weights.reshape((-1,)+(1,)*(rho.ndim-1))
    mass = np.sum(w*rho, axis=0)
    duration = weights.sum()
    mean_r = np.sum(w[..., None]*velocity, axis=0)/duration
    mean_f = np.sum((w*rho)[..., None]*velocity, axis=0)/mass[..., None]
    dr, df = velocity-mean_r, velocity-mean_f
    cov_r = np.einsum('t...,t...i,t...j->...ij', w, dr, dr)/duration
    cov_f = np.einsum('t...,t...i,t...j->...ij', w*rho, df, df)/mass[..., None, None]
    density = mass/duration
    shape = density.shape
    flat = lambda a: a.swapaxes(-1, -2).reshape((*shape, 9))
    return np.concatenate([
        np.full((*shape, 1), duration), density[..., None], mean_r, mean_f,
        np.sqrt(np.diagonal(cov_r, axis1=-2, axis2=-1)),
        np.sqrt(np.diagonal(cov_f, axis1=-2, axis2=-1)),
        flat(cov_r), flat(cov_f), flat(density[..., None, None]*cov_f)], axis=-1)


def compare(actual, expected):
    # Approved D7: test squared RMS and report unsquared differences separately.
    a, b = actual.copy(), expected.copy()
    a[..., 8:14] **= 2
    b[..., 8:14] **= 2
    np.testing.assert_allclose(a, b, atol=2e-10, rtol=0)
    return {'max_abs': float(np.max(np.abs(a-b))),
            'rms_max_abs': float(np.max(np.abs(actual[..., 8:14]-expected[..., 8:14])))}


def check_case(case, ranks, read_sample, device=False):
    directory = Path(case)/'outdat'
    result = []
    outputs = sorted(directory.glob('sample.statistics.step*.rank00000000.bin'))
    if len(outputs) != 3:
        raise ValueError('Expected three covered statistics outputs')
    for path in outputs:
        h, metadata, _ = read_statistics(path)
        step = int(h[1])
        times = [s*1e-3 for s in range(step+1)]
        np.testing.assert_array_equal(metadata[:3], [times[-1], 0.0005, 0.0025])
        weights = endpoint_weights(times, metadata[1:3])
        regional_signal = np.zeros((step+1, 11), dtype=np.longdouble)
        regional_signal[:, 5] = 1
        variance_sum = np.zeros(3, dtype=np.longdouble)
        ownership = np.zeros((32,32,32), dtype=int)
        for rank in range(ranks):
            suffix = f'.rank{rank:08d}.bin'
            hh, mm, actual = read_statistics(directory/f'sample.statistics.step{step:08d}{suffix}')
            np.testing.assert_array_equal(mm, metadata)
            samples = np.stack([read_sample(directory/f'sample.canonical.step{s:08d}{suffix}',
                                           canonical=True)[2] for s in range(step+1)])
            expected = reference(samples, weights)
            result.append(compare(actual, expected))
            if device:
                dh, dm, dv = read_statistics(directory/f'sample.device_statistics.step{step:08d}{suffix}')
                expected_header = hh.copy()
                expected_header[3:6] -= 1
                np.testing.assert_array_equal(dh, expected_header)
                np.testing.assert_array_equal(dm, mm)
                result.append({'device_oracle': compare(dv, expected[:-1,:-1,:-1]),
                               'device_host': compare(dv, actual[:-1,:-1,:-1])})
            np.testing.assert_allclose(actual[...,0], float(weights.sum()), atol=1e-18, rtol=0)
            offset, extent = hh[6:9], hh[3:6]-1
            region = tuple(slice(int(o), int(o+n)) for o,n in zip(offset, extent))
            ownership[region] += 1
            regional_signal[:,6:9] += samples[:,:-1,:-1,:-1,6:9].astype(np.longdouble).sum(axis=(1,2,3))/32**3
            variance_sum += expected[:-1,:-1,:-1,[14,18,22]].sum(axis=(0,1,2))/32**3
        np.testing.assert_array_equal(ownership, np.ones_like(ownership))
        regional = reference(regional_signal, weights)
        np.testing.assert_allclose(metadata[3], (2*np.pi)**3, atol=2e-10, rtol=0)
        np.testing.assert_allclose(metadata[4:7]**2, variance_sum, atol=2e-10, rtol=0)
        np.testing.assert_allclose(metadata[7:10]**2, regional[8:11]**2, atol=2e-10, rtol=0)
    return result
