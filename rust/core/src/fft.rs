//! In-house radix-2 FFT, just enough for the 0-1 autocovariance.
//!
//! The kernels that need a transform are few and small, so the crate carries
//! its own Cooley–Tukey pass rather than a dependency. Only power-of-two
//! lengths are supported; callers zero-pad up. Twiddle factors come from a
//! table of `cos`/`sin` of `2πk/M` computed directly, not by recurrence, so
//! the rounding stays near machine precision.
//!
//! The transform is unnormalised in the forward direction and divided by `M`
//! on the inverse, matching `numpy.fft`: `fft` then `fft(inverse = true)`
//! returns the input.

/// The `M`-th roots of unity `exp(-2πi k / M)` for `k = 0..M/2`, split into
/// cosine and sine tables.
///
/// Every twiddle a radix-2 pass of length `M` needs is one of these roots:
/// stage `m` uses `exp(-2πi j / m) = W_M^{j·(M/m)}` for `j < m/2`. The table
/// depends only on `M`, so a caller running many transforms of one size
/// builds it once.
pub(crate) struct Roots {
    /// `cos(2πk/M)` for `k = 0..M/2`.
    re: Vec<f64>,
    /// `-sin(2πk/M)` for `k = 0..M/2`.
    im: Vec<f64>,
}

impl Roots {
    /// Table for a transform of length `m`; `m` must be a power of two.
    pub(crate) fn of_len(m: usize) -> Self {
        let half = m / 2;
        let mut re = Vec::with_capacity(half);
        let mut im = Vec::with_capacity(half);
        for k in 0..half {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / m as f64;
            re.push(angle.cos());
            im.push(-angle.sin());
        }
        Self { re, im }
    }

    /// The transform length `M` this table was built for.
    pub(crate) fn len(&self) -> usize {
        self.re.len() * 2
    }
}

/// In-place radix-2 Cooley–Tukey FFT of the complex vector `(re, im)`.
///
/// `re` and `im` must have the same power-of-two length `M` and `roots` must
/// be `Roots::of_len(M)`. With `inverse` the transform conjugates the roots
/// and scales the result by `1/M`, so `fft(fft(x), inverse = true) == x`.
pub(crate) fn fft(re: &mut [f64], im: &mut [f64], roots: &Roots, inverse: bool) {
    let m = re.len();
    debug_assert!(m.is_power_of_two() && im.len() == m && roots.re.len() == m / 2);
    if m < 2 {
        return;
    }

    // Bit-reversal permutation.
    let shift = usize::BITS - m.trailing_zeros();
    for i in 0..m {
        let j = i.reverse_bits() >> shift;
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Iterative butterflies: stage `m2` combines pairs `m2` apart with
    // twiddle W_M^{j·(M/(2·m2))} = roots[j·step].
    let sign = if inverse { -1.0 } else { 1.0 };
    let mut half = 1usize;
    while half < m {
        let step = m / (2 * half);
        for base in (0..m).step_by(2 * half) {
            for j in 0..half {
                let wre = roots.re[j * step];
                let wim = sign * roots.im[j * step];
                let a = base + j;
                let b = a + half;
                let tre = re[b] * wre - im[b] * wim;
                let tim = re[b] * wim + im[b] * wre;
                re[b] = re[a] - tre;
                im[b] = im[a] - tim;
                re[a] += tre;
                im[a] += tim;
            }
        }
        half *= 2;
    }

    if inverse {
        let scale = 1.0 / m as f64;
        for v in re.iter_mut() {
            *v *= scale;
        }
        for v in im.iter_mut() {
            *v *= scale;
        }
    }
}
