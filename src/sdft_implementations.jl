export SDFT

## Basic SDFT

"""
    SDFT(n)
    SDFT(C, n)

Basic method to compute a Sliding Discrete Fourier Transform of window length `n`,
through the recursive formula:

\$X_{i+1}[k] = e^{j2{\\pi}k/n} \\cdot (X_{i}[k] + x[i+n] - x[i])\$

The transfer function for the `k`-th bin of this method is:

\$H(z) = \\frac{1 - z^{-n}}{1 - e^{j2{\\pi}k/n} z^{-1}}\$

Use `SDFT(C, n)` to obtain results with the precision of the complex data type `C`.
`C == ComplexF64` by default.

`SDFT` is a subtype of [`AbstractSDFT`](@ref).
See the documentation of that type for further details about its usage.

## References

Jacobsen, E. & Lyons, R. (2003). "The sliding DFT," *IEEE Signal Processing Magazine*, 20(2), 74-80.
doi:[10.1109/MSP.2003.1184347](https://doi.org/10.1109/MSP.2003.1184347)
"""
struct SDFT{T,C} <: AbstractSDFT
    n::T
    factor::C
end

function SDFT{T,C}(n) where {T,C}
    factor::C = exp(2π*im/n)
    SDFT(T(n),factor)
end

SDFT(C::DataType, n) = SDFT{typeof(n), C}(n)
SDFT(n) = SDFT(ComplexF64, n)

# Required functions

sdft_windowlength(method::SDFT) = method.n

function sdft_update!(dft, x, method::SDFT{T,C}, state) where {T,C}
    twiddle = one(C)
    for k in eachindex(dft)
        dft[k] = twiddle * (dft[k] + sdft_nextdata(state) - sdft_previousdata(state))
        twiddle *= method.factor
    end
end

sdft_dataoffsets(::SDFT) = 0
