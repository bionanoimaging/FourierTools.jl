export frft

"""
    frft(arr, p; method=:garcia)

Calculates the fractional Fourier transform of the order `p` of `arr`.

## Methods
Several implementation exists. The following are implemented: 

* `method=:garcia`: García, J., Mas, D., & Dorsch, R. G. (1996). Fractional-Fourier-transform calculation through the fast-Fourier-transform algorithm. Applied Optics, 35(35), 7013. doi:10.1364/ao.35.007013 

"""
function frft(arr, p; method=:garcia)
    if method == :garcia
        return _frft_garcia(arr, p)
    else
        throw(ArgumentError("method=$method is unknown or not implemented"))
    end
end

function _frft_garcia(arr::AbstractArray{T, 2}, _p) where T
    p = T(_p)
    
    # [0,1,2,..., -2,-1]
    m = fftfreq.(size(arr), T.(size(arr)))
    t1 = -1im .* T(π) ./ size(arr) .* tan(p * T(π) / 4)
    φ1 = exp.(t1[1] .* m[1].^2 .+
              t1[2] .* m[2]'.^2)

    t2 = -1im .* T(π) ./ size(arr) .* sin(p * T(π) / 2)
    φ2 = exp.(t2[1] .* m[1].^2 .+
              t2[2] .* m[2]'.^2)

    # different phase factor, paper might be wrong. paper claims + π/4.
    Mp = exp(-1im * T(π) * sign(sin(p * T(π) / 2)) / 4 + 1im * p * T(π) / 4 + T(π) / 4)#1im * T(π) / 2)
    
    res = Mp .* φ1 .* ifft(φ2 .* fft(arr .* φ1))

    return res
end
