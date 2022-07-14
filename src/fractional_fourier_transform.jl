export frfft

"""
    frfft(arr, p; shift=false, method=:garcia)

Calculates the fractional Fourier transform (FRFFT) of the order `p` of `arr`.

If `shift=false` the fraction FFT is calculated around the first entry.
If `shift=true` the FRFT


## Methods
Several implementation exists. The following are implemented: 

* `method=:garcia`: García, J., Mas, D., & Dorsch, R. G. (1996). Fractional-Fourier-transform calculation through the fast-Fourier-transform algorithm. Applied Optics, 35(35), 7013. doi:10.1364/ao.35.007013 

"""
function frfft(_arr, p; shift=false, method=:garcia)
    arr = let
        if shift
            ifftshift(_arr)
        else
            _arr
        end
    end


    out = let
        if method == :garcia
            _frfft_garcia(arr, p)
        else
            throw(ArgumentError("method=$method is unknown or not implemented"))
        end
    end

    res = let 
        if shift
            fftshift(out)
        else
            out
        end
    end

    return res
end

function _frfft_garcia(arr::AbstractVector{T}, _p) where T
    p = T(_p)
    
    m = fftfreq(length(arr), T(length(arr)))
    t1 = -1im * T(π) / length(arr) * tan(p * T(π) / 4)
    φ1 = exp.(t1 .* m.^2)
    t2 = -1im * T(π) / length(arr) * sin(p * T(π) / 2)
    φ2 = exp.(t2 .* m.^2)
    Mp = exp(-1im * T(π) * sign(sin(p * T(π) / 2)) / 4 + 1im * p * T(π) / 4)
    res = Mp .* φ1 .* ifft(φ2 .* fft(arr .* φ1))

    return res 
end

function _frfft_garcia(arr::AbstractMatrix{T}, _p) where T
    p = T(_p)
    
    m = fftfreq.(size(arr), T.(size(arr)))

    t1 = -1im .* T(π) ./ size(arr) .* tan(p * T(π) / 4)
    φ1 = exp.(t1[1] .* m[1].^2 .+
              t1[2] .* m[2]'.^2)

    t2 = -1im .* T(π) ./ size(arr) .* sin(p * T(π) / 2)
    φ2 = exp.(t2[1] .* m[1].^2 .+ t2[2] .* m[2]'.^2)

    Mp = exp(-1im * T(π) * sign(sin(p * T(π) / 2)) / 4 + 1im * p * T(π) / 4)
    
    res = Mp .* φ1 .* ifft(φ2 .* fft(arr .* φ1))
    return res
end
