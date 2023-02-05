export frfft


# TODO: circular convolution introduces wrap around artifacts which have not been handled yet

"""
frfft(arr, p; shift=false, method=:garcia)

Calculates the fractional Fast Fourier transform (FRFFT) of the order `p` of `arr`.
No `dims` argument is supported yet.

If `shift=false` the FRFFT is calculated around the first entry.
If `shift=true` the FRFFT is calculated aroound the center.


## Methods
Several implementation exists. The following are implemented: 

* `method=:garcia`: A convolutional approach based on 2 FFTs. See García, J., Mas, D., & Dorsch, R. G. (1996). Fractional-Fourier-transform calculation through the fast-Fourier-transform algorithm. Applied Optics, 35(35), 7013. doi:10.1364/ao.35.007013 

"""
function frfft(_arr, p; shift=true, method=:garcia, p_change=true)

    if p_change
        # reduce p to operations which are nice with sampling
        # p is afterwards ∈[-2, 2)
        p = -2 + mod(p + 2, 4)
        
        if 1 ≤ p 
            if shift
                _arr = reverse(_arr)
            else
                _arr = ifftshift(reverse(fftshift(_arr)))
            end
            p = p - 2
        elseif p ≤ -1 
            if shift
                _arr = reverse(_arr)
            else
                _arr = ifftshift(reverse(fftshift(_arr)))
            end
            p = p + 2        
        end

        # p∈[-1, 1]
        if 0.5 < p
            if shift
                _arr = fftshift(fft(ifftshift(_arr))) ./ sqrt(eltype(_arr)(length(_arr)))
            else
                _arr = fft(_arr) ./ sqrt(eltype(_arr)(length(_arr)))
            end
            p = p - 1
        elseif p < -0.5
            if shift
                _arr = fftshift(ifft(ifftshift(_arr))) .* sqrt(eltype(_arr)(length(_arr)))
            else
                _arr = ifft(_arr) .* sqrt(eltype(_arr)(length(_arr)))
            end

            p = p + 1
        end
    end

    # p ∈ [-0.5, 0.5)

    # trivial case
    if iszero(p)
        return complex.(_arr)
    end

    # handle shifting
    arr = let
        if shift
            ifftshift(_arr)
        else
            _arr
        end
    end


    # do non-trivial operations
    out = let
        if method == :garcia
            _frfft_garcia(arr, p)
        else
            throw(ArgumentError("method=$method is unknown or not implemented"))
        end
    end

    # handle shifts
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
    Mp = exp(-1im * T(π) * sign(sin(p * T(π) / 2)) / 4 + 1im * p * T(π) / 4 + sign(p) * 1im * T(π) /4)
    # can do an in-place fft
    res = Mp .* φ1 .* ifft!(φ2 .* fft!(arr .* φ1))

    return res 
end


function _frfft_garcia(arr::AbstractMatrix{T}, _p) where T 
    p = real(eltype(arr))(_p)
    
    mel = similar(arr, real(eltype(arr)), 1)
    m = Vector{typeof(mel)}(undef, ndims(arr)) 
    m .= collect(fftfreq.(size(arr), T.(size(arr))))

    t1 = -1im .* T(π) ./ size(arr) .* tan(p * T(π) / 4)
    φ1 = exp.(t1[1] .* m[1].^2 .+
              t1[2] .* m[2]'.^2)

    t2 = -1im .* T(π) ./ size(arr) .* sin(p * T(π) / 2)
    φ2 = exp.(t2[1] .* m[1].^2 .+ t2[2] .* m[2]'.^2)

    Mp = exp(-1im * T(π) * sign(sin(p * T(π) / 2)) / 4 + 1im * p * T(π) / 4 + sign(p) * 1im .* T(π) / 4)
    # can do an in-place fft
    res = Mp .* φ1 .* ifft!(φ2 .* fft!(arr .* φ1))
    return res
end


# specialized for n dimensions
function _frfft_garcia(arr::AbstractArray{T, N}, _p) where {N, T}
    p = T(_p)
    
    
    mel = similar(arr, real(eltype(arr)), 1)
    m = Vector{typeof(mel)}(undef, ndims(arr)) 
    m .= collect(fftfreq.(size(arr), T.(size(arr))))

    p1 = -1im .* T(π) ./ size(arr) .* tan(p * T(π) / 4)
    t1 = similar(arr, complex(T))
    fill!(t1, zero(T))


    p2 = -1im .* T(π) ./ size(arr) .* sin(p * T(π) / 2)
    t2 = similar(arr, complex(T))
    fill!(t2, zero(T))
    for d in 1:ndims(arr)
        ns = ntuple(i -> i == d ? size(arr, i) : 1, Val(ndims(arr)))
        t1 .+= reshape(p1[d] .* m[d].^2, ns...)
        t2 .+= reshape(p2[d] .* m[d].^2, ns...)
    end

    φ1 = exp.(t1)
    φ2 = exp.(t2)
    Mp = exp(-1im * T(π) * sign(sin(p * T(π) / 2)) / 4 + 1im * p * T(π) / 4 + sign(p) * 1im * T(π) / 4)
    # can do an in-place fft
    res = Mp .* φ1 .* ifft!(φ2 .* fft!(arr .* φ1))
    return res
end
