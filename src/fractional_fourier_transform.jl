export frft

"""
    frft(arr, p; method=:garcia)

Calculates the fractional Fourier transform of the order `p` of `arr`.

## Methods
Several implementation exists. The following are implemented: 

* `method=:garcia`: García, J., Mas, D., & Dorsch, R. G. (1996). Fractional-Fourier-transform calculation through the fast-Fourier-transform algorithm. Applied Optics, 35(35), 7013. doi:10.1364/ao.35.007013 

"""
function frft(arr, p; padding=false, method=:garcia)
    arr_l = let
        if padding
            NDTools.select_region(arr, new_size=size(arr) .* 2) 
        else
            arr
        end
    end

    out = let 
        if method == :garcia
            _frft_garcia(arr_l, p; padding)
        else
            throw(ArgumentError("method=$method is unknown or not implemented"))
        end
    end

    out_s = let
        if padding
            NDTools.select_region(out, new_size=size(arr)) 
        else
            out 
        end
    end

    return out_s
end

function _frft_garcia(arr::AbstractArray{T, 1}, _p; padding=false) where T
    p = T(_p)
    
    #arr = select_region(arr, new_size=size(arr) .* 2)
    
    # [0,1,2,..., -2,-1]
    m = fftshift(fftfreq(length(arr), 1.0 * T(length(arr))))
    t1 = -1im * T(π) / length(arr) * tan(p * T(π) / 4)
    φ1 = exp.(1 .* t1 .* m.^2)
    φ2 = exp.(1 .* t1 .* m.^2)
    m3 = fftshift(fftfreq(length(arr) ÷ 2, 1.0 * T(length(arr) ÷ 2)))
    t3 = -1im * T(π) / (length(arr) ÷2)  * tan(p * T(π) / 4)
    φ3 = exp.(1 .* t3 .* m3.^2)

    t2 = -1im * T(π) / length(arr) * sin(p * T(π) / 2)
    #φ2 = collect(ft(window_hanning(eltype(arr), size(arr), border_in=0.45, border_out=0.50) .* ift(exp.(1 .* t2 .* m.^2))))
    φ2 = exp.(1 .* t2 .* m.^2)

    Mp = exp(-1im * T(π) * sign(sin(p * T(π) / 2)) / 4 + 1im * p * T(π) / 4 + T(π) / 4)
    
    #res = Mp .* φ3 .* ift(select_region(φ2 .* ft(arr .* φ1), new_size=size(arr) .÷ 2))
    res = Mp .* φ1 .* ift(φ2 .* ft(arr .* φ1))

    return res 
    return select_region(res, new_size=size(res).÷2)
end

function _frft_garcia(arr::AbstractArray{T, 2}, _p; padding=true) where T
    p = T(_p)
    
    # [0,1,2,..., -2,-1]
    m = fftshift.(fftfreq.(size(arr), T.(size(arr))))
    m_cent = fftshift.(fftfreq.(size(arr), T.(size(arr))))
    t1 = -1im .* T(π) ./ size(arr) .* tan(p * T(π) / 4)
    φ1 = exp.(t1[1] .* m_cent[1].^2 .+
              t1[2] .* m_cent[2]'.^2)

    t2 = -1im .* T(π) ./ size(arr) .* sin(p * T(π) / 2)
    φ2 = collect(ft(window_hanning(eltype(arr), size(arr), border_in=0.45, border_out=0.55) .* ift(exp.(t2[1] .* m[1].^2 .+
                                                            t2[2] .* m[2]'.^2))))

    #return φ2
    # different phase factor, paper might be wrong. paper claims + π/4.
    #Mp = exp(-1im * T(π) * sign(sin(p * T(π) / 2)) / 4 + 1im * p * T(π) / 4 + T(π) / 4)#1im * T(π) / 2)
    Mp = exp(-1im * T(π) * sign(sin(p * T(π) / 2)) / 4 + 1im * p * T(π) / 4 + 1im * T(π) / 4)
    
    res = Mp .* φ1 .* ift(φ2 .* ft(arr .* φ1))

    return res
end
