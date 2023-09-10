export ccorr, plan_ccorr, ccorr_psf, plan_ccorr_psf, plan_ccorr_buffer, plan_ccorr_psf_buffer

"""
    ccorr(u, v[, dims]; centered=false)

Calculates the cross-correlation between `u` and `v` along `dims`.
`centered=true` moves the output of the cross-correlation to the Fourier center.

If `u` and `v` are both a real valued array we use `rfft` and hence
the output is real as well.
If either `u` or `v` is complex we use `fft` and output is hence complex.

Per default the correlation is performed along `min(ndims(u), ndims(v))`.

```jldoctest
julia> ccorr([1,1,0,0], [1,1,0,0], centered=true)
4-element Vector{Float64}:
 0.0
 1.0
 2.0
 1.0

julia> ccorr([1,1,0,0], [1,1,0,0])
4-element Vector{Float64}:
 2.0
 1.0
 0.0
 1.0

julia> ccorr([1im,0,0,0], [0,1im,0,0])
4-element Vector{ComplexF64}:
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 1.0 + 0.0im

julia> ccorr([1im,0,0,0], [0,1im,0,0], centered=true)
4-element Vector{ComplexF64}:
 0.0 + 0.0im
 1.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
```
"""
function ccorr(u::AbstractArray{T, N}, v::AbstractArray{D, M}, 
              dims=ntuple(+, min(N, M));
              centered=false) where {T, D, N, M}
    out = ifft(fft(u, dims) .* conj.(fft(v, dims)), dims)
    
    if centered
        return fftshift(out)
    else
        return out
    end
end

function ccorr(u::AbstractArray{<:Real, N}, v::AbstractArray{<:Real, M}, 
              dims=ntuple(+, min(N, M));
              centered=false) where {N, M}
    out = irfft(rfft(u, dims) .* conj.(rfft(v, dims)), size(u, dims[1]), dims)
    
    if centered
        return fftshift(out)
    else
        return out
    end
end

function ccorr_psf(u::AbstractArray{T, N}, psf::AbstractArray{D, M}, dims=ntuple(+, min(N, M))) where {T, D, N, M}
    return ccorr(u, ifftshift(psf, dims), dims)
end

function p_ccorr_aux(P, P_inv, u, v_ft)
    return (P_inv.p * ((P * u) .* conj(v_ft) .* P_inv.scale))
end

function plan_ccorr(u::AbstractArray{T1, N}, v::AbstractArray{T2, M}, dims=ntuple(+, N);
                   kwargs...) where {T1, T2, N, M}
    eltype_error(T1, T2)
    plan = get_plan(T1)
    # do the preplanning step
    P = let
        # FFTW.MEASURE flag might overwrite input! Hence copy!
        if (:flags in keys(kwargs) && 
            (getindex(kwargs, :flags) == FFTW.MEASURE || getindex(kwargs, :flags) == FFTW.PATIENT)) 
            plan(copy(u), dims; kwargs...)
        else
            plan(u, dims; kwargs...)
        end
    end

    v_ft = fft_or_rfft(T1)(v, dims)
    # construct the efficient conv function
    # P and P_inv can be understood like matrices
    # but their computation is fast
    ccorr = let P = P,
               P_inv = inv(P),
               # put a different name here! See https://discourse.julialang.org/t/type-issue-with-captured-variables-let-workaround-failed/85661
               v_ft = v_ft
        ccorr(u, v_ft=v_ft) = p_ccorr_aux(P, P_inv, u, v_ft)
    end
    
    return v_ft, ccorr
end

function plan_ccorr_psf(u::AbstractArray{T, N}, psf::AbstractArray{T, M}, dims=ntuple(+, N);
    kwargs...) where {T, N, M}
return plan_ccorr(u, ifftshift(psf, dims), dims; kwargs...)
end

function plan_ccorr_buffer(u::AbstractArray{T1, N}, v::AbstractArray{T2, M}, dims=ntuple(+, N);
    kwargs...) where {T1, T2, N, M}
    eltype_error(T1, T2)
    plan = get_plan(T1)
    # do the preplanning step
    P_u = plan(u, dims; kwargs...)
    P_v = plan(v, dims)

    u_buff = P_u * u
    v_ft = P_v * v
    conj!(v_ft)
    uv_buff = u_buff .* v_ft

    # for fourier space we need a new plan
    P = plan(u .* v, dims; kwargs...)
    P_inv = inv(P)
    out_buff = P_inv * uv_buff

    # construct the efficient conv function
    # P and P_inv can be understood like matrices
    # but their computation is fast
    function ccorr(u, v_ft=v_ft)
        mul!(u_buff, P_u, u)
        uv_buff .= u_buff .* v_ft
        mul!(out_buff, P_inv, uv_buff)
        return out_buff
    end

    return v_ft, ccorr
end

function plan_ccorr_psf_buffer(u::AbstractArray{T, N}, psf::AbstractArray{T, M}, dims=ntuple(+, N);
    kwargs...) where {T, N, M}
    return plan_ccorr_buffer(u, ifftshift(psf, dims), dims; kwargs...)
end
