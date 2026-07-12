export conv, plan_conv, conv_psf, plan_conv_psf
export plan_conv_buffer, plan_conv_psf_buffer


"""
    conv(u, v[, dims])

Convolve `u` with `v` over `dims` dimensions with an FFT based method.
Note, that this method introduces wrap-around artifacts without
proper padding/windowing.

# Arguments
* `u` is an array in real space.
* `v` is the array to be convolved in real space as well.
* Per default `ntuple(+, min(N, M)))` means that we perform the convolution 
    over all dimensions of that array which has less dimensions. 
    If `dims` is an array with integers, we perform convolution 
    only over these dimensions. Eg. `dims=[1,3]` would perform the convolution
    over the first and third dimension. Second dimension is not convolved.

If `u` and `v` are both a real valued array we use `rfft` and hence
the output is real as well.
If either `u` or `v` is complex we use `fft` and output is hence complex.


 # Examples
1D with FFT over all dimensions. We choose `v` to be a delta peak.
Therefore convolution should act as identity.
```jldoctest
julia> u = [1 2 3 4 5]
1×5 Array{Int64,2}:
 1  2  3  4  5
julia> v = [0 0 1 0 0]
1×5 Array{Int64,2}:
 0  0  1  0  0

julia> conv(u, v)
1×5 Matrix{Float64}:
 4.0  5.0  1.0  2.0  3.0
```

2D with FFT with different `dims` arguments.
```jldoctest
julia> u = 1im .* [1 2 3; 4 5 6]
2×3 Matrix{Complex{Int64}}:
 0+1im  0+2im  0+3im
 0+4im  0+5im  0+6im

julia> v = [1im 0 0; 1im 0 0]
2×3 Matrix{Complex{Int64}}:
 0+1im  0+0im  0+0im
 0+1im  0+0im  0+0im

julia> conv(u, v)
2×3 Matrix{ComplexF64}:
 -5.0+0.0im  -7.0+0.0im  -9.0+0.0im
 -5.0+0.0im  -7.0+0.0im  -9.0+0.0im
```
"""
function conv(u::AbstractArray{T, N}, v::AbstractArray{D, M}, dims=ntuple(+, min(N, M))) where {T, D, N, M}
    return ifft(fft(u, dims) .* fft(v, dims), dims)
end

function conv(u::AbstractArray{<:Real, N}, v::AbstractArray{<:Real, M}, dims=ntuple(+, min(N, M))) where {N, M}
    return irfft(rfft(u, dims) .* rfft(v, dims), size(u, dims[1]), dims)
end

"""
    conv_psf(u, psf[, dims])

`conv_psf` is a shorthand for `conv(u,ifftshift(psf))`. For examples see `conv`.
"""
function conv_psf(u::AbstractArray{T, N}, psf::AbstractArray{D, M}, dims=ntuple(+, min(N, M))) where {T, D, N, M}
    return conv(u, ifftshift(psf, dims), dims)
end



"""
    plan_conv(u, v [, dims]; kwargs...)

Pre-plan an optimized convolution for arrays shaped like `u` and `v` (based on pre-plan FFT)
along the given dimenions `dims`.
`dims = 1:ndims(u)` per default.
The 0 frequency of `u` must be located at the first entry.

We return two arguments: 
The first one is `v_ft` (obtained by `fft(v)` or `rfft(v)`).
The second return is the convolution function `pconv`.
`pconv` itself has two arguments. `pconv(u, v_ft=v_ft)` where `u` is the object and `v_ft` the v_ft.
This function achieves faster convolution than `conv(u, u)`.
Depending whether `u` is real or complex we do `fft`s or `rfft`s
Additionally, it is possible to provide `flags=FFTW.MEASURE` as `kwargs` 
to change the planning of the FFT.


# Examples
```jldoctest
julia> u = [1 2 3 4 5]
1×5 Matrix{Int64}:
 1  2  3  4  5

julia> v = [1 0 0 0 0]
1×5 Matrix{Int64}:
 1  0  0  0  0

julia> v_ft, pconv = plan_conv(u, v);

julia> pconv(u, v_ft)
1×5 Matrix{Float64}:
 1.0  2.0  3.0  4.0  5.0

julia> pconv(u)
1×5 Matrix{Float64}:
 1.0  2.0  3.0  4.0  5.0
```
"""
function plan_conv(u::AbstractArray{T1, N}, v::AbstractArray{T2, M}, dims=ntuple(+, N);
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
    P_inv = inv(P)
    conv = CallableConvPlan(P, P_inv, v_ft)
    
    return v_ft, conv
end

# """
#     plan_conv_buffer(u, v [, dims]; kwargs...)

# Similar to [`plan_conv`](@ref) but instead uses buffers to prevent memory allocations.
# The three buffers are internal to the function and are not exposed to the user.
# Not AD friendly!

# """
# function plan_conv_buffer(u::AbstractArray{T1, N}, v::AbstractArray{T2, M}, dims=ntuple(+, N);
#                    kwargs...) where {T1, T2, N, M}
#     eltype_error(T1, T2)
#     plan = get_plan(T1)
#     # do the preplanning step
#     P_u = plan(u, dims; kwargs...)
#     P_v = plan(v, dims)

#     u_buff = P_u * u
#     v_ft = P_v * v
#     uv_sz = bc_size(u_buff, v_ft)
#     # this saves memory allocations:
#     uv_buff = (uv_sz == size(u_buff)) ? u_buff : u_buff .* v_ft;
    
#     # for fourier space we need a new plan
#     P = plan(u .* v, dims; kwargs...)
#     P_inv = inv(P)
#     out_buff = P_inv * uv_buff

#     # construct the efficient conv function
#     # P and P_inv can be understood like matrices
#     # but their computation is fast
#     function conv(u, v_ft=v_ft)
#         mul!(u_buff, P_u, u)
#         uv_buff .= u_buff .* v_ft
#         mul!(out_buff, P_inv, uv_buff)
#         return out_buff
#     end

#     return v_ft, conv
# end

"""
    plan_conv_psf_buffer(u, psf [, dims]; kwargs...) where {T, N}

`plan_conv_psf_buffer` is a shorthand for `plan_conv_buffer(u, ifftshift(psf))`. For examples see `plan_conv`.
"""
function plan_conv_psf_buffer(u::AbstractArray{T, N}, psf::AbstractArray{T, M}, dims=ntuple(+, N);
                       kwargs...) where {T, N, M}
    return plan_conv_buffer(u, ifftshift(psf, dims), dims; kwargs...)
end

# Define the struct for non-buffered planned convolutions
struct CallableConvPlan{CT<:AbstractArray}
    P
    P_inv
    v_ft::CT
end

function (c::CallableConvPlan{CT})(u::AbstractArray, v_ft::CT) where {CT<:AbstractArray}
    return p_conv_apply(c, u, v_ft)
end

function (c::CallableConvPlan{CT})(u::AbstractArray) where {CT<:AbstractArray}
    return p_conv_apply(c, u, c.v_ft)
end

function p_conv_apply(c::CallableConvPlan, u, v_ft)
    return p_conv_aux(c.P, c.P_inv, u, v_ft)
end

function ChainRulesCore.rrule(::typeof(p_conv_apply), c::CallableConvPlan, u, v_ft)
    Y = p_conv_apply(c, u, v_ft)
    function conv_pullback(barx)
        barx2 = _materialize_barx(barx, u)
        conj_v = eltype(v_ft) <: Real ? v_ft : conj(v_ft)
        ∇ = p_conv_aux(c.P, c.P_inv, barx2, conj_v)
        return NoTangent(), NoTangent(), ∇, NoTangent()
    end
    return Y, conv_pullback
end

# Define the struct for buffered planned convolutions
struct CallableBufferPlan{IAT<:AbstractArray, CT1<:AbstractArray, CT2<:AbstractArray, CT3<:AbstractArray}
    P_u
    # This is only needed due to a bug in CUDA freeing the plan when wrapped it in "inv":
    P_for_inv
    P_inv
    v_ft::CT1
    u_buff::CT2 # dimensions can be different
    uv_buff::CT3 # final dimension can be different again
    out_buff::IAT # final datatype and size can also vary
end

# Define the call method for the struct
function (c::CallableBufferPlan{IAT, CT1, CT2, CT3})(u::AbstractArray, v_ft::CT1) where {IAT<:AbstractArray, CT1<:AbstractArray, CT2<:AbstractArray, CT3<:AbstractArray}
    return p_conv_apply_buffer(c, u, v_ft)
end

function (c::CallableBufferPlan{IAT, CT1, CT2, CT3})(u::AbstractArray) where {IAT<:AbstractArray, CT1<:AbstractArray, CT2<:AbstractArray, CT3<:AbstractArray}
    return p_conv_apply_buffer(c, u, c.v_ft)
end

function p_conv_apply_buffer(c::CallableBufferPlan, u, v_ft)
    return p_conv_aux!(c.P_u, c.P_inv, u, v_ft, c.u_buff, c.uv_buff, c.out_buff)
end

function ChainRulesCore.rrule(::typeof(p_conv_apply_buffer), c::CallableBufferPlan, u, v_ft)
    Y = p_conv_apply_buffer(c, u, v_ft)
    function conv_pullback(barx)
        barx2 = _materialize_barx(barx, u)
        conj_v = eltype(v_ft) <: Real ? v_ft : conj(v_ft)
        ∇ = p_conv_aux!(c.P_u, c.P_inv, barx2, conj_v, c.u_buff, c.uv_buff, copy(c.out_buff))
        return NoTangent(), NoTangent(), ∇, NoTangent()
    end
    return Y, conv_pullback
end


"""
    plan_conv_buffer(u, v [, dims])

Pre-plan an optimized convolution for arrays shaped like `u` and `v` (based on pre-plan FFT)
along the given dimensions `dims`.
`dims = 1:ndims(u)` per default.
The 0 frequency of `u` must be located at the first entry.
We return two arguments: 
The first one is `v_ft` (obtained by `fft(v)` or `rfft(v)`).
The second return is the convolution function `pconv`.
`pconv` itself has two arguments. `pconv(u, v_ft=v_ft)` where `u` is the object and `v_ft` the v_ft.
This function achieves faster convolution than `conv(u, u)`.
Depending whether `u` is real or complex we do `fft`s or `rfft`s

# Warning
The resulting output of the `pconv` function is a reference to an internal, allocated array.
If you use the `pconv` function for different tasks, 
a new call to `pconv` will change the previous result (since the previous result was only a reference, not a new array). 


# Examples
```jldoctest
julia> u = [1 2 3 4 5]
1×5 Matrix{Int64}:
 1  2  3  4  5
julia> v = [1 0 0 0 0]
1×5 Matrix{Int64}:
 1  0  0  0  0
julia> v_ft, pconv = plan_conv(u, v);
julia> pconv(u, v_ft)
1×5 Matrix{Float64}:
 1.0  2.0  3.0  4.0  5.0
julia> pconv(u)
1×5 Matrix{Float64}:
 1.0  2.0  3.0  4.0  5.0
```
"""
function plan_conv_buffer(u::AbstractArray{T, N}, v::AbstractArray{T, M}, dims=ntuple(+, N)) where {T, N, M}
    plan = get_plan(T)
    # do the preplanning step
    P = plan(u, dims)
    # inpout FT storage
    u_buff = P * u
    # P_inv = inv(P)
    # out = u .* v # similar(u)
    out_buff = similar(u, Base.Broadcast.broadcast_shape(size(u),size(v)))
    u = expand_dims(u, Val(max(N,M)))
    v = expand_dims(v, Val(max(N,M)))

    v_ft = fft_or_rfft(T)(v, dims)

    uv_buff = similar(v_ft, Base.Broadcast.broadcast_shape(size(u_buff),size(v_ft)))
    
    P_for_inv = plan(out_buff, dims)
    P_inv = inv(P_for_inv)

    # construct the efficient conv function
    # P and P_inv can be understood like matrices
    # but their computation is fast
    conv = CallableBufferPlan(P, P_for_inv, P_inv, v_ft, u_buff, uv_buff, out_buff)
    return v_ft, conv
end

# axiliary function to use with planned convolutions
function p_conv_aux!(P, P_inv, u, v_ft, u_buff, uv_buff, out) # , P_for_inv
    #return P_inv.scale .* (P_inv.p * ((P * u) .* v_ft))  
    mul!(u_buff, P, u)
    # may be in place or out-of-place:
    uv_buff .= u_buff .* v_ft .* P_inv.scale
    mul!(out, P_inv.p, uv_buff)
    #out2 = out .* P_inv.scale
   return out
end

function _materialize_barx(barx, u)
    barx_val = barx
    for _ in 1:3
        unthunked = ChainRulesCore.unthunk(barx_val)
        if unthunked === barx_val
            break
        end
        barx_val = unthunked
    end

    if barx_val isa AbstractArray
        T = promote_type(eltype(u), eltype(barx_val))
        out = similar(u, T)
        copyto!(out, barx_val)
        return out
    elseif barx_val isa Number
        T = promote_type(eltype(u), typeof(barx_val))
        return fill!(similar(u, T), convert(T, barx_val))
    else
        return fill!(similar(u, eltype(u)), zero(eltype(u)))
    end
end

function ChainRulesCore.rrule(::typeof(p_conv_aux!), P, P_inv, u, v_ft, u_buff, uv_buff, out)
    Y = p_conv_aux!(P, P_inv, u, v_ft, u_buff, uv_buff, out)
    function conv_pullback(barx)
        barx2 = _materialize_barx(barx, u)
        conj_v = eltype(v_ft) <: Real ? v_ft : conj(v_ft)
        ∇ = p_conv_aux!(P, P_inv, barx2, conj_v, u_buff, uv_buff, copy(out))
        return NoTangent(), NoTangent(), ∇, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    return Y, conv_pullback
end



"""
    plan_conv_psf(u, psf [, dims]; kwargs...) where {T, N}

`plan_conv_psf` is a shorthand for `plan_conv(u, ifftshift(psf))`. For examples see `plan_conv`.
"""
function plan_conv_psf(u::AbstractArray{T, N}, psf::AbstractArray{T, M}, dims=ntuple(+, N);
                       kwargs...) where {T, N, M}
    return plan_conv(u, ifftshift(psf, dims), dims; kwargs...)
end

function p_conv_aux(P, P_inv, u, v_ft)
    return (P_inv.p * ((P * u) .* v_ft .* P_inv.scale))
end

function ChainRulesCore.rrule(::typeof(p_conv_aux), P, P_inv, u, v_ft)
    Y = p_conv_aux(P, P_inv, u, v_ft)
    function conv_pullback(barx)
        barx2 = similar(u, promote_type(eltype(u), eltype(barx)))
        barx2 .= barx
        conj_v = eltype(v_ft) <: Real ? v_ft : conj(v_ft)
        ∇ = p_conv_aux(P, P_inv, barx2, conj_v)
        return NoTangent(), NoTangent(), ∇, NoTangent()
    end
    return Y, conv_pullback
end

"""
    fft_or_rfft(T)

Small helper function to decide whether a real
or a complex valued FFT is appropriate.
"""
function fft_or_rfft(::Type{<:Real})
    return rfft
end

function fft_or_rfft(::Type{T}) where T
    return fft
end


"""
    get_plan(T)

Small helper function to decide whether a real
or a complex valued FFT plan is appropriate.
"""
function get_plan(::Type{<:Real})
    return plan_rfft
end

function get_plan(::Type{T}) where T
    return plan_fft
end


