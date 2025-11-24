export czt, iczt, plan_czt

"""
    get_kernel_1d(arr::AbstractArray{T,D}, N::Integer, M::Integer; a= 1.0, w = cispi(-2/N), extra_phase=0.0, global_phase=0.0) where {T,D}

calculates the kernel for the Bluestein algorithm. Note that the length depends on the destination size.
Note the the resulting kernel-size is computed based on the minimum required length for the task.
Use `size(result)` to know what to calculate with. The center of the resulting `kernel` is set to the standard
convention position: 0 
The code is based on Rabiner, Schafer & Rader  1969, IEEE Trans. on Audio and Electroacoustics, 17,86-92

# Arguments
    `RT`: the datatype of the array to transform
    `N`: an integer for the length of the source array to transform.
    `M`: an integer for the length of the destination array.
    `a`: the (complex valued) phasor defining the start of the sampling in the source array
    `w`: the (complex valued) phasor defining consecutive sample positions
    `extra_phase`: a phase ramp to apply to the final result, which enables to change the interpretation 
                   of the central source positon. 
    `global_phase`: the start phase of the `extra_phase` to apply.
    returns: a tuple of three arrays for the initial multiplication (A*W), the convolution 
             (already fourier-transformed) and the post multiplication.
"""
function get_kernel_1d(arr::AT, N::Integer, M::Integer; a= 1.0, w = cispi(-2/N), extra_phase=0.0, global_phase=0.0) where {T,D, AT <: AbstractArray{T,D}}
    # the size is needed to avoid wrap
    RT = real(T)
    CT = (RT <: Real) ? Complex{RT} : RT
    RT = real(CT)

    # converts MutableShiftedArrays.CircShiftedArray into a plain array type:
    tmp = similar(arr, RT, (1,))
    RAT = real_arr_type(typeof(tmp), Val(1))

    # the maximal size where the convolution does not yield zero 
    # max_size = 2*N-1
    # the minimum size needed for the convolution
    # kernel-size for the deconvolution
    L = N + M -1 # max(min(max_size, dsize), nowrap_size)
    
    #Note that the source size ssz is used here
     # W of the source for FFTs. 
    n = RAT(0:N-1)
    # pre-calculate the product of a.^(-n) and w to be later multiplied with the input x
    # late casting is important, since the rounding errors are overly large if the original calculations are done in Float32.
    aw = CT.((a .^ (-n)) .* w .^ ((n .^ 2) ./ 2))

    conv_kernel = similar(arr, CT, L) # Array{CT}(undef, L)
    fill!(conv_kernel, zero(CT))

    m = RAT(0:M-1)
    conv_kernel[1:M] .= w .^ (-(m .^ 2) ./ 2)
    right_start = L-N+1
    n = (1:N-1)
    conv_kernel[L:-1:right_start+1] .= w .^ (.-(n .^ 2) ./ 2)
    # calculate W^(k^2/2), we can reuse the conv_kernel definition 1..M and just invert it 
    #
    # The extra_phase accounts for the nominal center position of the source array. Note that this is 
    # by default selected to be the geometric center and not the Fourier-center result in a 
    # real array upon zooming a symmetric even array.
    # Note that the casting is intentionally performed after the calculation, as for many cases 
    # the calculation precision in the coarse datatype is insufficient. 
    wd = (abs(w)≈1.0) ? CT.(conj.(conv_kernel[1:M]) .* global_phase .* extra_phase.^m) : CT.(inv.(conv_kernel[1:M]) .* global_phase .* extra_phase.^m)
    return aw, fft(conv_kernel), wd
end

# type for planning. The arrays are 1D but oriented
"""
    CZTPlan_1D{CT<:Complex, D<:Integer, AT<:AbstractArray{CT, D}, PT<:Number, PFFT<:AbstractFFTs.Plan, PIFFT<:AbstractFFTs.ScaledPlan} 

type used for the onedimensional plan of the chirp Z transformation (CZT).
containing
# Members:
    `d`: dimension (only one!) to transform with this plan
    `pad_value`: the value to pad wrapped data with (zero is already handled by the `wd` term, if wanted). 
    `pad_ranges` :: tuple of two ranges of invalid positions, which can be replaced by pad values
    `aw`: factor to multiply input with
    `fft_fv`: fourier-transform (FFTW) of the convolutio kernel
    `wd`: factor to multiply the result of the convolution by
    `fftw_plan!`: plan for the forward FFTW of the convolution kernel
    `ifftw_plan!`: plan for the inverse FFTW of the convolution kernel
"""
struct CZTPlan_1D{CT<:Complex, AT<:AbstractArray{CT}, PT<:Number, PFFT<:AbstractFFTs.Plan, PIFFT<:AbstractFFTs.ScaledPlan}
    d :: Int
    pad_value :: PT
    pad_ranges :: NTuple{2, UnitRange{Int64}}
    aw :: AT 
    fft_fv :: AT 
    wd :: AT 
    fftw_plan! :: PFFT 
    ifftw_plan! :: PIFFT 
end

"""
    CZTPlan_ND{CT, D} # <: AbstractArray{T,D}

type used for the onedimensional plan of the chirp Z transformation (CZT).
containing
# Members:
    `plans`: vector of CZTPlan_1D for each of the directions of the ND array to transform
"""
struct CZTPlan_ND{CT<:Complex, AT<:AbstractArray{CT}, PT<:Number, PFFT<:AbstractFFTs.Plan, PIFFT<:AbstractFFTs.ScaledPlan} 
    plans :: Vector{CZTPlan_1D{CT, AT, PT, PFFT, PIFFT}}
end

function get_invalid_ranges(sz, scaled, dsize, dst_center)
    start_range = 1:0
    stop_range = 1:0
    if (scaled*sz < dsize)
        ceil(Int64, scaled * sz)
        valid_start = floor(Int, dst_center - (scaled * sz)/2)
        valid_end = ceil(Int, dst_center + (scaled * sz)/2)
        start_range = let  
            if valid_start > 1
                if valid_end <= dsize
                    1:valid_start-1
                else
                    1+valid_end - dsize:valid_start-1
                end
            else
                1:0
            end
        end
        stop_range = let 
            if valid_end < dsize
                if valid_start < 1
                    valid_end+1:dsize+valid_start-1
                else
                    valid_end+1:dsize
                end
            else
                1:0
            end
        end
    end
    return (start_range, stop_range)
end

"""
    plan_czt_1d(xin, scaled, d, dsize=size(xin,d); a=nothing, w=nothing, damp=1.0, src_center=(size(xin,d)+1)/2, 
                dst_center=dsize÷2+1, remove_wrap=false, fft_flags=FFTW.ESTIMATE)

creates a plan for an one-dimensional chirp z-transformation (CZT). The generated plan is then applied via 
muliplication. For details about the arguments, see `czt_1d()`.
"""
function plan_czt_1d(xin::AT, scaled, d, dsize=size(xin,d); a=nothing, w=nothing, extra_phase=nothing, global_phase=nothing, damp=1.0, src_center=(size(xin,d)+1)/2, 
                     dst_center=dsize÷2+1, remove_wrap=false, pad_value=zero(eltype(xin)), fft_flags=FFTW.ESTIMATE) where {AT}

    a = isnothing(a) ? exp(-1im*(dst_center-1)*2pi/(scaled*size(xin,d))) : a
    w = isnothing(w) ? cispi(-2/(scaled*size(xin,d))) : w
    
    w = w * damp 
    extra_phase = isnothing(extra_phase) ? exp(1im*2pi*(src_center-1)/(scaled*size(xin,d))) : extra_phase
    global_phase = isnothing(global_phase) ? a ^ (src_center-1) : global_phase

    aw, fft_fv, wd = get_kernel_1d(xin, size(xin, d), dsize; a=a, w=w, extra_phase=extra_phase, global_phase=global_phase)

    # set pad ranges to empty ranges:
    start_range = 1:0
    stop_range = 1:0

    if remove_wrap
        start_range, stop_range = get_invalid_ranges(size(xin, d), scaled, dsize, dst_center)

        wd[start_range] .= zero(eltype(wd))
        wd[stop_range] .= zero(eltype(wd))
    end

    nsz = ntuple((dd) -> (d==dd) ? size(fft_fv, 1) : size(xin, dd), Val(ndims(xin))) 
    y = similar(xin, eltype(aw), nsz) 

    fft_p! = (typeof(y) <: Array) ? plan_fft!(y, (d,); flags=fft_flags) : plan_fft!(y, (d,))
    ifft_p! = (typeof(y) <: Array) ? plan_ifft!(y, (d,); flags=fft_flags) : plan_ifft!(y, (d,))
    
    plan = CZTPlan_1D(d, pad_value, (start_range, stop_range), reorient(aw, d, Val(ndims(xin))), reorient(fft_fv, d, Val(ndims(xin))), reorient(wd, d, Val(ndims(xin))), fft_p!, ifft_p!)
    return plan
end

"""
    plan_czt(xin, scale, dims, dsize=size(xin); a=nothing, w=nothing, damp=ones(ndims(xin)), 
             src_center=size(xin).÷2 .+1, dst_center=dsize.÷2 .+1, remove_wrap=false, fft_flags=FFTW.ESTIMATE)

creates a plan for an N-dimensional chirp z-transformation (CZT). The generated plan is then applied via 
muliplication. For details about the arguments, see `czt()`.
"""
function plan_czt(xin::AbstractArray{U,D}, scale, dims, dsize=size(xin); a=nothing, w=nothing, damp=ones(ndims(xin)),
                  src_center=size(xin).÷2 .+1, dst_center=dsize.÷2 .+1, remove_wrap=false, pad_value=zero(eltype(xin)), fft_flags=FFTW.ESTIMATE) where {U,D}
    CT = (eltype(xin) <: Real) ? Complex{eltype(xin)} : eltype(xin)
    sz = size(xin)
    xin = similar(xin) # Array{eltype(xin)}(undef, sz)

    d = dims[1]
    p = plan_czt_1d(xin, scale[d], d, dsize[d]; a=a, w=w, damp=damp[d], src_center=src_center[d], dst_center=dst_center[d], remove_wrap=remove_wrap, pad_value=pad_value, fft_flags=fft_flags)
    plans =  Vector{typeof(p)}(undef, length(dims))
    sz = ntuple((dd)-> (dd==d) ? dsize[d] : sz[dd], ndims(xin))
    n=1
    plans[n]=p 
    n+=1
    for d in dims[2:end]
        xin = similar(xin, sz) # Array{eltype(xin)}(undef, sz)
        p = plan_czt_1d(xin, scale[d], d, dsize[d]; a=a, w=w, damp=damp[d], src_center=src_center[d], dst_center=dst_center[d], remove_wrap=remove_wrap, pad_value=pad_value, fft_flags=fft_flags)
        sz = ntuple((dd)-> (dd==d) ? dsize[d] : sz[dd], ndims(xin))
        plans[n]=p 
        n += 1
    end
    return CZTPlan_ND(plans)
end

function Base.:*(p::CZTPlan_ND, xin::AbstractArray{U,D}; kargs...)::AbstractArray{complex(U),D} where {U,D} 
    xout = xin
    for pd in p.plans
        xout = czt_1d(xout, pd)
    end
    return xout
end

# for being called with the less stringent (real) datatype
# function Base.:*(p::CZTPlan_ND, f::AbstractArray{RT,D}; kargs...) where {RT <: Real, D}
#         return p * f
# end


"""
    czt_1d(xin , scaled , d; remove_wrap=false, pad_value=zero(eltype(xin)))

Chirp z transform along a single direction d of an ND array `xin`.
Note that the result type is defined by `eltype(xin)` and not by `scales`.

The code is based on Rabiner, Schafer & Rader  1969, IEEE Trans. on Audio and Electroacoustics, 17,86-92

# Arguments:
+ `xin`: array to transform
+ `scaled`: factor to zoom into during the 1-dimensional czt. 
+ `d`: single dimension to transform (as a tuple)
+ `dsize`: size of the destination array
+ `a`: defines the starting phase of the result CZT. This relates to the where the center of the destination 
       array should be. The default is `nothing` which means it is calculated from the `src_center` argument.
+ `w`: defines the consecutive phases of the result array, i.e. the zoom. It is (default `nothing`) usually automatically calculated from the `scaled` and the `damp` argument.
       You only need to state it, if you want to use the low-level interface (e.g. for the Laplace transform).
+ `damp`: a multiplicative factor to apply as a damping coefficient to `w`.
+ `src_center`: position of the nominal central (zero-position) pixel in the source array. By default the F
                ourier-center `size(src).÷2 .+1` is used.
+ `dst_center`: the center (zero-position) of the destination array. By default the 
                Fourier-center `size(dst).÷2 .+1` is used.
+ `extra_phase`: a phase ramp to apply to the final result relating to the src_center. By default `nothing` which calculates this phase according to the `src_center`.
+ `global_phase`: the initial phase of the destitation array. By default `nothing` which calculates this phase according to the centers.
+ `remove_wrap`: if true, the positions that represent a wrap-around will be set to zero
+ `pad_value`: the value to pad wrapped data with. 
"""
function czt_1d(xin::AbstractArray{U,D}, scaled, d, dsize=size(xin,d); a=nothing, w=nothing, damp=1.0, src_center=size(xin,d)÷2+1,
                dst_center=dsize÷2+1, extra_phase=nothing, global_phase=nothing, remove_wrap=false, pad_value=zero(U), fft_flags=FFTW.ESTIMATE)::AbstractArray{complex(U), D}  where {U,D}
    plan = plan_czt_1d(xin, scaled, d, dsize; a=a, w=w, extra_phase=extra_phase, global_phase=global_phase, damp, src_center=src_center, dst_center=dst_center, remove_wrap=remove_wrap, pad_value=pad_value, fft_flags=fft_flags);
    return plan * xin
end

function Base.:*(p::CZTPlan_1D, xin::AbstractArray{U,D}; kargs...)::AbstractArray{complex(U), D} where {U,D}       # Complex{U}
    return czt_1d(xin, p)
end

# for being called with the less stringent (real) datatype
# function Base.:*(p::CZTPlan_1D, f::AbstractArray{RT,D}; kargs...) where {RT <: Real, D}
#         return p * f
# end


"""
    czt_1d(xin , plan::CZTPlan_1D)

Chirp z transform along a single direction d of an ND array `xin`.
Note that the result type is defined by `eltype(xin)` and not by `scales`.
The plan can also be applied via multiplication with `xin`.

The code is based on Rabiner, Schafer & Rader  1969, IEEE Trans. on Audio and Electroacoustics, 17,86-92

# Arguments
    `plan`:   A plan created via plan_czt_1d()
"""
function czt_1d(xin::AbstractArray{U,D}, plan::CZTPlan_1D)::AbstractArray{complex(U), D} where {U,D}
    # destination position
    # cispi(-1/scaled * half_pix_shift) 
    #
    # The extra_phase accounts for the nominal center position of the source array.
    # Note that the default for src_center is the Fourier-center
    # which (intentionally) leads to non-real results for even-sized arrays at non-unit zoom

    L = size(plan.fft_fv, plan.d)
    nsz = ntuple((dd) -> (dd==plan.d) ? L : size(xin, dd), Val(D)) 
    # append zeros
    tmp = eltype(plan.aw).(xin .* plan.aw)
    
    corner = ntuple((x)->1, Val(D))
    y = NDTools.select_region(tmp, nsz; center=corner, dst_center=corner)

    # in-place application to y:
    plan.fftw_plan! * y 
    y .*= plan.fft_fv
    # in-place application to y:
    plan.ifftw_plan! * y

    # dsz = ntuple((dd) -> (d==dd) ? dsize : size(xin), Val(ndims(xin))) 
    # return only the wanted (valid) part
    myrange = ntuple((dd) -> (dd==plan.d) ? (1:size(plan.wd, plan.d)) : (1:size(xin, dd)), Val(D)) 
    res = y[myrange...] .* plan.wd
    # pad_value=0 means that it is either already handled by plan.wd or no padding is wanted.
    if plan.pad_value != 0
        # first the start_range (plan.pad_ranges[1]):
        myrange = ntuple((dd) -> (dd==plan.d) ? plan.pad_ranges[1] : Colon(), Val(D)) 
        res[myrange...] .= plan.pad_value
        # first the stop_range (plan.pad_ranges[2]):
        myrange = ntuple((dd) -> (dd==plan.d) ? plan.pad_ranges[2] : Colon(), Val(D)) 
        res[myrange...] .= plan.pad_value
    end
    return res
end

"""
    czt(xin, scale, dims=1:ndims(xin), dsize=size(xin,d); a=nothing, w=nothing, damp=ones(ndims(xin)), 
        src_center=size(xin,d)÷2+1, dst_center=dsize÷2+1, remove_wrap=false, fft_flags=FFTW.ESTIMATE)

Chirp z transform of the ND array `xin`
The tuple `scale` defines the zoom factors in the Fourier domain. Each has to be bigger than one.

# See also: `iczt`, `czt_1d`
The code is based on Rabiner, Schafer & Rader  1969, IEEE Trans. on Audio and Electroacoustics, 17,86-92

# Arguments:
+ `xin`: array to transform
+ `scale`: a tuple of factors (one for each dimension) to zoom into during the czt. 
   Note that a factor of nothing (or 1.0) needs to be provided, if a dimension is not transformed.
+ `dims`: a tuple of dimensions over which to apply the czt.
+ `dsize`: a tuple specifying the destination size
+ `a`: defines the starting phase of the result CZT. This relates to the where the center of the destination 
       array should be. The default is `nothing` which means it is calculated from the `src_center` argument.
+ `w`: defines the consecutive phases of the result array, i.e. the zoom. It is (default `nothing`) 
       usually automatically calculated from the `scaled` and the `damp` argument.
       You only need to state it, if you want to use the low-level interface (e.g. for the Laplace transform).
+ `damp`: a multiplicative factor to apply as a damping coefficient to `w`.
+ `src_center`: position of the nominal central (zero-position) pixel in the source array. By default the 
                Fourier-center `size(src).÷2 .+1` is used.
+ `dst_center`: the center (zero-position) of the destination array. By default the 
                Fourier-center `size(dst).÷2 .+1` is used.
+ `remove_wrap`: if true, the positions that represent a wrap-around will be set to zero

# Example:

```jldoctest
julia> using IndexFunArrays

julia> sz = (10,10);

julia> xin = disc(sz,4)
10×10 Matrix{Float64}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  0.0  0.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0

julia> xft = czt(xin,(1.2,1.3));

julia> zoomed = real.(ift(xft))
10×10 Matrix{Float64}:
 -0.0197423    0.0233008  -0.0449251   0.00295724   0.205593  -0.166546   0.205593   0.00295724  -0.0449251   0.0233008
  0.0239759   -0.028264    0.0541186  -0.0116475   -0.261294   0.312719  -0.261294  -0.0116475    0.0541186  -0.028264
 -0.0569       0.0666104  -0.122277    0.140354     0.78259    1.34381    0.78259    0.140354    -0.122277    0.0666104
  0.00540611  -0.0117886   0.0837357   1.30651      1.8283     1.50127    1.8283     1.30651      0.0837357  -0.0117886
  0.11892     -0.147731    0.368046    1.76537      1.33218    1.66119    1.33218    1.76537      0.368046   -0.147731
 -0.00389861   0.0145979   1.21842     1.52989      1.67375    1.543      1.67375    1.52989      1.21842     0.0145979
  0.11892     -0.147731    0.368046    1.76537      1.33218    1.66119    1.33218    1.76537      0.368046   -0.147731
  0.00540611  -0.0117886   0.0837357   1.30651      1.8283     1.50127    1.8283     1.30651      0.0837357  -0.0117886
 -0.0569       0.0666104  -0.122277    0.140354     0.78259    1.34381    0.78259    0.140354    -0.122277    0.0666104
  0.0239759   -0.028264    0.0541186  -0.0116475   -0.261294   0.312719  -0.261294  -0.0116475    0.0541186  -0.028264
```
"""
function czt(xin::AbstractArray{T,D}, scale, dims=1:D, dsize=size(xin);
            a=nothing, w=nothing, damp=ones(D), src_center=size(xin).÷2 .+1, dst_center=dsize.÷2 .+1,
            remove_wrap=false, pad_value=zero(T), fft_flags=FFTW.ESTIMATE)::AbstractArray{complex(T), D} where {T,D}
    xout = xin
    if length(scale) != ndims(xin)
        error("Every of the $(ndims(xin)) dimension needs exactly one corresponding scale (zoom) factor, which should be equal to 1.0 for dimensions not contained in the dims argument.")
    end
    # check all the dims:
    for d = 1:D 
        if !(d in dims) && scale[d] != 1.0 && !isnothing(scale[d])
            error("The scale factor $(scale[d]) needs to be nothing or 1.0, if this dimension is not in the list of dimensions to transform.")
        end
    end

    for d in dims
        # in-place assignement is not possible, since with a zoom the size always changes.
        xout = czt_1d(xout, scale[d], d, dsize[d]; a=a, w=w, damp=damp[d], src_center=src_center[d], dst_center=dst_center[d], remove_wrap=remove_wrap, pad_value=pad_value, fft_flags=fft_flags)
    end
    return xout
end

"""
    iczt(xin ,scale, dims=1:length(size(xin)), dsize=size(xin,d); a=nothing, w=nothing, damp=1.0, 
         src_center=size(xin,d)÷2+1, dst_center=dsize÷2+1, remove_wrap=false, fft_flags=FFTW.ESTIMATE)

Inverse chirp z transform of the ND array `xin`
The tuple `scale` defines the zoom factors in the Fourier domain. Each has to be bigger than one.    
The code is based on Rabiner, Schafer & Rader  1969, IEEE Trans. on Audio and Electroacoustics, 17,86-92

# Arguments:
+ `xin`: array to transform
+ `scaled`: factor to zoom into during the 1-dimensional czt. 
+ `d`: single dimension to transform (as a tuple)
+ `dsize`: size of the destination array
+ `a`: defines the starting phase of the result CZT. This relates to the where the center of the destination 
       array should be. The default is `nothing` which means it is calculated from the `src_center` argument.
+ `w`: defines the consecutive phases of the result array, i.e. the zoom. It is (default `nothing`) usually 
       automatically calculated from the `scaled` and the `damp` argument.
       You only need to state it, if you want to use the low-level interface (e.g. for the Laplace transform).
+ `damp`: a multiplicative factor to apply as a damping coefficient to `w`.
+ `src_center`: position of the nominal central (zero-position) pixel in the source array. By default the 
                Fourier-center `size(src).÷2 .+1` is used.
+ `dst_center`: the center (zero-position) of the destination array. By default the 
                Fourier-center `size(dst).÷2 .+1` is used.
+ `remove_wrap`: if true, the positions that represent a wrap-around will be set to zero
+ `pad_value`: the value to pad wrapped data with. 

See also: `czt`, `czt_1d`

# Example

```jldoctest

julia> using IndexFunArrays

julia> sz = (10,10);

julia> xin = disc(sz,4)
10×10 Matrix{Float64}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  0.0  0.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0

julia> xft = ft(xin);

julia> iczt(xft,(1.2,1.3))
10×10 Matrix{ComplexF64}:
 0.00648614+0.0213779im  0.0165456+0.0357733im  0.0389356+0.0482465im  -0.235491-0.156509im    …  0.178484-0.0730099im  -0.245418-5.88331e-5im  0.0471654-0.0814548im  0.0141525+0.0734892im
  -0.104602-0.160481im   -0.163859-0.125535im    0.495205+0.135059im    0.660335+0.00736904im     0.764045-0.0497007im    0.67753+0.263814im      0.48095-0.0775406im  -0.159713-0.0637132im
   0.139304+0.111348im    0.454973+0.106869im    0.970263-0.0249785im    1.25999-0.166495im        1.07328-0.0481437im    1.24013-0.14664im      0.986722-0.0414382im   0.450186+0.111656im
  -0.035645-0.0311352im    1.03899-0.0589268im     1.1463-0.0940003im   0.790545+0.283668im       0.994255+0.134865im     0.80774-0.0124851im     1.13205+0.151519im     1.04314-0.130321im
   0.292575+0.0853233im   0.929883+0.0687029im    1.06514-0.0649952im   0.989483-0.019913im        1.02311+0.018235im    0.979555-0.136654im      1.07337+0.0317868im    0.92749+0.0405597im
    1.12254-0.0464723im    1.03467-0.0239316im    0.92709+0.0822984im     1.0521-0.0992709im   …  0.983655-0.0663123im     1.0521+0.0992709im     0.92709-0.0822984im    1.03467+0.0239316im
   0.287928-0.0306724im    0.92749-0.0405597im    1.07337-0.0317868im   0.979555+0.136654im        1.01648+0.0597475im   0.989483+0.019913im      1.06514+0.0649952im   0.929883-0.0687029im
 -0.0275957+0.169775im     1.04314+0.130321im     1.13205-0.151519im     0.80774+0.0124851im       1.00574+0.0629632im   0.790545-0.283668im       1.1463+0.0940003im    1.03899+0.0589268im
   0.130009-0.120643im    0.450186-0.111656im    0.986722+0.0414382im    1.24013+0.14664im         1.06002+0.0348813im    1.25999+0.166495im     0.970263+0.0249785im   0.454973-0.106869im
 -0.0965531+0.0404296im  -0.159713+0.0637132im    0.48095+0.0775406im    0.67753-0.263814im        0.77553-0.121603im    0.660335-0.00736904im   0.495205-0.135059im   -0.163859+0.125535im
```
"""
function iczt(xin ,scale, dims=1:ndims(xin), dsize=size(xin); a=nothing, w=nothing, damp=ones(ndims(xin)), src_center=size(xin).÷2 .+1, dst_center=dsize.÷2 .+1, remove_wrap=false, pad_value=zero(eltype(xin)), fft_flags=FFTW.ESTIMATE)
    factor = prod(size(xin)[[dims...]])
    conj(czt(conj(xin), scale, dims, dsize; a=a, w=w, damp=damp, src_center=src_center, dst_center=dst_center, remove_wrap=remove_wrap, pad_value=pad_value*factor, fft_flags=fft_flags)) / factor
end
