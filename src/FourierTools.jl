module FourierTools

include("utils.jl")

using PaddedViews, ShiftedArrays
using FFTW

export ft,ift, rft, irft, resample, resample_by_FFT, resample_by_RFFT

#= # This is the setindex function that used to be in PaddedViews
# copied from commit https://github.com/JuliaArrays/PaddedViews.jl/commit/ff689b1f5d41545f3decf1f00b94c5ad7b1d5ac8
Base.@propagate_inbounds function Base.setindex!(A::PaddedView{T, N}, v, i::Vararg{Int, N}) where {T, N}
    @boundscheck begin
        # This gives some performance boost https://github.com/JuliaLang/julia/issues/33273
        _throw_argument_error() = throw(ArgumentError("PaddedViews do not support (re)setting the padding value. Consider making a copy of the array first."))
        _throw_bounds_error(A, i) = throw(BoundsError(A, i))
        if checkbounds(Bool, A, i...)
            # checkbounds(Bool, parent(A), i...) || _throw_argument_error()
            # just ignore assignments in this region
        else
            _throw_bounds_error(A, i)
        end
    end
    setindex!(parent(A), v, i...)
    return A
end
 =#

##  This View checks for the index to be L1 or the mirrored version (L2)
# and then replaces the value by half of the parent at L1
struct FourierDuplicate{T,N, AA<:AbstractArray{T, N}} <: AbstractArray{T,N}
    parent::AA
    D::Int # dimension along which to apply to copy
    L1::Int # low index position to copy from (and half)
    L2::Int # high index positon to copy to (and half)

    function FourierDuplicate(parent::AA, D::Int,L1::Int) where {T,N, AA<:AbstractArray{T, N}}
        if ndims(parent) != N
            throw(DimensionMismatch("parent and indices should have the same dimension, instead they're $(ndims(parent)) and $N."))
        end
        mid = fft_center(size(parent)[D])
        L2 = mid + (mid-L1)
        return new{T,N, AA}(parent, D, L1, L2)
    end
end

Base.IndexStyle(::Type{FS}) where {FS<:FourierDuplicate} = IndexStyle(parenttype(FourierDuplicate))
parenttype(::Type{FourierDuplicate{T,N,AA}}) where {T,N,AA} = AA
parenttype(A::FourierDuplicate) = parenttype(typeof(FourierDuplicate))
Base.parent(A::FourierDuplicate) = A.parent 
Base.size(A::FourierDuplicate) = size(parent(A))

Base.size(A::FourierDuplicate) = size(parent(A))

@inline function Base.getindex(A::FourierDuplicate{T,N, <:AbstractArray{T, N}}, i::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, i...)
    if i[A.D]==A.L1
        @inbounds return parent(A)[i...] / 2
    elseif i[A.D]==A.L2
        @inbounds return parent(A)[replace_dim(i,A.D,A.L1)...] / 2
    else 
        @inbounds return parent(A)[i...]
    end
end

## This View checks for the index to be L1 
# and then replaces the value by add the value at the mirrored position L2
struct FourierSum{T,N, AA<:AbstractArray{T, N}} <: AbstractArray{T, N}
    parent::AA
    D::Int # dimension along which to apply to copy
    L1::Int # low index position to copy from (and half)
    L2::Int # high index positon to copy to (and half)

    function FourierSum(parent::AA, D::Int,L1::Int) where {T, N, AA<:AbstractArray{T, N}}
        if ndims(parent) != N
            throw(DimensionMismatch("parent and indices should have the same dimension, instead they're $(ndims(parent)) and $N."))
        end
        mid = fft_center(size(parent)[D])
        L2 = mid + (mid-L1)
        return new{T, N, AA}(parent, D, L1, L2)
    end
end
Base.IndexStyle(::Type{FS}) where {FS<:FourierSum} = IndexStyle(parenttype(FourierSum))
parenttype(::Type{FourierSum{T,N,AA}}) where {T,N,AA} = AA
parenttype(A::FourierSum) = parenttype(typeof(FourierSum))
Base.parent(A::FourierSum) = A.parent
Base.size(A::FourierSum) = size(parent(A))

@inline function Base.getindex(A::FourierSum{T,N, <:AbstractArray{T, N}}, i::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, i...)
    if i[A.D]==A.L1
        @inbounds return parent(A)[i...] + parent(A)[replace_dim(i,A.D,A.L2)...]
    else 
        @inbounds return parent(A)[i...]
    end
end


function extract(mat; new_size=size(mat), center=ft_center_0(mat).+1)
    oldcenter = ft_center_0(new_size).+1
    PaddedView(0,mat,new_size, oldcenter .- center.+1);
end

function ft_pad(mat, new_size)
    return extract(mat;new_size=new_size)
end

function rft_pad(mat, new_size)
    c2 = rft_center_0(mat)
    c2 = replace_dim(c2,1,new_size[1].÷2);
    return extract(mat;new_size=new_size, center=c2.+1)
end

function ft_fix_before(mat, size_old, size_new; start_dim=1)
    for d = start_dim:ndims(mat)
        sn = size_new[d]
        so = size_old[d]
        if sn < so && iseven(sn)
            L1 = (size_old[d] -size_new[d] )÷2 +1
            mat = FourierSum(mat, d, L1)
        end
    end
    return mat
end

function ft_fix_after(mat,size_old,size_new; start_dim=1)
    for d=start_dim:ndims(mat)
        sn = size_new[d]
        so = size_old[d]
        if sn > so && iseven(so)
            L1 = (size_new[d]-size_old[d])÷2+1
            mat = FourierDuplicate(mat,d,L1)
        end
        # if equal do nothing
    end
    return mat
end


function rft_fix_before(mat,size_old,size_new)
    ft_fix_before(mat,size_old,size_new;start_dim=2) # ignore the first dimension
end



function rft_fix_after(mat,size_old,size_new)
    ft_fix_after(mat,size_old,size_new;start_dim=2) # ignore the first dimension
end

"""
performs the necessary Fourier-space operations of resampling
in the space of ft (meaning the already circshifted version of fft).
"""
function extract_ft(mat,new_size)
    old_size = size(mat)
    mat_fixed_before = ft_fix_before(mat,old_size,new_size)
    mat_pad = ft_pad(mat_fixed_before,new_size)
    # afterwards we add the highest pos. frequency to the highest lowest one 
    return ft_fix_after(mat_pad ,old_size,new_size)
end

function resample_by_FFT(mat, new_size; take_real=false) 
    # for complex arrays we don't need to restore hermitian property
    if eltype(mat) <: Complex
        res = ft_pad(ft(mat),new_size)
    else
        # for real arrays we apply an operation so that mat_fixed_before is hermitian
        res = ft(mat)
        res = extract_ft(res, new_size)
    end
    # go back to real space
    # @show typeof(res)
    # return res
    res = ift(res)
    if eltype(mat) <: Real && take_real
        real(res)
    else
        res
    end
end
"""
performs the necessary Fourier-space operations of resampling
in the space of rft (meaning the already circshifted version of rfft).
"""
function extract_rft(mat,new_size)
    rft_old_size = size(mat)
    rft_new_size = replace_dim(new_size,1,new_size[1]÷2 +1)
    return rft_fix_after(rft_pad(
        rft_fix_before(mat,rft_old_size,rft_new_size),
        rft_new_size),rft_old_size,rft_new_size)
end

function resample_by_RFFT(mat, new_size) where {T}
    rf = rft(mat)
    irft(extract_rft(rf,new_size),new_size[1])
end
function resample(mat, new_size)
    if eltype(mat) <: Complex
        resample_by_FFT(mat,new_size,TypeFT)
    else
        resample_by_RFFT(mat,new_size,TypeRFT)
    end
end
end # module
