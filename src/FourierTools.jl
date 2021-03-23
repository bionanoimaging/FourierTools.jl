module FourierTools

include("utils.jl")

using PaddedViews, ShiftedArrays
using FFTW

export ft,ift, rft, irft, resample, resample_rfft

#= # This is the setindex function that used to be in PaddedViews
# copied from commit https://github.com/JuliaArrays/PaddedViews.jl/commit/ff689b1f5d41545f3decf1f00b94c5ad7b1d5ac8
Base.@propagate_inbounds function Base.setindex!(A::PaddedView{T, N}, v, i::Vararg{Int, N}) where {T, N}
    @boundscheck begin
        # This gives some performance boost https://github.com/JuliaLang/julia/issues/33273
        _throw_argument_error() = throw(ArgumentError("PaddedViews do not support (re)setting the padding value. Consider making a copy of the array first."))
        _throw_bounds_error(A, i) = throw(BoundsError(A, i))
        if checkbounds(Bool, A, i...)
            # checkbounds(Bool, A.data, i...) || _throw_argument_error()
            # just ignore assignments in this region
        else
            _throw_bounds_error(A, i)
        end
    end
    setindex!(A.data, v, i...)
    return A
end
 =#

##  This View checks for the index to be L1 or the mirrored version (L2)
# and then replaces the value by half of the data at L1
struct FourierDuplicate{T,N, AA<:AbstractArray{T, N}} <: AbstractArray{T,N}
    data::AA
    D::Int # dimension along which to apply to copy
    L1::Int # low index position to copy from (and half)
    L2::Int # high index positon to copy to (and half)

    function FourierDuplicate(data::AA, D::Int,L1::Int) where {T,N, AA<:AbstractArray{T, N}}
        if ndims(data) != N
            throw(DimensionMismatch("data and indices should have the same dimension, instead they're $(ndims(data)) and $N."))
        end
        mid = fft_center(size(data)[D])
        L2 = mid + (mid-L1)
        return new{T,N, AA}(data, D, L1, L2)
    end
end

Base.size(A::FourierDuplicate) = size(A.data)

@inline function Base.getindex(A::FourierDuplicate{T,N, <:AbstractArray{T, N}}, i::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, i...)
    if i[A.D]==A.L1
        @inbounds return A.data[i...] / 2
    elseif i[A.D]==A.L2
        @inbounds return A.data[replace_dim(i,A.D,A.L1)...] / 2
    else 
        @inbounds return A.data[i...]
    end
end

## This View checks for the index to be L1 
# and then replaces the value by add the value at the mirrored position L2
struct FourierSum{T,N, AA<:AbstractArray{T, N}} <: AbstractArray{T, N}
    data::AA
    D::Int # dimension along which to apply to copy
    L1::Int # low index position to copy from (and half)
    L2::Int # high index positon to copy to (and half)

    function FourierSum(data::AA, D::Int,L1::Int) where {T, N, AA<:AbstractArray{T, N}}
        if ndims(data) != N
            throw(DimensionMismatch("data and indices should have the same dimension, instead they're $(ndims(data)) and $N."))
        end
        mid = fft_center(size(data)[D])
        L2 = mid + (mid-L1)
        return new{T, N, AA}(data, D, L1, L2)
    end
end

Base.size(A::FourierSum) = size(A.data)

@inline function Base.getindex(A::FourierSum{T,N, <:AbstractArray{T, N}}, i::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, i...)
    if i[A.D]==A.L1
        @inbounds return A.data[i...] + A.data[replace_dim(i,A.D,A.L2)...]
    else 
        @inbounds return A.data[i...]
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



function resample(mat, new_size; take_real=true)
    old_size = size(mat)
    # for complex arrays we don't need to restore hermitian property
    if eltype(mat) <: Complex
        res = ft_pad(ft(mat),new_size)
    else
        # for real arrays we apply an operation so that mat_fixed_before is hermitian
        mat_fixed_before = ft_fix_before(ft(mat),old_size,new_size)
        mat_pad = ft_pad(mat_fixed_before,new_size)
        # afterwards we add the highest pos. frequency to the highest lowest one 
        res = ft_fix_after(mat_pad ,old_size,new_size)
    end
    # go back to real space
    @show typeof(res)
    res = ift(res)
    if eltype(mat) <: Real && take_real
        real(res)
    else
        res
    end
end

function resample_rfft(mat, new_size)
    rf = rft(mat)
    rft_old_size = size(rf)
    rft_new_size = replace_dim(new_size,1,new_size[1]÷2 +1)
    irft(rft_fix_after(rft_pad(
        rft_fix_before(rf,rft_old_size,rft_new_size),
        rft_new_size),rft_old_size,rft_new_size),new_size[1])
end

end # module
