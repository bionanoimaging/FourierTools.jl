module FieldPropagation

# import Napari
using PaddedViews, ShiftedArrays

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


nap = (x) -> Napari.napari.view_image(x)
function vv(mat; gamma=nothing, viewer=nothing)
    #if isa(mat,FFTView)
    #    start = CartesianIndex(size(mat).÷2)
    #    stop = CartesianIndex(size(mat).÷2 .+size(mat))
    #    mat = mat[start:stop]
    #end
    if isnothing(gamma)
        if  eltype(mat) <: Complex
            gamma=0.2
        else
            gamma=1.0
        end
    end
    mat = (abs.(collect(mat))).^gamma
    mat = (mat./maximum(mat))
    if isnothing(viewer)
        return Gray.(mat)
    else
        viewer(mat)
    end
end

function size_d(x::AbstractArray{T},dims::NTuple{N,Int}; keep_dims=true) where{T,N}
    if ~keep_dims
        return map(n->size(x,n),dims)
    end
    sz=ones(Int, ndims(x))
    for n in dims
        sz[n]=size(x,n) 
    end
    return Tuple(sz)
end 

@inline function replace_dim(iterable::NTuple{T,N}, dim, val) where{T,N}
    Tuple(d == dim ? val : iterable[d] for d in 1:length(iterable))
end
# attention: all the center functions are zero-based as they are applied in shifts!

function ft_center_0(sz::NTuple) 
    (sz.÷2)
end 

function ft_center_0(mat :: AbstractArray) 
    ft_center_0(size(mat))
end

function rft_center_0(sz::NTuple)
    Tuple(d == 1 ? 0 : sz[d].÷2 for d in 1:length(sz))
end

function rft_center_0(mat :: AbstractArray) 
    rft_center_0(size(mat))
end

##  This View checks for the index to be L1 or the mirrored version (L2)
# and then replaces the value by half of the data at L1
struct FourierDuplicate{T,N,A} <: AbstractArray{T,N}
    data::A
    D::Int64 # dimension along which to apply to copy
    L1::Int64 # low index position to copy from (and half)
    L2::Int64 # high index positon to copy to (and half)

    function FourierDuplicate{T,N,A}(data,
                                     D::Int64,L1::Int64) where {T,N,A}
        ndims(data) == N || throw(DimensionMismatch("data and indices should have the same dimension, instead they're $(ndims(data)) and $N."))
        mid=size(data)[D]÷2+1
        L2=mid+(mid-L1)
        new{T,N,A}(data, D, L1,L2)
    end
end
function FourierDuplicate(data::AbstractArray{T,N},D,L1) where {T,N}
    FourierDuplicate{T,N,AbstractArray{T,N}}(data,D,L1)
end
Base.size(A::FourierDuplicate) = size(A.data)

Base.@propagate_inbounds function Base.getindex(A::FourierDuplicate{T,N}, i::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, i...)
    if i[A.D]==A.L1
        return A.data[i...] / 2
    elseif i[A.D]==A.L2
        return A.data[replace_dim(i,A.D,A.L1)...] / 2
    else 
        return A.data[i...]
    end
end

## This View checks for the index to be L1 
# and then replaces the value by add the value at the mirrored position L2
struct FourierSum{T,N,A} <: AbstractArray{T,N}
    data::A
    D::Int64 # dimension along which to apply to copy
    L1::Int64 # low index position to copy from (and half)
    L2::Int64 # high index positon to copy to (and half)

    function FourierSum{T,N,A}(data,
                                     D::Int64,L1::Int64) where {T,N,A}
        ndims(data) == N || throw(DimensionMismatch("data and indices should have the same dimension, instead they're $(ndims(data)) and $N."))
        mid=size(data)[D]÷2+1
        L2=mid+(mid-L1)
        new{T,N,A}(data, D, L1,L2)
    end
end
function FourierSum(data::AbstractArray{T,N},D,L1) where {T,N}
    FourierSum{T,N,AbstractArray{T,N}}(data,D,L1)
end
Base.size(A::FourierSum) = size(A.data)

Base.@propagate_inbounds function Base.getindex(A::FourierSum{T,N}, i::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, i...)
    if i[A.D]==A.L1
        return A.data[i...] + A.data[replace_dim(i,A.D,A.L2)...]
    else 
        return A.data[i...]
    end
end

ft_shift = (mat) -> ShiftedArrays.circshift(mat,ft_center_0(mat))
ift_shift = (mat) -> ShiftedArrays.circshift(mat,.-(ft_center_0(mat)))
rft_shift = (mat) -> ShiftedArrays.circshift(mat,rft_center_0(mat))
irft_shift = (mat) -> ShiftedArrays.circshift(mat,.-(rft_center_0(mat)))

ft = (mat) -> ft_shift(fft(mat));
ift = (mat) -> ifft(collect(ift_shift(mat)));
rft = (mat) -> rft_shift(rfft(mat));
irft = (mat,d) -> irfft(collect(irft_shift(mat)),d);

function extract(mat; newsize=size(mat), center=ft_center_0(mat).+1)
    oldcenter = ft_center_0(newsize).+1
    PaddedView(0,mat,newsize, oldcenter .- center.+1);
end

function ft_pad(mat, newsize)
    return extract(mat;newsize=newsize)
end

function rft_pad(mat, newsize)
    c2 = rft_center_0(mat)
    c2 = replace_dim(c2,1,newsize[1].÷2);
    return extract(mat;newsize=newsize, center=c2.+1)
end

function ft_fix_before(mat,size_old,size_new; startDim=1)
    for d=startDim:ndims(mat)
        sn = size_new[d]
        so = size_old[d]
        if sn < so && iseven(sn)
            L1 = (size_old[d]-size_new[d])÷2+1
            mat = FourierSum(mat,d,L1)
        end
        # if equal do nothing
    end
    mat
end

function ft_fix_after(mat,size_old,size_new; startDim=1)
    for d=startDim:ndims(mat)
        sn = size_new[d]
        so = size_old[d]
        if sn > so && iseven(so)
            L1 = (size_new[d]-size_old[d])÷2+1
            mat = FourierDuplicate(mat,d,L1)
        end
        # if equal do nothing
    end
    mat
end

function rft_fix_before(mat,size_old,size_new)
    ft_fix_before(mat,size_old,size_new;startDim=2) # ignore the first dimension
end
function rft_fix_after(mat,size_old,size_new)
    ft_fix_after(mat,size_old,size_new;startDim=2) # ignore the first dimension
end

# Note that for complex no fft_fix needs to be applied
function ft_resize(mat, newsize; keep_complex=false)
    oldsize = size(mat)
    if eltype(mat) <: Complex
        res=ift(ft_pad(ft(mat),newsize))
    else
        res=(ift(ft_fix_after(ft_pad(
            ft_fix_before(ft(mat),oldsize,newsize),
            newsize),oldsize,newsize)))
    end
    if keep_complex
        res
    else
        real(res)
    end
end

function rft_resize(mat, newsize)
    @time rf = rft(mat)
    rft_oldsize = size(rf)
    rft_newsize = replace_dim(newsize,1,newsize[1]÷2 +1)
    @time irft(rft_fix_after(rft_pad(
        rft_fix_before(rf,rft_oldsize,rft_newsize),
        rft_newsize),rft_oldsize,rft_newsize),newsize[1])
end

end # module

## for testing
using FFTW, TestImages, Colors # , PaddedViews
img = testimage("resolution_test_512.tif"); # fullname
# imgg = Gray.(img);
mat0 = convert(Array{Float64}, img);
mat=mat0[1:511,1:512];

vv(mat)
newsize=(1024,500)
@time res = ft_resize(mat, newsize; keep_complex=true);
maximum(imag(res))
vv(res)
@time res = rft_resize(mat, newsize);
vv(res)

w=ft(mat)
q=ft_pad(w,newsize)
r=ift(q)
