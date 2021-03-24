function fft_center(x)
    return x ÷ 2 + 1
end

ft(mat) = ft_shift(fft(mat));
ift(mat) = ifft(collect(ift_shift(mat)));
rft(mat) = rft_shift(rfft(mat));
irft(mat,d) = irfft(collect(irft_shift(mat)),d);

ft_shift(mat) = ShiftedArrays.circshift(mat, ft_center_0(mat))
ift_shift(mat) = ShiftedArrays.circshift(mat, .-(ft_center_0(mat)))
rft_shift(mat) = ShiftedArrays.circshift(mat, rft_center_0(mat))
irft_shift(mat) = ShiftedArrays.circshift(mat ,.-(rft_center_0(mat)))

@inline function replace_dim(iterable::NTuple{N,T}, dim::Int, val::Int)::NTuple{N, T} where{N,T}
    return ntuple(d -> d==dim ? val : iterable[d], Val(N))
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

