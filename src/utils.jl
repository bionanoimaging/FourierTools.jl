export ft,ift, rft, irft


"""
    fft_center(x)

Returns the center of a size in Fourier sense and Julia 
1-based indices.
"""
function fft_center(x)
    return x ÷ 2 + 1
end


"""
    ft(A [, dims])

Result is semantically equivalent to `fftshift(fft(A, dims), dims)`
However, the shift is done with `ShiftedArrays` and therefore doesn't allocate memory.
"""
function ft(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    return fftshift_view(fft(mat, dims), dims)
end

"""
    ift(A [, dims])

Result is semantically equivalent to `ifft(ifftshift(A), dims), dims)`
"""
function ift(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    # remove ift shift
    # return ifft(collect(ifftshift_view(mat, dims)), dims);
    return ifft(ifftshift(mat, dims), dims)
end

"""
    fftshift_view(A [, dims])

Result is semantically equivalent to `fftshift(A, dims)` but returns 
a view view instead. 
"""
function fftshift_view(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    ShiftedArrays.circshift(mat, ft_center_0(size(mat), dims))
end

"""
    ifftshift_view(A [, dims])

Result is semantically equivalent to `fftshift(A, dims)` but returns 
a view view instead. 
"""
function ifftshift_view(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    ShiftedArrays.circshift(mat, .-(ft_center_0(size(mat), dims)))
end




function rft(mat)
    rfftshift_view(rfft(mat));
end

function irft(mat,d)
    irfft(collect(irfftshift_view(mat)),d);
end


function rfftshift_view(mat)
    ShiftedArrays.circshift(mat, rft_center_0(size(mat), dims))
end

function irfftshift_view(mat)
    ShiftedArrays.circshift(mat ,.-(rft_center_0(size(mat), dims)))
end



@inline function replace_dim(iterable::NTuple{N,T}, dim::Int, val::Int)::NTuple{N, T} where{N,T}
    return ntuple(d -> d==dim ? val : iterable[d], Val(N))
end


# attention: all the center functions are zero-based as they are applied in shifts!
# function ft_center_0(sz::NTuple) 
#     (sz.÷2)
# end 

function ft_center_0(s::NTuple{N, T}, dims=ntuple(identity, Val(N))) where {N, T}
    ntuple(i -> i ∈ dims ?  s[i] ÷ 2 : 0 , N)
end


function rft_center_0(sz::NTuple)
    Tuple(d == 1 ? 0 : sz[d].÷2 for d in 1:length(sz))
end

function rft_center_0(mat :: AbstractArray) 
    rft_center_0(size(mat))
end


function selectsizes(x::AbstractArray{T},dim::NTuple{N,Int};
                    keep_dims=true) where{T,N}
    if ~keep_dims
        return map(n->size(x,n),dim)
    end
    sz = ones(Int, ndims(x))
    for n in dim
        sz[n] = size(x,n) 
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

