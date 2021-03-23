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

function replace_dim(iterable::NTuple{T,N}, dim, val) where{T,N}
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
