export rft_size, fft_center, fftpos
export expanddims, fourierspace_pixelsize, realspace_pixelsize
export δ


 #get_RFT_scale(real_size) = 0.5 ./ (max.(real_size ./ 2, 1))  # The same as the FFT scale but for the full array in real space!

"""
    δ([T,] sz, pos=FourierTools.fft_center.(sz))

Return an array which has `1` at `pos` in the 
array of size `sz`.

 # Examples
```jldoctest
julia> δ((3, 3))
3×3 Matrix{Int64}:
 0  0  0
 0  1  0
 0  0  0

julia> δ(Float32, (4, 3))
4×3 Matrix{Float32}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  0.0

julia> δ(Float32, (3, 3), (1,1))
3×3 Matrix{Float32}:
 1.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
```
"""
function δ(T::Type, sz, pos=fft_center.(sz))
    z = zeros(T, sz)
    z[pos...] = one(T)
    return z 
end

function δ(sz, pos=fft_center.(sz))
    return δ(Int, sz, pos)
end



"""
    fftpos(L, N)
Construct a range from -L/2 to L/2.
However, we ensure that everything is centered around the center
in a way that a FFT interpretes it correctly.
For odd sequences it is indeed in the real center.
For even sequences the center is at `N/2 + 1`.
 # Examples
```jldoctest
julia> collect(fftpos(1, 4))
4-element Array{Float64,1}:
 -0.5
 -0.25
  0.0
  0.25
julia> collect(fftpos(1, 5))
5-element Array{Float64,1}:
 -0.5
 -0.25
  0.0
  0.25
  0.5
```
"""
function fftpos(l, N)
    if N % 2 == 0
        dx = l / N
        return range(-l/2, l/2-dx, length=N)
    else
        return range(-l/2, l/2, length=N) 
    end
end




"""
    fft_center(x)

Returns the center of a size in Fourier sense and Julia 
1-based indices.
"""
function fft_center(x)
    return x ÷ 2 + 1
end

"""
    rft_size(sz::NTuple{Int})

Returns the size of an rft or rfft performed on the data x, without performing the rfft.
sz: corresponding real space size to obtain the rft size for
"""
function rft_size(sz::NTuple{N, Int}, dim=1) where {N}
    return Base.setindex(sz,sz[dim]÷2+1,dim)
end

"""
    rft_size(arr)

Returns the size of an rft or rfft performed on the data x, without performing the rfft.

arr: array to optain the corresponding rft size for
"""
function rft_size(arr, dim=1)
    return rft_size(size(arr),dim)
end


"""
    ft_center_diff(s [, dims])

Calculates how much each dimension must be shifted that the
center frequency is at the Fourier center.
This if for a normal `fft`
"""
function ft_center_diff(s::NTuple{N, T}, dims=ntuple(identity, Val(N))) where {N, T}
    ntuple(i -> i ∈ dims ?  s[i] ÷ 2 : 0 , N)
end


"""
    rft_center_diff(s [, dims])

Calculates how much each dimension must be shifted that the
center frequency is at the Fourier center.
This is for `rfft`. The `dims[1]` must be therefore not shifted!
"""
function rft_center_diff(sz::NTuple{N, T}, dims=ntuple(identity, Val(N))::NTuple) where {N,T}
    ntuple(i -> i == first(dims) ? 0 : i ∈ dims ? sz[i] ÷ 2 : 0, N)
    # Tuple(d == 1 ? 0 : sz[d].÷2 for d in 1:length(sz))
end




"""
    selectsizes(x, dism; keep_dims=true)

Select the sizes of `x` for all `dims`
If `keep_dims=true` the non-selected dimensions are
returned as 1.

# Examples
```jldoctest
julia> FourierTools.selectsizes(randn((4,3,2)), (2,3))
(1, 3, 2)

julia> FourierTools.selectsizes(randn((4,3,2)), (2,3), keep_dims=false)
(3, 2)
```

"""
function selectsizes(x::AbstractArray{T},dims::NTuple{N,Int};
                    keep_dims=true) where{T,N}
    if ~keep_dims
        return map(n->size(x,n),dims)
    end
    sz = ones(Int, ndims(x))
    for n in dims
        sz[n] = size(x,n) 
    end
    return Tuple(sz)
end 



"""
    slice(arr, dim, index)
Return a `N` dimensional slice (where one dimensions has size 1) of the N-dimensional `arr` at the index position
`index` in the `dim` dimension of the array.
It holds `size(out)[dim] == 1`.
# Examples
```jldoctest
julia> x = [1 2 3; 4 5 6; 7 8 9]
3×3 Matrix{Int64}:
 1  2  3
 4  5  6
 7  8  9

julia> FourierTools.slice(x, 1, 1)
1×3 view(::Matrix{Int64}, 1:1, 1:3) with eltype Int64:
 1  2  3
```
"""
function slice(arr::AbstractArray{T, N}, dim::Integer, index::Integer) where {T, N}
    inds = slice_indices(axes(arr), dim, index)
    return @view arr[inds...]
end

"""
    slice_indices(a, dim, index)

`a` should be the axes obtained by `axes(arr)` of an array.
`dim` is the dimension to be selected and `index` the index of it.

# Examples
```jldoctest
julia> FourierTools.slice_indices((1:10, 1:20, 1:12, 1:33), 1, 3)
(3:3, 1:20, 1:12, 1:33)
```
"""
function slice_indices(a::NTuple{N, T}, dim::Integer, index::Integer) where {T, N}
    inds = ntuple(i -> i == dim ? (a[i][index]:a[i][index])
                                : (first(a[i]):last(a[i])), 
                  Val(N))
    return inds
end


"""
    rfft_size(size, dims)

Returns the size `rfft` would return if applied
to a real array. `size` is the input size to `rfft` 
and `dims` the dimensions the `rfft` transforms over.
Actually we only would need `first(dims)`.

```jldoctest
julia> using FFTW

julia> rfft((ones((4,3,2))), (2,3)) |> size
(4, 2, 2)

julia> FourierTools.rfft_size((4,3,2), (2, 3))
(4, 2, 2)
```
"""
function rfft_size(size, dims)
    dim = first(dims)
    Base.setindex(size, size[dim] ÷ 2 + 1, dim)
end



"""
    expanddims(x, ::Val{N})
    expanddims(x, N::Number)

expands the dimensions of an array to a given number of dimensions.

Try to prefer the `Val` version because this is type-stable.
`Val(N)` encapsulates the number in a type from which the compiler
can then infer the return type.

# Examples
The result is a 5D array with singleton dimensions at the end
```jldoctest
julia> expanddims(ones((1,2,3)), Val(5))
1×2×3×1×1 Array{Float64, 5}:
[:, :, 1, 1, 1] =
 1.0  1.0

[:, :, 2, 1, 1] =
 1.0  1.0

[:, :, 3, 1, 1] =
 1.0  1.0

julia> expanddims(ones((1,2,3)), 5)
1×2×3×1×1 Array{Float64, 5}:
[:, :, 1, 1, 1] =
 1.0  1.0

[:, :, 2, 1, 1] =
 1.0  1.0

[:, :, 3, 1, 1] =
 1.0  1.0
```
"""
function expanddims(x, N::Number)
    return reshape(x, (size(x)..., ntuple(x -> 1, (N - ndims(x)))...))
end

function expanddims(x, ::Val{N}) where N
    return reshape(x, (size(x)..., ntuple(x -> 1, (N - ndims(x)))...))
end

"""
    get_indices_around_center(i_in, i_out)
A function which provides two output indices `i1` and `i2`
where `i2 - i1 = i_out`
The indices are chosen in a way that the set `i1:i2`
cuts the interval `1:i_in` in a way that the center frequency
stays at the center position.
Works for both odd and even indices
"""
function get_indices_around_center(i_in, i_out)
    if (mod(i_in, 2) == 0 && mod(i_out, 2) == 0 
     || mod(i_in, 2) == 1 && mod(i_out, 2) == 1) 
        x = (i_in - i_out) ÷ 2
        return 1 + x, i_in - x
    elseif mod(i_in, 2) == 1 && mod(i_out, 2) == 0
        x = (i_in - 1 - i_out) ÷ 2
        return 1 + x, i_in - x - 1 
    elseif mod(i_in, 2) == 0 && mod(i_out, 2) == 1
        x = (i_in - (i_out - 1)) ÷ 2
        return 1 + x, i_in - (x - 1)
    end
end


"""
    center_extract(arr, new_size_array)
Extracts a center of an array. 
`new_size_array` must be list of sizes indicating the output
size of each dimension. Centered means that a center frequency
stays at the center position. Works for even and uneven.
If `length(new_size_array) < length(ndims(arr))` the remaining dimensions
are untouched and copied.
# Examples
```jldoctest
julia> FourierTools.center_extract([1 2; 3 4], [1])
1×2 view(::Matrix{Int64}, 2:2, 1:2) with eltype Int64:
 3  4

julia> FourierTools.center_extract([1 2; 3 4], [1, 1])
1×1 view(::Matrix{Int64}, 2:2, 2:2) with eltype Int64:
 4

julia> FourierTools.center_extract([1 2 3; 3 4 5; 6 7 8], [2 2])
2×2 view(::Matrix{Int64}, 1:2, 1:2) with eltype Int64:
 1  2
 3  4
```
"""
function center_extract(arr::AbstractArray, new_size_array)
    new_size_array = collect(new_size_array)

    # we construct two lists
    # the reason is, that we don't change higher dimensions which are not 
    # specified in new_size_array
    out_indices1 = [get_indices_around_center(size(arr)[x], new_size_array[x]) 
                    for x = 1:length(new_size_array)]
    
    out_indices1 = [x[1]:x[2] for x = out_indices1]
    
    # out_indices2 contains just ranges covering the full size of each dimension
    out_indices2 = [1:size(arr)[i] for i = (1 + length(new_size_array)):ndims(arr)]
    return @view arr[out_indices1..., out_indices2...]
end


"""
    center_set!(arr_large, arr_small)
Puts the `arr_small` central into `arr_large`.
The convention, where the center is, is the same as the definition
as for FFT based centered.
Function works both for even and uneven arrays.
# Examples
```jldoctest
julia> FourierTools.center_set!([1, 1, 1, 1, 1, 1], [5, 5, 5])
6-element Array{Int64,1}:
 1
 1
 5
 5
 5
 1
```
"""
function center_set!(arr_large, arr_small)
    out_is = []
    for i = 1:ndims(arr_large)
        a, b = get_indices_around_center(size(arr_large)[i], size(arr_small)[i])
        push!(out_is, a:b)
    end

    #rest = ones(Int, ndims(arr_large) - 3)
    arr_large[out_is...] = arr_small
    
    return arr_large
end


"""
    center_pos(x)
Calculate the position of the center frequency.
Size of the array is `x`
# Examples
```jldoctest
julia> FourierTools.center_pos(3)
2
julia> FourierTools.center_pos(4)
3
```
"""
function center_pos(x::Integer)
    # integer division
    return div(x, 2) + 1
end



"""
    fourierspace_pixelsize(realspace_pixelsize, mysize)
converts a real space pixel pitch `realspace_pixelsize` into a Fourier-space pixel size.
This applies to all types of fft, ft, rft, or rfft alike.
Arguments:
+ realspace_pixelsize: pixel pitch in real space
+ mysize: the size of the array that is fourier transformed.
"""
function fourierspace_pixelsize(realspace_pixelsize, mysize)
    1.0 ./ (realspace_pixelsize .* mysize)
end

"""
    realspace_pixelsize(fourier_pixelsize, mysize)
converts a fourier space pixel pitch `fourier_pixelsize` into a pixel pitch in real space.
This applies to all types of ifft, ift, irft, or irfft alike.
Arguments:
+ fourier_pixelsize: pixel pitch in real space
+ mysize: the size of the array that is fourier transformed.
"""
function realspace_pixelsize(fourier_pixelsize, mysize)
    1.0 ./ (fourier_pixelsize .* mysize) 
end



"""
    eltype_error(T1, T2)

Throws an error of `T1 != T2`
"""
function eltype_error(T1, T2)
    if T1 != T2
        throw(ArgumentError("The element types of the first and second array are different ($T1) != $T2)). Please convert them to the same eltype."))
    end
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
