export rft_size, fft_center, fftpos
export expanddims, fourierspace_pixelsize, realspace_pixelsize
export δ
export fourier_reverse!

"""
    bc_size(arr1, arr2)

Calculates the size of the broadcasted array of `arr1` and `arr2`.

# Arguments
- `arr1`: first array
- `arr2`: second array

# Examples
```jldoctest
julia> FourierTools.bc_size(rand(5, 2, 3), rand(1, 2))
(5, 2, 3)
```
"""
function bc_size(arr1, arr2)
    md = max(ndims(arr1), ndims(arr2))
    return ntuple((d) -> max(size(arr1, d), size(arr2, d)), md)
end

"""
    similar_zeros(arr::AbstractArray, sz::NTuple)

Creates a similar array to `arr` with zeros. This is useful to also support CuArrays.
There are specializations for `Array` and `CuArray` which use the original `zeros` function.

# parameters
- `arr`: array to copy the type and size from
- `sz`: size of the new array. Default is the size of `arr`.

# Examples
```jldoctest
julia> FourierTools.similar_zeros([1, 2, 3], (3,))
3-element Vector{Int64}:
 0
 0
 0
```
"""
function similar_zeros(arr::AbstractArray, sz::NTuple=size(arr))
    res = similar(arr, sz)
    fill!(res, zero(eltype(res)))
    return res
end

function similar_zeros(arr::Array, sz::NTuple=size(arr))
    zeros(eltype(arr), sz)
end

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
    fftpos(L, N, around=CenterFirst::Center)

Construct a range from -L/2 to L/2 around `around`

However, we ensure that those positions are in a way
which they are useful for FFT operations.
This means, that depending on the center a small 
offset is subtracted.

See `NDTools.Center` for all center options.
You need to load `using NDTools` to access all center options.

 # Examples
```jldoctest
julia> collect(fftpos(1,4))
4-element Vector{Float64}:
 0.0
 0.2916666666666667
 0.5833333333333334
 0.875

julia> collect(fftpos(1,5))
5-element Vector{Float64}:
 0.0
 0.225
 0.45
 0.675
 0.9

julia> using NDTools

julia> collect(fftpos(1,4, CenterFirst))
4-element Vector{Float64}:
 0.0
 0.2916666666666667
 0.5833333333333334
 0.875

julia> collect(fftpos(1,4, CenterFT))
4-element Vector{Float64}:
 -0.5833333333333333
 -0.29166666666666663
  3.70074341541719e-17
  0.2916666666666667

julia> collect(fftpos(1,4, CenterMiddle))
4-element Vector{Float64}:
 -0.4375
 -0.14583333333333334
  0.14583333333333334
  0.4375
```
"""
function fftpos(l, N)
    # default
    fftpos(l, N, CenterFirst)
end

function fftpos(l, N, around::Type{CenterFirst})
    return fftpos(l, N, 1)
end

function fftpos(l, N, around::Type{CenterLast})
    return fftpos(l, N, N) 
end

function fftpos(l, N, around::Type{CenterFT})
    return fftpos(l, N, N ÷ 2 + 1) 
end

function fftpos(l, N, around::Type{CenterMiddle})
    return fftpos(l, N, (N+1) / 2)
end

"""
    fftpos(l, N, around)

Another `fftpos` method where the range is constructed
around `around`. `around` is here a number indicating
the index position around the range is constructed
"""
function fftpos(l::AbstractFloat, N, around::Number)
    dx = l / N
    fraction = typeof(l)(around - 1) / (N - 1)
    return range(0 - (l-dx) * fraction, 
                 0 + (l-dx) * (1-fraction),
                 length=N)
end

function fftpos(l::Integer, N, around::Number)
    fftpos(Float64(l), N, around)
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
    get_indices_around_center(i_in, i_out)

A function which provides two output indices `i1` and `i2`
where `i2 - i1 = i_out`
The indices are chosen such that the set `i1:i2`
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
    get_indexrange_around_center(arr_1, arr_2)

A function which provides a range of output indices `i1:i2`
where `i2 - i1 = i_out`
The indices are chosen in a way that the set `i1:i2`
cuts the interval `1:i_in` such that the center frequency
stays at the center position.
Works for both odd and even indices
"""
function get_indexrange_around_center(arr_1, arr_2)
    sz1 = size(arr_1)
    sz2 = size(arr_2)
    all_rng = ntuple((d) -> begin a,b = get_indices_around_center(sz1[d], sz2[d]); a:b end, ndims(arr_1))
    return all_rng
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
    arr_large[get_indexrange_around_center(arr_large, arr_small)...] = arr_small
    
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

 """
    odd_view(arr)

creates a view of `arr` that for each even dimension excludes the
starting index yielding a view of the array with only odd dimensions.
This is useful for operations in Fourier-space which should leave the first index unaltered
such as reverse!
Note that an array reversal can also be achieved by using two ffts instead of one fft and one ifft.

# Examples
```jldoctest
julia> odd_view([1 2 3; 4 5 6])
1×3 view(::Matrix{Int64}, 2:2, 1:3) with eltype Int64:
 4  5  6
```
"""
function odd_view(arr)
    s_idx =  ntuple((d)->firstindex(arr,d) + iseven.(size(arr,d)), Val(ndims(arr)))
    ids = ntuple((d)->s_idx[d]:lastindex(arr,d), Val(ndims(arr)))
    return @view arr[ids...]
end

"""
    fourier_reverse!(arr; dims=1:ndims(arr))

reverses the dimensions of the input array `arr` in place. This effectively mirrors these array.
Note that for even-sized dimensions the first index is excluded from the reverse operation along this dimensions. 

# Example
```jldoctest
julia> a = [1 2 3;4 5 6;7 8 9;10 11 12]
4×3 Matrix{Int64}:
  1   2   3
  4   5   6
  7   8   9
 10  11  12

julia> fourier_reverse!(a);

julia> a
4×3 Matrix{Int64}:
  3   2   1
 12  11  10
  9   8   7
  6   5   4
```
"""
function fourier_reverse!(arr; dims=ntuple((d)->d,Val(ndims(arr))))
    reverse!(odd_view(arr),dims=dims)
    for d = 1:ndims(arr)
        if iseven(size(arr,d))
            fv = slice(arr,d,firstindex(arr,d))
            fourier_reverse!(fv; dims=dims)
        end
    end
    return arr
end
