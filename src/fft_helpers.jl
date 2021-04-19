export ft,ift, rft, irft
export ffts, ffts!, iffts, rffts, irffts

"""
    optional_collect(a)

Only collects certain arrays, for a pure `Array` there is no collect
and it returns simply `a`.
"""
# collect
optional_collect(a::AbstractArray) = collect(a)
# no need to collect
optional_collect(a::Array) = a 

# for CircShiftedArray we only need collect if shifts is non-zero
function optional_collect(csa::ShiftedArrays.CircShiftedArray)
    if all(iszero.(csa.shifts))
        return optional_collect(parent(csa))
    else
        return collect(csa)
    end
end



"""
    ffts(A [, dims])

Result is semantically equivalent to `fftshift(fft(A, dims), dims)`
However, the shift is done with `ShiftedArrays` and therefore doesn't allocate memory.

See also: [`ft`](@ref ift), [`ift`](@ref ift), [`rft`](@ref rft), [`irft`](@ref irft),
          [`ffts`](@ref ffts),  [`iffts`](@ref iffts),  [`ffts!`](@ref ffts!), [`rffts`](@ref rffts), [`irffts`](@ref irffts!), 
"""
function ffts(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    return fftshift_view(fft(mat, dims), dims)
end


"""
    ffts!(A [, dims])

Result is semantically equivalent to `fftshift(fft!(A, dims), dims)`.
`A` is in-place modified.
However, the shift is done with `ShiftedArrays` and therefore doesn't allocate memory.

See also: [`ft`](@ref ift), [`ift`](@ref ift), [`rft`](@ref rft), [`irft`](@ref irft),
          [`ffts`](@ref ffts),  [`iffts`](@ref iffts),  [`ffts!`](@ref ffts!), [`rffts`](@ref rffts), [`irffts`](@ref irffts!), 
"""
function ffts!(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    return fftshift_view(fft!(mat, dims), dims)
end

"""
    iffts(A [, dims])

Result is semantically equivalent to `ifft(ifftshift(A, dims), dims)`.
`A` is in-place modified.
However, the shift is done with `ShiftedArrays` and therefore doesn't allocate memory.

See also: [`ft`](@ref ift), [`ift`](@ref ift), [`rft`](@ref rft), [`irft`](@ref irft),
          [`ffts`](@ref ffts),  [`iffts`](@ref iffts),  [`ffts!`](@ref ffts!), [`rffts`](@ref rffts), [`irffts`](@ref irffts!), 
"""
function iffts(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    return ifft(optional_collect(ifftshift_view(mat, dims)), dims)
end


"""
    rffts(A [, dims])

Calculates a `rfft(A, dims)` and then shift the frequencies to the center.
`dims[1]` is not shifted, because there is no negative and positive frequency.
The shift is done with `ShiftedArrays` and therefore doesn't allocate memory.

See also: [`ft`](@ref ift), [`ift`](@ref ift), [`rft`](@ref rft), [`irft`](@ref irft),
          [`ffts`](@ref ffts),  [`iffts`](@ref iffts),  [`ffts!`](@ref ffts!), [`rffts`](@ref rffts), [`irffts`](@ref irffts!), 
"""
function rffts(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    rfftshift_view(rfft(mat, dims), dims);
end

"""
    irffts(A, d, [, dims])

Calculates a `irfft(A, d, dims)` and then shift the frequencies back to the corner.
`dims[1]` is not shifted, because there is no negative and positive frequency.
The shift is done with `ShiftedArrays` and therefore doesn't allocate memory.

See also: [`ft`](@ref ift), [`ift`](@ref ift), [`rft`](@ref rft), [`irft`](@ref irft),
          [`ffts`](@ref ffts),  [`iffts`](@ref iffts),  [`ffts!`](@ref ffts!), [`rffts`](@ref rffts), [`irffts`](@ref irffts!), 
"""
function irffts(mat::AbstractArray{T, N}, d::Int, dims=ntuple(identity, Val(N))) where {T, N}
    irfft(optional_collect(irfftshift_view(mat, dims)), d, dims)
end



"""
    ft(A [, dims])

Digital Fourier-transformation centered in both spaces.
The result is semantically equivalent to `fftshift(fft(ifftshift(A, dims), dims), dims)`
This is a digital Fourier transformation with both coordinate systems in real and Fourier-space being
centered at position CtrFT == size÷2+1

The following identities are true:
```jldoctest
julia> sz = (5,5)
(5, 5)

julia> ft(ones(sz)) ≈ prod(sz) .* δ(sz)
true

julia> ft(δ(sz)) ≈ ones(sz)
true
```

See also: [`ft`](@ref ift), [`ift`](@ref ift), [`rft`](@ref rft), [`irft`](@ref irft),
          [`ffts`](@ref ffts),  [`iffts`](@ref iffts),  [`ffts!`](@ref ffts!), [`rffts`](@ref rffts), [`irffts`](@ref irffts!), 

"""
function ft(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    return fftshift_view(fft(optional_collect(ifftshift_view(mat, dims)), dims), dims)
end


"""
    ift(A [, dims])

Digital inverse Fourier-transformation centered in both spaces.
The result is semantically equivalent to `fftshift(ifft(ifftshift(A, dims), dims), dims)`
This is a digital Fourier transformation with both coordinate systems in real and Fourier-space being
centered at position CtrFT == size÷2+1

The following identities are true:

```jldoctest
julia> sz = (5,6,7)
(5, 6, 7)

julia> ift(ones(sz)) ≈ δ(sz)
true

julia> ift(δ(sz)) ≈ ones(sz) ./ prod(sz)
true
```


See also: [`ft`](@ref ift), [`ift`](@ref ift), [`rft`](@ref rft), [`irft`](@ref irft),
          [`ffts`](@ref ffts),  [`iffts`](@ref iffts),  [`ffts!`](@ref ffts!), [`rffts`](@ref rffts), [`irffts`](@ref irffts!), 
    
"""
function ift(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    return fftshift_view(ifft(optional_collect(ifftshift_view(mat, dims)), dims), dims)  # is faster than the exp mulitplication
end


"""
    rft(A [, dims])

Digital real-valued Fourier-transformation centered in both spaces.
The result is semantically equivalent to `fftshift(rfft(ifftshift(A, dims), dims), dims)`
This is a digital Fourier transformation with the coordinate systems in real space centered at CtrFT == size÷2+1
and in (half) Fourier-space being centered at CtrRFT == setindex(size÷2 +1,1,1).

The following identities are true:
```jldoctest
julia> sz = (6,6)
(6, 6)

julia> rft(ones(sz)) ≈ prod(sz) .* δ(rft_size(sz), offset=CtrRFT)
true

julia> rft(δ(sz)) ≈ ones(rft_size(sz))
true
```

See also: [`ft`](@ref ift), [`ift`](@ref ift), [`rft`](@ref rft), [`irft`](@ref irft),
          [`ffts`](@ref ffts),  [`iffts`](@ref iffts),  [`ffts!`](@ref ffts!), [`rffts`](@ref rffts), [`irffts`](@ref irffts!), 

"""
function rft(mat::AbstractArray{T, N}, dims=ntuple(identity, Val(N))) where {T, N}
    return rfftshift_view(rfft(optional_collect(ifftshift_view(mat, dims)), dims), dims);
end

"""
    irft(A, d, [, dims])

Digital real-valued inverse Fourier-transformation centered in both spaces.
The result is semantically equivalent to `fftshift(irfft(ifftshift(A, dims), dims), dims)`
This is a digital Fourier transformation with the coordinate systems in real space centered at CtrFT == size÷2+1
and in (half) Fourier-space being centered at CtrRFT == setindex(size÷2 +1,1,1).
Note that the size `d` of the first transform direction [1] is a required argument.

The following identities are true:
```jldoctest
julia> sz = (6,6)
(6, 6)

julia> irft(ones(rft_size(sz)),sz[1]) ≈ δ(sz)
true

julia> irft(δ(rft_size(sz),offset=CtrRFT),sz[1]) ≈ ones(sz) ./ prod(sz)
true
```

See also: [`ft`](@ref ift), [`ift`](@ref ift), [`rft`](@ref rft), [`irft`](@ref irft),
          [`ffts`](@ref ffts),  [`iffts`](@ref iffts),  [`ffts!`](@ref ffts!), [`rffts`](@ref rffts), [`irffts`](@ref irffts!), 

"""
function irft(mat::AbstractArray{T, N}, d::Int, dims=ntuple(identity, Val(N))) where {T, N}
    fftshift_view(irfft(optional_collect(irfftshift_view(optional_collect(mat), dims)), d, dims), dims);
end

