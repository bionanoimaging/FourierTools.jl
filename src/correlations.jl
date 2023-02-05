export ccorr

"""
    ccorr(u, v[, dims]; centered=false)

Calculates the cross-correlation between `u` and `v` along `dims`.
`centered=true` moves the output of the cross-correlation to the Fourier center.

If `u` and `v` are both a real valued array we use `rfft` and hence
the output is real as well.
If either `u` or `v` is complex we use `fft` and output is hence complex.

Per default the correlation is performed along `min(ndims(u), ndims(v))`.

```jldoctest
julia> ccorr([1,1,0,0], [1,1,0,0], centered=true)
4-element Vector{Float64}:
 0.0
 1.0
 2.0
 1.0

julia> ccorr([1,1,0,0], [1,1,0,0])
4-element Vector{Float64}:
 2.0
 1.0
 0.0
 1.0

julia> ccorr([1im,0,0,0], [0,1im,0,0])
4-element Vector{ComplexF64}:
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 1.0 + 0.0im

julia> ccorr([1im,0,0,0], [0,1im,0,0], centered=true)
4-element Vector{ComplexF64}:
 0.0 + 0.0im
 1.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
```
"""
function ccorr(u::AbstractArray{T, N}, v::AbstractArray{D, M}, 
              dims=ntuple(+, min(N, M));
              centered=false) where {T, D, N, M}
    out = ifft(fft(u, dims) .* conj.(fft(v, dims)), dims)
    
    if centered
        return fftshift(out)
    else
        return out
    end
end

function ccorr(u::AbstractArray{<:Real, N}, v::AbstractArray{<:Real, M}, 
              dims=ntuple(+, min(N, M));
              centered=false) where {N, M}
    out = irfft(rfft(u, dims) .* conj.(rfft(v, dims)), size(u, dims[1]), dims)
    
    if centered
        return fftshift(out)
    else
        return out
    end
end
