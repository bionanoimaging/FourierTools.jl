using FFTW
using LinearAlgebra

function shift_c!(arr::AbstractArray{<:Complex, N}, shifts) where {N}
    for (d, shift) in pairs(shifts)
        freqs = reshape(fftfreq(size(arr, d)), ntuple(i -> 1, d-1)..., size(arr,d)) 
        # allocates a 1D slice of exp values 
        ϕ = exp.(-1im .* freqs .* 2pi .* shift)
        # in even case, set one value to real
        if iseven(size(arr, d))
            s = size(arr, d) ÷ 2 + 1
            ϕ[s] = real(ϕ[s])
        end
        # go to fourier space and apply ϕ
        fft!(arr, d)
        arr .*= ϕ
        ifft!(arr, d)
    end

    return arr
end


function shift_r(arr::AbstractArray{<:Real, N}, shifts) where {T, N}
    for (d, shift) in pairs(shifts)
        s = size(arr, d) ÷ 2 + 1
        freqs = reshape(fftfreq(size(arr, d))[1:s], ntuple(i -> 1, d-1)..., s) 
        ϕ = exp.(-1im .* freqs .* 2pi .* shift)
        if iseven(size(arr, d))
            ϕ[s] = real(ϕ[s])
        end
        p = plan_rfft(arr, d)

        arr_ft = p * arr
        arr_ft .*= ϕ
        arr .= inv(p) *arr_ft

        # in principle mul! could improve performance since in-place
        # however, does not work currently
        #inds = top_left_indices(rft_size(size(arr), d))
        #mul!(arr_ft2, p, arr)
        #arr_ft2 .= p * arr
        #arr_ft2 .= rfft(arr, d)
        #arr_ft2 .*= ϕ
        #p_inv = plan_irfft(arr_ft2, size(arr, d), d,  flags=FFTW.UNALIGNED)
        
        #arr_ft2 = @view arr_ft[inds...]
        #mul!(arr, inv(p), arr_ft2)
        #arr .= irfft(arr_ft2, size(arr, d), d)
    end

    return arr
end

function rft_size(sizes, dim)
    Base.setindex(sizes, sizes[dim] ÷ 2 + 1, dim)
end

function top_left_indices(sizes::NTuple{N, T}) where {T, N}
    return ntuple(i -> 1:sizes[i], Val(N))
end
