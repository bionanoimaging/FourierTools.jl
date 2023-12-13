# This file contains workarounds to make Cuda FFTs work even for non-consecutive directions

function head_view(arr, d)
    front_ids = ntuple((dd)->Colon(), d)
    ids = ntuple((dd)->ifelse(dd <= d,Colon(),1), ndims(arr))
    return (@view arr[ids...]), front_ids
end

function fft!(arr::CuArray, d)
    # @show "fft!"
    if isa(d, Number) && d>1 && d < ndims(arr)
        hv, front_ids = head_view(arr,d)
        p = plan_fft!(hv, d)
        for c in CartesianIndices(size(arr)[d+1:end])
            p * @view arr[front_ids..., Tuple(c)...]
        end
        return arr
    else
        return CUDA.CUFFT.fft!(arr, d)
    end
end

function ifft!(arr::CuArray, d)
    # @show "ifft!"
    if isa(d, Number) && d>1 && d < ndims(arr)
        @show "problematic d"
        hv, front_ids = head_view(arr,d)
        p = plan_ifft!(hv, d)
        for c in CartesianIndices(size(arr)[d+1:end])
            p * @view arr[front_ids..., Tuple(c)...]
        end
        return arr
    else
        CUDA.CUFFT.ifft!(arr, d)
        # invoke(ifft!, Tuple{CuArray, Any}, (arr, d))
    end
end

function fft(arr::CuArray, d)
    if isa(d, Number) && d>1 && d < ndims(arr)
        if isa(eltype(arr), Complex)
            res = copy(arr) 
        else
            res = complex.(arr)
        end
        return fft!(res, d)
    else
        return CUDA.CUFFT.fft(arr, d)
    end
end

function fft(arr, d)
    return FFTW.fft(arr,d)
end

function ifft(arr, d)
    return FFTW.ifft(arr,d)
end

function ifft(arr::CuArray, d)
    # @show "ifft 1"
    if isa(d, Number) && d>1 && d < ndims(arr)
        if isa(eltype(arr), Complex)
            res = copy(arr) 
        else
            res = complex.(arr)
        end
        return ifft!(res, d)
    else
        return CUDA.CUFFT.ifft(arr, d)
    end
end

struct CuFFT_new{B}
    p::B
    in_place::Bool
end

Base.size(p::CuFFT_new) = size(p.p)

# note that all functions are defined inplace, even if used out-of-place 
new_plan_rfft(arr, d) = new_plan(arr, d; func=plan_rfft)
new_plan_fft(arr, d) = new_plan(arr, d; func=plan_fft!)
new_plan_fft!(arr, d) = new_plan(arr, d; func=plan_fft!, in_place=true)

function new_plan(arr, d::Int; func=plan_fft, in_place=false) # ::CuArray{T,N} where {T<:Union{Float32, Float64}, N}
    if isa(arr, CuArray) && d>1 && d < ndims(arr)
        hv, _ = head_view(arr,d)
        CuFFT_new(func(hv, d), in_place)
    else
        # use the conventional way of planning FFTs
        return func(arr, d)
    end
end

function apply_rft_plan(p::CuFFT_new, src::CuArray)
    d = 1 + ndims(src) - length(size(p))
    sz = (size(src)[d-1]..., rft_size(size(src)[d:d])..., size(src)[d+1]...)
    arr = similar(src, complex(eltype(src)), sz)
    _, front_ids = head_view(src, d)
    for c in CartesianIndices(size(src)[d+1:end])
        arr[front_ids..., Tuple(c)...] .= p.p * @view src[front_ids..., Tuple(c)...]
    end
    return arr
end

function apply_irft_plan(dst::CuArray, p::CuFFT_new, src::CuArray)
    @show "apply_irft_plan"
    d = 1 + ndims(src) - length(size(p))
    @show d
    # sz = (size(src)[d-1]..., rft_size(size(src)[d:d])..., size(src)[d+1]...)
    # arr = similar(src, complex(eltype(src)), sz)
    _, front_ids = head_view(src, d)
    for c in CartesianIndices(size(src)[d+1:end])
        dv = @view dst[front_ids..., Tuple(c)...] 
        sv = @view src[front_ids..., Tuple(c)...]
        ldiv!(dv, p.p, sv)
    end
    return dst
end

function Base. *(p::CuFFT_new, arr::CuArray)
    if (!p.in_place)
        if isa(p.p, CUDA.CUFFT.cCuFFTPlan)
            arr = copy(arr)
        else # rFFT
            return apply_rft_plan(p, arr)
        end
    end
    @show (d = 1 + ndims(arr) - length(size(p)))
    _, front_ids = head_view(arr,d)
    for c in CartesianIndices(size(arr)[d+1:end])
        p.p * @view arr[front_ids..., Tuple(c)...]
    end
    return arr
end

function ldiv!(dst::CuArray, p::CuFFT_new, src::CuArray)
    @show "special ldiv!"
    if (!p.in_place)
        return apply_irft_plan(dst, p, src)
    end
    @show (d = 1 + ndims(arr) - length(size(p)))
    _, front_ids = head_view(arr,d)
    for c in CartesianIndices(size(arr)[d+1:end])
        sv = @view src[front_ids..., Tuple(c)...]
        dv = @view dst[front_ids..., Tuple(c)...]
        ldiv!(dv, p.p, sv)
    end
    return dst
end
