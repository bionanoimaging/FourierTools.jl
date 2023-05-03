# This file contains workarounds to make Cuda FFTs work even for non-consecutive directions

function head_views(arr, d)
    front_ids = ntuple((dd)->Colon(), d)
    ids = ntuple((dd)->ifelse(dd <= d,Colon(),1), ndims(arr))
    return @view arr[ids...], front_ids
end

function fft!(arr::CuArray, d::Int)
    if d>1 && d < ndims(arr)
        front_ids = ntuple((dd)->Colon(), d)
        ids = ntuple((dd)->ifelse(dd <= d,Colon(),1), ndims(arr))
        p = plan_fft!((@view arr[ids...]), d)
        for c in CartesianIndices(size(arr)[d+1:end])
            p * @view arr[front_ids..., Tuple(c)...]
        end
    else
        CUDA.CUFFT.fft!(arr, d)
    end
end

function fft(arr::CuArray, d::Int)
    if d>1 && d < ndims(arr)
        res = similar(arr, Complex(eltype(arr)))
        return fft!(res, d)
    else
        return CUDA.CUFFT.fft(arr, d)
    end
end

struct rCuFFT_new{B}
    p::B
end

function new_plan_rfft(arr::CuArray{T,N}, d::Int) where {T<:Union{Float32, Float64}, N}
    @show "myplan"
    if d>1 && d < ndims(arr)
        myview = 
        rCuFFT_new(plan_rfft(myview, d))
    else
        return plan_rfft(arr, d)
    end
end

