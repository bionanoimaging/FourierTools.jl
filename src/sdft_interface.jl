export AbstractSDFT

## Required functions

"""
    sdft_windowlength(method)

Return the length of the window used by `method`.
"""
function sdft_windowlength end

"""
    sdft_update!(dft, x, method, state)

Update the values of a sliding Discrete Fourier Transform (DFT) of a data series,
according to the algorithm of the provided method, for a given state of the sliding DFT.

`dft` is a mutable collection of complex values with length equal to `sdft_windowlength(method)`,
containing the value returned by the last iteration of the sliding DFT.

`x` is the data series for which the sliding DFT is computed, at least as long as
`sdft_windowlength(method)`.

`method` is the object that defines the method to compute the sliding DFT.

`state` is an object generated automatically at each iteration
of an SDFT iterator made from `method` and `x`,
(an object of the type [`FourierTools.SDFTIterator`](@ref)).

The information that is needed to update the sliding DFT can be extracted from `state`
with the following functions:

* [`FourierTools.sdft_previousdft`](@ref) to get the DFTs of previous iterations.
* [`FourierTools.sdft_previousdata`](@ref) to get a previous value of the data series.
* [`FourierTools.sdft_nextdata`](@ref) to get the next value of the data series.
"""
function sdft_update! end

## Conditionally required functions

"""
    sdft_backindices(method)

Return an integer or a vector of positive integers with the indices of the previous iterations
that are needed by the given method to compute a sliding DFT.

If the code of `FourierTools.updatepdf!` for the type of `method` uses the function `FourierTools.sdft_previousdft`,
this function must return the integers that are used as the third argument (`back`) of that function.

If that function is not needed, this one may return `nothing` to reduce memory allocations.
"""
sdft_backindices(::Any) = nothing

"""
    sdft_dataoffsets(method)

Return an integer or a vector of integers with the offsets of data samples
that are needed by the given method to compute a sliding DFT.

If the code of `FourierTools.updatepdf!` that dispatches on the type of `method` uses the function `FourierTools.sdft_previousdata`,
this function must return the integers that are used as the third argument (`offset`) of that function.

If that function is not needed (no past samples are used), this one may return `nothing` to reduce memory allocations.
"""
sdft_dataoffsets(::Any) = nothing


## States

# State data used by `sdft_update!`
struct SDFTStateData{H, F, N}
    dfthistory::H       # record of back dfts if needed, otherwise nothing
    fragment::F         # previous data fragment if needed, otherwise nothing
    nextdatapoint::N    # next data point after the fragment
    windowlength::Int   # window length
    iteration::Int      # iteration count
end

hasdfthistory(::SDFTStateData{Nothing}) = false
hasdfthistory(::SDFTStateData) = true
haspreviousdata(::SDFTStateData{<:Any, Nothing}) = false
haspreviousdata(::SDFTStateData) = true

"""
    sdft_previousdft(state[, back=0])

Return the DFT computed in the most recent iteration
of the sliding DFT represented by `state`, or in a number of previous
iterations equal to `back`.

If the DFT computed in the most recent iteration corresponds to the
fragment of the data series between its positions `i` and `i+n`,
then this function returns its DFT for the fragment between `i-back` and `i+n-back`.
"""
function sdft_previousdft(state::SDFTStateData, back=0)
    !hasdfthistory(state) && throw(ErrorException(
        "previous DFT results not available; the SDFT method has no valid definition of `FourierTools.sdft_backindices`"
        ))
    dfthistory = state.dfthistory
    n = state.windowlength
    nh = length(dfthistory) ÷ n
    offset = rem(state.iteration - back - 1, nh)
    rg = (offset * n) .+ (1:n)
    return view(dfthistory, rg)
end

"""
    sdft_previousdata(state[, offset=0])

Return the first value of the fragment of the data series that was used
in the most recent iteration of the sliding DFT represented by `state`,
or at `offset` positions after the beginning of that fragment.

If the DFT computed in the most recent iteration corresponds to the
fragment of the data series between its positions `i` and `i+n`,
then this function returns the `i+offset`-th value.
"""
function sdft_previousdata(state::SDFTStateData, offset=0)
    !haspreviousdata(state) && throw(ErrorException(
        "previous data values not available; the SDFT method has no valid definition of `FourierTools.sdft_dataoffsets`"
        ))
    fragment = state.fragment
    adjustedoffset = rem(offset + state.iteration - 1, length(fragment))
    return fragment[firstindex(fragment) + adjustedoffset]
end


"""
    sdft_nextdata(state)

Return the next value after the fragment of the data series that was used
in the most recent iteration of the sliding DFT represented by `state`.

If the DFT computed in the most recent iteration corresponds to the
fragment of the data series between its positions `i` and `i+n`,
then this function returns the `i+n+1`-th value.

There is no defined behavior if such value does not exist
(i.e. if the end of a finite data series was reached).
"""
sdft_nextdata(state::SDFTStateData) = state.nextdatapoint

"""
    sdft_iteration(state)

Return the number of iterations done for the sliding DFT represented by `state`.

If the DFT computed in the most recent iteration corresponds to the
fragment of the data series between its positions `i` and `i+n`,
then this function returns  the number `i`.
"""
sdft_iteration(state) = state.iteration

# State of the iterator
struct SDFTState{T, H, F, S}
    dft::Vector{Complex{T}}     # dft returned
    dfthistory::H               # (see StateData)
    fragment::F                 # (see StateData)
    nextdatastate::S            # state used to get the next data point
    iteration::Int              # iteration count
end

# Return the updated state of the iterator, or `nothing` if the data series is consumed.
function sdft_updatestate(state::SDFTState, method, x)
    nextiter = iterate(x, state.nextdatastate)
    if isnothing(nextiter)
        return nothing
    end
    nextdatapoint, nextdatastate = nextiter
    dfthistory = state.dfthistory
    fragment = state.fragment
    n = sdft_windowlength(method)
    # explicit fft of shifted fragment until dfthistory is complete
    if !isnothing(dfthistory) && state.iteration <= maximum(sdft_backindices(method))
        sdft_updatefragment!(fragment, nextdatapoint, state.iteration)
        dft = shiftedfft(fragment, state.iteration)
    else
        dft = state.dft
        statedata = SDFTStateData(dfthistory, fragment, nextdatapoint, n, state.iteration)
        sdft_update!(dft, x, method, statedata)
        sdft_updatefragment!(fragment, nextdatapoint, state.iteration)
    end
    sdft_updatedfthistory!(dfthistory, dft, n, state.iteration + 1)
    return SDFTState(dft, dfthistory, fragment, nextdatastate, state.iteration + 1)
end

function shiftedfft(x, delay)
    n = length(x)
    y = fft(x)
    delayfactor = exp(2π*im*delay/n)
    updatedfactor = one(delayfactor)
    for k in eachindex(y)
        y[k] *= updatedfactor
        updatedfactor *= delayfactor
    end
    return y
end

function sdft_updatedfthistory!(dfthistory, dft, n, iteration)
    offset = rem((iteration - 1) * n, length(dfthistory))
    dfthistory[(1:n) .+ offset] .= dft
end

function sdft_updatedfthistory!(::Nothing, args...) end

function sdft_updatefragment!(fragment, nextdatapoint, iteration)
    n = length(fragment)
    offset = rem(iteration - 1, n)
    fragment[begin + offset] = nextdatapoint
end

function sdft_updatefragment!(::Nothing, ::Any, ::Any) end

## Iterator

"""
    SDFTIterator(method, x[, safe=true])

Return an iterator to produce a sliding DFT of `x` using the given `method`.
If `safe == true` (the default value), this iterator produces a new vector at each iteration;
otherwise the vector produced in the first iteration can be reused in subsequent iterations.
"""
struct SDFTIterator{M, T}
    method::M
    data::T
    safe::Bool
end

SDFTIterator(method, x) = SDFTIterator(method, x, true)

getmethod(iterator::SDFTIterator) = iterator.method
getdata(iterator::SDFTIterator) = iterator.data
issafe(iterator::SDFTIterator) = iterator.safe

Base.IteratorSize(::SDFTIterator{<:Any,T}) where T = IteratorSizeWrapper(Base.IteratorSize(T))
IteratorSizeWrapper(::Base.HasShape) = Base.HasLength() # do not inherit shape
IteratorSizeWrapper(iteratorsizetrait) = iteratorsizetrait # inherit everything else

function Base.length(iterator::SDFTIterator)
    method = getmethod(iterator)
    data = getdata(iterator)
    return length(data) - sdft_windowlength(method) + 1
end

Base.eltype(::SDFTIterator{M,T}) where {M,T} = Vector{Complex{eltype(T)}}

Base.isdone(iterator::SDFTIterator) = Base.isdone(getdata(iterator))
Base.isdone(iterator::SDFTIterator, state::SDFTState) = Base.isdone(getdata(iterator), state.nextdatastate)

function Base.iterate(itr::SDFTIterator)
    windowed_data, datastate = initializesdft(itr)
    dft = fft(windowed_data)
    method = getmethod(itr)
    backindices = sdft_backindices(method)
    dfthistory = create_sdfthistory(dft, backindices)
    state = SDFTState(dft, dfthistory, windowed_data, datastate, 1)
    returned_dft = itr.safe ? copy(dft) : dft
    return returned_dft, state
end

function Base.iterate(itr::SDFTIterator, state)
    method = getmethod(itr)
    x = getdata(itr)
    newstate = sdft_updatestate(state, method, x)
    if isnothing(newstate)
        return nothing
    end
    dft = newstate.dft
    returned_dft = itr.safe ? copy(dft) : dft
    return returned_dft, newstate
end

# Get the window of the first chunk of data and the state of the data iterator at the end
function initializesdft(itr)
    x = getdata(itr)
    n = sdft_windowlength(getmethod(itr))
    firstiteration = iterate(x)
    if isnothing(firstiteration)
        throw(ErrorException("insufficient data to compute a sliding DFT of window length = $n"))
    end
    datapoint, datastate = firstiteration
    windowed_data = fill(datapoint, n)
    i = 1
    while i < n
        iteration = iterate(x, datastate)
        if isnothing(iteration)
            throw(ErrorException("insufficient data to compute a sliding DFT of window length = $n"))
        end
        datapoint, datastate = iteration
        i += 1
        windowed_data[i] = datapoint
    end
    return windowed_data, datastate
end

create_sdfthistory(::Any, ::Nothing) = nothing
create_sdfthistory(dft, n::Integer) = repeat(dft, n+1)
create_sdfthistory(dft, indices) = create_sdfthistory(dft, maximum(indices))

"""
    (m::AbstractSDFT)(x[, safe=true])

`AbstractSDFT` is an abstract type for callable structs that implement
Sliding Discrete Fourier Transforms.

If `typeof(m) <: AbstractSDFT`, then `m(x)` will return an iterator
that produces a sliding DFT of `x` according to the method `m`. 

By default that iterator produces a new vector at each iteration.
Set the optional argument `safe=false` to improve performance by reducing allocations,
at the expense of unexpected behavior if the resulting vector is mutated between iterations.

# Example
```julia
sdft = SDFT(10) # basic [`SDFT`](@ref) with a window of length 10

for dft in sdft(x)
    # here `dft` is a vector with 10 complex values
end
```

# Available methods:
See some methods for sliding DFTs at:
https://nanoimaging.de/FourierTools.jl/stable/slidingdft/#Methods-for-SDFTs
"""
abstract type AbstractSDFT end

(m::AbstractSDFT)(args...) = SDFTIterator(m, args...)
