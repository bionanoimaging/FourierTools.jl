# Sliding Discrete Fourier Transforms

Computation of [Sliding Discrete Fourer Transforms](https://en.wikipedia.org/wiki/Sliding_DFT) over one-dimensional series of values.

## Usage

The basic Sliding Discrete Fourier Transform (SDFT) of a one-dimensional series of values `x`, using a window of length `n`, is calculated as follows:

**Step 1**: Setup the method for an SDFT of length `n`:

```julia
sdft = SDFT(n)
```

**Step 2**: Apply the created method to the data series `x`. This is typically used in a loop:

```julia
for spectrum in sdft(x)
    # `spectrum` is a `Vector{Complex(eltype(x))}` of length `n`
end
```

### Considerations for stateful iterators

By default, SDFTs are computed traversing sequentially the data series `x`, which can be any kind of iterator. In the case of stateful iterators (i.e. those that are modified upon each iteration, like `Base.Channel`s),  `sdft(method, x)` will also be a stateful iterator that will "consume" as many items of `x` as the length of the computed DFT in the first iteration, and one additional item in every subsequent iteration.

Apart from that consideration, it is safe to apply SDFTs to stateful iterators, since past samples of `x` already used in previous iterations, which are often required for the computations, are temporarily stored in an array — in internal variables that users do not need to deal with.

## Methods for SDFTs

```@autodocs
Modules = [FourierTools]
Pages = ["sdft_implementations.jl"]
```

## Developing new SDFTs

### Theoretical basis

An SDFT is generally implemented as a recursive equation, such that if $X_{i}[k]$ is the DFT of $x[j]$ for $j = i, \ldots, i+n$ at frequency $k$, the next iteration is:

$X_{i+1}[k] = f(k, X_{1}[k], \ldots, X_{i}[k], x[i], \ldots x[i+n], x[i+n+1])$

Such an equation depends on:

* The frequency $k$
* The values of the DFT in one or more previous iterations, $X_{p}[k]$ for $p = 1, \ldots, i$.
* The values of the data series used in the most recent iteration, $x[j]$ for $j = i, \ldots, i+n$.
* The next value of the data series after the fragment used in the most recent iteration, $x[i+n+1]$.

For instance, the [basic definition of the SDFT](https://www.researchgate.net/publication/3321463_The_sliding_DFT) uses the following formula: 

$X_{i+1}[k] = W[k] \cdot (X_{i}[k] + x[i+n] - x[i]),$

which depends only on the most recent DFT ($X_{i}[k]$), the first data point of the fragment used in that DFT ($x[i]$), and the next data point after that fragment ($x[i+n+1]$), plus a "twiddle" factor $W[k]$ that only depends on the frequency $k$, equal to $\exp(j2{\pi}k/n)$.
Other variations of the SDFT may use formulas that depend on previous iterations or other values of the data series in that fragment.

### Implementation in Julia object types

A method to compute an SDFT is defined by three kinds of object types:

* One for the method (e.g. `sdft = SDFT(n)` in the previous example), which contains the fixed parameters that are needed by the algorithm to compute the SDFT.
* Another for the iterator created by calling the method as a function (`sdft(x)` in the example), which is thus bound with the target data series.
* And yet another for the state of the iterator, which holds the information needed by the algorithm that depends on the data series and changes at each iteration.

The internals of this package take care of the design of the iterator and state types, and of the creation of their instances. The only thing that has to be defined to create a new kind of SDFT is the `struct` of the method with the fixed parameters, and a few function methods dispatching on that type.

Before explaining how to define such a struct, it is convenient to know the functions that can be used to extract the information that is stored in the state of SDFT iterators. There is a function for each one of the three kinds of variables presented in the general recursive equation above.

* [`FourierTools.sdft_previousdft`](@ref) for the values of the DFT in previous iterations.
* [`FourierTools.sdft_previousdata`](@ref) for the values of the data series used in the most recent iteration.
* [`FourierTools.sdft_nextdata`](@ref) for the next value of the data series after the fragment used in the most recent iteration.

For instance, the values used in the formula of the basic SDFT may be obtained from a `state` object as:
* `FourierTools.sdft_previousdft(state, 0)` for $X_{i}$.
* `FourierTools.sdft_previousdata(state, 0)` for $x[i]$.
* `FourierTools.sdft_nextdata(state)` for $x[i+n+1]$.

Notice that the second arguments of `sdft_previousdft` and `sdft_previousdata` might have been ommited in this case, since they are zero by default.

For methods that need to know how many steps of the SDFT have been done, this can also be extracted with the function [`FourierTools.sdft_iteration`](@ref).

The design of the `struct` representing a new SDFT type is free, but it is required to be a subtype of [`AbstractSDFT`](@ref), and implement the following methods dispatching on that type:

* [`FourierTools.sdft_windowlength`](@ref) to return the length of the DFT window.
* [`FourierTools.sdft_update!`](@ref) with the implementation of the recursive equation, extracting the information stored in the state with the functions commented above (`sdft_previousdft`, etc.) .

Depending on the functions that are used in the particular implementation of `sdft_update!` for a given type, the following methods should be defined too:

* [`FourierTools.sdft_backindices`](@ref) if `FourierTools.sdft_previousdft` is used.
* [`FourierTools.sdft_dataoffsets`](@ref) if `FourierTools.sdft_previousdata` is used.

### Example

The formula of the basic SDFT formula could be implemented for a type `MyBasicSDFT` as follows:

```julia
import FourierTools: sdft_update!, sdft_windowlength, sdft_nextdata, sdft_previousdata

function sdft_udpatedft!(dft, x, method::MyBasicSDFT, state)
    n = sdft_windowlength(method)
    for k in eachindex(dft)
        X_i = dft[k]
        x_iplusn = sdft_nextdata(state)
        x_i = sdft_previousdata(state)
        Wk = exp(2π*im*k/n)
        dft[k] = Wk * (X_i + x_iplusn - x_i)
    end
end
```

(The type [`SDFT`](@ref) actually has as a similar, but not identical definition.)

The implementation of `sdft_update!` given in the previous example does use `sdft_previousdata` - with the default offset value, equal to zero - so the following is required in this case:

```julia
FourierTools.sdft_dataoffsets(::MyBasicSDFT) = 0
```

On the other hand there is no need to define `FourierTools.sdft_backindices` in this case, since the interface of of `sdft_update!` assumes that the most recent DFT is already contained in its first argument `dft`, so it is not necessary to use the function `sdft_previousdft` to get it.

### Alternative to subtyping `AbstractSDFT`

Types that represent SDFT methods are required to be subtypes of `AbstractSDFT` in order to make them callable and return an iterator of the SDFT (i.e. objects of the type `FourierTools.SDFTIterator`). If for some reason that subtyping is not desired or possible (e.g. if the said type already has another supertype), the same behavior can be obtained by defining them explicitly as [functors](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects), in the following fashion:

```julia
(m::MyBasicSDFT)(args...) = FourierTools.SDFTIterator(args...)
```


### SDFT development API

```@autodocs
Modules = [FourierTools]
Pages = ["sdft_interface.jl"]
```
