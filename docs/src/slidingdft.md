# Sliding Discrete Fourier Transforms

Computation of [Sliding Discrete Fourer Transforms](https://en.wikipedia.org/wiki/Sliding_DFT) over one-dimensional series of values.

## Usage

The basic Sliding Discrete Fourier Transform (SDFT) of a one-dimensional series of values `x`, using a window of length `n`, is calculated as follows:

**Step 1**: Setup the method for an SDFT of length `n`:

```julia
method = SDFT(n)
```

**Step 2**: Create an iterator of the SDFT over `x`, with the function `sdft`. This is typically used in a loop:

```julia
for spectrum in sdft(method, x)
    # `spectrum` is a `Vector{Complex(eltype(x))}` of length `n`
end
```

### Considerations for stateful iterators

By default, SDFTs are computed traversing sequentially the data series `x`, which can be any kind of iterator. In the case of stateful iterators (i.e. those that are modified upon each iteration, like `Base.Channel`s),  `sdft(method, x)` will also be a stateful iterator that will "consume" as many items of `x` as the length of the computed DFT in the first iteration, and one additional item in every subsequent iteration.

Apart from that consideration, it is safe to apply SDFTs to stateful iterators, since past samples of `x` already used in previous iterations, which are often required for the computations, are temporarily stored in an array — in internal variables that users do not need to deal with.

## Methods for SDFTs

```@autodocs
Modules = [SlidingDFTs]
Pages = ["sdft_implementations.jl"]
```

## Functions

```@docs
sdft
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

* One for the method, which contains the fixed parameters that are needed by the algorithm to compute the SDFT.
* Another for the iterator created by the function `sdft`, which binds a method with the target data series.
* And yet another for the state of the iterator, which holds the information needed by the algorithm that depends on the data series and changes at each iteration.

The internals of this package, implemented in the module `FourierTools.SlidingDFTs`, take care of the design of the iterator and state types, and of the creation of their instances. The only thing that has to be defined to create a new kind of SDFT is the `struct` of the method with the fixed parameters, and a few function methods dispatching on that type.

Before explaining how to define such a struct, it is convenient to know the functions that can be used to extract the information that is stored in the state of SDFT iterators. There is a function for each one of the three kinds of variables presented in the general recursive equation above.

* [`SlidingDFTs.previousdft`](@ref) for the values of the DFT in previous iterations.
* [`SlidingDFTs.previousdata`](@ref) for the values of the data series used in the most recent iteration.
* [`SlidingDFTs.nextdata`](@ref) for the next value of the data series after the fragment used in the most recent iteration.

For instance, the values used in the formula of the basic SDFT may be obtained from a `state` object as:
* `SlidingDFTs.previousdft(state, 0)` for $X_{i}$.
* `SlidingDFTs.previousdata(state, 0)` for $x[i]$.
* `SlidingDFTs.nextdata(state)` for $x[i+n+1]$.

Notice that the second arguments of `previousdft` and `previousdata` might have been ommited in this case, since they are zero by default.

For methods that need to know how many steps of the SDFT have been done, this can also be extracted with the function [`SlidingDFTs.iterationcount`](@ref).

The design of the `struct` representing a new SDFT type is free, but it is required to implement the following methods dispatching on that type:

* [`SlidingDFTs.windowlength`](@ref) to return the length of the DFT window.
* [`SlidingDFTs.updatepdf!`](@ref) with the implementation of the recursive equation, extracting the information stored in the state with the functions commented above (`previousdft`, etc.) .

Depending on the functions that are used in the particular implementation of `updatepdf!` for a given type, the following methods should be defined too:

* [`SlidingDFTs.dftback`](@ref) if `SlidingDFTs.previousdft` is used.
* [`SlidingDFTs.dataoffsets`](@ref) if `SlidingDFTs.previousdata` is used.

### Example

The formula of the basic SDFT formula could be implemented for a type `MyBasicSDFT` as follows:

```julia
import FourierTools.SlidingDFTs: updatedft!, windowlength, nextdata, previousdata

function udpatedft!(dft, x, method::MyBasicSDFT, state)
    n = windowlength(method)
    for k in eachindex(dft)
        X_i = dft[k]
        x_iplusn = nextdata(state)
        x_i = previousdata(state)
        Wk = exp(2π*im*k/n)
        dft[k] = Wk * (X_i + x_iplusn - x_i)
    end
end
```

(The type [`SDFT`](@ref) actually has as a similar, but not identical definition.)

The implementation of `updatepdf!` given in the previous example does use `previousdata` - with the default offset value, equal to zero - so the following is required in this case:

```julia
SlidingDFTs.dataoffsets(::MyBasicSDFT) = 0
```

On the other hand there is no need to define `SlidingDFTs.dftback` in this case, since the interface of of `updatedft!` assumes that the most recent DFT is already contained in its first argument `dft`, so it is not necessary to use the function `previousdft` to get it.

### SDFT development API

```@autodocs
Modules = [FourierTools]
Pages = ["sdft_interface.jl"]
Filter = !isequal(SlidingDFTs.sdft)
```
