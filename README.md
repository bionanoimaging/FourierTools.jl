# FourierTools.jl


| **Documentation**                       | **Build Status**                          | **Code Coverage**               |
|:---------------------------------------:|:-----------------------------------------:|:-------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][CI-img]][CI-url] | [![][codecov-img]][codecov-url] |


This package contains various functions that are useful for working with and in Fourier space.

## Installation
`FourierTools.jl` is available for all version equal or above Julia 1.6.
Since it is not registered yet, it can be installed with the following command

```julia
julia> ] add FourierTools
```

## Features
* sinc interpolation allows to up and downsample a (bandlimited) signal
* FFT based convolutions
* array rotation 
* array shifting (including noteworthy subpixel shifts)
* array shearing
* several tools like `ffts`, `ft` etc. allowing simpler use with Fourier transforms

Have a look in the [examples folder](examples/) for interactive examples. The [documentation](https://bionanoimaging.github.io/FourierTools.jl/dev/) offers a quick overview.


[docs-dev-img]: https://img.shields.io/badge/docs-dev-pink.svg
[docs-dev-url]: https://bionanoimaging.github.io/FourierTools.jl/dev/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-darkgreen.svg
[docs-stable-url]: https://bionanoimaging.github.io/FourierTools.jl/stable/

[CI-img]: https://github.com/bionanoimaging/FourierTools.jl/actions/workflows/ci.yml/badge.svg
[CI-url]: https://github.com/bionanoimaging/FourierTools.jl/actions/workflows/ci.yml

[codecov-img]: https://codecov.io/gh/bionanoimaging/FourierTools.jl/branch/main/graph/badge.svg?token=6XWI1M1MPB
[codecov-url]: https://codecov.io/gh/bionanoimaging/FourierTools.jl
