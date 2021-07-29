export czt, iczt

"""
    czt_1d(xin , scalx , scaly , d)
    chirp z transform along a single direction d of an ND array `xin` into the ND array 'xout'.
    Note that xin and xout can be the same array for inplace operations.
    This code is based on a 2D Matlab version of the CZT, written by H. Gross et al.
    
    References:
    Rabiner, Schafer, Rader, The Cirp z-Transform Algorithm, IEEE Trans AU 17(1969) p. 86

    Example: Transformation of a Gaussian, at different sampling and zooms
"""
function czt_1d(xin, scaled, d)
    sz=size(xin)
    dsize = sz[d]
    nn = (0:dsize-1)
    kk = ((-dsize+1):(dsize-1))
    kk2 = (kk .^ 2) ./ 2
    
    w = cispi(-2/(dsize*scaled)) # scalar factor
    a = cispi(-1/scaled) # scalar factor
    ww = w .^ kk2
    aa = a .^ (-nn)
    aa = aa .* ww[dsize .+ nn] # is a 1d list of factors
    tofft = collect(NDTools.select_region(1 ./ ww[1:(2*dsize-1)], new_size=2*dsize, center=dsize .+1))
    fv = fft(tofft) # is always 1d
    fak =  ww[dsize:(2*dsize-1)] .* cispi.(ramp(1,dsize, scale=1/scaled)) # is a 1d list of factors

    y = xin .* reorient(aa,d)
    nsz = sz .* NDTools.single_dim_size(d,2,length(sz)) # twice the size along direction d
    tofft = collect(NDTools.select_region(y, new_size=nsz, center=nsz.÷2 .+1))
    g = ifft(fft(tofft, d) .* reorient(fv,d), d)
    oldctr = sz[d]÷2 + 1
    newctr = size(g) .÷ 2 .+1
    ctr = collect((md == d) ? newctr[md] + oldctr - 2 : newctr[md] for md in 1:length(newctr))
    xout = select_region(g, new_size=sz,center=ctr) .* reorient(fak,d)
    return xout
    # xout .= g[dsize:(2*dsize-1)] .* reorient(fak,d)
end

"""
    czt(xin , scale, dims=1:length(size(xin)))
    chirp z transform of the ND array `xin`
    This code is based on a 2D Matlab version of the CZT, written by H. Gross.
    The tuple `scale` defines the zoom factors in the Fourier domain. Each has to be bigger than one.
    
    References:
    Rabiner, Schafer, Rader, The Cirp z-Transform Algorithm, IEEE Trans AU 17(1969) p. 86

    Example: Transformation of a Gaussian, at different sampling and zooms
    using IndexFunArrays
    sz = (10,10)
    xin = disc(sz,4)
    xft = ift(xin)
    czt(xft,(1.2,1.3))

"""
function czt( xin , scale, dims=1:length(size(xin)))
    sz = size(xin)
    
    xout = xin
    for d in dims
        xout = czt_1d(xout, scale[d], d)
    end
    return xout
end

"""
    iczt(xin , scale, dims=1:length(size(xin)))
    inverse chirp z transform of the ND array `xin`
    This code is based on a 2D Matlab version of the CZT, written by H. Gross.
    The tuple `scale` defines the zoom factors in the Fourier domain. Each has to be bigger than one.
    
    References:
    Rabiner, Schafer, Rader, The Cirp z-Transform Algorithm, IEEE Trans AU 17(1969) p. 86

    Example: Transformation of a Gaussian, at different sampling and zooms
    using IndexFunArrays
    sz = (10,10)
    xin = disc(sz,4)
    xft = ft(xin)
    iczt(xft,(1.2,1.3))
"""
function iczt( xin , scale, dims=1:length(size(xin)))
    conj(czt(conj(xin), scale, dims)) / prod(size(xin))
end
