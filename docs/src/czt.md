# CZTs
Chirp Z Transformations: Allows Fourier-transformation and at the same time zooming into the result,
which is why it is also called the Zoomed-FFT algorithm.
The algorithm is based on a publication [Rabiner, Schafer, Rader, The Cirp z-Transform Algorithm, IEEE Trans AU 17(1969) p. 86] and a matlab Version written by the group of H. Gross. It currently needs three FFTs to perform its work.
As one of these FFTs only depends on the datasize and zoom parameters, it can be moved to a plan in future implementations.

```@docs
FourierTools.czt
FourierTools.iczt
FourierTools.czt_1d
```
