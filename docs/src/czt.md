# CZTs
Chirp Z Transformations: Allows Fourier-transformation and at the same time zooming into the result,
which is why it is also called the Zoomed-FFT algorithm.
The algorithm is loosely based on a publication [Rabiner, Schafer, Rader, The Cirp z-Transform Algorithm, IEEE Trans AU 17(1969) p. 86] and a 2D Matlab Version written by N.G. Worku & H. Gross, with their consent (28. Oct. 2020) to make it openly available. It currently needs three FFTs to perform its work.
As one of these FFTs only depends on the datasize and zoom parameters, it can be moved to a plan in future implementations.

```@docs
FourierTools.czt
FourierTools.iczt
FourierTools.czt_1d
```
