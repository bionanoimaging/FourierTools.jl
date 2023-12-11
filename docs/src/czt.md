# CZTs
Chirp Z Transformations: Allows Fourier-transformation and at the same time zooming into the result,
which is why it is also called the Zoomed-FFT algorithm.
The algorithm is loosely based on a publication [Rabiner, Schafer, Rader, The Chirp z-Transform Algorithm, IEEE Trans AU 17(1969) p. 86]. It needs three FFTs to perform its work but one can be precalculated by using `plan_czt`.
Variable zooms, transform dimensions, array center positions as well as output sizes are supported along wiht a low-level interface by specifingy `a` and `w`. 
```@docs
FourierTools.czt
FourierTools.plan_czt
FourierTools.iczt
FourierTools.czt_1d
FourierTools.plan_czt_1d
FourierTools.CZTPlan_1D
FourierTools.CZTPlan_ND
```
