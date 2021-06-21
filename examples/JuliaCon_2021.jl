### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ e6db292c-d2be-11eb-1c25-05ed50a9256c
using FourierTools, TestImages, ImageShow, Plots, Colors, FFTW, Noise, PlutoUI

# ╔═╡ 4958079f-da85-4357-94df-b122cf01c958
FFTW.set_num_threads(12)

# ╔═╡ b71a5e39-0805-4e3f-aa83-c3972f6c54af
begin
	img = Float32.(testimage("fabio_gray"));
	img_1D = img[:, 200]
end;

# ╔═╡ c39440fc-17e6-440c-9e10-7b2c9a7e9331
md"### FourierTools.jl | Working with the Frequency Space

* **Felix Wechsler**: Master student at the IPHT Jena, Germany
* **Rainer Heintzmann**: Head of the biological nanoimaging research group at the IPHT and FSU Jena, Germany
"

# ╔═╡ e28dc936-04da-48cb-b324-73632d008290
md"
### Definition of Fourier Transform

Complex valued integral 

$\mathcal{F}[U](\mathbf k) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} U(\mathbf r) \cdot \exp[-i  (\mathbf x \cdot \mathbf k)] \, \mathrm d \mathbf x$

$\mathcal{F}^{-1}[\tilde U](\mathbf x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} U(\mathbf r) \cdot \exp[i  (\mathbf x \cdot \mathbf k)] \, \mathrm d \mathbf k$

* visually it decomposes a signal in harmonic basis functions
* the so called _Fourier_ or _Frequency_ space

### FFT
* naive integral calculation takes $\mathcal O(N^2)$ operations for a 1D dataset with $N$ points 
* FFT algorithms evaluates this with $\mathcal O(N \cdot log(N))$ operations
* In Julia _FFTW.jl_
"

# ╔═╡ e8f1e951-26dd-4c0e-83f9-4a2c60b0ebed
md" ## FFT of an dataset with FFTW.jl"

# ╔═╡ 3adab086-0c77-40e5-990a-6b89b183cc4d
begin
	x = range(0, 50, length=999)
	f(x) = (0.1 * sin(17.3 + x) + 0.5 * cos(1.2*x) + 0.3 * sin(5*x)^2 + 0.05 * sin(13.1*x) + sin(27.2 * x)) * exp(-x/20)
	data = add_gauss(f.(x), 0.1)
end

# ╔═╡ 850869e6-32da-48cf-b2a0-081cc9f8e524
plot(x, data)

# ╔═╡ d6f9bd7a-ddcb-40b5-8509-f4dac3f8bd5d
begin
	plot(real(fft(data)), title="FFT of the dataset", label="real part")
	plot!(imag(fft(data)), label="imaginary part")
end

# ╔═╡ f0746bc6-f50a-49f1-96d2-0379498a0168
begin
	plot(fftshift(fftfreq(size(data, 1))), real(ffts(data)), label="real part", title="FFT of the dataset but shifted")
	plot!(fftshift(fftfreq(size(data, 1))), imag(ffts(data)), label="imaginary part")
end

# ╔═╡ 979d641e-83ad-4484-bc28-912947711936
fftshift(fft(data)) == ffts(data)

# ╔═╡ 24298466-c360-4c7d-b408-9f7d155faea5


# ╔═╡ dfdfb8ff-6c30-4a95-8d6b-efbd9d943fd8


# ╔═╡ 2004c39e-f650-440e-858e-20b9f81c6632


# ╔═╡ cc160200-5911-4c80-b651-a54e1608f186
begin
	N_low = 64
	x_min = 0.0
	x_max = 8*2π
	
	xs_low = range(x_min, x_max, length=N_low+1)[1:N_low]
	xs_high = range(x_min, x_max, length=5000)[1:end-1]
	f2(x) = sin(0.5*x) + cos(x) + cos(2 * x) + sin(0.25*x)
	arr_low = f2.(xs_low)
	arr_high = f2.(xs_high)
end;

# ╔═╡ a725ae98-e2cc-439c-9cc1-692972ac98e7
md"## Calculate sinc interpolation"

# ╔═╡ d5717c9c-2c16-47da-8533-a28acf189e55
begin
	N = 1000
	xs_interp = range(x_min, x_max, length=N+1)[1:N]
	arr_interp = resample(arr_low, N)
end;

# ╔═╡ 3d8f11a3-4ebf-4d36-a227-b2cafa13e1d6
begin
	scatter(xs_low, arr_low, legend=:bottomleft, markersize=2, label="Low sampling")
	plot!(xs_interp, arr_interp, label="FFT based sinc interpolation", linestyle=:dash)
	plot!(xs_high, arr_high, linestyle=:dashdotdot, label="High sampling")
end

# ╔═╡ eb1de74c-266e-4d7e-9bb2-b34f241b0c20
md"## Downsampling
32 samples in the downsampled signal should be sufficient for Nyquist sampling.
And as we can see, the downsampled signal still matches the original one.
"

# ╔═╡ a5977c5c-dbcd-402d-b7a3-56614968ccb7
begin
	N_ds = 32
	xs_ds = range(x_min, x_max, length=N_ds+1)[1:N_ds]
	arr_ds = real(resample(arr_high, (N_ds,)))
end;

# ╔═╡ f411f944-f08a-4c69-a334-35780b6d7418
begin
	scatter(xs_low, arr_low, legend=:bottomleft, markersize=2, label="Low sampling")
	plot!(xs_interp, arr_interp, label="FFT based sinc interpolation", linestyle=:dash)
	plot!(xs_ds, arr_ds, label="downsampled array", linestyle=:dot)	
end

# ╔═╡ 7f3be670-cfe3-4e60-b38f-ebacb4eaf8d3
md"## Fourier Shift Theorem

$\mathcal{F}[U(x+ Δx)](\mathbf k) = \mathcal{F}[U(x)](\mathbf k) \exp(i k \Delta x)$

$\Longrightarrow$
$U(x+ Δx) = \mathcal{F}^{-1}\bigg[\mathcal{F}[U(x)](\mathbf k) \exp(i k \Delta x)\bigg](x)$

* which allows for efficient sub-pixel shifting
"

# ╔═╡ 1ab61f99-799f-4c34-b0d1-205c01c20dbc
@bind offset Slider(-10:0.01:10, show_value=true)

# ╔═╡ 55667950-4542-4505-9d21-e44ea8b6074d
begin
	f3(x) = cos(4π * x / 30)
	x1 = 1:30
	x2 = x1 .+ 3
	y1 = f3.(x1)
	y2 = f3.(x2)
	y3 = shift(y2, tuple(offset))
end;

# ╔═╡ cfb02910-fb2d-4de0-bc81-8ecea50e5cb5
begin
	plot(y1, label="Original signal")
	plot!(y2, label="Shifted signal")
	plot!(y3, label="Fourier shifted with $offset")
end

# ╔═╡ 73c916e8-5761-40f4-8082-df5d6920d50b


# ╔═╡ 16f61a24-afe4-4e22-aa9c-a941b1819a90


# ╔═╡ 105344de-8617-47fd-aeed-84102f19eaeb
md"## Based on Shifting $\Longrightarrow$ Rotation

"

# ╔═╡ 344f9221-4213-4540-bb40-8a2de2d9b48c
begin
	z = zeros(Float32, round.(Ref(Int), 1.5 .* size(img)))
	FourierTools.center_set!(z, img)
end;

# ╔═╡ 27a7bcae-e50f-4225-bd7f-45934ec55be3
@bind ϕ Slider(-180:1:180, show_value=true)

# ╔═╡ bf2c54f1-8110-4e25-b89c-7276cf55f42a
Gray.(FourierTools.rotate(z, ϕ))

# ╔═╡ a5ea8226-ff14-432c-974e-68e1a79155cf
Gray.(FourierTools.rotate(img, ϕ))

# ╔═╡ Cell order:
# ╠═e6db292c-d2be-11eb-1c25-05ed50a9256c
# ╠═4958079f-da85-4357-94df-b122cf01c958
# ╠═b71a5e39-0805-4e3f-aa83-c3972f6c54af
# ╟─c39440fc-17e6-440c-9e10-7b2c9a7e9331
# ╠═e28dc936-04da-48cb-b324-73632d008290
# ╟─e8f1e951-26dd-4c0e-83f9-4a2c60b0ebed
# ╟─3adab086-0c77-40e5-990a-6b89b183cc4d
# ╟─850869e6-32da-48cf-b2a0-081cc9f8e524
# ╠═d6f9bd7a-ddcb-40b5-8509-f4dac3f8bd5d
# ╟─f0746bc6-f50a-49f1-96d2-0379498a0168
# ╠═979d641e-83ad-4484-bc28-912947711936
# ╠═24298466-c360-4c7d-b408-9f7d155faea5
# ╠═dfdfb8ff-6c30-4a95-8d6b-efbd9d943fd8
# ╠═2004c39e-f650-440e-858e-20b9f81c6632
# ╟─cc160200-5911-4c80-b651-a54e1608f186
# ╟─a725ae98-e2cc-439c-9cc1-692972ac98e7
# ╟─d5717c9c-2c16-47da-8533-a28acf189e55
# ╟─3d8f11a3-4ebf-4d36-a227-b2cafa13e1d6
# ╠═eb1de74c-266e-4d7e-9bb2-b34f241b0c20
# ╟─a5977c5c-dbcd-402d-b7a3-56614968ccb7
# ╟─f411f944-f08a-4c69-a334-35780b6d7418
# ╠═7f3be670-cfe3-4e60-b38f-ebacb4eaf8d3
# ╟─1ab61f99-799f-4c34-b0d1-205c01c20dbc
# ╟─55667950-4542-4505-9d21-e44ea8b6074d
# ╟─cfb02910-fb2d-4de0-bc81-8ecea50e5cb5
# ╠═73c916e8-5761-40f4-8082-df5d6920d50b
# ╠═16f61a24-afe4-4e22-aa9c-a941b1819a90
# ╠═105344de-8617-47fd-aeed-84102f19eaeb
# ╠═344f9221-4213-4540-bb40-8a2de2d9b48c
# ╟─27a7bcae-e50f-4225-bd7f-45934ec55be3
# ╠═bf2c54f1-8110-4e25-b89c-7276cf55f42a
# ╠═a5ea8226-ff14-432c-974e-68e1a79155cf
