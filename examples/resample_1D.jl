### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ f054ac12-47b3-11eb-0d47-43990d3906a8
using Revise, FourierTools, Plots, FFTW

# ╔═╡ c5555382-4924-11eb-15d1-17cb887ca876
md"#### Preparing data

Please note, that we always exclude `x_max` from the x-positions since the FFT based interpolation is basically a high resolution evaluation of the Fourier series which assumes a periodic signal.
"

# ╔═╡ 0511e9da-47b4-11eb-3bde-c78c192be0cf
begin
	N_low = 64
	x_min = 0.0
	x_max = 8*2π
	
	xs_low = range(x_min, x_max, length=N_low+1)[1:N_low]
	xs_high = range(x_min, x_max, length=5000)[1:end-1]
	f(x) = sin(0.5*x) + cos(x) + cos(2 * x) + sin(0.25*x)
	arr_low = f.(xs_low)
	arr_high = f.(xs_high)
end

# ╔═╡ 5e5cc26a-49eb-11eb-01a7-150d675fee58
md"#### Calculate the interpolation" 

# ╔═╡ b8fe21d8-47b5-11eb-1854-09254d3cbba1
begin
	N = 1000
	xs_interp = range(x_min, x_max, length=N+1)[1:N]
	arr_interp = resample(arr_low, N)
end

# ╔═╡ bf11f160-4924-11eb-3812-777bb547f499
md"#### Plot the final results"

# ╔═╡ 0511d792-47b4-11eb-316a-3d0148d68406
begin
	scatter(xs_low, arr_low, legend=:bottomleft, markersize=2, label="Low sampling")
	plot!(xs_interp, arr_interp, label="FFT based sinc interpolation", linestyle=:dash)
	plot!(xs_high, arr_high, linestyle=:dashdotdot, label="High sampling")
end

# ╔═╡ 20f9491e-511b-11eb-1a6e-c3af6f5e11a5
md"#### Downsampling
32 samples in the downsampled signal should be sufficient for Nyquist sampling.
And as we can see, the downsampled signal still matches the original one.
"

# ╔═╡ 2e7e1800-511b-11eb-3334-4ddf5076143e
begin
	N_ds = 32
	xs_ds = range(x_min, x_max, length=N_ds+1)[1:N_ds]
	arr_ds = real(resample(arr_high, (N_ds,))) # currently slightly wrong because of a bug
	arr_ds2 = real(resample(0im .+ arr_high, (N_ds,)))
end

# ╔═╡ 48c79aa6-8fbc-11eb-0596-b125f1331482
arr_ds ≈ arr_ds2

# ╔═╡ 29db9534-511b-11eb-1bdf-37ea0594681f
begin
	scatter(xs_low, arr_low, legend=:bottomleft, markersize=2, label="Low sampling")
	plot!(xs_interp, arr_interp, label="FFT based sinc interpolation", linestyle=:dash)
	plot!(xs_ds, arr_ds2, label="downsampled array", linestyle=:dot)	
end

# ╔═╡ Cell order:
# ╠═f054ac12-47b3-11eb-0d47-43990d3906a8
# ╟─c5555382-4924-11eb-15d1-17cb887ca876
# ╠═0511e9da-47b4-11eb-3bde-c78c192be0cf
# ╟─5e5cc26a-49eb-11eb-01a7-150d675fee58
# ╠═b8fe21d8-47b5-11eb-1854-09254d3cbba1
# ╟─bf11f160-4924-11eb-3812-777bb547f499
# ╠═0511d792-47b4-11eb-316a-3d0148d68406
# ╠═20f9491e-511b-11eb-1a6e-c3af6f5e11a5
# ╠═2e7e1800-511b-11eb-3334-4ddf5076143e
# ╠═48c79aa6-8fbc-11eb-0596-b125f1331482
# ╠═29db9534-511b-11eb-1bdf-37ea0594681f
