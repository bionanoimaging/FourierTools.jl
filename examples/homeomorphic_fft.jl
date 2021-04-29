### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ 2c1154be-a8fb-11eb-2dfa-d3fc55d26ba0
using Revise, FourierTools, Plots, ForwardDiff, LinearAlgebra, PyCall, FFTW

# ╔═╡ 8e53f88c-5a90-4b20-8c03-6e7780e6d7bd
@pyimport scipy.interpolate as si

# ╔═╡ b5767f75-b84a-44bc-9d30-1f6630b37382
begin
	N = 8192
	N_disp = 256
	N_small = 100
	ρ_1D = range(-2e-3, 2e-3, length=N)
	ρ_1D_small = range(-2e-3, 2e-3, length=N_small)

	ρ = [[ρ_1D[i], ρ_1D[j]] for i = 1:N, j=1:N]
	ρ_small = [[a, b] for a in ρ_1D_small, b in ρ_1D_small]
end;

# ╔═╡ c2ea9c8b-0104-4932-b49e-6a0436ac9ebb
U_hard = zeros((N, N));

# ╔═╡ 571a8e1b-38f6-4552-9fde-fc909c9bf196
begin
	U_hard[N÷2:3*N÷5, 4*N÷7:N÷7*5] .= 1
	U_hard[N÷20 * 5: N÷20 *8, N÷10*4: N÷10*7] .= 1
end;

# ╔═╡ e47d7441-6431-41da-ab54-a9b003f4cea0
begin
	psf = exp.(.- (ρ_1D.^2 .+ ρ_1D'.^2)./5e-9)
	psf ./= sum(psf)
	U = conv_psf(U_hard, psf)
	U_small = resample(U, (N_small, N_small))
end;

# ╔═╡ 46fca5cf-1c9a-4a96-a083-7bc6e9246dc3
heatmap(U_small)

# ╔═╡ 24904425-0d25-45d3-bc81-fb6e49124826
sign(-1)

# ╔═╡ 534cc94f-3029-4952-91f4-bd7424630b3a
ψsph(ρ, R=2e-3, λ=550e-9, k0=2π/λ) = sign(R) * k0 * sqrt(dot(ρ, ρ) + R^2)

# ╔═╡ 16113ae5-dae3-4b87-b6b7-68d70577f28a
ψsph([1e-3, 2e-3])

# ╔═╡ 0d1e7725-1ed9-4db1-9523-0bd31672c992
Ũ, κ = FourierTools.hfft(U_small, ρ_small, ψsph)

# ╔═╡ 17b70cd4-251a-43a0-bbec-bb50da27b0ff
begin
	Ũ_lin = reshape(Ũ[:, end:-1:1], (length(Ũ, )))
	κ_lin = reshape(κ[:, end:-1:1], (length(κ, )))
	#Ũ_lin = circshift(reshape(Ũ, (length(Ũ, ))), (-100))
	#κ_lin = circshift(reshape(κ, (length(κ, ))), (-100))
end

# ╔═╡ 66f797f8-4033-4627-ac59-e6fa1e25d45e
κ[end][1]

# ╔═╡ 61442cb9-801b-4cd3-8f3b-b2ae27a71fb1
begin
	o = ones((N_disp, N_disp))
	κ_reg_x = o .* range(κ[1][2], κ[end][2], length=N_disp)'
	κ_reg_y = o .* range(κ[1][1], κ[end][1], length=N_disp)
end;

# ╔═╡ 15f4a321-3949-4a9a-91cb-10879139393c
Ũ_interp = si.griddata(κ_lin, abs.(Ũ_lin), (κ_reg_y, κ_reg_x), method="cubic");

# ╔═╡ aee7354d-5e2f-4dec-a1f4-40db579c26dc
heatmap(abs.(Ũ_interp))

# ╔═╡ 60cd58b1-9786-41d8-afd4-6b365991c918
heatmap(abs.(Ũ))

# ╔═╡ d6ae639c-3908-495a-93b2-bd2cc1adc720
U_fft = resample(abs.(ffts(U .* exp.(1im .* ψsph.(ρ)))), (N_disp, N_disp))

# ╔═╡ dfeaafff-394f-4328-a509-0c266c0b568b
heatmap(U_fft)

# ╔═╡ Cell order:
# ╠═2c1154be-a8fb-11eb-2dfa-d3fc55d26ba0
# ╠═8e53f88c-5a90-4b20-8c03-6e7780e6d7bd
# ╠═b5767f75-b84a-44bc-9d30-1f6630b37382
# ╠═c2ea9c8b-0104-4932-b49e-6a0436ac9ebb
# ╠═571a8e1b-38f6-4552-9fde-fc909c9bf196
# ╠═e47d7441-6431-41da-ab54-a9b003f4cea0
# ╠═46fca5cf-1c9a-4a96-a083-7bc6e9246dc3
# ╠═24904425-0d25-45d3-bc81-fb6e49124826
# ╠═534cc94f-3029-4952-91f4-bd7424630b3a
# ╠═16113ae5-dae3-4b87-b6b7-68d70577f28a
# ╠═0d1e7725-1ed9-4db1-9523-0bd31672c992
# ╠═17b70cd4-251a-43a0-bbec-bb50da27b0ff
# ╠═66f797f8-4033-4627-ac59-e6fa1e25d45e
# ╠═61442cb9-801b-4cd3-8f3b-b2ae27a71fb1
# ╠═15f4a321-3949-4a9a-91cb-10879139393c
# ╠═aee7354d-5e2f-4dec-a1f4-40db579c26dc
# ╠═60cd58b1-9786-41d8-afd4-6b365991c918
# ╠═dfeaafff-394f-4328-a509-0c266c0b568b
# ╠═d6ae639c-3908-495a-93b2-bd2cc1adc720
