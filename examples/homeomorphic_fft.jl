### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ 2c1154be-a8fb-11eb-2dfa-d3fc55d26ba0
begin
	using Revise, FourierTools, Plots, ForwardDiff, LinearAlgebra, PyCall, FFTW, PlutoUI, PhysicalOptics
	FFTW.set_num_threads(12)
end

# ╔═╡ 8e53f88c-5a90-4b20-8c03-6e7780e6d7bd
# needed to interpolation
@pyimport scipy.interpolate as si

# ╔═╡ d8111917-6b7e-49bf-a065-9ad9f7b59b42
md"### Create some spatial coordinates in different formats and sizes"

# ╔═╡ b5767f75-b84a-44bc-9d30-1f6630b37382
begin
	N = 8192
	N_disp = 300
	N_small = 300
	ρ_1D = range(-2f-3, 2f-3, length=N)
	ρ_1D_small = range(-2f-3, 2f-3, length=N_small)

	ρ = [[i, j] for i in ρ_1D, j in ρ_1D]
	ρ_small = [[a, b] for a in ρ_1D_small, b in ρ_1D_small]
end;

# ╔═╡ ea21c127-c762-43ef-b7c9-9e0fa6df4b43
typeof(ρ)

# ╔═╡ b399e58e-12f7-48f2-911b-52de291e8b74
md"## Create a nice Image :)"

# ╔═╡ 571a8e1b-38f6-4552-9fde-fc909c9bf196
begin
	U_hard = zeros(Float32, (N, N));
	U_hard[N÷4:3N÷4, N÷4:3N÷4] .= 1
	U_hard[round(Int, 1.3*N÷4):round(Int, 2.7N÷4), round(Int, 1.3*N÷4):round(Int, 2.7N÷4)] .= 0
	U_hard[N÷20 * 5: N÷20 *8, N÷10*4: N÷10*7] .= 1
end;

# ╔═╡ e47d7441-6431-41da-ab54-a9b003f4cea0
begin
	psf = Float32.(exp.(.- (ρ_1D.^2 .+ ρ_1D'.^2)./5e-9))
	psf ./= sum(psf)
	U = FourierTools.conv_psf(U_hard, psf)
	U_small = resample(U, (N_small, N_small))
end;

# ╔═╡ 46fca5cf-1c9a-4a96-a083-7bc6e9246dc3
heatmap(U_small)

# ╔═╡ a0f9fc70-8636-4df6-bb02-f0f9d2b6cb63
md"## Analytical spherical phase

`ψ_sph` depends only on `ρ` and `R` was already passed to it.
`R` is defined in front of the figures
"

# ╔═╡ 82f7f1e2-8ca7-40e7-a165-4adb4eb77e04
ψsph_full(ρ, R, λ=550f-9, k0=oftype(λ, 2π/λ)) = sign(R) * k0 * sqrt(dot(ρ, ρ) + R^2)

# ╔═╡ 534cc94f-3029-4952-91f4-bd7424630b3a
ψsph(ρ, R=-5f-3) = ψsph_full(ρ, Float32(-5f-3))

# ╔═╡ 44e8d5eb-29c9-4fa4-a63d-0ce46d83ec2c
md"# Homeomorphic FFT"

# ╔═╡ 0d1e7725-1ed9-4db1-9523-0bd31672c992
begin
	Ũ, κ = FourierTools.hfft(U_small, ρ_small, ψsph)
end

# ╔═╡ 10b92bf4-3816-4365-bad7-bc434594a4c0
κ

# ╔═╡ 8f36f985-8755-44e7-87a7-2039cdb8b216
Ũ2, κ2 = FourierTools.hfft(U_small, ρ_small, ψsph, ϕ=0, Δz=4.97f-3)

# ╔═╡ c9b32da7-6b51-4974-8b6d-4287b2387906
heatmap(abs2.(ift(Ũ2)))

# ╔═╡ 3db5598c-4285-4da0-8d9c-17ae7cf1de45
heatmap(imag.(Ũ2) .+ real.(Ũ2))

# ╔═╡ 75599c42-7a48-4533-a310-06b3506f964c
heatmap(angle.(Ũ2))

# ╔═╡ 650a28a2-020a-45d0-a528-98c74c57a115
heatmap(abs2.(Ũ2))

# ╔═╡ 69d22dc0-384a-4cc5-bfe1-b04d62381d9e
heatmap(angle.(out))

# ╔═╡ aaf22e28-9faa-4b7e-85ac-ac470e0011f3
typeof(Ũ)

# ╔═╡ 5b05249c-c244-41ca-97ef-263a083e9bac
md"### Interpolation to regular grid with scipy

The output is on a irregular grid, therefore we need a sophisticated interpolation routine
"

# ╔═╡ 17b70cd4-251a-43a0-bbec-bb50da27b0ff
begin
	Ũ_lin = reshape(Ũ2[:, end:-1:1], (length(Ũ2, )))
	κ_lin = reshape(κ[:, end:-1:1], (length(κ, )))
	κ_lin_x = [i[1] for i in κ_lin]	
	κ_lin_y = [i[2] for i in κ_lin]
end

# ╔═╡ f25bd1d3-a582-4095-93b7-9e392036caea
typeof(κ_lin_x)

# ╔═╡ 61442cb9-801b-4cd3-8f3b-b2ae27a71fb1
begin
	o = ones((N_disp, N_disp))
	κ_reg_x_1D = range(κ[1][2], κ[end][2], length=N_disp)'
	κ_reg_y_1D = range(κ[1][2], κ[end][2], length=N_disp)
	κ_reg_x = o .* κ_reg_x_1D
	κ_reg_y = o .* κ_reg_y_1D
end;

# ╔═╡ 324b29ca-b194-429d-97ab-fffc3d1c0397
Ũ_interp_imag = si.griddata(κ_lin, imag.(Ũ_lin), (κ_reg_y, κ_reg_x), method="cubic");

# ╔═╡ 4195f594-10f8-4f81-b31c-79cb7a4035e7
Ũ_interp_real = si.griddata(κ_lin, real.(Ũ_lin), (κ_reg_y, κ_reg_x), method="cubic");

# ╔═╡ 12aceeef-e296-4bab-81d8-845d023b36c8
Ũ_interp_compl = Ũ_interp_real .+ 1im .* Ũ_interp_imag

# ╔═╡ 15f4a321-3949-4a9a-91cb-10879139393c
Ũ_interp_abs = si.griddata(κ_lin, abs.(Ũ_lin), (κ_reg_y, κ_reg_x), method="linear");

# ╔═╡ 50e2cc4a-c8ab-4422-9aa0-8c44d99ae553
md"## Regular FFT
To compare with HFFT
"

# ╔═╡ d6ae639c-3908-495a-93b2-bd2cc1adc720
U_fft_abs = resample(abs.(ffts(U .* exp.(1im .* ψsph.(ρ)))), (N_disp, N_disp))

# ╔═╡ 6abfdc7a-0cec-4da1-8a98-61d8448e2e1f
md"## Display results"

# ╔═╡ aee7354d-5e2f-4dec-a1f4-40db579c26dc
heatmap(.- κ_reg_x_1D', .- κ_reg_y_1D, abs.(Ũ_interp_abs), title="Homeomorphic FFT $N_small x $N_small")

# ╔═╡ 16d396d8-d8b3-43d4-8e41-bc1663b15038
heatmap(.- κ_reg_x_1D', .- κ_reg_y_1D, abs.(Ũ_interp_imag .* 1im .+ Ũ_interp_real), title="Homeomorphic FFT $N_small x $N_small")

# ╔═╡ dfeaafff-394f-4328-a509-0c266c0b568b
heatmap(ρ_1D_small, ρ_1D_small, U_fft_abs, title="FFT $N x $N")

# ╔═╡ 60cd58b1-9786-41d8-afd4-6b365991c918
heatmap(abs.(Ũ))

# ╔═╡ 60e52d82-b531-40bd-9f61-ae476b075f88
κ_reg_x[100, 1] - κ_reg_x[100, 2]

# ╔═╡ 8003ee99-f455-4137-ad04-04386fdbf8f9
κ_reg_x[100, 101] - κ_reg_x[100, 102]

# ╔═╡ d9aa5146-4a5b-459e-b171-fecd3e351d9f
κ_reg_x

# ╔═╡ 7375365f-e363-4ae2-bb96-4ef1e59fd05f


# ╔═╡ 70155a80-e73c-4ff2-8908-7a2f0be4f97c


# ╔═╡ 5ecf467c-d02e-4631-aef7-34f46414b1ac
# 	using Dierckx
#begin#
#spl = Spline2D(κ_lin_x, κ_lin_y, abs.(Ũ_lin), s=1e-4)
#Ũ_interp_abs2 = evalgrid(spl, κ_reg_y_1D, κ_reg_x_1D')
#ed

# heatmap(κ_reg_x_1D', κ_reg_y_1D, Ũ_interp_abs2, title="Homeomorphic FFT $N_small x $N_small")



# begin
#κ_lin_d = reshape_vec(κ_lin)
#Ũ_lin_d = reshape_vec(Ũ_lin)
#nd;

# ╔═╡ Cell order:
# ╠═2c1154be-a8fb-11eb-2dfa-d3fc55d26ba0
# ╠═8e53f88c-5a90-4b20-8c03-6e7780e6d7bd
# ╠═ea21c127-c762-43ef-b7c9-9e0fa6df4b43
# ╠═d8111917-6b7e-49bf-a065-9ad9f7b59b42
# ╠═b5767f75-b84a-44bc-9d30-1f6630b37382
# ╠═b399e58e-12f7-48f2-911b-52de291e8b74
# ╠═571a8e1b-38f6-4552-9fde-fc909c9bf196
# ╠═e47d7441-6431-41da-ab54-a9b003f4cea0
# ╠═46fca5cf-1c9a-4a96-a083-7bc6e9246dc3
# ╠═a0f9fc70-8636-4df6-bb02-f0f9d2b6cb63
# ╠═82f7f1e2-8ca7-40e7-a165-4adb4eb77e04
# ╠═534cc94f-3029-4952-91f4-bd7424630b3a
# ╠═44e8d5eb-29c9-4fa4-a63d-0ce46d83ec2c
# ╠═0d1e7725-1ed9-4db1-9523-0bd31672c992
# ╠═10b92bf4-3816-4365-bad7-bc434594a4c0
# ╠═8f36f985-8755-44e7-87a7-2039cdb8b216
# ╠═c9b32da7-6b51-4974-8b6d-4287b2387906
# ╠═3db5598c-4285-4da0-8d9c-17ae7cf1de45
# ╠═75599c42-7a48-4533-a310-06b3506f964c
# ╠═650a28a2-020a-45d0-a528-98c74c57a115
# ╠═69d22dc0-384a-4cc5-bfe1-b04d62381d9e
# ╠═aaf22e28-9faa-4b7e-85ac-ac470e0011f3
# ╠═5b05249c-c244-41ca-97ef-263a083e9bac
# ╠═17b70cd4-251a-43a0-bbec-bb50da27b0ff
# ╠═f25bd1d3-a582-4095-93b7-9e392036caea
# ╠═61442cb9-801b-4cd3-8f3b-b2ae27a71fb1
# ╠═324b29ca-b194-429d-97ab-fffc3d1c0397
# ╠═4195f594-10f8-4f81-b31c-79cb7a4035e7
# ╠═12aceeef-e296-4bab-81d8-845d023b36c8
# ╠═15f4a321-3949-4a9a-91cb-10879139393c
# ╠═50e2cc4a-c8ab-4422-9aa0-8c44d99ae553
# ╠═d6ae639c-3908-495a-93b2-bd2cc1adc720
# ╠═6abfdc7a-0cec-4da1-8a98-61d8448e2e1f
# ╠═aee7354d-5e2f-4dec-a1f4-40db579c26dc
# ╠═16d396d8-d8b3-43d4-8e41-bc1663b15038
# ╠═dfeaafff-394f-4328-a509-0c266c0b568b
# ╠═60cd58b1-9786-41d8-afd4-6b365991c918
# ╠═60e52d82-b531-40bd-9f61-ae476b075f88
# ╠═8003ee99-f455-4137-ad04-04386fdbf8f9
# ╠═d9aa5146-4a5b-459e-b171-fecd3e351d9f
# ╠═7375365f-e363-4ae2-bb96-4ef1e59fd05f
# ╠═70155a80-e73c-4ff2-8908-7a2f0be4f97c
# ╠═5ecf467c-d02e-4631-aef7-34f46414b1ac
