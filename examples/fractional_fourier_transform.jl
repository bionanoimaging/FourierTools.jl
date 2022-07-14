### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ a696290a-0122-11ed-01e5-a39256aed683
begin
	using Pkg;
	Pkg.activate(".")
	using Revise
end

# ╔═╡ cc0f07d5-9192-48cd-9a90-50eebb06783f
Pkg.add("IndexFunArrays")

# ╔═╡ 459649ed-ca70-426e-8273-97b146b5bcd5
using FourierTools, FFTW, NDTools, TestImages, Colors, ImageShow, PlutoUI, Napari, IndexFunArrays, Plots

# ╔═╡ 55894157-a2d1-4567-99a8-a052d5335dd1
begin
	"""
	    complex_show(arr)
	Displays a complex array. Color encodes phase, brightness encodes magnitude.
	Works within Jupyter and Pluto.
	"""
	function complex_show(cpx::AbstractArray{T, N}) where {T<:Complex, N}
	    Tr = real(T)
		ac = abs.(cpx)
	    HSV.(angle.(cpx)./Tr(2pi)*256, ones(Tr,size(cpx)), ac./maximum(ac))
	end
	
	
	"""
	    gray_show(arr; set_one=false, set_zero=false)
	Displays a real gray color array. Brightness encodes magnitude.
	Works within Jupyter and Pluto.
	## Keyword args
	* `set_one=false` divides by the maximum to set maximum to 1
	* `set_zero=false` subtracts the minimum to set minimum to 1
	"""
	function gray_show(arr; set_one=true, set_zero=false)
	    arr = set_zero ? arr .- minimum(arr) : arr
	    arr = set_one ? arr ./ maximum(arr) : arr
	    Gray.(arr)
	end
	
end

# ╔═╡ 18b0700c-20fc-4e58-8950-ca09fe34ea19
import FractionalTransforms

# ╔═╡ d90b7f67-4166-44fa-aab7-de2c4f38fc00
img = Float32.(testimage("resolution_test_512"));

# ╔═╡ 6dfdacdb-15f6-4eed-b1f8-ad7b6622c308
arr4 = reshape(1:16.0, (4,4))

# ╔═╡ 41aa4b60-209e-4011-b57f-ef7c76943fe6
arr5 = reshape(1:25.0, (5,5))

# ╔═╡ 764bf083-d03c-40b3-91b6-c2e04da1ba0f
frft(frft(arr4, 0.1), 0.15)

# ╔═╡ 372a12e4-47ce-40d5-a318-2aa064b57269
frft(arr4, 0.25)

# ╔═╡ bb95233f-afd8-4f50-b640-2079d03dbc93
ft(arr4)

# ╔═╡ 3ed30a2a-65f2-4cc8-a87b-4056e4be0c62
frft(frft(arr5, 0.5), 0.5)

# ╔═╡ 26de86fd-c354-461e-afe2-67e5ea50f927
ft(arr5) ./ length(arr5)

# ╔═╡ 6405c755-fa66-46e0-b342-a3b73720a98b
Gray.(abs.(frft(img, 0.0)))

# ╔═╡ 24901666-4cc4-497f-a6ff-68c3e7ead629
@bind s Slider(0:0.01:4, show_value=true)

# ╔═╡ 29e999fb-7d24-4907-9d90-66828cf2e90e
complex_show([frft(img, s) frft(frft(img, s/2), s/2)])

# ╔═╡ 9808fdb8-0457-4aa6-84fd-e9e67a1e1da5
complex_show(log1p.(frft(img, s)))

# ╔═╡ c123c36d-c33a-45e8-bfa9-18ac464e1b2f
complex_show((frft(img, 0.5)))

# ╔═╡ 44805dd0-9fd5-4c51-b89f-be01569e9595
complex_show((frft(frft(img, 0.5), 0.5)))

# ╔═╡ 10312daf-7908-4269-bdf2-0d2b25ca98a1
complex_show(ft(img))

# ╔═╡ 7c445baa-d970-4954-a3dc-df828971bfd7
gray_show(log1p.(abs.(ft(img))))

# ╔═╡ a9468610-df7d-45ab-b2ef-00021e4fc39d
gray_show(log1p.(abs.(frft(img, 1))))

# ╔═╡ 77b3fa5c-9d78-4485-8873-d0b820fcad63
gray_show(log1p.(abs.(frft(frft(img, 0.5), 0.5))))

# ╔═╡ 4dd6fdc9-72d5-4ab3-a1f6-8a58d1a46ff9
size(img)

# ╔═╡ 2e2b0cb1-99bc-43e6-9aad-192f552e2d6e
# ╠═╡ disabled = true
#=╠═╡
begin
	v = Napari.view_image(log1p.(abs.(frft(frft(img, 0.5), 0.5))))
	@add_image v log1p.(abs.(ft(img)))
end
  ╠═╡ =#

# ╔═╡ fe9900ca-effd-49be-af28-d5a804e91ee3
begin
	a = frft(img, 1)
	a ./= a[257, 257]
end

# ╔═╡ 2c7284f4-d483-4c23-bf9a-971d22c8633d
begin
	b = ft(img)
	b ./= b[257, 257]
end

# ╔═╡ e8c2bb34-3b79-46ac-b9d4-e37be206ace6
b[256, 256]

# ╔═╡ 7df75507-ac7c-4e1f-b35d-752f4fb8384a
a[256, 256]

# ╔═╡ c59fea9b-a381-4b4e-b5c0-b57064b24e9a
sum(b)

# ╔═╡ a1977492-506c-434f-9b6e-098e5e902568
sum(a)

# ╔═╡ b674160a-9b2a-4d5a-aebe-2f11a8bab551
complex_show(frft(img, 1))

# ╔═╡ 3d882cf6-bab6-4f3c-90ae-7f1c82dd2010
x = randn((21)) .* (1:21)

# ╔═╡ 5fdc76d5-2c39-4e49-990f-9d6b74b89103
gray_show(abs.(hcat(frft(frft(x, 0.5), 0.5), FractionalTransforms.frft(FractionalTransforms.frft(x, 0.5), 0.5))))

# ╔═╡ 3015cd73-cfae-45bd-a84d-19dfdce91482
frft(x, 1.0)

# ╔═╡ 8ed744b8-2a33-4e05-ba02-7597d9bb68c9
frft(x, 0.5)

# ╔═╡ 52a6f0a5-3142-4d1b-8529-b1131af53b40
FractionalTransforms.frft(x, 0.999)

# ╔═╡ b1b15e01-37bb-4109-9f70-9f0b7b44fbea
FractionalTransforms.frft(x, 1)

# ╔═╡ 24b3a010-28a7-49d5-bf70-9e49051a6c1d
FractionalTransforms.frft(FractionalTransforms.frft(x, 0.5), 0.5)

# ╔═╡ ed366d8f-cb63-4f9a-a169-a04020776b50
ft(x) ./ sqrt(length(x))

# ╔═╡ 1915c023-69cf-4d18-90cb-b47465dbef69


# ╔═╡ bae3c5b7-8964-493b-9e7b-d343e092219c
r = box(Float64, (101,), (50,))

# ╔═╡ ef1ba833-9b5d-43dd-837e-11c32b26fae6
plot(r)

# ╔═╡ 9370a6f9-12e7-44fe-b9c4-5dd648a988e0
Revise.errors()

# ╔═╡ 31c06af0-f6e2-4fca-8139-feb928d1a8ce
plot(abs.(frft(frft(r, 0.5), 0.5)))

# ╔═╡ 5e6cdaae-c81f-4d1a-829a-556807c3fea0
Revise.errors()

# ╔═╡ 2852c352-2d60-43b0-b33a-70e514684fe3
plot(real.(frft(frft(frft(r, 0.33333), 0.33333), 0.33333)))

# ╔═╡ 07d2b3b6-3584-4c64-9c4a-138beb3d6b88
@bind s2 Slider(0:0.01:4, show_value=true)

# ╔═╡ 609930b2-cc78-4401-97bc-5260983ef767
plot(real.(frft(r, s2)))

# ╔═╡ dab8899b-b1b2-4a8b-91c2-477b5d199160
plot(real.(FractionalTransforms.frft(collect(r), s2)))

# ╔═╡ b5e4073f-6f60-4a99-9734-91f99e26aa61
plot(real.(ft(r)))

# ╔═╡ 392463f5-0d07-4afa-8024-8c7e273e16af
plot(abs.(FractionalTransforms.frft(FractionalTransforms.frft(collect(r), s2/2), s2/2)))

# ╔═╡ Cell order:
# ╠═a696290a-0122-11ed-01e5-a39256aed683
# ╟─55894157-a2d1-4567-99a8-a052d5335dd1
# ╠═cc0f07d5-9192-48cd-9a90-50eebb06783f
# ╠═18b0700c-20fc-4e58-8950-ca09fe34ea19
# ╠═459649ed-ca70-426e-8273-97b146b5bcd5
# ╠═d90b7f67-4166-44fa-aab7-de2c4f38fc00
# ╠═6dfdacdb-15f6-4eed-b1f8-ad7b6622c308
# ╠═41aa4b60-209e-4011-b57f-ef7c76943fe6
# ╠═764bf083-d03c-40b3-91b6-c2e04da1ba0f
# ╠═372a12e4-47ce-40d5-a318-2aa064b57269
# ╠═bb95233f-afd8-4f50-b640-2079d03dbc93
# ╠═3ed30a2a-65f2-4cc8-a87b-4056e4be0c62
# ╠═26de86fd-c354-461e-afe2-67e5ea50f927
# ╠═6405c755-fa66-46e0-b342-a3b73720a98b
# ╠═24901666-4cc4-497f-a6ff-68c3e7ead629
# ╠═29e999fb-7d24-4907-9d90-66828cf2e90e
# ╠═9808fdb8-0457-4aa6-84fd-e9e67a1e1da5
# ╠═c123c36d-c33a-45e8-bfa9-18ac464e1b2f
# ╠═44805dd0-9fd5-4c51-b89f-be01569e9595
# ╠═10312daf-7908-4269-bdf2-0d2b25ca98a1
# ╠═7c445baa-d970-4954-a3dc-df828971bfd7
# ╠═a9468610-df7d-45ab-b2ef-00021e4fc39d
# ╠═77b3fa5c-9d78-4485-8873-d0b820fcad63
# ╠═4dd6fdc9-72d5-4ab3-a1f6-8a58d1a46ff9
# ╠═2e2b0cb1-99bc-43e6-9aad-192f552e2d6e
# ╠═fe9900ca-effd-49be-af28-d5a804e91ee3
# ╠═2c7284f4-d483-4c23-bf9a-971d22c8633d
# ╠═e8c2bb34-3b79-46ac-b9d4-e37be206ace6
# ╠═7df75507-ac7c-4e1f-b35d-752f4fb8384a
# ╠═c59fea9b-a381-4b4e-b5c0-b57064b24e9a
# ╠═a1977492-506c-434f-9b6e-098e5e902568
# ╠═b674160a-9b2a-4d5a-aebe-2f11a8bab551
# ╠═3d882cf6-bab6-4f3c-90ae-7f1c82dd2010
# ╠═5fdc76d5-2c39-4e49-990f-9d6b74b89103
# ╠═3015cd73-cfae-45bd-a84d-19dfdce91482
# ╠═8ed744b8-2a33-4e05-ba02-7597d9bb68c9
# ╠═52a6f0a5-3142-4d1b-8529-b1131af53b40
# ╠═b1b15e01-37bb-4109-9f70-9f0b7b44fbea
# ╠═24b3a010-28a7-49d5-bf70-9e49051a6c1d
# ╠═ed366d8f-cb63-4f9a-a169-a04020776b50
# ╠═1915c023-69cf-4d18-90cb-b47465dbef69
# ╠═bae3c5b7-8964-493b-9e7b-d343e092219c
# ╠═ef1ba833-9b5d-43dd-837e-11c32b26fae6
# ╠═9370a6f9-12e7-44fe-b9c4-5dd648a988e0
# ╠═31c06af0-f6e2-4fca-8139-feb928d1a8ce
# ╠═5e6cdaae-c81f-4d1a-829a-556807c3fea0
# ╠═2852c352-2d60-43b0-b33a-70e514684fe3
# ╠═07d2b3b6-3584-4c64-9c4a-138beb3d6b88
# ╠═609930b2-cc78-4401-97bc-5260983ef767
# ╠═dab8899b-b1b2-4a8b-91c2-477b5d199160
# ╠═b5e4073f-6f60-4a99-9734-91f99e26aa61
# ╠═392463f5-0d07-4afa-8024-8c7e273e16af
