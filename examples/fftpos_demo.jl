### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ 4d38c306-5ced-11ec-32c8-354a64d62d72
using Pkg

# ╔═╡ feec0e1a-0b9c-4e18-9131-bdaa4ba8981d
Pkg.develop(path="../.")

# ╔═╡ 94a14d53-eda6-4840-a389-51c3df60f017
using Plots, FourierTools, NDTools

# ╔═╡ 80c0e146-0ddf-42f5-a4e2-b528b8e73b40
x = 2π .* fftpos(1,5,3)

# ╔═╡ 75810e0b-442f-46d1-a484-f78d638cf576
f(x) = sin(x)

# ╔═╡ 9903e739-58eb-403e-a903-94a3b4815ab9
collect(fftpos(1, 5, 3))

# ╔═╡ 97dd506f-46f8-46a8-bbd8-a6bd6b58ef79
collect(fftpos(1, 101, 51))

# ╔═╡ 4735a86c-0ecc-45d9-91c7-26eed0f51804
collect(fftpos(1, 100, 51))

# ╔═╡ e0ecb758-1825-46c8-9d72-2baa39c1eaf6
md"
The even interpolation somehow extrapolates (!) only the left tail. the right tail
"

# ╔═╡ 82b62bf7-587a-4787-81ae-46fc328bf87a
plot(fftpos(1, 20, 11), circshift(resample(circshift(f.(x), -2), 20), 10))

# ╔═╡ 7475a27b-4423-49db-9b69-0ec15c7c2a89
plot(fftpos(1, 21, 11), circshift(resample(circshift(f.(x), -2), 21), 10))

# ╔═╡ Cell order:
# ╠═4d38c306-5ced-11ec-32c8-354a64d62d72
# ╠═feec0e1a-0b9c-4e18-9131-bdaa4ba8981d
# ╠═94a14d53-eda6-4840-a389-51c3df60f017
# ╠═80c0e146-0ddf-42f5-a4e2-b528b8e73b40
# ╠═75810e0b-442f-46d1-a484-f78d638cf576
# ╟─9903e739-58eb-403e-a903-94a3b4815ab9
# ╠═97dd506f-46f8-46a8-bbd8-a6bd6b58ef79
# ╠═4735a86c-0ecc-45d9-91c7-26eed0f51804
# ╟─e0ecb758-1825-46c8-9d72-2baa39c1eaf6
# ╠═82b62bf7-587a-4787-81ae-46fc328bf87a
# ╠═7475a27b-4423-49db-9b69-0ec15c7c2a89
