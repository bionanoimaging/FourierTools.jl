using FourierTools


f(x) = sin(x * 2π)
x = [0.0, 0.2, 0.4, 0.6, 0.8] 

@show(f.(x))
@show(f.(x .- 0.5))
@show(shift(f.(x), -2.5))



plot(x, f.(x))
#plot!(range(0, 1, length=100), resample(f.(x), (100,)))
plot!(x .- 0.5, f.(x .- 0.5))
plot!(x .- 0.5, shift(f.(x), -2.5))
