using FourierTools, FFTW, PGFPlotsX


N_low = 64
x_min = 0.0
x_max = 8*2π

xs_low = range(x_min, x_max, length=N_low+1)[1:N_low]
xs_high = range(x_min, x_max, length=5000)[1:end-1]
f(x) = sin(0.5*x) + cos(x) + cos(2 * x) + sin(0.25*x)
arr_low = f.(xs_low)
arr_high = f.(xs_high)

N = 1000
xs_interp = range(x_min, x_max, length=N+1)[1:N]
arr_interp = resample(arr_low, N)




plt1 = @pgf PGFPlotsX.Axis(
    { 
        xmin=0, xmax=50, ymin=-3, ymax=3.5, 
        legend_pos="north east", legend_entries={"low sampling", "sinc interpolated", "high sampling"},
    },
    
    Plot({mark="*", style="only marks", color="green",mark_options = {scale=0.3},
        },
        Table([xs_low, arr_low])
    ),
    
    Plot({"sharp plot", color="red"
        },
        Table([xs_interp, arr_interp])
    ),
    

    Plot({"sharp plot", style="dashed", color="blue"
        },
        Table([xs_high, arr_high])
    ),
)
pgfsave("../../paper/figures/resampling.pdf", plt1)
