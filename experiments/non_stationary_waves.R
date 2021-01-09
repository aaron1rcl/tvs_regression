
# Time range (50 cycles)
t = 1:1000
t = t/100
#t = rep(t, 10)

plot(sin(2*pi*t), type="lines")

# Create a non stationary series determining the frequency
f = rep(NA, length(t))
f[1] = 2*pi
for (i in 2:length(t)){
  f[i] = f[i - 1] + rnorm(1, 0, 0.04*pi)
}