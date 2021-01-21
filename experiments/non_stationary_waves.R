
# Time range (50 cycles)
t = 1:1000
t = t/100
#t = rep(t, 10)

plot(sin(2*pi*t), type="lines")

# Create a non stationary series determining the frequency
f = rep(NA, length(t))
f[1] = 2*pi
for (i in 2:length(t)){
  f[i] = 0.9*f[i - 1] + rnorm(1, 0, 0.001*pi)
}

plot(f, type="lines")

plot(sin(f*t), type="lines")
lines(sin(2*pi*t), col="red")

y = sin(f*t)

stft(y)
