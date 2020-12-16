library("mvtnorm")
library("tidyverse")


a = 2
x = runif(10,0,10)
xi = x + rnorm(10,0,2)
plot(x, xi)
y = a*x + rnorm(10,0,1)
plot(xi, y)
lm(y~xi)

oy = c(1:100)/10
ox = c(1:100)/10
a = 1.5 + 1:10/10
c = c(1:100)/10

monte_carlo = as.data.frame(expand.grid(a, c, oy, ox))
monte_carlo$likelihood = NA
colnames(monte_carlo) = c("alpha","c","oy","ox","likelihood")
monte_carlo$r = monte_carlo$ox/monte_carlo$oy

# Known ratio of the standard deviations
monte_carlo = subset(monte_carlo, monte_carlo$r == 2)

# Remove a bunch to speed it up
#monte_carlo = monte_carlo[sample(1:nrow(monte_carlo), size=50000),]


points((1:100)/10, 1.6*((1:100)/10) + 3, col="red")
points((1:100)/10, 1.3*((1:100)/10) + 4, col="blue")


for (i in 1:nrow(monte_carlo)){
  print(i)
  # Assume independent 
  l = all_likelihoods(xi, y, x_theta=monte_carlo[i,'ox'], y_theta=monte_carlo[i,'oy'], alpha=monte_carlo[i,'alpha'], c=monte_carlo[i,'c'])
  monte_carlo[i,'likelihood'] = l
}

monte_carlo = monte_carlo[order(monte_carlo$likelihood),]

plot(monte_carlo$ox, monte_carlo$likelihood, ylim=c(0,50))

all_likelihoods = function(xi, y, x_theta, y_theta, alpha=alpha, c=c){
  
  shifts = NULL
  sum_l = 0
  for (i in 1:length(xi)){
    #print(i)
    point_l = point_likelihood(xi[i], y[i], x_theta, y_theta, alpha=alpha, c=c)
    #print(point_l)
    sum_l = sum_l + point_l$likelihood
    shifts = append(shifts, point_l$shift_z)
  }
  # Return the total shift likelihood across all points
  return(sum_l)
}



point_likelihood = function(xi0, yi, x_theta, y_theta, alpha=2, c=0){
  
  z_optim = optim(0, 
        fn=single_likelihood, 
        x_theta = x_theta,
        y_theta=y_theta, 
        yi=yi, 
        xi = xi0,
        alpha=alpha,
        c=c, 
        method="Brent",
        lower=-6*x_theta,
        upper=6*x_theta,
        type="mv")
  
  shift_z = z_optim$par
  max_l = z_optim$value

  return(list("shift_z"=(shift_z), "likelihood"=max_l))
}

single_likelihood = function(params, x_theta, y_theta, xi0, yi, alpha, c, type="individual"){
  
    z = params[1]
    xl = dnorm(z, 0, sd=x_theta, log=TRUE)
    y_p = alpha*(z + xi0) + c
    res = yi - y_p
    yl = dnorm(res, 0, y_theta, log=TRUE)
    #print(paste0("XL: ",xl, " , YL: ", yl, ", Shift: ",z, ", Res: ",res))
    
    if (type == "individual"){
      return (-(yl + xl))
    } else {
      # Calculate the likelihood from a Multivariate Gaussian
      # Assume independence of the parameters
      covariance_mat = matrix(c(x_theta, 0, 0, y_theta), nrow=2, ncol=2)
      means = c(0,0)
      m_l = dmvnorm(c(z, res), means, covariance_mat, log = TRUE)
      return(-m_l)
    }
}


# Exploration of non-identifiability
plot(xi, y, xlim=c(0,15), ylim = c(0,22))
lines((1:100)/10, 2*(1:100)/10, col="red")
points(xi0, yi, col="blue")

# For different values of 
z_optim = optim(0, 
                fn=single_likelihood, 
                x_theta = x_theta,
                y_theta = y_theta, 
                yi=yi, 
                xi = xi0,
                alpha=alpha,
                method="Brent",
                lower=-6*x_theta,
                upper=6*x_theta,
                type="mv")


# Joint distributions of two different distributions
jx = rnorm(100000, 0, 10)
jy = rgamma(100000, shape= 2, rate = 1)

j_pdf = kde2d(jx, jy, n=1000)
jpdf_wide = (j_pdf$z)
colnames(jpdf_wide) = j_pdf$y
jpdf_wide = as.data.frame(jpdf_wide)
jpdf_wide$x = j_pdf$x

jpdf_wide = as_tibble(jpdf_wide)
jpdf_wide = jpdf_wide %>% pivot_longer(., !x)
colnames(jpdf_wide)[2] = "y"

# Plot the joint PDF plot
v <- ggplot(jpdf_wide, aes(x=x, y=as.numeric(y), z = value)) + 
  ylim(0,5)
v + geom_contour_filled()




# Multivariate likelihood vs individual likelihoods
x = rnorm(1000,0,1)
y = rnorm(1000,0,3) + 10*x

# Build the multviariate covariance
u = c(mean(x), mean(y))
c = cov(x, y)
var(x)
var(y)
covariance_mat = matrix(c(var(x), c, c, var(y)), nrow=2, ncol=2)


j_pdf = kde2d(x, y, n=25)
jpdf_wide = (j_pdf$z)
colnames(jpdf_wide) = j_pdf$y
jpdf_wide = as.data.frame(jpdf_wide)
jpdf_wide$x = j_pdf$x

jpdf_wide = as_tibble(jpdf_wide)
jpdf_wide = jpdf_wide %>% pivot_longer(., !x)
colnames(jpdf_wide)[2] = "y"

# Plot the joint PDF plot
v <- ggplot(jpdf_wide, aes(x=x, y=as.numeric(y), z = value))
v + geom_contour_filled()
v + geom_point(aes(x=k, y=m), col="red")

# Print 
mls = NULL
ils = NULL
for (i in 1:1000){
  k = rnorm(1,0,1)
  m = rnorm(1,0,3)
  points(k, m, col="red")
  ml = dmvnorm(c(k, m), u, covariance_mat, log = TRUE)
  
  kl = dnorm(k, u[1], sqrt(var(x)), log=TRUE)
  il = dnorm(m, u[2], sqrt(var(y)), log=TRUE)
  
  mls = append(mls, ml)
  ils = append(ils, kl + il)
}

plot(mls,ils)



# Multivariate likelihood vs individual likelihoods
x = rnorm(1000,0,1)
y = rnorm(1000,0,3) + 10*x

# Build the multviariate covariance
u = c(mean(x), mean(y))
c = cov(x, y)
var(x)
var(y)
covariance_mat = matrix(c(var(x), c, c, var(y)), nrow=2, ncol=2)


j_pdf = kde2d(x, y, n=25)
jpdf_wide = (j_pdf$z)
colnames(jpdf_wide) = j_pdf$y
jpdf_wide = as.data.frame(jpdf_wide)
jpdf_wide$x = j_pdf$x

jpdf_wide = as_tibble(jpdf_wide)
jpdf_wide = jpdf_wide %>% pivot_longer(., !x)
colnames(jpdf_wide)[2] = "y"

# Plot the joint PDF plot
v <- ggplot(jpdf_wide, aes(x=x, y=as.numeric(y), z = value))
v + geom_contour_filled()
v + geom_point(aes(x=k, y=m), col="red")

# Print 
mls = NULL
ils = NULL
for (i in 1:1000){
  k = rnorm(1,0,1)
  m = rnorm(1,0,3)
  points(k, m, col="red")
  ml = dmvnorm(c(k, m), u, covariance_mat, log = TRUE)
  
  kl = dnorm(k, u[1], sqrt(var(x)), log=TRUE)
  il = dnorm(m, u[2], sqrt(var(y)), log=TRUE)
  
  mls = append(mls, ml)
  ils = append(ils, kl + il)
}

plot(mls,ils)




# Multivariate likelihood vs individual likelihoods
x = rnorm(1000,0,1)
y = rgamma(1000,1,3) + 0.1*x

# Build the multviariate covariance
u = c(mean(x), mean(y))
c = cov(x, y)
var(x)
var(y)
covariance_mat = matrix(c(var(x), c, c, var(y)), nrow=2, ncol=2)


j_pdf = kde2d(x, y, n=25)
jpdf_wide = (j_pdf$z)
colnames(jpdf_wide) = j_pdf$y
jpdf_wide = as.data.frame(jpdf_wide)
jpdf_wide$x = j_pdf$x

jpdf_wide = as_tibble(jpdf_wide)
jpdf_wide = jpdf_wide %>% pivot_longer(., !x)
colnames(jpdf_wide)[2] = "y"

# Plot the joint PDF plot
v <- ggplot(jpdf_wide, aes(x=x, y=as.numeric(y), z = value))
v + geom_contour_filled()
v + geom_point(aes(x=k, y=m), col="red")

# Print 
mls = NULL
ils = NULL
for (i in 1:1000){
  k = rnorm(1,0,1)
  m = rnorm(1,0,3)
  points(k, m, col="red")
  ml = dmvnorm(c(k, m), u, covariance_mat, log = TRUE)
  
  kl = dnorm(k, u[1], sqrt(var(x)), log=TRUE)
  il = dnorm(m, u[2], sqrt(var(y)), log=TRUE)
  
  mls = append(mls, ml)
  ils = append(ils, kl + il)
}

plot(mls,ils)







