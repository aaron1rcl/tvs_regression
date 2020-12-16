library("ggplot2")

u = runif(1000,-20,20)
n = rnorm(200,0,3)

hist(c(u,n))



b=(1:100)/10
oy = (1:10)
width=10:20
width = width[width %% 2 == 0]
ot=(1:30)/10
ox=(1:2)
#b=1
#oy=1
#width = 10
#ot = 3
#ox = 1
mc = expand.grid(b, oy, width, ot, ox)
mc$estimate_u = NA
mc$estimate_sd = NA
mc$t_u = NA
mc$t_sd = NA
colnames(mc)[1:5] = c("b","oy","width","ot","ox")
estimate_sd = rep(NA, nrow(mc))
t_sd = rep(NA, nrow(mc))
# Shuffle
mc = mc[sample(1:nrow(mc)),]
for (j in 1:nrow(mc)){
  print(j)
  out = get_distributions(mc[j,'b'],
                          mc[j,'oy'],
                          mc[j,'width'],
                          mc[j,'ot'],
                          mc[j,'ox'])
  
  estimate_sd[j] = out$esd
  t_sd[j] = out$tsd
  #mc[j,'estimate_u'] = out$eu
  #mc[j,'estimate_sd'] = out$esd
  #mc[j,'t_u'] = out$tu
  #mc[j,'t_sd'] = out$tsd
  
}

plot(mc$ot/mc$oy, estimate_sd/t_sd)

library(mgcv)
mc$estimate_sd = estimate_sd
mc$t_sd = t_sd
gam_fit = mgcv::gam(estimate_sd/t_sd~s(b, k=3) + s(oy, k=3) + width + s(ot,k=3) + ox, data=mc)
predict(gam_fit)


get_distributions = function(b, oy, width, ot, ox){
  # Select a random time point t between 1 & 20, by some shift distribution
  estimated_shifts = NULL
  ts = NULL
  for (i in 1:1000){
    y = rnorm(width + 1,0,oy)
    t = round(rnorm(1,0,ot))
    x = rnorm(1,0,ox)
    y[width/2 + 1 + t] = y[width/2 + 1 + t] + b*x
    res = y - b*x
    
    # Check for the likelihood
    shift_l = dnorm((-width/2):(width - width/2), 0, ot, log=TRUE)
    res_l = dnorm(res, 0, oy, log=TRUE)
    
    l = shift_l + res_l
    max_l = which.max(l)
    estimated_shift = max_l - (width/2 + 1)
    estimated_shifts = append(estimated_shifts, estimated_shift)
    ts = append(ts, t)
  }
  
  return(list("eu" = mean(estimated_shifts),
              "esd" = sd(estimated_shifts),
              "tu" = mean(ts),
              "tsd" = sd(ts)))
}

# Plot the distributions of the actual shifted value and the estimated shifted values
hist(estimated_shifts)
# ts=round(rnorm(10000,0.009,sd=1.17612)) Confirmed this is still a normal distribution - now need to derive
# estimated sd and mean, from the relationship of the various parameters
hist(ts)


plot_df = c(estimated_shifts, ts)
label = c(rep("t_hat",length(ts)), rep("t", length(ts)))
plot_df = as.data.frame(cbind(plot_df, label))
colnames(plot_df) = c("shift","label")
plot_df$shift = as.numeric(plot_df$shift)

ggplot(plot_df, aes(shift, fill = label)) + geom_density(alpha = 0.2)
sd(estimated_shifts)
sd(ts)



# Gaussian likelihood function
b = 10
lt = dnorm((-30:30)/10,0,1) + dnorm(rnorm(1000,0,1),0,1)
hist(lt)
le = dnorm((-100:100)/10,0,3) + dnorm(rnorm(1000,0,1),0,1)
hist(le)
