library(ggplot2)
library(devtools)
install_github("rforge/partykit/pkg/partykit")
library(partykit)
install_github("rforge/partykit/pkg/disttree")
library(disttree)

n <- 300
k = 2

x <- runif(n, min = -0.4, max = 1)

mu <- 10*(tanh(3*exp(-(4*x-2)^2*k)))
plot(x, mu)

sigma <- 0.5 + (1*sample(x, length(x)))
plot(x, sigma)
  
y <- rnorm(n, mean = mu, sd = sigma)

ggplot(data.frame(x = x, y = y)) + geom_point(aes(x = x, y = y), alpha = 0.5) + theme_bw()

data <- data.frame(y = y, x = x)

distrf <- disttree(y ~ ., data = data, family = NOF())






