data <- data.frame(x = seq(1,1000,1), y = 0)
data$y <- (sin(data$x/400) + rnorm(nrow(data), mean = 0, sd = 0.06))*10
data$x <- (data$x-mean(data$x))/sd(data$x)

plot(data)

l1n1 <- tanh(data$x*0.25)
plot(data$x, l1n1)

l1n2 <- tanh(data$x*0.5)
plot(data$x, l1n2)

l1n3 <- tanh(data$x*-0.5)
plot(data$x, l1n3)

l2n1 <- (l1n1*20) + (l1n2*0.1) + (l1n3*20)

plot(data$x, l2n1, ylim = c(-20,20))
points(data$x, data$y, col = "red")

residuals <- data$y-l2n1
plot(residuals)

w <- c(0.25,0.5,0.5,0.1,0.8,-0.6)

l1n1 <- tanh(data$x*w[1])
plot(data$x, l1n1)

l1n2 <- tanh(data$x*w[2])
plot(data$x, l1n2)

l1n3 <- tanh(data$x*w[3])
plot(data$x, l1n3)

l2n1 <- (l1n1*w[4]) + (l1n2*w[5]) + (l1n3*w[6])

plot(data$x, l2n1, ylim = c(-20,20))
points(data$x, data$y, col = "red")

residuals <- data$y-l2n1
plot(residuals)
