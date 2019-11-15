library(data.table)
library(ggplot2)

n <- 1000

scoresA <- rnorm(n = n, m = 0, sd = 1)
scoresA

scoresB <- rnorm(n = n, m = 0.1, sd = 1)
scoresB

t.test(scoresA, scoresB)

ggplot(melt(data.frame(A = scoresA, B = scoresB))) + geom_jitter(aes(x = variable, y = value, col = variable), alpha = 0.33, shape = 16) + theme_classic()
