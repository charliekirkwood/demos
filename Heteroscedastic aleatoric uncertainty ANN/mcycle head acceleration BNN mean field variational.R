library(keras)
library(MASS)
library(ggplot2)
library(data.table)

### Load data, and normalise predictors for ANN

data <- as.data.table(mcycle)

ggplot(data) + geom_point(aes(x = times, y = accel))

meantime <- mean(data$times)
sdtime <- sd(data$times)

datanorm <- copy(data)
datanorm[, times := (times-meantime)/sdtime, ]

ggplot(datanorm) + geom_point(aes(x = times, y = accel))

### Split data into train and test

testsamps <- sample(1:nrow(datanorm), nrow(datanorm)/10)

x_train <- as.matrix(datanorm[!testsamps, times])
y_train <- as.matrix(datanorm[!testsamps, accel])

x_test <- as.matrix(datanorm[testsamps, times])
y_test <- as.matrix(datanorm[testsamps, accel])

### Construct neural network

prior_trainable <-
  function(kernel_size,
           bias_size = 0,
           dtype = NULL) {
    n <- kernel_size + bias_size
    keras_model_sequential() %>%
      # we'll comment on this soon
      # layer_variable(n, dtype = dtype, trainable = FALSE) %>%
      layer_variable(n, dtype = dtype, trainable = TRUE) %>%
      layer_distribution_lambda(function(t) {
        tfd_independent(tfd_normal(loc = t, scale = 1),
                        reinterpreted_batch_ndims = 1)
      })
  }

posterior_mean_field <-
  function(kernel_size,
           bias_size = 0,
           dtype = NULL) {
    n <- kernel_size + bias_size
    c <- log(expm1(1))
    keras_model_sequential(list(
      layer_variable(shape = 2 * n, dtype = dtype),
      layer_distribution_lambda(
        make_distribution_fn = function(t) {
          tfd_independent(tfd_normal(
            loc = t[1:n],
            scale = 1e-5 + tf$nn$softplus(c + t[(n + 1):(2 * n)])
          ), reinterpreted_batch_ndims = 1)
        }
      )
    ))
  }

model <- keras_model_sequential() %>%
  layer_dense_variational(
    units = 1,
    make_posterior_fn = posterior_mean_field,
    make_prior_fn = prior_trainable,
    kl_weight = 0.5 / n
  ) %>%
  layer_distribution_lambda(function(x)
    tfd_normal(loc = x, scale = 1))

negloglik <- function(y, model) - (model %>% tfd_log_prob(y))
model %>% compile(optimizer = optimizer_adam(lr = 0.0001), loss = negloglik)

### Train neural network

history <- model %>% fit(
  x_train, y_train,
  batch_size = 32,
  epochs = 100,
  verbose = 2,
  validation_data = list(x_test, y_test),
  callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 100, verbose = 1))
)

yhats <- purrr::map(1:100, function(x) model(tf$constant(x_test)))

means <-
  purrr::map(yhats, purrr::compose(as.matrix, tfd_mean)) %>% abind::abind()

lines <- data.frame(cbind(x_test, means)) %>%
  gather(key = run, value = value,-X1)

mean <- apply(means, 1, mean)

ggplot(data.frame(x = x_test, y = y_test, mean = as.numeric(mean)), aes(x, y)) +
  geom_point() +
  geom_line(aes(x = x_test, y = mean), color = "violet", size = 1.5) +
  geom_line(
    data = lines,
    aes(x = X1, y = value, color = run),
    alpha = 0.3,
    size = 0.5
  ) +
  theme(legend.position = "none")
