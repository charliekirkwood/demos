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
                    
### Define loss function for aleatoric uncertainty

# R6 wrapper class, a subclass of KerasWrapper
ConcreteDropout <- R6::R6Class("ConcreteDropout",
                               
                               inherit = KerasWrapper,
                               
                               public = list(
                                 weight_regularizer = NULL,
                                 dropout_regularizer = NULL,
                                 init_min = NULL,
                                 init_max = NULL,
                                 is_mc_dropout = NULL,
                                 supports_masking = TRUE,
                                 p_logit = NULL,
                                 p = NULL,
                                 
                                 initialize = function(weight_regularizer,
                                                       dropout_regularizer,
                                                       init_min,
                                                       init_max,
                                                       is_mc_dropout) {
                                   self$weight_regularizer <- weight_regularizer
                                   self$dropout_regularizer <- dropout_regularizer
                                   self$is_mc_dropout <- is_mc_dropout
                                   self$init_min <- k_log(init_min) - k_log(1 - init_min)
                                   self$init_max <- k_log(init_max) - k_log(1 - init_max)
                                 },
                                 
                                 build = function(input_shape) {
                                   super$build(input_shape)
                                   
                                   self$p_logit <- super$add_weight(
                                     name = "p_logit",
                                     shape = shape(1),
                                     initializer = initializer_random_uniform(self$init_min, self$init_max),
                                     trainable = TRUE
                                   )
                                   
                                   self$p <- k_sigmoid(self$p_logit)
                                   
                                   input_dim <- input_shape[[2]]
                                   
                                   weight <- private$py_wrapper$layer$kernel
                                   
                                   kernel_regularizer <- self$weight_regularizer * 
                                     k_sum(k_square(weight)) / 
                                     (1 - self$p)
                                   
                                   dropout_regularizer <- self$p * k_log(self$p)
                                   dropout_regularizer <- dropout_regularizer +  
                                     (1 - self$p) * k_log(1 - self$p)
                                   dropout_regularizer <- dropout_regularizer * 
                                     self$dropout_regularizer * 
                                     k_cast(input_dim, k_floatx())
                                   
                                   regularizer <- k_sum(kernel_regularizer + dropout_regularizer)
                                   super$add_loss(regularizer)
                                 },
                                 
                                 concrete_dropout = function(x) {
                                   eps <- k_cast_to_floatx(k_epsilon())
                                   temp <- 0.1
                                   
                                   unif_noise <- k_random_uniform(shape = k_shape(x))
                                   
                                   drop_prob <- k_log(self$p + eps) - 
                                     k_log(1 - self$p + eps) + 
                                     k_log(unif_noise + eps) - 
                                     k_log(1 - unif_noise + eps)
                                   drop_prob <- k_sigmoid(drop_prob / temp)
                                   
                                   random_tensor <- 1 - drop_prob
                                   
                                   retain_prob <- 1 - self$p
                                   x <- x * random_tensor
                                   x <- x / retain_prob
                                   x
                                 },
                                 
                                 call = function(x, mask = NULL, training = NULL) {
                                   if (self$is_mc_dropout) {
                                     super$call(self$concrete_dropout(x))
                                   } else {
                                     k_in_train_phase(
                                       function()
                                         super$call(self$concrete_dropout(x)),
                                       super$call(x),
                                       training = training
                                     )
                                   }
                                 }
                               )
)

# function for instantiating custom wrapper
layer_concrete_dropout <- function(object, 
                                   layer,
                                   weight_regularizer = 1e-6,
                                   dropout_regularizer = 1e-5,
                                   init_min = 0.1,
                                   init_max = 0.1,
                                   is_mc_dropout = TRUE,
                                   name = NULL,
                                   trainable = TRUE) {
  create_wrapper(ConcreteDropout, object, list(
    layer = layer,
    weight_regularizer = weight_regularizer,
    dropout_regularizer = dropout_regularizer,
    init_min = init_min,
    init_max = init_max,
    is_mc_dropout = is_mc_dropout,
    name = name,
    trainable = trainable
  ))
}

### Construct neural network

# sample size (training data)
n_train <- nrow(x_train)
# sample size (validation data)
n_val <- nrow(x_test)
# prior length-scale
l <- 1e-5
# initial value for weight regularizer 
wd <- l^2/n_train
# initial value for dropout regularizer
dd <- 2/n_train

input <- layer_input(shape = 1)
output <- input %>% layer_concrete_dropout(
  layer_dense(units = 64, activation = 'relu', kernel_initializer = 'lecun_normal'),
  weight_regularizer = wd,
  dropout_regularizer = dd
  ) %>% layer_concrete_dropout(
  layer_dense(units = 32, activation = 'relu', kernel_initializer = 'lecun_normal'),
  weight_regularizer = wd,
  dropout_regularizer = dd
  )

mean <- output %>% layer_concrete_dropout(
  layer = layer_dense(units = 1),
  weight_regularizer = wd,
  dropout_regularizer = dd
)

log_var <- output %>% layer_concrete_dropout(
  layer_dense(units = 1),
  weight_regularizer = wd,
  dropout_regularizer = dd
)

output <- layer_concatenate(list(mean, log_var))

model <- keras_model(input, output)

heteroscedastic_loss <- function(y_true, y_pred) {
  mean <- y_pred[, 1:1]
  log_var <- y_pred[, (1 + 1):(1 * 2)]
  precision <- k_exp(-log_var)
  k_sum(precision * (y_true - mean) ^ 2 + log_var, axis = 2)
}

model %>% compile(
  optimizer = optimizer_adam(lr = 0.0001),
  loss = heteroscedastic_loss,
  metrics = c(custom_metric("heteroscedastic_loss", heteroscedastic_loss))
)


history <- model %>% fit(
  x_train, y_train,
  batch_size = 64,
  epochs = 1000,
  verbose = 2,
  validation_split = 0.1,
  callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 1000, verbose = 1),
                   callback_model_checkpoint(paste0(getwd(),"/mcycle_two_layer_concrete_dropout.hdf5"), save_best_only = TRUE))
)

num_MC_samples <- 100

MC_samples <- array(0, dim = c(num_MC_samples, nrow(datanorm), 2 * 1))
for (k in 1:num_MC_samples) {
  MC_samples[k, , ] <- (model %>% predict(datanorm$times))
}

# the means are in the first output column
means <- MC_samples[, , 1:1]  
# average over the MC samples
predictive_mean <- apply(means, 2, mean)

epistemic_uncertainty <- apply(means, 2, var)

logvar <- MC_samples[, , (1 + 1):(1 * 2)]
aleatoric_uncertainty <- exp(colMeans(logvar))

df <- data.frame(
  x = datanorm$times,
  y_pred = predictive_mean,
  e_u_lower = predictive_mean - sqrt(epistemic_uncertainty),
  e_u_upper = predictive_mean + sqrt(epistemic_uncertainty),
  a_u_lower = predictive_mean - sqrt(aleatoric_uncertainty),
  a_u_upper = predictive_mean + sqrt(aleatoric_uncertainty),
  u_overall_lower = predictive_mean - 
    sqrt(epistemic_uncertainty) - 
    sqrt(aleatoric_uncertainty),
  u_overall_upper = predictive_mean + 
    sqrt(epistemic_uncertainty) + 
    sqrt(aleatoric_uncertainty)
)

ggplot(df, aes(x, y_pred)) + 
  geom_point() + 
  geom_ribbon(aes(ymin = e_u_lower, ymax = e_u_upper), alpha = 0.3) +
  geom_point(data = datanorm, aes(x = times, y = accel), col = "red")

### Load best model

#model <- load_model_hdf5(paste0(getwd(),"/mcycle_two_layer_mlp_relu.hdf5"))

### Evaluate performance on held out test data

eval_dat <- data.frame(obs = y_test, preds = predict(model, x_test))
ggplot(eval_dat) + geom_point(aes(x = obs, y = preds)) + geom_abline(intercept = 0, slope = 1) +
  ggtitle(label = paste0("RMSE = ", round(sqrt(mean((eval_dat$obs-eval_dat$preds)^2)), 2), "   R\U00B2 =", round(cor(eval_dat$obs,eval_dat$preds)^2, 2)))

### Predict for full range of input data

predictions <- data.frame(times = seq(min(datanorm$times), max(datanorm$times), length.out =  1000), accel_preds = NA)
predictions$accel_preds <- predict(model, predictions$times)

ggplot(datanorm) + geom_point(aes(x = times, y = accel)) + geom_line(data = predictions, aes(x = times, y = accel_preds))

          