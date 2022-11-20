## librerías
library(tidyverse)
library(tidymodels)

## adquisición de datos
dt00 <- read_csv("material/bmw.csv")

glimpse(dt00)


## limpieza
dt01 <- dt00 %>% 
  mutate(across(where(is.character), as_factor))


## data budget
bmw_split <- initial_split(dt01, strata = price)

bmw_train <- bmw_split %>% training()

bmw_test <- bmw_split %>% testing()


## recipe
bmw_recipe <- recipe(price ~ .,
                     data = bmw_train) %>% 
  step_other(model, threshold = 0.15) %>% 
  step_nzv(all_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())


prep(bmw_recipe) %>% bake(new_data = NULL) %>% 
  corrr::correlate() %>% 
  corrr::focus(price) %>% 
  ggplot(aes(x = price, 
             y = fct_reorder(term, price),
             fill =  price > 0)) +
  geom_col()


## modeling
# linear regression
# spec
lr_model <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression")

# workflow
lr_wf <- workflow() %>% 
  add_model(lr_model) %>% 
  add_recipe(bmw_recipe)

# fit
lr_fit <- lr_wf %>% 
  fit(bmw_train)

# evaluación
lr_results <- bmw_test %>% 
  select(price) %>% 
  mutate(predict(lr_fit, new_data = bmw_test),
         model = "linear regression")

# calculo de rmse
lr_results %>% 
  mutate(error = (price - .pred) ^ 2) %>% 
  summarize(rmse = sqrt(mean(error)))
  
# metricas
lr_metrics <- metrics(lr_results, 
                      truth = price, 
                      estimate = .pred) %>% 
  mutate(model = "linear regression")

# plot
lr_results %>% 
  ggplot(aes(x = .pred, y = price)) +
  geom_point(alpha = 0.1) +
  geom_abline(color = "darkred")



# random forest
# spec
rf_model <- rand_forest() %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("regression")

# workflow
rf_wf <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(bmw_recipe)

# fit
rf_fit <- rf_wf %>% 
  fit(bmw_train)

# evaluación
rf_results <- bmw_test %>% 
  select(price) %>% 
  mutate(predict(rf_fit, new_data = bmw_test),
         model = "random forest")

# metricas
rf_metrics <- metrics(rf_results, truth = price, estimate = .pred) %>% 
  mutate(model = "random forest")

# plot
rf_results %>% 
  ggplot(aes(x = .pred, y = price)) +
  geom_point(alpha = 0.1) +
  geom_abline(color = "darkred")


# k nearest neighbors
# spec
knn_model <- nearest_neighbor() %>% 
  set_engine("kknn") %>% 
  set_mode("regression")

# workflow
knn_wf <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(bmw_recipe)

# fit
knn_fit <- knn_wf %>% 
  fit(bmw_train)

# evaluación
knn_results <- bmw_test %>% 
  select(price) %>% 
  mutate(predict(knn_fit, new_data = bmw_test),
         model = "nearest neighbors")

# metricas
knn_metrics <- metrics(knn_results, 
                       truth = price, 
                       estimate = .pred) %>% 
  mutate(model = "nearest neighbors")

# plot
knn_results %>% 
  ggplot(aes(x = .pred, y = price)) +
  geom_point(alpha = 0.1) +
  geom_abline(color = "darkred")


# glm
# spec
glm_model <- linear_reg(penalty = 0.01, mixture = 0.5) %>% 
  set_engine("glmnet") %>% 
  set_mode("regression")

# workflow
glm_wf <- workflow() %>% 
  add_model(glm_model) %>% 
  add_recipe(bmw_recipe)

# fit
glm_fit <- glm_wf %>% 
  fit(bmw_train)

# evaluación
glm_results <- bmw_test %>% 
  select(price) %>% 
  mutate(predict(glm_fit, new_data = bmw_test),
         model = "generalized linear")

# metricas
glm_metrics <- metrics(glm_results, 
                       truth = price, 
                       estimate = .pred) %>% 
  mutate(model = "generalized linear")

# plot
glm_results %>% 
  ggplot(aes(x = .pred, y = price)) +
  geom_point(alpha = 0.1) +
  geom_abline(color = "darkred")


# comparison
bind_rows(lr_results, rf_results, knn_results, glm_results) %>% 
  ggplot(aes(x = .pred, y = price, color = model)) +
  geom_point(alpha = 0.1) +
  geom_abline(color = "darkred") +
  facet_wrap(~ model) +
  theme(legend.position = "none") +
  coord_obs_pred()


bind_rows(lr_metrics, rf_metrics, knn_metrics, glm_metrics) %>% 
  filter(.metric == "rmse") %>% 
  arrange(.estimate)


## predicciones
datos_prediccion <- tibble(
  model = "X5",
  year =  2019,
  transmission =  "Manual",
  mileage =  45531,
  fuelType =  "Petrol",
  tax =  165,
  mpg =  35.7,
  engineSize = 3,
)

# regresion lineal
lr_fit %>% 
  predict(new_data = datos_prediccion)

# random forest
rf_fit %>% 
  predict(new_data = datos_prediccion)

# knn
knn_fit %>% 
  predict(new_data = datos_prediccion)


# feature importance
rf_fit %>% 
  extract_fit_parsnip() %>% 
  vip::vip(aesthetics =list(alpha = 0.75,
                            fill = "darkgreen"))

