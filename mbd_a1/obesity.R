pacman::p_load(tidyverse, tidymodels, ranger, plotly, patchwork)

od <- read_csv("./data/ObesityDataSet.csv")
od

od <- od %>% 
  mutate(NObeyesdad = ifelse(grepl("Obesity", od$NObeyesdad), 1, 0)) %>% 
  mutate(NObeyesdad = factor(NObeyesdad)) %>% 
  select(-Weight, -Height)
od

set.seed(1942340)
split_od <- initial_split(od, prop = 0.7)
od_train <- training(split_od)
od_test <- testing(split_od)

od_recipe <- recipe(NObeyesdad ~ .,
                    data = od) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) 

od_M1 <- logistic_reg() %>% 
  set_engine('glm') %>% 
  set_mode('classification')
  
od_WF1 <- workflow() %>% 
  add_recipe(od_recipe) %>% 
  add_model(od_M1)

od_fit1 <- od_WF1 %>% fit(od_train)
od_fit1 %>% tidy()



od_M2 <- rand_forest() %>% 
  set_engine('ranger', importance = 'impurity') %>% 
  set_mode('classification')

od_WF2 <- workflow() %>% 
  add_recipe(od_recipe) %>% 
  add_model(od_M2)

od_fit2 <- od_WF2 %>% fit(od_train)
od_fit2 

od_pred1 <- od_test %>% 
  bind_cols(
    predict(od_fit1, od_test),
    predict(od_fit1, od_test, type = "prob"),
  ) %>% 
  select(NObeyesdad, starts_with(".pred"))
od_pred1


od_pred2 <- od_test %>% 
  bind_cols(
    predict(od_fit2, od_test),
    predict(od_fit2, od_test, type = "prob"),
  ) %>% 
  select(NObeyesdad, starts_with(".pred")) 
od_pred2

roc_data1 <- od_pred1 %>%
  roc_curve(truth = NObeyesdad, .pred_0) %>% 
  mutate(model = "Logistic Regression") 

roc_data2 <- od_pred2 %>%
  roc_curve(truth = NObeyesdad, .pred_0)  %>% 
  mutate(model = "Random Forest") 
roc_data <- bind_rows(
  roc_data1,
  roc_data2
) 

ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
  scale_color_manual(values = c("Logistic Regression" = "#F66359", "Random Forest" = "#1AC5CA")) +
  labs(color = "model") +
  coord_equal() +
  theme_bw() +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    legend.position = "right"
  )



od_fit1 %>% extract_fit_parsnip() %>% vip::vip() | od_fit2 %>% extract_fit_parsnip() %>% vip::vip()

