# 0. Set env -------------------------------------------------------------------
set.seed(54147)

library(tidymodels)
library(tidyverse)
library(data.table)
library(patchwork)
library(skimr)
library(GGally)
library(doParallel)

out_dir <- "Result/" 
dir.create(out_dir, showWarnings = F, recursive = T)

options(tidymodels.dark = TRUE)

# 1. load data --------------------------------------------------------------------
data_raw_train <- fread("Rawdata/train.csv")
data_raw_test  <- fread("Rawdata/test.csv") %>% 
  dplyr::mutate(Survived = NA)

data_raw_train
data_raw_test

skim(data_raw_train)


# 2. EDA -----------------------------------------------------------------------
hist(data_raw_train$Fare)

## pair plot -------------------------------------------------------------------
g_ggpairs <-
  data_raw_train %>% 
  dplyr::select(-PassengerId, -Name, -Ticket, -Cabin) %>% 
  dplyr::mutate_at(.vars = vars(Survived, Pclass, SibSp, Parch), 
                   .funs = as.factor) %>% 
  GGally::ggpairs(., 
                  mapping = aes(color = Survived))


skim(data_raw_train)
skim(data_raw_test)

## Ticket group ---------------------------------------------------------------
Ticket_mod <- data_raw_train %>% 
  dplyr::bind_rows(data_raw_test) %>% 
  dplyr::mutate(Ticket_mod = str_remove_all(Ticket, "^[a-zA-Z].*\\s")) %>% 
  # dplyr::mutate(Ticket_mod = str_remove_all(Ticket_mod, "^[a-zA-Z]*")) %>% 
  dplyr::select(Ticket, Ticket_mod)

Ticket_group <- Ticket_mod %>% 
  dplyr::group_by(Ticket_mod) %>% 
  summarise(n = n()) %>% 
  ungroup()

df_Ticket <- Ticket_mod %>% 
  dplyr::full_join(Ticket_group)

df_Ticket %>% 
  view(.)

data_raw_train %>% 
  dplyr::bind_rows(data_raw_test) %>% 
  dplyr::full_join(df_Ticket) %>% 
  dplyr::select(Ticket, Ticket_mod, n) %>% 
  dplyr::filter(!complete.cases(.))

summary_ratio <- 
  data_raw_train %>% 
  dplyr::bind_rows(data_raw_test) %>% 
  dplyr::full_join(df_Ticket) %>% 
  dplyr::filter(!is.na(Survived)) %>% 
  # dplyr::mutate(Survived = as.numeric(Survived)) %>% 
  dplyr::group_by(Ticket_mod) %>% 
  dplyr::summarise(rate = sum(Survived)/n()) %>% 
  dplyr::ungroup() 

summary_ratio %>% 
  dplyr::left_join(df_Ticket) %>% 
  dplyr::arrange(n) %>% 
  dplyr::mutate(n = fct_inorder(as.character(n))) %>% 
  ggplot(., aes(x = n, y = rate)) +
  geom_violin()+
  geom_point(position = position_jitter(width = 0.2))

summary_ratio %>% 
  dplyr::left_join(df_Ticket) %>% 
  dplyr::group_by(n) %>% 
  dplyr::summarise(mean = mean(rate), sd = sd(rate)) %>% 
  dplyr::ungroup() %>% 
  dplyr::mutate(n = fct_inorder(as.character(n))) %>% 
  ggplot(., aes(x = n, y = mean)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = mean - sd,
                    ymax = mean + sd))

## Name ------------------------------------------------------------------------ 


# 3. age prediction ------------------------------------------------------------
## make data set ---------------------------------------------------------------
data_model <- data_raw_train %>% 
  dplyr::bind_rows(data_raw_test) %>% 
  dplyr::mutate(Survived = as.factor(Survived))

skim(data_model)
  
data_split_1 <- 
  data_model %>% 
  dplyr::arrange(Age) %>% 
  rsample::initial_time_split(., prop = (1309 - 263)/1309)

data_train_1 <- training(data_split_1)
data_test_1 <- testing(data_split_1)

skim(data_train_1)
str(data_train_1)

skim(data_test_1)

data_train_1 %>% 
  dplyr::filter(Embarked == "")


## make recipe -----------------------------------------------------------------
recipe_Age_pred <- 
  recipes::recipe(Age ~ ., data = data_train_1) %>% 
  recipes::update_role(PassengerId, new_role = "uid") %>% 
  
  recipes::step_select(all_outcomes(), all_numeric_predictors(), Sex, Embarked) %>% 
  
  recipes::step_filter(!is.na(Fare)) %>% 
  recipes::step_filter(Embarked != "") %>% 
  
  recipes::step_mutate_at(any_of(!!c("Pclass", "SibSp", "Parch")),
                          fn = function(x){as.character(x)}) %>% 
  recipes::step_novel(Pclass, SibSp, Parch) %>% 
  
  recipes::step_dummy(all_predictors(), -Fare, one_hot = TRUE)  %>% 
  recipes::step_nzv(all_predictors())
  
recipe_Age_pred  

## make model ------------------------------------------------------------------
model_age_pred <- 
  boost_tree(
    trees = 1000,
    tree_depth = tune(),
    min_n = tune(),
    mtry = tune(),
    sample_size = tune(),
    learn_rate = 0.1
  ) %>%
  set_engine("xgboost", 
             lambda = tune(),
             alpha  = tune(),
             # params=list(tree_method = 'gpu_hist')
  ) %>%
  set_mode("regression")

model_age_pred

## make workflow ---------------------------------------------------------------
wf_age_pred_1 <- workflow() %>% 
  add_recipe(recipe_Age_pred) %>% 
  add_model(model_age_pred)

wf_age_pred_1

## Make Grid -------------------------------------------------------------------
param <-
  wf_age_pred_1 %>% 
  hardhat::extract_parameter_set_dials() %>% 
  finalize(recipes::prep(recipe_Age_pred) %>% recipes::bake(new_data = NULL) %>% dplyr::select(-all_outcomes()))
param
param$object

hyper_grid <-
  param %>% 
  dials::grid_latin_hypercube(size = 50)
plot(hyper_grid)

## Grid search -----------------------------------------------------------------
data_train_1_vFc <-
  vfold_cv(data_train_1,
           v = 5, 
           repeats = 2,
           # repeats = 1,
           strata = Age)
data_train_1_vFc

all_cores <- parallel::detectCores(all.tests = TRUE, logical = FALSE) - 2
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

wf_age_pred_1_res <-
  wf_age_pred_1 %>% 
  tune_grid(
    data_train_1_vFc,
    grid = hyper_grid,
    metrics = metric_set(rmse),
    control = control_grid(verbose = TRUE,
                           parallel_over = "everything")
  )

registerDoSEQ()
stopCluster(cl)

autoplot(wf_age_pred_1_res)
show_best(wf_age_pred_1_res, n = 5)[2,]

## Add param -------------------------------------------------------------------
wf_age_pred_1 <-
  wf_age_pred_1 %>%
  finalize_workflow(show_best(wf_age_pred_1_res, n = 5)[2,]) 

wf_age_pred_1

## train and validation --------------------------------------------------------
wf_age_pred_fit <- 
  wf_age_pred_1 %>% 
  fit(data_train_1)

wf_age_pred_fit

collect_metrics(wf_age_pred_fit %>% fit_resamples(data_train_1_vFc))

## finalize model --------------------------------------------------------------
wf_age_pred_last <-
  wf_age_red_1 %>% 
  finalize_workflow(show_best(wf_age_pred_1_res, n = 5)[2,]) %>% 
  last_fit(data_split_1)

collect_predictions(wf_age_pred_last, summarize = TRUE) %>% dim()

## update dataset --------------------------------------------------------------
data_pred_age <- collect_predictions(wf_age_pred_last, summarize = TRUE) %>% 
  pull(.pred)

data_model_mod <- 
  data_test_1 %>% 
  dplyr::mutate(Age = data_pred_age) %>% 
  bind_rows(data_train_1)

fwrite(data_model_mod, str_c(out_dir, "data_model_mod.csv"))

# 4. Predict survive -----------------------------------------------------------
## 1. split data ---------------------------------------------------------------
skim(data_model_mod)

data_split_sv <-
  data_model_mod %>% 
  dplyr::arrange(Survived) %>%
  rsample::initial_time_split(., 
                              prop = dim(data_raw_train)[1]/dim(data_model)[1])

data_train_sv <- rsample::training(data_split_sv)
data_test_sv <- rsample::testing(data_split_sv)

skim(data_train_sv)
skim(data_test_sv)


data_train_sv$Cabin %>% str_extract(., "^.")


## Make recipe -----------------------------------------------------------------
### prep regex -----------------------------------------------------------------
params_Officer <- c('Capt', 'Col', 'Major', 'Dr', 'Rev') %>% glue::glue_collapse(., sep = "|")
params_Royalty <- c('Don', 'Sir',  'the Countess', 'Lady', 'Dona') %>% glue::glue_collapse(., sep = "|")
params_Mrs     <- c('Mme', 'Ms') %>% glue::glue_collapse(., sep = "|")
params_Miss    <- c('Mlle') %>% glue::glue_collapse(., sep = "|")
params_Master  <- c('Jonkheer') %>% glue::glue_collapse(., sep = "|")

### prep ticket group ----------------------------------------------------------
param_t_mid <- df_Ticket %>% 
  dplyr::filter(n %in% c(1, 5:8)) %>% 
  dplyr::pull(Ticket)

param_t_high <- df_Ticket %>% 
  dplyr::filter(n %in% c(2, 3, 4)) %>% 
  dplyr::pull(Ticket)

data_test_sv %>% 
  dplyr::mutate(ticket_mod = case_when(Ticket %in% param_t_mid  ~ 1,
                                       Ticket %in% param_t_high ~ 2,
                                       TRUE ~ 0)) %>% 
  pull(ticket_mod) %>% 
  table()

data_train_sv %>% 
  dplyr::mutate(ticket_mod = case_when(Ticket %in% param_t_mid  ~ 1,
                                       Ticket %in% param_t_high ~ 2,
                                       TRUE ~ 0)) %>% 
  pull(ticket_mod) %>% 
  table()

### prep cabin factor ----------------------------------------------------------
param_cabin <- data_model_mod %>% 
  dplyr::mutate(Cabin = stringr::str_extract(Cabin, "^.")) %>% 
  dplyr::arrange(Cabin) %>%
  dplyr::filter(!is.na(Cabin)) %>% 
  dplyr::pull(Cabin) %>% 
  unique(.)

## recipes --------------------------------------------------------------------
recipe_sv <- recipes::recipe(Survived ~ ., data = data_train_sv) %>% 
  recipes::update_role(PassengerId, new_role = "uid") %>% 
  
  # create feature from name
  recipes::step_regex(Name, pattern = params_Officer, result = "Officer") %>% 
  recipes::step_regex(Name, pattern = params_Royalty, result = "Royalty") %>% 
  recipes::step_regex(Name, pattern = params_Mrs, result = "Mrs") %>% 
  recipes::step_regex(Name, pattern = params_Miss, result = "Miss") %>% 
  recipes::step_regex(Name, pattern = params_Master, result = "Master") %>% 
  
  # Fare impute NA by median
  recipes::step_impute_median(Fare) %>% 
  
  # Cabin: extract head and fill NA as unknown
  recipes::step_mutate(Cabin = stringr::str_extract(Cabin, "^.")) %>% 
  recipes::step_mutate(Cabin = forcats::as_factor(Cabin)) %>% 
  recipes::step_mutate(Cabin = forcats::fct_expand(Cabin, !!param_cabin)) %>% 
  recipes::step_unknown(Cabin) %>% 
  
  # Embarked: impute by mode
  recipes::step_mutate(Embarked = if_else(Embarked == "", NA_character_, as.character(Embarked))) %>% 
  recipes::step_impute_mode(Embarked) %>% 
  recipes::step_string2factor(Embarked) %>% 
  
  # Ticket group
  recipes::step_mutate(Ticket = dplyr::case_when(Ticket %in% !!param_t_mid  ~ 1,
                                                 Ticket %in% !!param_t_high ~ 2,
                                                 TRUE ~ 0)) %>% 
  recipes::step_mutate(Ticket = as.character(Ticket)) %>%
  recipes::step_novel(Ticket) %>%
  
  # drop name
  recipes::step_select(-Name) %>% 
  
  # factor to dummy
  recipes::step_dummy(all_nominal_predictors(), one_hot = TRUE) %>% 
  
  # filter zero variance
  recipes::step_zv(all_predictors())

recipe_sv %>% prep() %>% bake(new_data = NULL)
recipe_sv %>% prep() %>% bake(new_data = data_test_sv)

## make Model ------------------------------------------------------------------
model_sv_pred <- 
  boost_tree(
    trees = 1000,
    tree_depth = tune(),
    min_n = tune(),
    mtry = tune(),
    sample_size = tune(),
    learn_rate = 0.1
  ) %>%
  set_engine("xgboost", 
             lambda = tune(),
             alpha  = tune(),
             # params=list(tree_method = 'gpu_hist')
  ) %>%
  set_mode("classification")

model_sv_pred

## make workflow ---------------------------------------------------------------
wf_sv_pred_1 <- workflow() %>% 
  add_recipe(recipe_sv) %>% 
  add_model(model_sv_pred)

wf_sv_pred_1

## Make Grid -------------------------------------------------------------------
param <-
  wf_sv_pred_1 %>% 
  hardhat::extract_parameter_set_dials() %>% 
  finalize(recipes::prep(recipe_sv) %>%
             recipes::bake(new_data = NULL) %>%
             dplyr::select(-all_outcomes()))
param
param$object

hyper_grid <-
  param %>% 
  dials::grid_latin_hypercube(size = 100)
plot(hyper_grid)

## Grid search -----------------------------------------------------------------
data_train_sv_vFc <-
  vfold_cv(data_train_sv,
           v = 10, 
           repeats = 5,
           # repeats = 1,
           strata = Survived)
data_train_sv_vFc

all_cores <- parallel::detectCores(all.tests = TRUE, logical = FALSE) - 4
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

wf_sv_pred_1_res <-
  wf_sv_pred_1 %>% 
  tune_grid(
    data_train_sv_vFc,
    grid = hyper_grid,
    metrics = metric_set(mn_log_loss),
    control = control_grid(verbose = TRUE,
                           parallel_over = "everything")
  )

registerDoSEQ()
stopCluster(cl)

autoplot(wf_sv_pred_1_res)
save(wf_sv_pred_1_res, file = str_c(out_dir, "wf_sv_pred_1_res_2.Rdata"))

show_best(wf_sv_pred_1_res)

## fit result ------------------------------------------------------------------
wf_sv_pred_1_fit <-
  wf_sv_pred_1 %>%
  finalize_workflow(select_best(wf_sv_pred_1_res, "mn_log_loss")) %>%
  fit(data_train_sv)

res <- wf_sv_pred_1_fit %>% 
  fit_resamples(data_train_sv_vFc, 
                control = control_resamples(
                  verbose = TRUE,
                  allow_par = TRUE,
                  # extract = NULL,
                  # save_pred = FALSE,
                  # pkgs = NULL,
                  # save_workflow = FALSE,
                  # event_level = "first",
                  parallel_over = "everything"
                  )
                )

collect_notes(res) %>% view(.)

tmp <- data_train_sv_vFc %>% 
  dplyr::filter(id == "Repeat1") %>% 
  dplyr::filter(id2 == "Fold02") %>% 
  dplyr::mutate(train = map(splits, ~rsample::training(.))) %>% 
  dplyr::mutate(test  = map(splits, ~rsample::testing(.))) 

recipe_sv %>% prep() %>% bake(tmp$train[[1]])
recipe_sv %>% prep() %>% bake(tmp$test[[1]])

tmp$train[[1]] %>% 
  dplyr::mutate(Cabin = stringr::str_extract(Cabin, "^.")) %>% 
  pull(Cabin) %>% table(.)

tmp$test[[1]] %>% 
  dplyr::mutate(Cabin = stringr::str_extract(Cabin, "^.")) %>% 
  pull(Cabin) %>% table(.)


## Get prediction result -------------------------------------------------------
wf_sv_pred_1_last <-
  wf_sv_pred_1 %>%
  finalize_workflow(select_best(wf_sv_pred_1_res, "mn_log_loss")) %>%
  last_fit(data_split_sv)

res <- data_test_sv %>% 
  dplyr::mutate(Survived = collect_predictions(wf_sv_pred_1_last) %>% pull(.pred_class)) %>% 
  dplyr::arrange(PassengerId) %>% 
  dplyr::select(PassengerId, Survived)

fwrite(res, str_c(out_dir, "res.csv"), row.names = F)
data_raw_test


recipe_sv %>% prep() %>% bake(data_test_sv)
