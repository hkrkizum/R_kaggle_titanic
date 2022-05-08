# 0. Set env -------------------------------------------------------------------
set.seed(54147)

library(tidymodels)
library(tidyverse)
library(data.table)
library(patchwork)
library(skimr)
library(GGally)
library(doParallel)
library(vip)
library(ggsignif)
library(gghalves)

out_dir <- "Result/" 
dir.create(out_dir, showWarnings = F, recursive = T)

options(tidymodels.dark = TRUE)

# save.image(str_c(out_dir, "workspase.RData"))
# load(str_c(out_dir, "workspase.RData"))

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

# df_Ticket %>% 
#   view(.)

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

data_train_1$Pclass %>% table()
data_train_1$SibSp %>% table()
data_train_1$Parch %>% table()
data_train_1$Embarked %>% table()

## make recipe -----------------------------------------------------------------
recipe_Age_pred <- 
  recipes::recipe(Age ~ ., data = data_train_1) %>% 
  recipes::update_role(PassengerId, new_role = "uid") %>% 
  
  recipes::step_select(all_outcomes(), all_numeric_predictors(), Sex, Embarked) %>% 
  
  recipes::step_impute_median(Fare) %>% 
  recipes::step_impute_mode(Embarked) %>% 
  
  recipes::step_mutate(Pclass = forcats::as_factor(Pclass)) %>% 
  recipes::step_mutate(Pclass = forcats::fct_expand(Pclass, c("1", "2", "3"))) %>% 
  
  # recipes::step_mutate(SibSp = forcats::as_factor(SibSp)) %>% 
  # recipes::step_mutate(SibSp = forcats::fct_expand(SibSp, c("1", "2", "3"))) %>% 
  # 
  # recipes::step_mutate(Pclass = forcats::as_factor(Pclass)) %>% 
  # recipes::step_mutate(Pclass = forcats::fct_expand(Pclass, c("1", "2", "3"))) %>% 
  
  # recipes::step_mutate_at(any_of(!!c("Pclass", "SibSp", "Parch")),
  #                         fn = function(x){as.character(x)}) %>% 
  # recipes::step_novel(Pclass, SibSp, Parch) %>% 
  
  # Embarked: impute by mode

  recipes::step_mutate(Embarked = dplyr::if_else(Embarked == "",
                                                 NA_character_,
                                                 as.character(Embarked))) %>% 
  recipes::step_impute_mode(Embarked) %>% 
  recipes::step_string2factor(Embarked) %>% 
  
  recipes::step_dummy(all_nominal_predictors(), one_hot = TRUE)  %>% 
  
  recipes::step_normalize(Fare, SibSp, Parch) 
  
recipe_Age_pred  

recipe_Age_pred %>% prep() %>% bake(NULL) %>% skim()

## make model ------------------------------------------------------------------
model_age_pred <- 
  boost_tree(
    trees = tune(),
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
  finalize(recipes::prep(recipe_Age_pred) %>% 
             recipes::bake(new_data = NULL) %>% 
             dplyr::select(-all_outcomes()))
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
# all_cores <- 12*2
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

wf_age_pred_1_res <-
  wf_age_pred_1 %>% 
  tune_grid(
    data_train_1_vFc,
    grid = hyper_grid,
    metrics = metric_set(rmse, rsq, mae),
    control = control_grid(verbose = TRUE,
                           parallel_over = "everything")
  )

registerDoSEQ()
stopCluster(cl)

autoplot(wf_age_pred_1_res)
show_best(wf_age_pred_1_res, metric = "rmse")
show_best(wf_age_pred_1_res, n = 5)[2,]

## Add param -------------------------------------------------------------------
wf_age_pred_1 <-
  wf_age_pred_1 %>%
  finalize_workflow(show_best(wf_age_pred_1_res, n = 5, metric = "rmse")[1,]) 

wf_age_pred_1

## train and validation --------------------------------------------------------
wf_age_pred_fit <- 
  wf_age_pred_1 %>% 
  fit(data_train_1)

wf_age_pred_fit

collect_metrics(wf_age_pred_fit %>% fit_resamples(data_train_1_vFc))

## finalize model --------------------------------------------------------------
wf_age_pred_last <-
  wf_age_pred_fit %>% 
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
data_model_mod <- fread(str_c(out_dir, "data_model_mod.csv"))

skim(data_model_mod)

data_split_sv <-
  data_model_mod %>%
  dplyr::arrange(Survived) %>%
  rsample::initial_time_split(.,
                              prop = dim(data_raw_train)[1]/dim(data_model_mod)[1])

data_split_sv
data_train_sv <- rsample::training(data_split_sv)
data_test_sv <- rsample::testing(data_split_sv)

skim(data_train_sv)
skim(data_test_sv)

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

# data_test_sv %>% 
#   dplyr::mutate(ticket_mod = case_when(Ticket %in% param_t_mid  ~ 1,
#                                        Ticket %in% param_t_high ~ 2,
#                                        TRUE ~ 0)) %>% 
#   pull(ticket_mod) %>% 
#   table()
# 
# data_train_sv %>% 
#   dplyr::mutate(ticket_mod = case_when(Ticket %in% param_t_mid  ~ 1,
#                                        Ticket %in% param_t_high ~ 2,
#                                        TRUE ~ 0)) %>% 
#   pull(ticket_mod) %>% 
#   table()

### prep cabin factor ----------------------------------------------------------
param_cabin <- data_model_mod %>% 
  dplyr::mutate(Cabin = stringr::str_extract(Cabin, "^.")) %>% 
  dplyr::arrange(Cabin) %>%
  dplyr::filter(!is.na(Cabin)) %>% 
  dplyr::pull(Cabin) %>% 
  unique(.)

## recipes --------------------------------------------------------------------
recipe_sv <- 
  recipes::recipe(Survived ~ ., data = data_train_sv) %>% 
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
  recipes::step_mutate(Cabin = forcats::fct_expand(Cabin, all_of(!!param_cabin))) %>% 
  recipes::step_unknown(Cabin) %>% 
  
  # Embarked: impute by mode
  recipes::step_mutate(Embarked = if_else(Embarked == "", NA_character_, as.character(Embarked))) %>% 
  recipes::step_impute_mode(Embarked) %>% 
  recipes::step_string2factor(Embarked) %>% 
  
  # Ticket group
  recipes::step_mutate(Ticket = dplyr::case_when(Ticket %in% all_of(!!param_t_mid)  ~ 1,
                                                 Ticket %in% all_of(!!param_t_high) ~ 2,
                                                 TRUE ~ 0)) %>%
  recipes::step_mutate(Ticket = forcats::as_factor(Ticket)) %>%
  recipes::step_mutate(Ticket = forcats::fct_expand(Ticket, c("0", "1", "2"))) %>%
  
  # Family: SibSp + Parch + 1 
  recipes::step_mutate(Family = SibSp + Parch + 1) %>% 
  recipes::step_mutate(Family = dplyr::case_when(Family == 1 ~ 1,
                                                 Family >= 2 & Family <= 4 ~ 2,
                                                 Family >= 5 & Family <= 7 ~ 1,
                                                 Family >= 8 ~ 0)) %>% 
  recipes::step_mutate(Family = forcats::as_factor(Family)) %>%
  recipes::step_mutate(Family = forcats::fct_expand(Family, c("0", "1", "2"))) %>% 
  recipes::step_mutate(Family = forcats::fct_relevel(Family, "0", "1")) %>%
  
  # Pclass
  recipes::step_mutate(Pclass = forcats::as_factor(Pclass)) %>%
  recipes::step_mutate(Pclass = forcats::fct_expand(Pclass, c("1", "2", "3"))) %>% 
  recipes::step_mutate(Pclass = forcats::fct_relevel(Pclass, "1", "2")) %>%
  
  # drop name
  recipes::step_select(-Name) %>% 
  recipes::step_select(-Parch, -SibSp) %>% 
  
  # factor to dummy
  recipes::step_dummy(all_nominal_predictors(), one_hot = TRUE) %>% 
  
  # normalize numeric
  recipes::step_normalize(Fare, Age) %>% 
  
  # Outcome to factor
  recipes::step_mutate(Survived = forcats::as_factor(Survived)) %>% 
  recipes::step_mutate(Survived = forcats::fct_expand(Survived, c("0", "1"))) 
  
  # filter zero variance
  # recipes::step_zv(all_predictors())
  # recipes::step_nzv(all_predictors())

recipe_sv
recipe_sv %>% prep() %>% bake(new_data = NULL)  %>% skim()
recipe_sv %>% prep() %>% bake(new_data = data_test_sv) %>% skim()

## make Model ------------------------------------------------------------------
model_sv_pred <- 
  boost_tree(
    trees = tune(),
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
  dials::grid_latin_hypercube(size = 50)
plot(hyper_grid)

## Grid search -----------------------------------------------------------------
data_train_sv_vFc <-
  vfold_cv(data_train_sv,
           v = 10, 
           repeats = 3,
           # repeats = 1,
           strata = Survived)
data_train_sv_vFc

all_cores <- parallel::detectCores(all.tests = TRUE, logical = FALSE) - 2
# all_cores <- 14*2
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

wf_sv_pred_1_res <-
  wf_sv_pred_1 %>% 
  tune_grid(
    data_train_sv_vFc,
    grid = hyper_grid,
    metrics = metric_set(accuracy, roc_auc, precision, recall),
    control = control_grid(verbose = TRUE,
                           parallel_over = "everything")
  )

registerDoSEQ()
stopCluster(cl)

autoplot(wf_sv_pred_1_res)
save(wf_sv_pred_1_res, file = str_c(out_dir, "wf_sv_pred_1_res_5.Rdata"))

show_best(wf_sv_pred_1_res, metric = "roc_auc")
show_best(wf_sv_pred_1_res, metric = "accuracy")

## fit result ------------------------------------------------------------------
wf_sv_pred_1_fit <-
  wf_sv_pred_1 %>%
  finalize_workflow(show_best(wf_sv_pred_1_res, "accuracy")[1, ]) %>%
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
collect_metrics(res)

## Get prediction result -------------------------------------------------------
wf_sv_pred_1_last <-
  wf_sv_pred_1 %>%
  finalize_workflow(select_best(wf_sv_pred_1_res, "accuracy")) %>%
  last_fit(data_split_sv)

res <- data_test_sv %>% 
  dplyr::mutate(Survived = collect_predictions(wf_sv_pred_1_last) %>% pull(.pred_class)) %>% 
  dplyr::arrange(PassengerId) %>% 
  dplyr::select(PassengerId, Survived)

out_n <- lubridate::now(tzone = "Asia/Tokyo") %>%
  as.character() %>% 
  str_replace(., "\\s", "_") %>% 
  str_remove_all(., "-|:|\\s") %>% 
  str_c(out_dir, "MySubmission_all_feature_", ., ".csv")
out_n
fwrite(res, out_n, row.names = F)

# update recipes ---------------------------------------------------------------
## visualize vip ---------------------------------------------------------------
library(vip)
extract_workflow(wf_sv_pred_1_last) %>%
  extract_fit_parsnip() %>%
  vip(geom = "point", num_features = 30)

tmp <- 
  extract_workflow(wf_sv_pred_1_last) %>%
  extract_fit_parsnip() %>% 
  .$fit %>% 
  xgboost::xgb.importance(model = ., )

tmp

tmp$fit

data_train_sv
data_train_sv$Fare
data_train_sv$Pclass
data_train_sv$SibSp %>% hist()
data_train_sv$Parch %>% hist()


## plot feature effect ---------------------------------------------------------
procecced_data <- recipe_sv %>% 
  prep() %>% bake(NULL)
procecced_data %>% skim()

tmp <- procecced_data %>% 
  # dplyr::select(-Age) %>% 
  tidyr::pivot_longer(cols = c(-PassengerId, -Survived),
                      names_to = "feature",
                      values_to = "val") %>% 
  dplyr::group_by(feature) %>% 
  tidyr::nest() %>% 
  dplyr::ungroup() %>% 
  dplyr::mutate(type = case_when(feature %in% c("Fare", "Age") ~ "logistic",
                                 TRUE ~ "Category")) %>% 
  dplyr::mutate(stat = pmap(.l = list(feature, type, data),
                            .f = function(feature, type, data){
                              if(type == "logistic"){
                                t.test(data = data, val ~ Survived) %>% tidy()
                              } else{
                                tmp <- data %>% 
                                  dplyr::group_by(Survived, val) %>% 
                                  dplyr::summarise(n = n()) %>% 
                                  dplyr::ungroup() %>% 
                                  tidyr::pivot_wider(names_from = val,
                                                     names_prefix = "feature_",
                                                     values_from = n, 
                                                     values_fill = 0) 
                                if(dim(tmp)[2] == 3 & dim(tmp)[1] == 2){
                                  res <- tmp %>%
                                    tibble::column_to_rownames("Survived") %>%
                                    as.matrix() %>%
                                    fisher.test(.) %>% tidy()
                                }else{
                                  res <- NA
                                }
                                return(res)
                              }
                            })) %>% 
  dplyr::mutate(pval = map(.x = stat, 
                           .f = function(x){
                             if(!is.na(x[[1]])){
                              res <- x$p.value[1]
                             } else{
                               res <- NA
                             }
                             return(res)}) %>% unlist) %>% 
  dplyr::mutate(plot = pmap(.l = list(feature, type, data, pval),
                            .f = function(feature, type, data, pval){
                              p_annotate <- case_when(pval < 0.01 ~ "p < 0.01",
                                                      pval < 0.05 ~ "p < 0.05",
                                                      TRUE ~ "n.s")
                              
                              if(type == "logistic"){
                                g <- ggplot(data, 
                                            aes(x = Survived,
                                                y = val,
                                                color = Survived,
                                                fill = Survived)) +
                                  theme_bw(base_family = "Arial", base_size = 10) +
                                  geom_half_violin(nudge = 0.07) +
                                  geom_boxplot(outlier.shape = NA, 
                                               fill = "white",
                                               width = 0.1) +
                                  geom_half_point(width = 0.5,
                                                  transformation = 
                                                    position_jitter(height = 0,
                                                                    width = 0.05)) +
                                  ggsignif::geom_signif(
                                    textsize = 3,
                                    y_position = max(data$val)*1.05,
                                    xmin = 1, xmax = 2, 
                                    annotation = p_annotate,
                                    tip_length = 0,
                                    color = "black"
                                  ) +
                                  ggtitle(feature)
                              } else{
                                g <- data %>% 
                                  dplyr::group_by(Survived, val) %>% 
                                  dplyr::summarise(n = n()) %>% 
                                  dplyr::ungroup() %>% 
                                  ggplot(., 
                                         aes(x = as.factor(val),
                                             y = n,
                                             fill = Survived)) +
                                  theme_bw(base_family = "Arial", base_size = 10)+
                                  geom_bar(position="fill", stat="identity") +
                                  xlab("feature") +
                                  ylab("Popuration (%)") +
                                  ggtitle(feature) +
                                  ggsignif::geom_signif(textsize = 3,
                                                        y_position = 1.01,
                                                        xmin = 1, xmax = 2, 
                                                        annotation = p_annotate,
                                                        tip_length = 0,
                                                        color = "black")
                                }
                            })) %>% 
  dplyr::mutate(plot_group = case_when(feature %in% c("Age", "Fare") ~ "Numeric",
                                       feature %in% c("Officer", "Royalty",
                                                      "Mrs", "Miss", "Master") ~ "Name",
                                       feature  %>% str_detect(., "Pclass_") ~ "Pclass",
                                       feature  %>% str_detect(., "Sex_") ~ "Sex",
                                       feature  %>% str_detect(., "Ticket_") ~ "Ticket",
                                       feature  %>% str_detect(., "Cabin_") ~ "Cabin",
                                       feature  %>% str_detect(., "Embarked_") ~ "Embarked",
                                       feature  %>% str_detect(., "Family_") ~ "Family",
  )) %>% 
  dplyr::group_by(plot_group) %>% 
  tidyr::nest()

for (i in 1:dim(tmp)[1]) {
  print(i)
  g <- wrap_plots(tmp$data[[i]]$plot)
  plot(g)
  ggsave(plot = g,
         filename = str_c(out_dir, "Plot_Suvived_by_feature_", tmp$plot_group[[i]], ".png"),
         dpi = 300, units = "cm", width = 25, height = 17)
}

## make modified recipes -------------------------------------------------------
recipe_sv_v2 <- 
  recipes::recipe(Survived ~ ., data = data_train_sv) %>% 
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
  recipes::step_mutate(Cabin = dplyr::case_when(Cabin %in% all_of(!!c("A", "G"))  ~ 0,
                                                Cabin %in% all_of(!!c("B", "C", "D", "E", "F"))  ~ 1,
                                                Cabin %in% all_of(!!c("T"))  ~ 2,  
                                                is.na(Cabin) ~ 2,
                                                TRUE ~ 3)) %>% 
  recipes::step_mutate(Cabin = forcats::as_factor(Cabin)) %>% 
  recipes::step_mutate(Cabin = forcats::fct_expand(Cabin, all_of(!!as.character(0:2)))) %>% 

  # Embarked: impute by mode
  recipes::step_mutate(Embarked = if_else(Embarked == "", NA_character_, as.character(Embarked))) %>% 
  recipes::step_impute_mode(Embarked) %>% 
  recipes::step_string2factor(Embarked) %>% 
  
  # Ticket group
  recipes::step_mutate(Ticket = dplyr::case_when(Ticket %in% all_of(!!param_t_mid)  ~ 1,
                                                 Ticket %in% all_of(!!param_t_high) ~ 2,
                                                 TRUE ~ 0)) %>%
  recipes::step_mutate(Ticket = forcats::as_factor(Ticket)) %>%
  recipes::step_mutate(Ticket = forcats::fct_expand(Ticket, c("0", "1", "2"))) %>%
  
  # Family: SibSp + Parch + 1 
  recipes::step_mutate(Family = SibSp + Parch + 1) %>% 
  recipes::step_mutate(Family = dplyr::case_when(Family == 1 ~ 1,
                                                 Family >= 2 & Family <= 4 ~ 2,
                                                 Family >= 5 & Family <= 7 ~ 1,
                                                 Family >= 8 ~ 0)) %>% 
  recipes::step_mutate(Family = forcats::as_factor(Family)) %>%
  recipes::step_mutate(Family = forcats::fct_expand(Family, c("0", "1", "2"))) %>% 
  recipes::step_mutate(Family = forcats::fct_relevel(Family, "0", "1")) %>%
  
  # Pclass
  recipes::step_mutate(Pclass = forcats::as_factor(Pclass)) %>%
  recipes::step_mutate(Pclass = forcats::fct_expand(Pclass, c("1", "2", "3"))) %>% 
  recipes::step_mutate(Pclass = forcats::fct_relevel(Pclass, "1", "2")) %>%
  
  # drop name
  recipes::step_select(-Name) %>% 
  recipes::step_select(-Parch, -SibSp) %>% 
  
  # factor to dummy
  recipes::step_dummy(all_nominal_predictors(), one_hot = TRUE) %>% 
  
  # normalize numeric
  recipes::step_normalize(Fare, Age)
  
recipe_sv_v2

recipe_sv_v2 %>% prep() %>% bake(NULL) %>% skim()
recipe_sv_v2 %>% prep() %>% bake(data_test_sv) %>% skim()

tmp_cor <- recipe_sv_v2 %>% prep() %>% bake(NULL) %>% 
  dplyr::select(-PassengerId) %>% 
  dplyr::mutate_all(as.numeric) %>% 
  as.matrix(.) %>% 
  cor(., method = "spearman") %>% 
  round(., digits = 3)

g <- 
  recipe_sv_v2 %>% prep() %>% bake(NULL) %>% 
  dplyr::select(-PassengerId) %>% 
  dplyr::mutate_all(as.numeric) %>% 
  as.matrix(.) %>% 
  cor(., method = "spearman") %>% 
  round(., digits = 3) %>% 
  as.data.table(keep.rownames = T) %>% 
  dplyr::rename(Col_1 = 1) %>% 
  tidyr::pivot_longer(cols = -Col_1, 
                      names_to = "Col_2", 
                      values_to = "Corr") %>% 
  dplyr::mutate(Col_1 = fct_inorder(Col_1)) %>% 
  dplyr::mutate(Col_2 = fct_inorder(Col_2) %>% fct_rev) %>% 
  ggplot(.,
         aes(x = Col_1,
             y = Col_2,
             fill = Corr)) +
  geom_tile()+
  scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                       midpoint = 0) +
  scale_x_discrete(expand = c(0, 0),position = 'top') +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 0))
g
ggsave(plot = g, filename = str_c(out_dir, "Cor_feature.png"), dpi = 300)


## make Model ------------------------------------------------------------------
model_sv_pred <-
  boost_tree(
    trees = tune(),
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
wf_sv_pred_2 <- workflow() %>% 
  add_recipe(recipe_sv_v2) %>% 
  add_model(model_sv_pred)

wf_sv_pred_2

## Make Grid -------------------------------------------------------------------
param <-
  wf_sv_pred_2 %>% 
  hardhat::extract_parameter_set_dials() %>% 
  finalize(recipes::prep(recipe_sv_v2) %>%
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

all_cores <- parallel::detectCores(all.tests = TRUE, logical = FALSE) - 2
# all_cores <- 14*2
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

wf_sv_pred_2_res <-
  wf_sv_pred_2 %>% 
  tune_grid(
    data_train_sv_vFc,
    grid = hyper_grid,
    metrics = metric_set(accuracy, roc_auc, precision, recall),
    control = control_grid(verbose = TRUE,
                           parallel_over = "everything")
  )

registerDoSEQ()
stopCluster(cl)

collect_notes(wf_sv_pred_2_res)
autoplot(wf_sv_pred_2_res)

save(wf_sv_pred_2_res, file = str_c(out_dir, "wf_sv_pred_2_res_1.Rdata"))

autoplot(wf_sv_pred_2_res)
show_best(wf_sv_pred_2_res, metric = "roc_auc")
show_best(wf_sv_pred_2_res, metric = "accuracy")


## fit result ------------------------------------------------------------------
wf_sv_pred_2_fit <-
  wf_sv_pred_2 %>%
  finalize_workflow(show_best(wf_sv_pred_2_res, "accuracy")[1, ]) %>%
  fit(data_train_sv)
wf_sv_pred_2_fit

res <- wf_sv_pred_2_fit %>% 
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
collect_metrics(res)

## Get prediction result -------------------------------------------------------
wf_sv_pred_2_last <-
  wf_sv_pred_2_fit %>%
  last_fit(data_split_sv)

res <- data_test_sv %>% 
  dplyr::mutate(Survived = collect_predictions(wf_sv_pred_2_last) %>% pull(.pred_class)) %>% 
  dplyr::arrange(PassengerId) %>% 
  dplyr::select(PassengerId, Survived)

out_n <- lubridate::now(tzone = "Asia/Tokyo") %>%
  as.character() %>% 
  str_replace(., "\\s", "_") %>% 
  str_remove_all(., "-|:|\\s") %>% 
  str_c(out_dir, "MySubmission_Select_feature_", ., ".csv")

out_n
fwrite(res, out_n, row.names = F)

# param_submit <- str_c("kaggle competitions submit -c titanic -f ", out_n)
# param_submit
# system(param_submit)


# Keras ------------------------------------------------------------------------
## recipe ----------------------------------------------------------------------
recipe_sv

recipe_sv %>% prep() %>% bake(NULL) %>% skim()
## model -----------------------------------------------------------------------
model_sv_keras <- 
  mlp(
    epochs = 200, 
    hidden_units = tune(), 
    dropout = tune(),
    # penalty = tune(),
    activation = "relu",
    ) %>% # param to be tuned
  set_mode("classification") %>% # binary response var
  set_engine("keras", 
             verbose = 0)

model_sv_keras

## workflow --------------------------------------------------------------------
wf_sv_keras <- 
  workflow() %>% 
  add_recipe(recipe_sv) %>% 
  add_model(model_sv_keras)

## make grid -------------------------------------------------------------------
param <-
  wf_sv_keras %>% 
  hardhat::extract_parameter_set_dials() %>% 
  # update(activation = activation(c("softmax", "relu", "tanh"))) %>%
  # update(activation = activation(c("relu"))) %>%
  update(dropout = dropout(c(0, 0.999))) %>%
  finalize(dropout(),activation(),
           recipes::prep(recipe_sv) %>%
             recipes::bake(new_data = NULL) %>%
             dplyr::select(-all_outcomes()))
param
param$object

hyper_grid <-
  param %>% 
  # dials::grid_regular(levels = 3)
  # dials::grid_latin_hypercube(size = 50)
  dials::grid_max_entropy(size = 50)
plot(hyper_grid)

## Grid search -----------------------------------------------------------------
data_train_sv_vFc <-
  vfold_cv(data_train_sv,
           v = 5, 
           # repeats = 1,
           repeats = 1,
           strata = Survived)

# RUN, only single
wf_sv_keras_res <-
  wf_sv_keras %>% 
  tune_grid(
    data_train_sv_vFc,
    grid = hyper_grid,
    # iter = 10,
    # param_info = param,
    metrics = metric_set(accuracy, roc_auc, precision, recall),
    control = control_grid(verbose = TRUE,
                           parallel_over = "everything")
    # control = control_bayes(verbose = TRUE,
    #                         no_improve = 10L,
    #                         parallel_over = "everything")
  )
wf_sv_keras_res

save(wf_sv_keras_res, file = str_c(out_dir, "wf_sv_keras_res.Rdata"))
collect_notes(wf_sv_keras_res)$note[[1]]
collect_metrics(wf_sv_keras_res)
show_best(wf_sv_keras_res, n = 20, metric = "accuracy")
