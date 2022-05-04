# Set env ----------------------------------------------------------------------
library(tidymodels)
library(tidyverse)
library(data.table)
library(patchwork)
library(skimr)
library(GGally)
library(doParallel)

out_dir <- "Result/" 
dir.create(out_dir, showWarnings = F, recursive = T)

# load data --------------------------------------------------------------------
data_raw_train <- fread("Rawdata/train.csv")
data_raw_test  <- fread("Rawdata/test.csv") %>% 
  dplyr::mutate(Survived = NA)

data_raw_train
data_raw_test

skim(data_raw_train)


# EDA --------------------------------------------------------------------------
hist(data_raw_train$Fare)

g_ggpairs <-
  data_raw_train %>% 
  dplyr::select(-PassengerId, -Name, -Ticket, -Cabin) %>% 
  dplyr::mutate_at(.vars = vars(Survived, Pclass, SibSp, Parch), 
                   .funs = as.factor) %>% 
  GGally::ggpairs(., 
                  mapping = aes(color = Survived))

# Make data set ----------------------------------------------------------------
data_model <- data_raw_train %>% 
  dplyr::bind_rows(data_raw_test) %>% 
  dplyr::mutate(Survived = as.factor(Survived))

data_split <- rsample::initial_time_split(data = data_model,
                                          strata = Survived,
                                          prop = dim(data_raw_train)[1]/dim(data_model)[1])

data_train <- rsample::training(data_split)
data_test <- rsample::testing(data_split)

skim(data_train)
# Make recipe ------------------------------------------------------------------
recipe_1 <- recipes::recipe(Survived ~ ., data = data_train) %>% 
  recipes::update_role(PassengerId, new_role = "uid") %>% 
  
  recipes::step_rm(Name, Ticket) %>%
  
  step_mutate(Sex = if_else(Sex == "male", 1, 0)) %>% 
  



recipe_1
