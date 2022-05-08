# Base image
FROM rocker/ml:4.2.0-cuda11.1

ARG DEBIAN_FRONTEND=noninteractive

RUN sed -i 's@archive.ubuntu.com@jp.archive.ubuntu.com@g' /etc/apt/sources.list

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get clean all && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y  \
     libmagick++-dev \
     tk

RUN echo "options(repos='https://cran.ism.ac.jp/')\ninstall.packages(c('tidymodels'))" > tmp.R | Rscript tmp.R
RUN echo "options(repos='https://cran.ism.ac.jp/')\ninstall.packages(c('tidytext'))" > tmp.R | Rscript tmp.R
RUN echo "options(repos='https://cran.ism.ac.jp/')\ninstall.packages(c('finetune'))" > tmp.R | Rscript tmp.R

RUN echo "options(repos='https://cran.ism.ac.jp/')\ninstall.packages(c('GGally'))" > tmp.R | Rscript tmp.R
RUN echo "options(repos='https://cran.ism.ac.jp/')\ninstall.packages(c('doParallel'))" > tmp.R | Rscript tmp.R
RUN echo "options(repos='https://cran.ism.ac.jp/')\ninstall.packages(c('skimr'))" > tmp.R | Rscript tmp.R
RUN echo "options(repos='https://cran.ism.ac.jp/')\ninstall.packages(c('ggrepel'))" > tmp.R | Rscript tmp.R
RUN echo "options(repos='https://cran.ism.ac.jp/')\ninstall.packages(c('stopwords'))" > tmp.R | Rscript tmp.R

# RUN echo "options(repos='https://cran.ism.ac.jp/')\ninstall.packages(c('xgboost'))" > tmp.R | Rscript tmp.R
RUN wget https://github.com/dmlc/xgboost/releases/download/v1.6.0/xgboost_r_gpu_linux_1.6.0.tar.gz
RUN R CMD INSTALL ./xgboost_r_gpu_linux_1.6.0.tar.gz

RUN echo "options(repos='https://cran.ism.ac.jp/')\ninstall.packages(c('glmnet'))" > tmp.R | Rscript tmp.R
RUN echo "options(repos='https://cran.ism.ac.jp/')\ninstall.packages(c('lme4'))" > tmp.R | Rscript tmp.R
RUN echo "options(repos='https://cran.ism.ac.jp/')\ninstall.packages(c('vip'))" > tmp.R | Rscript tmp.R

RUN 
RUN echo /opt/venv/reticulate/bin/pip install --upgrade tensorflow-gpu==2.8.*