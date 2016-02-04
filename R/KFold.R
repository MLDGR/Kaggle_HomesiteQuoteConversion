

h2o.kfold <- function(k,training_frame,X,Y,algo,predict.fun,poll=FALSE) {
  folds <- 1+as.numeric(cut(h2o.runif(training_frame), seq(0,1,1/k), include.lowest=T))
  print(dim(folds))
  
  # launch models
  model.futures <- NULL
  for( i in 1L:k) {
    train <- training_frame[folds!=i,]
    if( is.null(model.futures) ) model.futures <- list(algo(train,X,Y))
    else                         model.futures <- c(model.futures, list(algo(train,X,Y)))
  }
  models <- model.futures
  if( poll ) {
    for( i in 1L:length(models) ) {
      models[[i]] <- h2o.getFutureModel(models[[i]])
    }
  }
  
  # perform predictions on the holdout data
  preds  <- NULL
  for( i in 1L:k) {
    valid <- training_frame[folds==i,]
    p <- predict.fun(models[[i]], valid)
    if( is.null(preds) ) preds <- p
    else                 preds <- h2o.rbind(preds,p)
  }
  
  # return the results
  list(models=models, predictions=preds)
}



# 10-fold Deeplearning:
h2o_dl  <- function(training_frame,X,Y){
  dp<-h2o.deeplearning(x=X,
                   y=Y,
                   training_frame=training_frame,
                   hidden=c(200,200,200),
                   activation="RectifierWithDropout", 
                   input_dropout_ratio=0.3, 
                   hidden_dropout_ratios=c(0.5,0.5,0.5), 
                   l1=1e-4)         # no future since each DL has high Duty Cycle
}


h2o_gbm <- function(training_frame,X,Y) { 
  h2o.gbm(
    x = setdiff(colnames(train), target_var),
    y = target_var, 
    training_frame = train,
    #validation_frame = valid,
    #ntrees = 100,
    ntrees = 1800,
    max_depth = 9,
    nbins = 40,
    distribution = "bernoulli",
    future=T  
  ) # future = TRUE launches model builds in parallel, careful!
}