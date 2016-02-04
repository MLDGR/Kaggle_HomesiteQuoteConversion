library(sfsmisc)
library(parallel)


train <- read.csv("train.csv")
test <- read.csv("test.csv")
train=train[,-c(128,133,138,181,191,192,199,207,213,217,283,291,293,295)]
lista<-c(128,133,138,181,191,192,199,207,213,217,283,291,293,295)
test=test[,-(lista-1)]

columnas=c(126,127,141,142,147,148,151,152,154,155,168,169,170,171,179:205,207:281)
lista=list(c(126,127),c(141,142),c(147,148),c(151,152),c(154,155),c(168,169),c(170,171),c(179:205),c(207:216),c(217:218),c(219:220),c(221:222),
           c(223:224),c(225:226),c(227:228),c(229:230),c(231:232),c(233:234),c(235:236),c(237:238),c(239:240),c(241:242),c(243:244),c(245:250),
           c(251:252),c(253:254),c(255:256),c(257:268),c(269:270),c(271:273),c(274:277),c(278:279),c(280:281))

temp=NULL
for(j in 1:length(lista)){
  cat(length(lista)-j,"\n")
  temp=cbind(temp,matrix(unlist(mclapply(1:nrow(train),function(i){
    as.intBase(as.matrix(as.integer(train[i,lista[[j]]])),base=25)
  },mc.preschedule = TRUE, mc.set.seed = TRUE,
  mc.silent = FALSE, mc.cores = getOption("mc.cores", 6L),
  mc.cleanup = TRUE, mc.allow.recursive = TRUE))))
}


for(i in 1:ncol(temp)){
  minimo=abs(min(temp[,i]))
  temp[,i]=t