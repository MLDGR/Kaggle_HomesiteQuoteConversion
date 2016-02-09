library(tsne)
library(Rtsne)
library(ggplot2)
trainSample<-train[sample(1:dim(train)[1],1000),]
features<-trainSample[,-c(1,2,3)]
for(i in 1:dim(features)[2]){
  if(class(features[,i])=="factor"){
    features[,i]<-as.integer(features[,i])
  }
}
features[is.na(features)]   <- 0
tsne <- Rtsne(as.matrix(features), check_duplicates = FALSE, pca = TRUE,perplexity=30, theta=0.5, dims=2)
tsne <- Rtsne(as.matrix(features))#, check_duplicates = FALSE, pca = TRUE,perplexity=30, theta=0.5, dims=2)


embedding <- as.data.frame(tsne$Y)
embedding$Class <- as.factor(sub("Class_", "", trainSample$QuoteConversion_Flag))

p <- ggplot(embedding, aes(x=V1, y=V2, color=Class)) +
  geom_point(size=1.25) +
  guides(colour = guide_legend(override.aes = list(size=6))) +
  xlab("") + ylab("") +
  ggtitle("t-SNE 2D Embedding of Products Data") +
  theme_light(base_size=20) +
  theme(strip.background = element_blank(),
        strip.text.x     = element_blank(),
        axis.text.x      = element_blank(),
        axis.text.y      = element_blank(),
        axis.ticks       = element_blank(),
        axis.line        = element_blank(),
        panel.border     = element_blank())

ggsave("tsne.png", p, width=8, height=6, units="in")
p
