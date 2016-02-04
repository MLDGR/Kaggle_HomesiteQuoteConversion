

Caracteristicas<-read_csv("ffs.csv")
Caracteristicas<-Caracteristicas[-1,]

name<-Caracteristicas[!Caracteristicas[,3]>=0.89,2]
name<-Caracteristicas[147:dim(Caracteristicas)[1],2]

