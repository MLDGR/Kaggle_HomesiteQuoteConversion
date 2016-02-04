localH2O = h2o.init(ip = "localhost",nthreads = -1, port = 54321, startH2O = TRUE, 
                    Xmx = '4g')

train1 =as.h2o(localH2O, train)
test1 =as.h2o(localH2O, test)
train1$class<-as.factor(train1$class)
n_repeat=2
n_size=c(1,90)
n_fold=3
n_tree = 20
balance = T

Caracteristicas <- h2o_feaSelect(train1[,-2],as.integer(train1[,2]),n_threads = -1,
                                 n_size=c(n_size[1]:n_size[2]),n_repeat=n_repeat,
                                 n_fold = n_fold,n_tree=n_tree,balance=T)