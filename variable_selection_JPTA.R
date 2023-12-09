library(reticulate)
library(fda)

np <- import("numpy")

# Data from the two views

#mat1 <- np$load("JPTA_exampleDataset1.npy")    # Data of Shape: (N,P,T)
#mat2 <- np$load("JPTA_exampleDataset2.npy")    # Data of shape: (N,P,T)



source("JPTA.R")
source("sub.functions.R")
source("JPTA.CV.R")

#Run JPTA to select the topfeau variables from view 1 and topfeav variables from view 2
out = JPTA(mat1, mat2, niter = 10, lambda = 1, sumabs = 0.7, sumabsu = NULL,
           sumabsv = NULL, topfeau = topfeau, topfeav = topfeav, feature.flag = TRUE,
           timevec = c(1:dim(mat1)[3]))

#out = JPTA(mat1, mat2, niter=10, lambda=0.1, sumabs=0.8, feature.flag=TRUE,
#           timevec=c(1:dim(mat1)[3]))

X = list()
Y = list()
for (i in 1:length(out$u))
{
  if (abs(out$u[i])>0.0000001)
  {
    X = append(X,i-1)
  }
}
for (i in 1:length(out$v))
{
  if (abs(out$v[i])>0.0000001)
  {
    Y = append(Y,i-1)
  }
}
X = array(unlist(X))
Y = array(unlist(Y))

# Save the indices of the variables
write.table(X, file = "sub0view_1.csv", row.names = FALSE, col.names = c('index'))
write.table(Y, file = "sub0view_2.csv", row.names = FALSE, col.names = c('index'))
return(5)
