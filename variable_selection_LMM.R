library(reticulate)
library(fda)
library(lme4)
np <- import("numpy")


#topfeat <- 5
#d = 1
#mat1 <- np$load("JPTA_exampleDataset1.npy")    # Data of Shape: (N,P,T)
#y <- sample(0:1, dim(mat1)[1], replace = TRUE)






variables = list()
weeks = list()
class = list()
subject = list()
variable_values = list()
c = 1
for (i in 1:dim(mat1)[2])
{
  for (j in 1:dim(mat1)[3])
  {
    for (k in 1:dim(mat1)[1])
    {
      variables[[c]] <- i-1
      variable_values[[c]] <- mat1[k,i,j]
      weeks[[c]] <- j-1
      subject[[c]] <- k-1
      class[[c]] <- y[k]
      c = c+1
    }
  }
}

df <- data.frame(unlist(variables), unlist(weeks), unlist(class), unlist(subject), unlist(variable_values))
colname = c("variables", "weeks", "class", "subject", "variable_values")
colnames(df) <- colname


pvalues = list()
imp = list()
for (var in 1:dim(mat1)[2])
{
  print(var)
  options(warn = -1)
  model.full = lmer(variable_values ~ class + weeks + (1 + weeks|subject), data = subset(df,variables == var-1), REML = FALSE)
  model.null = lmer(variable_values ~ weeks + (1 + weeks|subject), data = subset(df, variables == var-1), REML = FALSE)
  X = anova(model.full, model.null)
  pvalues = append(pvalues,X$`Pr(>Chisq)`[2])
}

top_indices <- order(unlist(pvalues), decreasing = FALSE)[1:topfeat]-1


write.csv(top_indices, paste("sub0view_",d,".csv", sep = ''), row.names = FALSE, col.names = FALSE)

