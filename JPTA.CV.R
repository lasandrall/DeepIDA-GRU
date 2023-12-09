#' Cross-valiation for Joint Principal Trend Analysis. 
#' @param x A N * P * T Longitudinal data tensor. N is the number of subjects, P is the number of features, T is the number of time points.
#' @param y A N * Q * T Longitudinal data tensor. Q is the number of features. 
#' @param timevec A vector for time points.
#' @param niter Number of iterations. 
#' @param lambdas Tuning parameter vector for smoothness of principal trends.
#' @param sumabss Tuning parameter vector for sparsity of features. This vector is used when feature.flag is FALSE.
#' @param topfeaus Tuning parameter vector for the number of nonzero features in u. This vector is used when feature.flag is TRUE.
#' @param topfeavs Tuning parameter vector for the number of nonzero features in v. This vector is used when feature.flag is TRUE.
#' @param feature.flag To indicate the way of feature selection.
#' @param nfolds The number of folds for cross-validation.
#' @param seed The seed argument in set.seed used in the cross-validation function.
#' @param trace Print out progress as iterations are performed. Default is TRUE.
#' @return returns a list with following objects.
#' \item{errmeans}{Means of cross-validation errors.}
#' \item{errses}{Standard errors of cross-validation errors.}
#' @references Joint Principal Trend Analysis for Longitudinal High-dimensional Data
#'             by Yuping Zhang and Zhengqing Ouyang
#' @keywords JPTA.CV 
#' @export
#' @importFrom fda create.bspline.basis getbasismatrix getbasispenalty 
#' @importFrom stats runif var
#' @examples
#' N = 10
#' P = 50
#' Q = 40
#' T = 10
#' x = array(NA, dim = c(N,P,T))
#' y = array(NA, dim = c(N,Q,T))
#' timevec = seq(from=0, to=2, length.out=T)
#' p1 = 40
#' q1 = 30
#' e=0.1
#' for(j in 1:N){
#'  for(i in 1:P){
#'     x[j, i, ] = ((i>0) & (i<=p1))*sin(pi*timevec) + rnorm(T, 0, e)                       
#'  }
#'  for(i in 1:Q){
#'    y[j, i, ] = ((i>0) & (i<=q1))*sin(pi*timevec)  + rnorm(T, 0, e)                           
#'  }
#' }
#' for(i in 1:P){
#'   x[,i,] = (x[,i,] - mean(x[,i,], na.rm= TRUE))
#' }
#' for(i in 1:Q){
#'   y[,i,] = (y[,i,] - mean(y[,i,], na.rm= TRUE))
#' }
#' lambdas = c(0.01, 0.1, 1)
#' sumabss=seq(from=0.5,to=1, by=0.01)
#' nfolds = 2
#' cv.obj = JPTA.CV(x, y, timevec=timevec, niter=5, lambdas=lambdas, sumabs=sumabss, feature.flag=FALSE, nfolds=nfolds)
#'

JPTA.CV = function(x, y, timevec=c(1:dim(x)[3]), niter=6, lambdas=0.5, sumabss=seq(0.5, 1, by = 0.1),topfeaus = c(5:20),topfeavs=c(5:20), feature.flag=TRUE, nfolds=10, seed=NULL, trace=TRUE){
    N = dim(x)[1]
    P = dim(x)[2]
    Q = dim(y)[2]
    T = dim(x)[3]
    if(N!=dim(y)[1]){cat("sample size does not match")}
    if(T!=dim(y)[3]){cat("time points do not match")}
    M = T+2
    nfolds = max(nfolds, 2);
    percentRemove <- min(.25, 1/nfolds)
    missing.x <- is.na(x)
    missing.y <- is.na(y)
    rands.x <- array(runif(dim(x)[1]*dim(x)[2]*dim(x)[3]), dim(x))
    rands.y <- array(runif(dim(y)[1]*dim(x)[2]*dim(x)[3]), dim(y))

    if(feature.flag){
        topfeaus = topfeaus[topfeaus>=5];
        topfeavs = topfeavs[topfeavs>=5];
        errlist = array(NA, c(length(lambdas), length(topfeaus),length(topfeavs), nfolds));     
        for( n in 1:nfolds){
            if(is.null(seed)){
                set.seed(n);
            }else{
                set.seed(seed);
            }
            if(trace) cat(" Fold ", n, " out of ", nfolds, "\n")
            rm.x <- ((n-1)*percentRemove < rands.x)&(rands.x < n*percentRemove)
            xrm <- x
            xrm[rm.x] <- NA 
            rm.y <- ((n-1)*percentRemove < rands.y)&(rands.y < n*percentRemove)
            yrm <- y
            yrm[rm.y] <- NA 
            err = array(dim=c(length(lambdas), length(topfeaus), length(topfeavs)))
            for(i in 1:length(lambdas)){
                for(j in 1:length(topfeaus)){
                    for(m in 1:length(topfeavs)){ 
                        prd.x = array(NA, c( N, P, T))
                        prd.y = array(NA, c( N, Q, T))
                        lambda = lambdas[i];
                        topfeau = topfeaus[j];
                        topfeav = topfeavs[m];
                        out = JPTA(xrm, yrm, niter=niter,lambda=lambda, topfeau=topfeau, topfeav=topfeav,feature.flag=feature.flag, timevec=timevec)
                        for(jj in 1:N){
                            prd.x[jj,,] = out$u %*% t(out$theta) %*% t(out$B);
                            prd.y[jj,,] = out$v %*% t(out$theta) %*% t(out$B);    
                        }
                        err[i,j,m] = sum((x-prd.x)[rm.x & !missing.x]^2)/sum(rm.x & !missing.x)+ sum((y-prd.y)[rm.y & !missing.y]^2)/sum(rm.y & !missing.y);
                        rm(prd.x);
                        rm(prd.y);
                        rm(out);    
                    }   
                }
            }
            errlist[,,,n] = err;
        }
        errmeans = apply(errlist, c(1,2,3), mean);
        dimnames(errmeans) = list(paste("lambda=",lambdas), paste("topfeau=", topfeaus), paste("topfeav=", topfeavs))
        errses = sqrt(apply(errlist, c(1,2,3), var)/nfolds);
        dimnames(errses) =  list(paste("lambda=",lambdas), paste("topfeau=", topfeaus), paste("topfeav=", topfeavs))
        object <- list(errmeans=errmeans, errses = errses)
    }else{
        errlist = array(NA, c(length(lambdas), length(sumabss), nfolds));
        for( n in 1:nfolds){
            if(is.null(seed)){
                set.seed(n);
            }else{
                set.seed(seed);
            }
            if(trace) cat(" Fold ", n, " out of ", nfolds, "\n")
            rm.x <- ((n-1)*percentRemove < rands.x)&(rands.x < n*percentRemove)
            xrm <- x
            xrm[rm.x] <- NA 
            rm.y <- ((n-1)*percentRemove < rands.y)&(rands.y < n*percentRemove)
            yrm <- y
            yrm[rm.y] <- NA 
            err = matrix(NA, nrow=length(lambdas), ncol=length(sumabss))
            for(i in 1:length(lambdas)){
                for(j in 1:length(sumabss)){        
                    prd.x = array(NA, c( N, P, T))
                    prd.y = array(NA, c( N, Q, T))
                    lambda = lambdas[i];
                    out = JPTA(xrm, yrm, niter = niter, lambda=lambda, sumabs=sumabss[j],feature.flag=feature.flag, timevec=timevec)
                    for(jj in 1:N){
                        prd.x[jj,,] = out$u %*% t(out$theta) %*% t(out$B);
                        prd.y[jj,,] = out$v %*% t(out$theta) %*% t(out$B);    
                    }
                    err[i,j] = sum((x-prd.x)[rm.x & !missing.x]^2)/sum(rm.x & !missing.x) + sum((y-prd.y)[rm.y & !missing.y]^2)/sum(rm.y & !missing.y);
                    rm(prd.x);
                    rm(prd.y);
                    rm(out);    
                }  
            }
            errlist[,,n] = err;
        }
        errmeans = apply(errlist, c(1,2), mean);
        dimnames(errmeans) = list(paste("lambda=",lambdas), paste("sumabs=", sumabss ))

        errses = sqrt(apply(errlist, c(1,2), var)/nfolds);
        dimnames(errses) = list(paste("lambda=",lambdas), paste("sumabs=", sumabss ))

        object <- list(errmeans=errmeans, errses = errses)
    }
    return(object)
}
