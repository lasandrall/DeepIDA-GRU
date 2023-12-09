#' Joint Principal Trend Analysis (JPTA) of two high-dimensional longitudinal datasets.
#' @param x A N * P * T longitudinal data tensor. N is the number of subjects, P is the number of features, T is the number of time points.
#' @param y A N * Q * T longitudinal data tensor. Q is the number of features. 
#' @param niter Number of iterations. 
#' @param lambda Tuning parameter for smoothness of principal trends.
#' @param sumabs A measure of sparsity for u and v vectors, between 0 and 1. It is used when feature.flag is FALSE and if sumabsu or sumabsv is not specified. 
#' In this case, sumabsu will be set as sumabs*sqrt(P) and sumabsv will be set as sumabs*sqrt(Q).
#' @param sumabsu Tuning parameter for feature selection in x when feature.flag is FALSE. It must be between 1 and the square root of P.
#' @param sumabsv Tuning parameter for feature selection in y when feature.flag is FALSE. It must be between 1 and the square root of Q.
#' @param topfeau The number of nonzero features in u for feature selection in x when feature.flag is TRUE.
#' @param topfeav The number of nonzero features in v for feature selection in y when feature.flag is TRUE.
#' @param feature.flag To indicate the way of feature selection.
#' @param timevec A vector for time points.
#' @return returns a list with following objects.
#' \item{u}{Loadings for features in x.}
#' \item{v}{Loadings for features in y.}
#' \item{theta}{Weights for basis functions.}
#' \item{B}{Basis function matrix}
#' \item{err}{Reconstruction error.}
#' \item{xprd}{JPTA reconstruction for x.}
#' \item{yprd}{JPTA reconstruction for y.}
#' @references Joint Principal Trend Analysis for Longitudinal High-dimensional Data
#'             by Yuping Zhang and Zhengqing Ouyang
#' @keywords JPTA 
#' @export
#' @importFrom fda create.bspline.basis getbasismatrix getbasispenalty 
#' @examples
#' N = 10
#' P = 50
#' Q = 40
#' T = 10
#' x = array(NA, dim = c(N,P,T))
#' y = array(NA, dim = c(N,Q,T))
#' timevec = seq(from=0, to=2, length.out=T)
#' e = 0.1
#' p1 = 40
#' q1 = 30
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
#' out = JPTA(x, y, niter=5, lambda=0.1, sumabs=0.8, feature.flag=FALSE, timevec=timevec)
#'

JPTA = function(x, y, niter=6, lambda=1, sumabs=0.7, sumabsu=NULL, sumabsv=NULL, topfeau=10, topfeav=10,feature.flag=TRUE,timevec=c(1:dim(x)[3])){
    ## time is ordered increasingly
    N = dim(x)[1]
    P = dim(x)[2]
    Q = dim(y)[2]
    T = dim(x)[3]
    if(N!=dim(y)[1]){cat("sample size does not match")}
    if(T!=dim(y)[3]){cat("time points do not match")}
    M = T+2
 
    xoo = x
    xoo[is.na(x)] = mean(x[!is.na(x)])
    xm = matrix(nrow=P, ncol=T)
	for(t in 1:T){
		if(N==1){xm[,t] = xoo[1,,t]}else{
            xm[, t] = apply(xoo[, ,t], 2, mean, na.rm= TRUE)
		}
    }
    if(P>1){
        u = matrix(svd(matrix(xm, nrow=P))$u[,1], ncol=1)
    }else{
        u = matrix(1, nrow=P, ncol=1)
    }
    x = xoo
    rm(xoo)

    yoo = y
    yoo[is.na(y)] = mean(y[!is.na(y)])
    ym = matrix(nrow=Q, ncol=T)
	for(t in 1:T){
		if(N==1){ym[,t] = yoo[1,,t]}else{
            ym[, t] = apply(yoo[, ,t], 2, mean, na.rm= TRUE)
		}
    }
    if(Q>1){
        v = matrix(svd(matrix(ym, nrow=Q))$u[,1], ncol=1)
    }else{
        v = matrix(1, nrow=Q, ncol=1)    
    }
    y = yoo
    rm(yoo)    

    
    BS = create.bspline.basis(timevec, norder=4)
	B = getbasismatrix(timevec, BS)
	om = getbasispenalty(BS)     
    

    theta = matrix(1, nrow=M, ncol=1)
    iter=1;
    while(iter < (niter +1)){
		cat(iter, fill=F)
        theta_new = trend.coef.func(u, v,  B, x, y, lambda, om, N)
        theta <- theta_new;
        u_new = fea.coef.func(theta, x, B)
        v_new = fea.coef.func(theta, y, B)
        if(feature.flag){
            if(topfeau>=P){        
                u_new = u_new/l2n(u_new)
            }else{
                lamu = sort(abs(u_new[,1]), decreasing=TRUE)[topfeau+1]
                u_new[,1] = matrix(soft(u_new[,1], lamu)/l2n(soft(u_new[,1],lamu)),ncol=1)
            }        
            if(topfeav>=Q){
                v_new = v_new/l2n(v_new)
            }else{
                lamv = sort(abs(v_new[,1]), decreasing=TRUE)[topfeav+1]
                v_new[,1] = matrix(soft(v_new[,1], lamv)/l2n(soft(v_new[,1],lamv)),ncol=1)
            }
        }else{
            if(is.null(sumabsu) || is.null(sumabsv)){
                sumabsu <- sqrt(P)*sumabs
                sumabsv <- sqrt(Q)*sumabs
            }
            lamu <- BinarySearch(u_new,sumabsu)
            u_new[,1] = matrix(soft(u_new[,1], lamu)/l2n(soft(u_new[,1],lamu)),ncol=1)
            lamv = BinarySearch(v_new,sumabsv)
            v_new[,1] = matrix(soft(v_new[,1], lamv)/l2n(soft(v_new[,1],lamv)),ncol=1)
        }
        u <- u_new;
        v <- v_new;
        iter = iter +1;
    }       
    err=0
    for(n in 1:N){
        err = err + sum((x[n,,] -  u %*% t(theta)%*% t(B))^2) 
        err = err + sum((y[n,,] -  v %*% t(theta)%*% t(B))^2)  
    }
    err = err/N
    out = list(u = u, v = v, theta=theta, B=B, err=err,  xprd= u %*% t(theta)%*% t(B), yprd = v %*% t(theta)%*% t(B))
    return(out) 
}
