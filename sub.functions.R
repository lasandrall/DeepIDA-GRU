#' Internal function
#' @keywords internal
soft <- function(vec,lam){
	return(sign(vec)*pmax(0,abs(vec)-lam))
}

#' Internal function
#' @keywords internal
l2n <- function(vec){
  return(sqrt(sum(vec^2)))
}

#' Internal function
#' @keywords internal
fea.coef.func = function(theta, x, B){
  u = (apply(x, c(2,3), sum))%*%B%*%theta
  return(u)
}

#' Internal function
#' @keywords internal
trend.coef.func = function(u, v,  B, x, y, lambda, om, N){
  theta = solve(kronecker(N*t(B)%*%B, t(u)%*%u) + kronecker(N*t(B)%*%B, t(v)%*%v) + lambda*om)%*%(t(B)%*%(t(apply(x, c(2,3), sum))%*%u +t(apply(y, c(2,3), sum))%*%v ))    
  return(theta)
}

#' Internal function
#' @keywords internal
BinarySearch <- function(argu,sumabs){
  if(l2n(argu)==0 || sum(abs(argu/l2n(argu)))<=sumabs) return(0)
  lam1 <- 0
  lam2 <- max(abs(argu))-1e-5
  iter <- 1
  while(iter < 150){
    su <- soft(argu,(lam1+lam2)/2)
    if(sum(abs(su/l2n(su)))<sumabs){
      lam2 <- (lam1+lam2)/2
    } else {
      lam1 <- (lam1+lam2)/2
    }
    if((lam2-lam1)<1e-6) return((lam1+lam2)/2)
    iter <- iter+1
  }
  warning("Didn't quite converge")
  return((lam1+lam2)/2)
}