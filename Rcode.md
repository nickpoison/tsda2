

## R Code Used in the Examples - tsda2 




R code in <a href="https://www.routledge.com/Time-Series-A-Data-Analysis-Approach-Using-R/Shumway-Stoffer/p/book/9781041031642">Time Series: A Data Analysis Approach Using R</a> (Edition 2) &mdash; coming Jan-Feb 2026 <br/><br/>

<img align="left" src="na.jpg" alt="tsda"  height="200"/>  

#### &#x2728;  See the [NEWS](https://github.com/nickpoison/astsa/blob/master/NEWS.md) for further details about the state of the package and the changelog.

#### &#10024; An intro to `astsa` capabilities can be found at  [FUN WITH ASTSA](https://github.com/nickpoison/astsa/blob/master/fun_with_astsa/fun_with_astsa.md)

#### &#10024; Here is [A Road Map](https://nickpoison.github.io/) if you want a broad view of what is available.

<br/><br/>

---
---

>  __Note:__ When you are in a code block below, you can __copy the contents of the block__ by moving your mouse to the upper right corner and clicking on the copy icon ( __&#10697;__ ) &hellip;





-----
-----

### Table of Contents

  * [Chapter 1 - Time Series Elements](#chapter-1)
  * [Chapter 2 - Correlation and Stationary Time Series ](#chapter-2)
  * [Chapter 3 - Time Series Regression and EDA ](#chapter-3)
  * [Chapter 4 - ARMA Models](#chapter-4)
  * [Chapter 5 - ARIMA Models](#chapter-5)
  * [Chapter 6 - Spectral Analysis and Filtering](#chapter-6)
  * [Chapter 7 - Spectral Estimation](#chapter-7)
  * [Chapter 8 - Additional Topics](#chapter-8)
  * [Elsewhere](#elsewhere)

---

## Chapter 1

<br/>
Example 1.1 

```r
library(astsa)   # 'astsa' should be loaded before each session

par(mfrow=2:1)
tsplot(jj, ylab="USD", type="o", col=4, main="Johnson & Johnson QEPS")
tsplot(jj, log='y', ylab="USD", type="o", col=4)
```

<br/>
Example 1.2  

```r
library(xts)  # assumes 'xts' has been installed

djia_return = diff(log(djia$Close))
par(mfrow=2:1)
plot(djia$Close, col=4)
plot(djia_return, col=4)
```
If `xts` isn't available, the following works for a similar plot.
```r
Close = ts(djia[,'Close'])                 # make it a ts object
x = cbind(Close, Return=diff(log(Close)))  # now columns are aligned
tsplot(timex(djia), x, col=4, lwd=2, main='DJIA')
```

Compare diff-log to actual returns:

```r
tsplot(diff(log(gdp)), type="o", col=4, ylab="GDP Growth")   # using diff log
points(diff(gdp)/lag(gdp,-1), pch="+", col=2)                # actual return
```

<br/>
Example 1.3  

```r
tsplot(cbind(gtemp_land, gtemp_ocean), spaghetti=TRUE, lwd=2, pch=20, type="o", col=astsa.col(c(4,2),.5), ylab="Temperature Deviations (\u00B0C)", main="Global Warming", addLegend=TRUE, location="topleft", legend=c("Land Surface", "Sea Surface"))
```

<br/>
Example 1.4  

```r
par(mfrow = 2:1)
tsplot(soi, ylab="", xlab="", main="Southern Oscillation Index", col=4)
text(1969, .91, "COOL", col=5, font=4)
text(1969,-.91, "WARM", col=6, font=4)
tsplot(rec, ylab=NA, main="Recruitment", col=4) 
```

<br/>
Example 1.5  

```r
tsplot(cbind(Hare,Lynx), col=astsa.col(c(2,4),.5), lwd=2, type="o", pch=c(0,2), ylab="Number", spaghetti=TRUE, addLegend=TRUE)
mtext("(\u00D71000)", side=2, adj=1, line=1.5, cex=.8)
```

<br/>
Example 1.6

```r
par(mfrow=c(3,1), cex=.8)
x = ts(fmri1[,4:9], start=0, freq=32)
names = c("Cortex","Thalamus","Cerebellum")
u = ts(rep(c(rep(.6,16), rep(-.6,16)), 4), start=0, freq=32)
for (i in 1:3){
 j = 2*i-1
 tsplot(x[,j:(j+1)], ylab="BOLD", xlab="", main=names[i], col=5:6, ylim=c(-.6,.6), lwd=2, xaxt="n", spaghetti=TRUE)
 axis(seq(0,256,64), side=1, at=0:4)
 lines(u, type="s", col=gray(.3))
}
mtext("seconds", side=1, line=1.75)
```

<br/>
Examples 1.7 - 1.8


> 😖 Don't forget to deal with the curse of `dplyr` if it's loaded. Either detach it `detach(package:dplyr)` or reverse its curse: set `filter = stats::filter` and `lag = stats::lag`.  You can set `dfilter = dplyr::filter` and `dlag = dplyr::lag` and use these versions if you want to have `dplyr` available while analyzing time series. 

```r
par(mfrow=2:1)
w = rnorm(250)                             # 250 N(0,1) variates
v = filter(w, sides=1, filter=rep(1/3,3))  # moving average
tsplot(w, col=4, main="white noise", gg=TRUE)
tsplot(v, ylim=c(-3,3), col=4, main="moving average", gg=TRUE)
```

<br/>
Example 1.9


```r
set.seed(90210)
w = rnorm(250 + 50)   # 50 extra to avoid startup problems
x = filter(w, filter=c(1.5,-.75), method="recursive")[-(1:50)]
tsplot(x, main="autoregression", col=4, gg=TRUE)
```

<br/>
Example 1.10

```r
set.seed(314159265)
w  = rnorm(200)
x  = cumsum(w)    # RW without drift
wd = w +.3
xd = cumsum(wd)   # RW with drift
tsplot(xd, ylim=c(-2,80), main="random walk", ylab=NA, col=4, gg=TRUE)
clip(0, 200, 0, 100)
abline(a=0, b=.3, lty=2, col=4)  # drift
lines(x, col=2)
abline(h=0, col=2, lty=2)  
```


<br/>
Example 1.11

```r
cs = 2*cos(2*pi*(1:500 + 15)/50)  # signal
w  = rnorm(500)                   # noise
par(mfrow=c(3,1), cex=.8)
tsplot(cs, col=4, main=bquote(2*cos(2*pi*(t+15)/50)), ylab=NA, gg=TRUE)
tsplot(cs+w, col=4, main=bquote(2*cos(2*pi*(t+15)/50)+N(0,1)), ylab=NA, gg=TRUE)
tsplot(cs+5*w, col=4, main=bquote(2*cos(2*pi*(t+15)/50)+N(0,5^2)), ylab=NA, gg=TRUE)
```


<br/>
Bad 32 bit RNG

```r
x = c(1)  # the bad seed (they are all bad)
for (n in 2:30){ x[n] = (12*x[n-1] + 4) %% 2^32 }
x
```

[<sub>top</sub>](#table-of-contents)

---



## Chapter 2

<br/>
Example 2.18

```r
ACF = c(0,0,0,1,2,3,2,1,0,0,0)/3
LAG = -5:5
tsplot(LAG, ACF, type="h", col=4, lwd=3, xlab="LAG", gg=TRUE)
abline(h=0, col=8)
points(LAG[-(4:8)], ACF[-(4:8)], pch=20, col=4)
axis(1, at=seq(-5, 5, by=2), col=gray(1))     
```



<br/>
Example 2.27

```r
( r = round( acf1(soi, 6, plot=FALSE), 2) ) # sample acf 
par(mfrow=c(1,2))
tsplot(lag(soi,-1), soi, col=4, type='p', xlab='lag(soi,-1)')
 legend("topleft", legend=r[1], bg="white", adj=.45, cex = 0.8)
tsplot(lag(soi,-6), soi, col=4, type='p', xlab='lag(soi,-6)')
 legend("topleft", legend=r[6], bg="white", adj=.25, cex = 0.8)
```

<br/>
Large Sample ACF Distribution

```R
set.seed(101010)
ACF = replicate(1000, acf1(rgamma(100, shape=4), plot=FALSE)) # H=20 here (by default)
round(c(mean(ACF), sd(ACF)), 3)
QQnorm(ACF)  
```

<br/>
Example 2.29

```r
set.seed(101011)
x  = sample(c(-2,2), 101, replace=TRUE)  # simulated coin tosses
y2 = 5 + filter(x, sides=1, filter=c(1,-.5))[-1]
y1 = y2[1:10]

tsplot(y1, type="s", col=4, yaxt="n", xaxt="n", gg=TRUE)
 axis(1, 1:10); axis(2, seq(2,8,2), las=1); box(col=gray(1))
 points(y1, pch=21, cex=1.1, bg=3)
 round( acf1(y1, 4, plot=FALSE), 2)
 round( acf1(y2, 4, plot=FALSE), 2)
```

<br/>
Example 2.32

```r
x = cos(2*pi*.1*1:100) + rnorm(100)
y = lag(x,-5) + rnorm(100)
ccf2(y, x, lwd=2, col=4, gg=TRUE)
text(11, .65, 'x leads')
text(-9, .65, 'y leads')
```



<br/>Example 2.33

```r
par(mfrow=c(3,1), cex=.8)
acf1(soi, 48, col=4, lwd=2, main="Southern Oscillation Index")
acf1(rec, 48, col=4, lwd=2, main="Recruitment")
ccf2(soi, rec, 48, col=4, lwd=2, main="SOI & Recruitment")
```

<br/>
Example 2.34 

```r
set.seed(90210)
num = 250 
t   = 1:num
X   = .01*t + rnorm(num,0,2)
Y   = .01*t + rnorm(num)
par(mfrow=c(3,1), cex=.8)
tsplot(cbind(X,Y), ylab="data", col=c(4,2), lwd=2, spag=TRUE, gg=TRUE)
ccf2(X, Y, ylim=c(-.3,.3), col=4, lwd=2, gg=TRUE)
ccf2(X, detrend(Y), ylim=c(-.3,.3), col=4, lwd=2, gg=TRUE)
```



[<sub>top</sub>](#table-of-contents)

---



## Chapter 3


<br/>
Example 3.1

```r
par(mfrow=2:1)
trend(salmon, lwd=2, results=TRUE, ci=FALSE)  # graphic and results
trend(chicken, lwd=2, ci=FALSE)               # graphic only
```



<br/>

$R^2$ is NOT a good measure of linear relationship (after eq 3.3)

```r
set.seed(1984)
t = 1:10;  w = rnorm(10)
x = t + w
summary( lm(x~ t) )$r.sq   # cor(x~t)^2 works in this case too
x = t + 3*w
summary( lm(x~ t) )$r.sq   
```



<br/>

Example 3.2

```r
gecon5  = diff(log(econ5));  names = colnames(econ5)
tsplot(cbind(econ5, gecon5), byrow=FALSE, ylab=names, ncol=2, col=2:6, lwd=2)
mtext('Actual',side=3, outer=TRUE, line=-1, adj=.25)
mtext('Growth Rate', side=3, outer=TRUE, line=-1, adj=.8)

ttable( lm(unemp~ time(unemp) + . , data=econ5), vif=TRUE) 

ttable( lm(unemp~ . , data=econ5), vif=TRUE)  # not shown

ttable( lm(unemp~ . , data=gecon5), vif=TRUE)  

gnpp = resid( lm(gnp~ consum + govinv + prinv, data=gecon5) )
ttable(lm(unemp~ gnpp + consum + govinv + prinv, data=gecon5), vif=TRUE)

res = resid( lm(unemp~ gnpp + consum + govinv + prinv, data=gecon5, na.action=NULL) )
dev.new()
par(mfrow=2:1)
tsplot(res)  
acf1(res)
```





<br/>
Example 3.6

```r
##-- Figure 3.3 --##
par(mfrow=c(3,1), cex=.8)
tsplot(cmort, main="Cardiovascular Mortality", col=6, type="o", pch=19, ylab=NA)
tsplot(tempr, main="Temperature", col=4, type="o", pch=19, ylab=NA)
tsplot(part, main="Particulates", col=2, type="o", pch=19, ylab=NA)

##-- Figure 3.4 --##
dev.new()
tsplot(cbind(cmort,tempr,part), col=astsa.col(2:4,.8), spaghetti=TRUE, addLegend=TRUE, legend=c("Mortality", "Temperature", "Pollution"), llwd=2)

##-- Figure 3.5 --##
dev.new()
tspairs(cbind(Mortality=cmort, Temperature=tempr, Particulates=part), hist=FALSE, col.diag=6)

##-- the final model --##
Z = cbind(trnd=time(cmort), tempr, tempr^2, part)
ttable( lm(cmort~ Z, na.action=NULL), vif=TRUE )

summary( aov(cmort~ Z) ) # Table 3.1
```

<br/>
Example 3.7

```r
Z = cbind(trnd=time(cmort), tempr, tempr^2, part)  # Z from previous example
ttable( lm(cmort~ Z + co, data=lap), vif=TRUE )  

cop  = resid(lm(co~ part, data=lap))  # partial out particulates from co
temp = tempr - mean(tempr)            # center temperature
Z    = cbind(trnd=time(cmort), temp, temp^2, part, cop)
ttable( lm(cmort~ Z), vif=TRUE)
```

<br/>
Example 3.8

```r
prdpry = ts.intersect(L=Lynx, L1=lag(Lynx,-1), H1=lag(Hare,-1), dframe=TRUE)
fit    = lm(L~ L1 + L1:H1, data=prdpry, na.action=NULL) 
ttable(fit)

# residuals
par(mfrow=1:2)
tsplot(resid(fit), col=4, main=NA)
acf1(  resid(fit), col=4, main=NA)
mtext("Lynx Residuals", outer=TRUE, line=-1.4, font=2)
```

<br/>
Example 3.12

```r
par(mfrow=2:1)
tsplot(detrend(salmon), col=4, main="detrended salmon price")
tsplot(diff(salmon), col=4, main="differenced salmon price")

dev.new()
par(mfrow=2:1)
acf1(detrend(salmon), 48, col=4, main="detrended salmon price")
acf1(diff(salmon), 48, col=4, main="differenced salmon price")
```

<br/>
Example 3.13 &#128561;

```r
par(mfrow=c(2,1))
tsplot(diff(gtemp_land), col=4, main="differenced global temperature")
acf1(diff(gtemp_land), col=4, nxm=0)
mean(window(diff(gtemp_land), end=1979))   # drift before 1980
#  [1] 0.00465
mean(window(diff(gtemp_land), start=1980)) # drift after 1980
#  [1] 0.04909   
```

<br/>
Example 3.14

```R
layout(matrix(1:4,2), widths=c(2.5,1))
tsplot(varve, main=NA, ylab=NA, col=4)
mtext("varve", side=3, line=.25, cex=1.1, font=2, adj=0)
tsplot(log(varve), main=NA, ylab=NA, col=4)
mtext("log(varve)", side=3, line=.25, cex=1.1, font=2, adj=0)
QQnorm(varve, main=NA, nxm=0)
QQnorm(log(varve), main=NA, nxm=0)
```

<br/>
Example 3.15

```r
lag1.plot(soi, 12, col=4, location='bottomrigh')  # Figure 3.12
dev.new()
lag2.plot(soi, rec, 8, col=4)                     # Figure 3.13
```

<br/>
Example 3.16

```r
set.seed(90210)                # so you can reproduce these results
x  = 2*cos(2*pi*1:500/50 + .6*pi) + rnorm(500,0,5)
z1 = cos(2*pi*1:500/50); z2 = sin(2*pi*1:500/50)
ttable(fit <- lm(x~ 0 + z1 + z2))  # zero to exclude the intercept
par(mfrow=c(2,1))
tsplot(x, col=4, gg=TRUE)
tsplot(x, ylab=bquote(hat(x)), col=astsa.col(4,.7), gg=TRUE)
lines(fitted(fit), col=6, lwd=2)
```


<br/>
Example 3.17

```r
set.seed(90210)
t = 1:500
x = 2*cos(2*pi*(t+15)/50) + rnorm(500,0,5)
acf1(x, 200)  # not displayed
summary(fit <- nls(x~ A*cos(2*pi*omega*t + phi), start=list(A=10,omega=1/55,phi=0)))

tsplot(x, ylab=bquote(hat(x)), col=4, gg=TRUE)  # not shown but looks like
 lines(fitted(fit), col=2, lwd=2)               # the bottom of Figure 3.14
```



<br/>
Example 3.18

```r
w = c(.5, rep(1,11), .5)/12
soif = filter(soi, sides=2, filter=w)
tsplot(soi, col=4)
lines(soif, lwd=2, col=6)
# insert
par(fig = c(0,.25,0,.25), new = TRUE, col=8)
w1 = c(rep(0,20), w, rep(0,20))
plot(w1, type="l", ylim = c(-.02,.1), xaxt="n", yaxt="n", ann=FALSE, col=4)
```

<br/>
Example 3.19

```r
tsplot(soi, col=4)
lines(ksmooth(time(soi), soi, "normal", bandwidth=1), lwd=2, col=6)
# insert
par(fig = c(0,.25,0,.25), new = TRUE, col=8)
curve(dnorm(x), -3, 3, xaxt="n", yaxt="n", ann=FALSE, col=4)

# change time unit
SOI = ts(soi, freq=1) # make the unit of time a month
tsplot(SOI, col=4)    # not shown
lines(ksmooth(time(SOI), SOI, "normal", bandwidth=12), lwd=2, col=6)
```

<br/>
Example 3.20

```r 
trend(soi, lowess=TRUE)      # trend (with default span)
lines(lowess(soi, f=.05), lwd=2, col=6)  # El Niño cycle
```

<br/>
Example 3.21

```r
tsplot(tempr, cmort, type='p', xlab="Temperature", ylab="Mortality", col=4)
lines(lowess(tempr,cmort), col=6, lwd=2)
```

<br/>
Example 3.22

```r
x = window(hor, start=2002)
plot(decompose(x))            # not shown
dev.new()
plot(stl(x, s.window="per"))  # seasons are periodic - not shown
dev.new()
plot(stl(x, s.window=15))     # better, but nicer version below  

dev.new()
par(mfrow = c(4,1))
x = window(hor, start=2002)
out = stl(x, s.window=15)$time.series
tsplot(x, main="Hawaiian Occupancy Rate", ylab="% rooms", col=8, type="c")
 text(x, labels=1:4, col=c(3,4,2,6), cex=1.25)
tsplot(out[,1], main="Seasonal", ylab="% rooms",col=8, type="c")
 text(out[,1], labels=1:4, col=c(3,4,2,6), cex=1.25)
tsplot(out[,2], main="Trend", ylab="% rooms", col=8, type="c")
 text(out[,2], labels=1:4, col=c(3,4,2,6), cex=1.25)
tsplot(out[,3], main="Noise", ylab="% rooms", col=8, type="c")
 text(out[,3], labels=1:4, col=c(3,4,2,6), cex=1.25)
```

[<sub>top</sub>](#table-of-contents)

---



## Chapter 4


<br/>
Example 4.2

```r
par(mfrow=c(2,1))
tsplot(sarima.sim(ar= .9, n=100), main=bquote(AR(1)~~~phi==+.9), ylab="x", col=4, gg=TRUE)
tsplot(sarima.sim(ar=-.9, n=100), main=bquote(AR(1)~~~phi==-.9), ylab="x", col=4, gg=TRUE)
```

<br/>
Example 4.3

```r
set.seed(8675309)
x = sarima.sim(ar=c(1.5,-.75), n=144, S=12)
psi = ts(c(1, ARMAtoMA(ar=c(1.5, -.75), ma=0, 50)), start=0, freq=12)
par(mfrow=c(2,1))
tsplot(x, main=bquote(AR(2)~~~phi[1]==1.5~~~phi[2]==-.75), col=4, xaxt="n", gg=TRUE)
 mtext(seq(0,144,by=12), side=1, at=0:12, cex=.8)
tsplot(psi, col=4, type="o", ylab=bquote(psi-weights), xaxt="n", xlab="Index", gg=TRUE)
 mtext(seq(0,48,by=12), side=1, at=0:4, cex=.8) 
```

<br/>
Examples 4.5

```r
par(mfrow = c(2,1))
tsplot(sarima.sim(ma= .9, n=100), main=bquote(MA(1)~~~theta==+.9), col=4, ylab="x", gg=TRUE)
tsplot(sarima.sim(ma=-.9, n=100), main=bquote(MA(1)~~~theta==-.9), col=4, ylab="x", gg=TRUE)
```

<br/>
Example 4.10

```r
set.seed(8675309)         # Jenny, I got your number
x = rnorm(150, mean=5)    # generate iid N(5,1)s
sarima(x, p=1, q=1, details=FALSE)  # estimation 
```

<br/>
Example 4.11

```r
AR = c(1, -.3, -.4) # original AR coefs on the left
polyroot(AR)
MA = c(1, .5)       # original MA coefs on the right
polyroot(MA)
```

<br/>
Example 4.12

```r
round( ARMAtoMA(ar=.8, ma=-.5, 10), 2) # first 10 psi-weights
round( ARMAtoAR(ar=.8, ma=-.5, 10), 2) # first 10 pi-weights
ARMAtoMA(ar=1, ma=0, 20)
```


<br/>
Example 4.19

```r
ACF  = ARMAacf(ar=c(1.5,-.75), ma=0, 24)[-1]
PACF = ARMAacf(ar=c(1.5,-.75), ma=0, 24, pacf=TRUE)
par(mfrow=1:2)
tsplot(ACF, type="h", xlab="LAG", ylim=c(-.8,1), col=4, lwd=2, gg=TRUE)
abline(h=0, col=gray(.7,.4))
tsplot(PACF, type="h", xlab="LAG", ylim=c(-.8,1), col=4, lwd=2, gg=TRUE)
abline(h=0, col=gray(.7,.4))    
```

<br/>
Example 4.22

```r
acf2(rec, 48, col=4)  # will produce values and a graphic
(regr = ar.ols(rec, order=2, demean=FALSE, intercept=TRUE))
regr$asy.se.coef  # standard errors of the estimates
```

<br/>
Example 4.25

```r
rec.yw = ar.yw(rec, order=2)
rec.yw$x.mean   # mean estimate
rec.yw$ar       # phi parameter estimates
sqrt(diag(rec.yw$asy.var.coef)) # their standard errors
rec.yw$var.pred # error variance estimate
```

<br/>
Example 4.26

```r
set.seed(1)
ma1 = sarima.sim(ma = 0.9, n = 100)
acf1(ma1, plot=FALSE)[1]

# generate 10000 MA(1)s and calculate first sample ACF
r = replicate(10^4, acf1(sarima.sim(ma=.9, n=100), max.lag=1, plot=FALSE))
mean(abs(r) >= .5)  # .5 exceedance prob
```

<br/>
Example 4.28

```r
tsplot(diff(log(varve)), col=4, ylab=bquote(nabla~log~X[~t]), main="Transformed Glacial Varves")
dev.new()
acf2(diff(log(varve)), col=4 )

 
x = diff(log(varve))       # data
r = acf1(x, 1, plot=FALSE) # acf(1)
c(0) -> z -> Sc -> Sz -> Szw -> para # initialize ...
c(x[1]) -> w                         # ... all variables
num = length(x)    

## Gauss-Newton Estimation
para[1] = (1-sqrt(1-4*(r^2)))/(2*r)  # MME to start (d)
niter   = 12             
for (j in 1:niter){ 
 for (t in 2:num){  w[t] = x[t]   - para[j]*w[t-1]
                    z[t] = w[t-1] - para[j]*z[t-1]
 }
 Sc[j]  = sum(w^2)
 Sz[j]  = sum(z^2)
 Szw[j] = sum(z*w)
para[j+1] = para[j] + Szw[j]/Sz[j]
}
## Results
cbind(iteration=1:niter-1, thetahat=para[1:niter], Sc, Sz)

## Plot conditional SS and results
c(0) -> cSS
th = -seq(.3, .94, .01)
for (p in 1:length(th)){  
 for (t in 2:num){  w[t] = x[t] - th[p]*w[t-1] 
 }
cSS[p] = sum(w^2)
}

dev.new()
tsplot(th, cSS, ylab=bquote(S[c](theta)), xlab=bquote(theta))
abline(v=para[1:12], lty=2, col=4)   # add previous results 
points(para[1:12], Sc[1:12], pch=16, col=4)
 text(para[1:12], 149, labels=0:11)
```



<br/>
Example 4.29

```r
sarima(diff(log(varve)), q=1, no.constant=TRUE)
```

<br/>
Example 4.30

```R
set.seed(666)
sarima(rnorm(1000), p=3, q=3)
```

<br/>
Example 4.31<br/>

and now, these guys do time series
 ![](dd.png) <br/>

```R
set.seed(666)     
x = rnorm(1000) 
library(forecast)
auto.arima(x)    # stepwise

auto.arima(x, stepwise=FALSE)  # all subsets
```

&#129315; &#129315; but let's get smart &#129300;
```r
ar(x)   # uses AIC by default
```

<br/>
Example 4.33

```r
set.seed(1)
x = sarima.sim(ar=.9, n=100)       # simulate an AR(1)
sarima(x,1,0,0, no.constant=TRUE, details=FALSE)  # fit AR(1)
sarima(x,2,0,0, no.constant=TRUE, details=FALSE)  # overfit AR(2)
```

<br/>
Example 4.34

```r
fish = sarima(rec, p=2)             # fit the model
sarima.for(rec, n.ahead=24, p=2)    # forecast
abline(h=fish[[1]]$coef["xmean"])   # display mean
```

<br/>
Example 4.41

```R
z = c(1,-1.5,.75)      # coefficients of the polynomial
(z1 = polyroot(z)[1])  # print one root = 1 + i/√3
arg = Arg(z1)/(2*pi)   # arg in cycles/pt
1/arg                  # period
```





[<sub>top</sub>](#table-of-contents)

---



## Chapter 5



<br/>
Example 5.2

```r
sarima(diff(log(varve)), q=1, no.constant=TRUE)
# equivalently
sarima(log(varve), d=1, q=1, no.constant=TRUE)
```

<br/>
Example 5.3

```r
ARMAtoMA(ar=1, ma=0, 20) # psi-weights for rw
```

<br/>
Example 5.4

```r
round( ARMAtoMA(ar=c(1.9,-.9), ma=0, 60), 1 )  # ain't no AR(2)

set.seed(12345)
x <- sarima.sim(ar=.9, d=1, n=150)
y <- window(x, start=1, end=100)
sarima.for(y, n.ahead=50, p=1, d=1, gg=TRUE, plot.all=TRUE)
text(85, 360, "PAST"); text(115, 360, "FUTURE")
abline(v=100, lty=2, col=4)
lines(x, col=4)
```

<br/>
Example 5.5

```r
set.seed(666)
x = sarima.sim(ma = -.8, d=1, n = 100)
(x.ima = HoltWinters(x, beta=FALSE, gamma=FALSE)) 
plot(x.ima, main="EWMA")
```

<br/>
Examples 5.6

```r
##-- Figure 5.3 --##
layout(1:2, heights=2:1)
tsplot(gnp, col=4)
acf1(gnp, 48, main="")

##-- Figure 5.4 --##
dev.new()
tsplot(diff(log(gnp)), ylab="GNP Growth Rate", col=4)
abline(h = mean(diff(log(gnp))), col=6)

##-- Figure 5.5 --##
dev.new()
acf2(diff(log(gnp)), main="")


sarima(diff(log(gnp)), q=2) # MA(2) on growth rate
dev.new()
sarima(diff(log(gnp)), p=1) # AR(1) on growth rate

round( ARMAtoMA(ar=.35, ma=0, 10), 3) # print psi-weights
```

<br/>
Example 5.7

```r
sarima(diff(log(gnp)), q=3, details=FALSE)  # try an MA(2+1)

sarima(diff(log(gnp)), p=2, details=FALSE)  # try an AR(1+1) 
```

<br/>Example 5.8

```r
ma2 = sarima(diff(log(gnp)), q=2, details=FALSE)
ar1 = sarima(diff(log(gnp)), p=1, details=FALSE)
rbind(ma2=ma2[[5]], ar1=ar1[[5]])  # compare ICs
```



<br/>
Example 5.9

```r
sarima(log(varve), 0, 1, 1, no.constant=TRUE) # ARIMA(0,1,1)
sarima(log(varve), 1, 1, 1, no.constant=TRUE) # ARIMA(1,1,1)
```

<br/>
Example 5.10

```r
# for the polynomial regression, it's better if
# the regressors don't get too large, so we center time
t    = time(USpop20) - 1960
# for the regression, 'raw' specifies not to use
# orthogonal polynomials because we want to easily
# get a curve for the prediction
reg  = lm( USpop20~ poly(t, 10, raw=TRUE) )
# the next 4 lines are to get the prediction curve
# X is the design matrix for the curve
b    = as.vector(coef(reg))
t    = 1900:2044
X    = outer(t - 1960, 0:10, FUN = "^")
pred = X %*% b
# now plot the prediction curve, and then add the data as points
tsplot(t, pred, ylab="Population", xlab='Year', cex.main=1, col=4,
            main="U.S. Population by Official Census")
points(time(USpop20), USpop20, pch=21, bg=rainbow(13), cex=1.25)
mtext(bquote('\u00D7'~10^6), side=2, line=1.5, adj=1, cex=.8)
```



<br/>
Example 5.11 

```r
set.seed(10101010)
SAR = sarima.sim(sar=.95, S=12, n=37) + 50
layout(matrix(c(1,2, 1,3), nc=2), heights=c(1.5,1))
tsplot(SAR, type="c", xlab="Year", gg=TRUE, ylab='SAR(1)', xaxt='n')
 abline(v=0:3, col=4, lty=2)
 points(SAR, pch=Months, cex=1.2, font=4, col=1:6)
 axis(1, at=0:3, col='white')
phi  = c(rep(0,11),.95)
ACF  = ARMAacf(ar=phi, ma=0, 100) 
PACF = ARMAacf(ar=phi, ma=0, 100, pacf=TRUE)
LAG  = 0:100/12
tsplot(LAG, ACF, type="h", xlab="LAG \u00F7 12", ylim=c(-.04,1), gg=TRUE, col=4)
 abline(h=0, col=8)
tsplot(LAG[-1], PACF, type="h", xlab="LAG \u00F7 12", ylim=c(-.04,1), gg=TRUE, col=4)
 abline(h=0, col=8)
```

<br/>
Example 5.12 

```r
##-- Figure 5.10 --##
par(mfrow=1:2)
phi = c(rep(0,11),.8)
ACF = ARMAacf(ar=phi, ma=-.5, 50)[-1]     
PACF = ARMAacf(ar=phi, ma=-.5, 50, pacf=TRUE)
LAG = 1:50/12
tsplot(LAG, ACF, type="h", xlab="LAG", ylim=c(-.4,.8), col=4, lwd=2) 
abline(h=0, col=8)
tsplot(LAG, PACF, type="h", xlab="LAG", ylim=c(-.4,.8), col=4, lwd=2)  
abline(h=0, col=8)

##-- birth series --##
par(mfrow=2:1)
tsplot(birth, col=4)          # monthly number of births in US
tsplot(diff(birth), col=4)
dev.new()
acf2(diff(birth))     # P/ACF of the differenced birth rate
```

<br/>
&#128009; Seasonal Persistence

```R
x = window(hor, start=2002)
par(mfrow = c(2,1)) 
tsplot(x, main='Hawaiian Occupancy Rate', ylab=' % rooms', col=8)
text(x, labels=1:4, col=c(3,4,2,6))
Qx = stl(x,15)$time.series[,1] 
tsplot(Qx, main="Seasonal Component", ylab=' % rooms', col=8)
text(Qx, labels=1:4, col=c(3,4,2,6))
```

<br/>
Example 5.15

```r
par(mfrow=c(2,1))
tsplot(cardox, ylab=bquote(CO[2]), main="Monthly Carbon Dioxide Readings - Mauna Loa Observatory",  col=4)
tsplot(diff(diff(cardox,12)), ylab=bquote(nabla~nabla[12]~CO[2]), col=4)

dev.new()
acf2(diff(diff(cardox,12)), col=4) 

dev.new()
sarima(cardox, 0,1,1, 0,1,1,12, col=4)
dev.new()
sarima(cardox, 1,1,1, 0,1,1,12, col=4)

dev.new()
sarima.for(cardox, 60, 1,1,1, 0,1,1,12, col=4, ylab=bquote(CO[2]))
abline(v=2018.9, lty=6)

##-- for comparison, try the first model --##
dev.new()
sarima.for(cardox, 60, 0,1,1, 0,1,1,12)  # not shown 
```

<br/>
Example 5.16

```r
pp = ts.intersect(L=Lynx, L1=lag(Lynx,-1), H1=lag(Hare,-1), dframe=TRUE)
# Original Regression
summary( fit <- lm(L~ L1 + L1:H1, data=pp, na.action=NULL) )

acf2(resid(fit), col=4)   # ACF/PACF of the residuls

# Try AR(2) errors
dev.new()
sarima(pp$L, p=2, xreg=cbind(L1=pp$L1, LH1=pp$L1*pp$H1), col=4)
```


[<sub>top</sub>](#table-of-contents)

---



## Chapter 6




<br/>
Aliasing

```r
t = seq(0, 24, by=.01)  
X = cos(2*pi*t*1/2)              # 1 cycle every 2 hours  
tsplot(t, X, xlab="Hours", gg=TRUE, col=7)
T = seq(1, length(t), by=250)    # observed every 2.5 hrs 
points(t[T], X[T], pch=19, col=4)
lines(t, cos(2*pi*t/10), col=4)
axis(1, at=t[T], labels=FALSE, lwd.ticks=3, col.ticks=5, col=gray(1))
```

<br/>
Example 6.1

```r
x1 = 2*cos(2*pi*1:100*6/100)  + 3*sin(2*pi*1:100*6/100)
x2 = 4*cos(2*pi*1:100*10/100) + 5*sin(2*pi*1:100*10/100)
x3 = 6*cos(2*pi*1:100*40/100) + 7*sin(2*pi*1:100*40/100)
x  = x1 + x2 + x3;  L=c(-10,10)
par(mfrow = c(2,2), cex=.9, font.main=1)
tsplot(x1, ylim=L, col=4, main=bquote(omega==6/100~~A^2==13),  gg=TRUE)
tsplot(x2, ylim=L, col=4, main=bquote(omega==10/100~~A^2==41), gg=TRUE)
tsplot(x3, ylim=L, col=4, main=bquote(omega==40/100~~A^2==85), gg=TRUE)
tsplot(x, main="sum", col=4, gg=TRUE)
```

<br/>Example 6.2

```r
set.seed(1)
x = rnorm(7)
t = 1:7
c1 = cos(2*pi*t*1/7); s1 = sin(2*pi*t*1/7)
c2 = cos(2*pi*t*2/7); s2 = sin(2*pi*t*2/7)
c3 = cos(2*pi*t*3/7); s3 = sin(2*pi*t*3/7)
reg = lm(x~ cbind(c1,s1,c2,s2,c3,s3))
rbind(x, xhat = fitted(reg))   
```

<br/>
Periodogram &mdash; just before Example 6.4

```r 
x1 = 2*cos(2*pi*1:100*6/100)  + 3*sin(2*pi*1:100*6/100)
x2 = 4*cos(2*pi*1:100*10/100) + 5*sin(2*pi*1:100*10/100)
x3 = 6*cos(2*pi*1:100*40/100) + 7*sin(2*pi*1:100*40/100)
x  = x1 + x2 + x3   # from Example 6.1

per  = Mod(fft(x)/sqrt(100))^2   
P    = (4/100)*per
Fr   = 0:99/100
tsplot(Fr, P, type="h", lwd=3, xlab="frequency", ylab="scaled periodogram", col=4, gg=TRUE)
abline(v=.5, lty=5, col=8)
axis(side=1, at=seq(.1,.9,by=.2), col='white', col.ticks=1)
```

<br/>
Example 6.5

```r
par(mfrow=c(3,2))
for(i in 4:9){
mvspec(fmri1[,i], main=colnames(fmri1)[i], ylim=c(0,3), xlim=c(0,.2), col=5, lwd=2, type="o", pch=20)
abline(v=1/32, col=4, lty=5) # stimulus frequency
}
```

<br/>
Examples 6.7, 6.9, and 6.10

```r
par(mfrow=c(3,1))
arma.spec(main="White Noise", col=4, gg=TRUE)
arma.spec(ma=.5, main="Moving Average", col=4, gg=TRUE)
arma.spec(ar=c(1,-.9), main="Autoregression", col=4, gg=TRUE)
```

<br/>
Example 6.12

```r
##-- Figure 6.7 --##
par(mfrow=c(3,1))
tsplot(soi, col=4, main='SOI')  
tsplot(diff(soi), col=4, main='First Difference')       
 k = kernel("modified.daniell", 6)   # MA weights
tsplot(kernapply(soi, k), col=4, main="Seasonal Moving Average")    

##-- Figure 6.8 - frequency responses --##
dev.new()
par(mfrow=c(2,1)) 
w = seq(0, .5, by=.001) 
FRdiff = abs(1-exp(2i*pi*w))^2
tsplot(12*w, FRdiff, col=4, ylab='', xlab='frequency (\u00D7 12)', main='First Difference', gg=TRUE, cex.main=1)
u = rowSums(cos(outer(w, 2*pi*1:5)))
FRma = ((1 + cos(12*pi*w) + 2*u)/12)^2
tsplot(12*w, FRma, col=4, ylab='', xlab='frequency (\u00D7 12)',  main='Seasonal Moving Average', gg=TRUE, cex.main=1)
```





[<sub>top</sub>](#table-of-contents)

---



## Chapter 7


<br/>
DFTs

```r
(dft = fft(1:4)/sqrt(4))
(idft = fft(dft, inverse=TRUE)/sqrt(4))
```

<br/>
Example 7.4

```r
par(mfrow=c(2,1))    
mvspec(soi, col=4, lwd=2)
  rect(1/7, -1e5, 1/3, 1e5, density=NA, col=gray(.5,.2))
  abline(v=1/4, lty=2, col=4)
  mtext('1/4', side=1, line=0, at=.25, cex=.75)
mvspec(rec, col=4, lwd=2) 
  rect(1/7, -1e5, 1/3, 1e5, density=NA, col=gray(.5,.2))
  abline(v=1/4, lty=2, col=4)
  mtext('1/4', side=1, line=0, at=.25, cex=.75)

#  log redux
dev.new()
par(mfrow=c(2,1)) 
mvspec(soi, col=4, lwd=2, log='yes')
  rect(1/7, 1e-5, 1/3, 1e5, density=NA, col=gray(.5,.2))
  abline(v=1/4, lty=2, col=4)
  mtext('1/4', side=1, line=0, at=.25, cex=.75)
mvspec(rec, col=4, lwd=2, log='yes')
  rect(1/7, 1e-5, 1/3, 1e5, density=NA, col=gray(.5,.2))
  abline(v=1/4, lty=2, col=4)
  mtext('1/4', side=1, line=0, at=.25, cex=.75)  
```

<br/>
Periodogram... Bad! &#128556;

```r
u = mvspec(rnorm(1000), col=8)    # periodogram
abline(h=1, col=2, lwd=5)         # true spectrum
sm = filter(u$spec, filter=rep(1,101)/101, circular=TRUE) # smooth
lines(u$freq, sm, col=5, lwd=2)   # add the smooth
```

<br/>
Example 7.5

```r
par(mfrow=c(2,1))
soi_ave = mvspec(soi, spans=9, col=4, lwd=2)
 rect(1/7, -1e5, 1/3, 1e5, density=NA, col=gray(.5,.2))
 abline(v=.25, lty=2, col=4)
 mtext('1/4', side=1, line=0, at=.25, cex=.75)
rec_ave = mvspec(rec, spans=9, col=4, lwd=2)
 rect(1/7, -1e5, 1/3, 1e5, density=NA, col=gray(.5,.2))
 abline(v=.25, lty=2, col=4)
 mtext('1/4', side=1, line=0, at=.25, cex=.75)

 # log scale
 dev.new()
 par(mfrow=c(2,1))
soi_ave = mvspec(soi, spans=9, col=4, lwd=2, log='y')
 rect(1/7, 1e-5, 1/3, 1e5, density=NA, col=gray(.5,.2))
 abline(v=.25, lty=2, col=4)
 mtext('1/4', side=1, line=0, at=.25, cex=.75)
rec_ave = mvspec(rec, spans=9, col=4, lwd=2, log='y')
 rect(1/7, 1e-5, 1/3, 1e5, density=NA, col=gray(.5,.2))
 abline(v=.25, lty=2, col=4)
 mtext('1/4', side=1, line=0, at=.25, cex=.75)
```


<br/>
Example 7.6

```r
y = ts(100:1 %% 20, freq=20)   # sawtooth signal
par(mfrow=2:1)
tsplot(1:100, y, ylab='sawtooth signal', col=4, gg=TRUE)
mvspec(y, main=NA, ylab='periodogram', col=5, gg=TRUE)
```

<br/>
Example 7.7

```r
(dm = kernel("modified.daniell", c(3,3)))            # for a list

# easy graphic
par(mfrow=1:2)
plot(dm, ylab=bquote(h[~k])) # for a plot
plot(kernel("modified.daniell", c(3,3,3)), ylab=bquote(h[~k])) 

# text version 
par(mfrow=1:2)
tsplot(kernel("modified.daniell", c(3,3)), ylab=bquote(h[~k]), cex.main=1, lwd=2, col=4, ylim=c(0,.16), xlab='k', type='h', main='mDaniell(3,3)', gg=TRUE, las=0)
tsplot(kernel("modified.daniell", c(3,3,3)), ylab=bquote(h[~k]), cex.main=1, lwd=2, col=4, ylim=c(0,.16), xlab='k', type='h', main='mDaniell(3,3,3)', gg=TRUE, las=0)

# smoothed specta
dev.new()
par(mfrow=c(2,1))
sois = mvspec(soi, spans=c(7,7), taper=.1, col=5, lwd=2)
 rect(1/7, -1e5, 1/3, 1e5, density=NA, col=gray(.5,.2))
 abline(v=.25, lty=2, col=4)
 mtext("1/4", side=1, line=0, at=.25, cex=.75)
recs = mvspec(rec, spans=c(7,7), taper=.1, col=5, lwd=2)
 rect(1/7, -1e5, 1/3, 1e5, density=NA, col=gray(.5,.2))
 abline(v=.25, lty=2, col=4)
 mtext("1/4", side=1, line=0, at=.25, cex=.75)

# log scale
dev.new()
par(mfrow=c(2,1))
sois = mvspec(soi, spans=c(7,7), taper=.1, col=5, lwd=2, log='y')
 rect(1/7, 1e-5, 1/3, 1e5, density=NA, col=gray(.5,.2))
 abline(v=.25, lty=2, col=4)
 mtext("1/4", side=1, line=0, at=.25, cex=.75)
recs = mvspec(rec, spans=c(7,7), taper=.1, col=5, lwd=2, log='y')
 rect(1/7, 1e-5, 1/3, 1e5, density=NA, col=gray(.5,.2))
 abline(v=.25, lty=2, col=4)
 mtext("1/4", side=1, line=0, at=.25, cex=.75)

# details 
sois$details[1:45,]
```

<br/>
Example 7.8

```r
layout(matrix(1:2,1), widths=c(3,1))
s0  = mvspec(soi, spans=c(7,7), plot=FALSE)             # no taper
s10 = mvspec(soi, spans=c(7,7), taper=.2,  plot=FALSE)  # 20%
s50 = mvspec(soi, spans=c(7,7), taper=.5, plot=FALSE)   # full taper
r = 1:60
tsplot(s0$freq[r], log(s0$spec[r]), gg=TRUE, col=4, lwd=2, ylab="log-spectrum", xlab="frequency")  
lines(s10$freq[r], log(s10$spec[r]), col=2, lwd=2)   
lines(s50$freq[r], log(s50$spec[r]), col=3, lwd=2)   
text(.7, -3.5, 'leakage', cex=.8)
arrows(.7, -3.6, .7, -4.5, length=0.05, angle=30)   
legend('bottomleft', legend=c('no taper', '20% taper', '50% taper'), lwd=2, col=c(4,2,3), bty='n')
# tapers
x = rep(1,100) 
tsplot(1:100/100, cbind(spec.taper(x, p=.2), spec.taper(x, p=.5)), col=2:3, gg=TRUE, spag=TRUE, xlab='t / n', lwd=2, ylab='tapers')
```


<br/>
Example 7.11

```r
par(mfrow=2:1)
spec.ic(soi, col=5, lwd=2, lowess=TRUE) -> u   # min AIC spec
 rect(1/7, -1e5, 1/2, 1e5, density=NA, col=gray(.6,.2))
 mtext("3/10", side=1, line=0, at=.3, cex=.75)
 abline(v=.3, lty=2, col=8)  # approximate El Niño Cycle
spec.ic(soi, col=6, lwd=2, lowess=TRUE, BIC=TRUE)  # min BIC spec
 rect(1/7, -1e5, 1/2, 1e5, density=NA, col=gray(.6,.2))
 mtext("3/10", side=1, line=0, at=.3, cex=.75)
 abline(v=.3, lty=2, col=8)

# AIC/BIC plots
dev.new()
tsplot(u[[1]][-1,1], u[[1]][-1,2:3], type="o", xlab="order", col=c(2,4), pch=c(19,17), ylab="AIC / BIC", cex=1.05, nxm=5, spag=TRUE, addLegend=TRUE, location="topleft")
```



<br/>
Example 7.13

```r
sr = mvspec(cbind(soi,rec), kernel=kernel("daniell",9), plot.type="coh", main="SOI & Recruitment", col=5, lwd=2)                 
( f = qf(.999, 2, sr$df-2) )  
( C = f/(18+f) )         
abline(h = C)
```



[<sub>top</sub>](#table-of-contents)

---



## Chapter 8

<br/>Figure 8.1

```r
library(xts)
djiar = diff(log(djia$Close))
layout(matrix(c(1,2,1,3), 2), heights=2:1)
plot(djiar, col=4)
acf1(djiar, ylim=c(-.1,.6))
acf1(djiar^2, ylim=c(-.1,.6))
```



<br/>Example 8.1

```r
res = resid( sarima(diff(log(gnp)), 1,0,0, details=FALSE)[[1]] )
acf2(res^2, 20)

library(fGarch)
gnpr = diff(log(gnp))
summary( garchFit(~arma(1,0) + garch(1,0), data = gnpr) )
```
&#127886; __NOTE:__ The regression summary gives 2 sided p-values for parameters that must be positive... so you have to 1/2 them. 	

<br/>
Example 8.2

```r
library(xts)
djiar = diff(log(djia$Close))[-1]
acf2(djiar)    # exhibits some autocorrelation
u = resid( sarima(djiar, 1,0,0, details=FALSE)$fit )
acf2(u^2)      # oozes autocorrelation
library(fGarch)
summary(djia.g <- garchFit(~arma(1,0)+garch(1,1), data=djiar, cond.dist="std"))
plot(djia.g, which=3)   
```

<br/>
Example 8.3

```r
lapply( c("xts", "fGarch"), library, char=TRUE) # load 2 packages in one line - amazing!
djiar = diff(log(djia$Close))[-1]
summary(djia.ap <- garchFit(~arma(1,0)+aparch(1,1), data=djiar, cond.dist="std"))
plot(djia.ap)   # to see all plot options 
```

<br/>
Compare sample ACF of RW vs long memory

```R
par(mfrow=2:1)
acf1(cumsum(rnorm(634)), 100, col=4, main='Series: random walk')
acf1(log(varve), 100, col=4, ylim=c(-.1,1))
```

<br/>
Example 8.5 

```r
library(tseries)
adf.test(log(varve), k=0) # DF test
adf.test(log(varve))      # ADF test
pp.test(log(varve))       # PP test
```

<br/>
Example 8.6

```r
library(arfima)
summary(varve.fd <- arfima(log(varve), order = c(0,0,0)))  

# residual analysis
innov = resid(varve.fd)[[1]]  # resid() produces a `list` 
sarima(innov, col=3, no.constant=TRUE)  # arfima residuals (* see Note)
dev.new()
sarima(log(varve), 1,1,1, col=4, no.constant=TRUE)  # arima residuals

# plot pi(d)
dev.new()
d = coef(varve.fd)[1]
p = c(1)
for (k in 1:30) { p[k+1] = (k-d)*p[k]/(k+1) } 
tsplot(1:30, p[-1], ylab=bquote(pi(d)), lwd=2, xlab="Index", type="h", col=4)
```
\* Note- regarding `sarima(innov)` ... what's going on is you're fitting an ARIMA(0,0,0) to `innov` and then looking at the resulting residual analysis graphic.

<br/>
Example 8.7

```r
library(arfima)
summary(varve1.fd <- arfima(log(varve), order=c(0,0,1)))
```

<br/>
Example 8.10

```R
fit = ssm(gtemp_land, A=1, alpha=.01, phi=1, sigw=.01, sigv=.1, fixphi=TRUE)

tsplot(gtemp_land, col=4, type="o", pch=20, ylab="Temperature Deviations")
lines(fit$Xs, col=6, lwd=2)
 xx = c(time(fit$Xs), rev(time(fit$Xs)))
 yy = c(fit$Xs-2*sqrt(fit$Ps), rev(fit$Xs+2*sqrt(fit$Ps)))
polygon(xx, yy, border=8, col=gray(.6, alpha=.25) )
```


<br/>
Example 8.11

```R
ccf2(cmort, part, col=4)  # Fig 8.9
dev.new()
acf2(diff(cmort), col=4)  # Fig 8.10
dev.new()
pre.white(cmort, part, diff=TRUE, max.lag=110, col=4)
```

<br/>
Section 8.6 &mdash; the whole freakin section is an example

```R
# data
set.seed(101010)
e   = rexp(150, rate=.5); u = runif(150,-1,1); de = e*sign(u)  
dex = 50 + sarima.sim(n=100, ar=.95, innov=de, burnin=50)
layout(matrix(1:2, nrow=1), widths=c(5,2))
tsplot(dex, col=4, ylab=bquote(X[~t]), gg=TRUE)
# densities for comparison
f = function(x) { .5*dexp(abs(x), rate = 1/sqrt(2))}
w = seq(-5, 5, by=.01)
tsplot(w, f(w), gg=TRUE, col=4, xlab='w', ylab='f(w)', ylim=c(0,.4)) 
lines(w, dnorm(w), col=2) 

# estimation
fit  = ar.yw(dex, aic=FALSE, order=1)
round(estyw <- c(mean=fit$x.mean, ar1=fit$ar, se=sqrt(fit$asy.var.coef), var=fit$var.pred), 3)

# finite sample distribution 
phi.yw = c()
for (i in 1:1000){
  e = rexp(150, rate=.5); u = runif(150,-1,1); de = e*sign(u)
  x = 50 + sarima.sim(n=100, ar=.95, innov=de, burnin=50)
  phi.yw[i] = ar.yw(x, order=1)$ar    
} 

# bootstrap
boots = ar.boot(dex, order=1, plot=FALSE)  # default is B = 500

# pictures
hist(boots[[1]], main=NA, prob=TRUE, ylim=c(0,15), xlim=c(.65,1.05), col=astsa.col(4,.4), xlab=bquote(hat(phi)))                
lines(density(phi.yw, bw=.02), lwd=2) # ture distribution 
u = seq(.75, 1.1, by=.001)            # normal approximation
lines(u, dnorm(u, mean=estyw[2], sd=estyw[3]), lty=2, lwd=2)
legend(.65, 15, bty="n", lty=c(1,0,2), lwd=c(2,0,2), col=1, pch=c(NA,22,NA), legend=c("true distribution", "bootstrap distribution", "normal approximation"), pt.bg=c(NA, astsa.col(4,.4), NA), pt.cex=2.5)

# CIs
alf = .025   # 95% CI
quantile(phi.star.yw, probs=c(alf, 1-alf))        # bootstrap
quantile(phi.yw, probs=c(alf, 1-alf))             # true
qnorm(c(alf, 1-alf), mean=estyw[2], sd=estyw[3])  # normal approx
```

<br/>
Example 8.12

```R
library(NTS)       # load package - install it first
flutar = uTAR(diff(flu), p1=4, p2=4)

sarima(resid(flutar), 0,0,0)  # residual analysis 

##-- graphic --##
dev.new()
innov = resid(flutar)
pred  = diff(flu)[-(1:4)] - innov
pred  = ts(pred, start=c(1968,6), freq=12)
tsplot(diff(flu), type='p', ylim=c(-.5,.5), pch=20, col=6, nym=2, ylab=bquote(nabla~flu[~t]))
lines(pred, col=4, lwd=2)
abline(h = flutar$thr, lty=6, col=5)
# error bnds
prde1 = sqrt(sum(resid(flutar$model1)^2)/flutar$model1$df)
prde2 = sqrt(sum(resid(flutar$model2)^2)/flutar$model2$df)
    x = time(diff(flu))[-(1:4)]
prde = ifelse(lag(x,-1) < flutar$thr, prde1, prde2)
   xx = c(x, rev(x))
   yy = c(pred - 2*prde, rev(pred + 2*prde))
polygon(xx, yy, border=gray(.6,.5), col=gray(.6,.2))
legend('bottomright', legend=c('observed', 'predicted'), lty=0:1, pch=c(20,NA), col=c(6,4), lwd=2)
```

<br/>

[<sub>top</sub>](#table-of-contents)

---

## Elsewhere

This is a collection of code used in the text not listed above.


<br/>

__Lotka-Volterra Equations__ &ndash; Figure 3.5 shows a simulation of the equations.  This is how the figure was generated.

```r
H = c(1); L =c(.5)
for (t in 1:66000){
H[t+1] = 1.0015*H[t] - .00060*L[t]*H[t] 
L[t+1] =  .9994*L[t] + .00025*L[t]*H[t]
}
L = ts(10*L, start=1850, freq=900)
H = ts(10*H, start=1850, freq=900)

tsplot(cbind(predator=L, prey=H), spag=TRUE, col=c(2,4), ylim=c(0,125), ylab="Population Size", gg=TRUE, addLegend=TRUE, location='topleft', horiz=TRUE)
```

<br/>   

Example 4.40 shows the __causal region of an AR(2)__:

```r
seg1   =  seq( 0, 2,  by=0.1)
seg2   =  seq(-2, 2,  by=0.1)
name1  =  bquote(phi[1])
name2  =  bquote(phi[2])
tsplot(seg1, (1-seg1), ylim=c(-1,1), xlim=c(-2,2), ylab=name2, xlab=name1, col=4, gg=TRUE, main='Causal Region of an AR(2)')
lines(-seg1, (1-seg1), ylim=c(-1,1), xlim=c(-2,2), col=4) 
lines(seg2, -(seg2^2 /4), ylim=c(-1,1), col=4)
lines(x=c(-2,2), y=c(-1,-1), ylim=c(-1,1), col=4)
clip(-1,1,-1,1)
abline(h=0, v=0, lty=2, col=8)
text(0, .35, 'real roots')
text(0, -.5, 'complex roots')
```

<br/>
Figure 7.9 &mdash; Spectral windows 


```r
# set up
w = seq(-.02,.02,.0001); n=864  
u = 0
for (i in -4:4){ 
 k  = i/n
 u  = u + sin(n*pi*(w+k))^2/sin(pi*(w+k))^2
}
fk  = u/(9*n)
u   = 0; wp = w+1/n; wm = w-1/n
for (i in -4:4){
 k  = i/n; wk = w+k; wpk = wp+k; wmk = wm+k
 z  =  complex(real=0,imag=2*pi*wk)
 zp = complex(real=0,imag=2*pi*wpk)
 zm = complex(real=0,imag=2*pi*wmk)
 d  =  exp(z)*(1-exp(z*n))/(1-exp(z))
 dp = exp(zp)*(1-exp(zp*n))/(1-exp(zp))
 dm = exp(zm)*(1-exp(zm*n))/(1-exp(zm))
 D  = .5*d - .25*dm*exp(pi*w/n)-.25*dp*exp(-pi*w/n)
 D2 = abs(D)^2
 u  = u+D2 }
sfk = u/(n*9)
fk[201] = fk[200]
sfk[201] = sfk[200]

# graphic
 par(mfrow=1:2)
 tsplot(w, fk, col=4, ylab="", xlab="frequency", main="Without Tapering", yaxt='n', gg=TRUE, cex.main=1)
  mtext(expression("|"), side=1, line=-.5, at=c(-0.005 , 0.005), cex=.75, col=2)
  segments(-0.005, -4, 0.005 , -4 , lty=1, lwd=3, col=2)
 tsplot(w, 2*sfk/3, col=4, ylab="", xlab="frequency", main="With Tapering", yaxt='n',gg=TRUE, cex.main=1 )
  mtext(expression("|"), side=1, line=-.5, at=c(-0.005, 0.005), cex=.75, col=2)
  segments(-0.005, -1, 0.005, -1 , lty=1, lwd=3, col=2)
```




<br/>

In Appendix A, Example A.7 shows a normal likelihood.  The code in the text is for a contour plot, but the figure (Fig A.4) is a perspective plot.  This is how the perspective plot was generated (the code is pretty involved, which is why it's not displayed in the text).

```r
# da data
set.seed(90210)
N = 200
xdata = rnorm(N, mean=100, sd=15)

# for the likelihood
normL = function(x, mu, sigma) {
   -sum(dnorm(x, mu, sigma, log=TRUE))
 }

# grid of parameter values
mu    = seq(80, 120, length.out=N)
sigma = seq(10, 20, length.out=N)
parm.grid = expand.grid(mu=mu, sigma=sigma)
# evaluate -log L over the grid
like = c()
for (i in 1:N^2) {
like[i] = normL(xdata, parm.grid[i,"mu"], parm.grid[i,"sigma"])
}
like = matrix(like, nrow=N, ncol=N)


# code to make a perspective plot with levels - from:
# https://stat.ethz.ch/pipermail/r-help/2003-July/036151.html
 levelpersp <- function(x, y, z, colors=rainbow, ...) {
  zz <- (z[-1,-1] + z[-1,-ncol(z)] + z[-nrow(z),-1] + z[-nrow(z),-ncol(z)])/4
  breaks <- hist(zz, breaks=20, plot=FALSE)$breaks
  cols <-  colors(length(breaks)-1, start=.1, end=1, v=.8)
  zzz <- cut(zz, breaks=breaks, labels=cols)
  persp(x, y, z, col=(as.character(zzz)), ...)
  }

# finally, the figure
par(mar=c(2,0,0,0), cex.axis=.9)
levelpersp(mu, sigma, like*.001, phi=35, theta=25, expand=.75, scale=TRUE,  border=NA, ticktype="detailed", xlab='\u03BC',  ylab="\u03C3", zlab= "-log L")
```

<br/>
Appendix A: Daniell and the CLT

```r
md = function(n){kernel("modified.daniell", m=rep(3,n))}
par(mfrow=c(2,3), cex=.8, oma=c(0,0,.5,0))
for (i in 1:6){
 ytop = ifelse(i<4,.2,.12)
 tsplot(md(i), ylab=NA, lwd=2, col=4, ylim=c(0,ytop), xlab=NA, type='h', gg=TRUE)
 if (i==1) { mtext(bquote(X[1]), side=3, line=-2, adj=.95) 
  } else { mtext(bquote(sum(X[j], j==1, .(i))), side=3, line=-3, adj=.9) }
}
 title('The CLT in Action', outer=TRUE, adj=.52, line=-.9)
```



<br/>
<br/>




[<sub>top</sub>](#table-of-contents)

---
---
