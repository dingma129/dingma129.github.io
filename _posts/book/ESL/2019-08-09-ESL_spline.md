---
title: Spline
layout: splash
excerpt: ""
categories: [Python]
tags: [Spline]
---
<span style="font-weight:bold;font-size:36px">0. Introduction</span>

In this blog, I will discuss about <span style="font-weight:bold"><u>natural cubic spline</u></span> and <span style="font-weight:bold"><u>smooth spline</u></span>.

In mathematics, a spline is a special function defined piecewise by polynomials. In interpolating problems, spline interpolation is often preferred to polynomial interpolation because it yields similar results, even when using low degree polynomials. 

When using splines in regression models, the data is fitted to a set of spline basis functions with a reduced set of knots, typically by least squares.

Smooth splines are function estimates in order to balance a measure of error (for example, mean square error) with a derivative based a measure of the smoothness (for example, the roughness penalty based on second derivative). They are similar to Ridge regression, which can be thought as a balance of mean square error with size of the model coefficients. The most familiar example is the cubic smoothing spline.

---

<span style="font-weight:bold;font-size:36px">1. Natural cubic spline and smooth spline</span>

<span style="font-weight:bold;font-size:32px">1.1 Natural cubic spline</span>

To generalize linear regression, instead using polynomial regression

<center><img src="https://latex.codecogs.com/png.latex?f(X)=\beta_0+\sum_{j=1}^{p} \beta_jX^j,"/> </center>
we can also use <span style="font-weight:bold"><u>cubic spline</u></span>

<center><img src="https://latex.codecogs.com/png.latex?f(X)=\beta_0+\beta_1X+\beta_2X^2+\beta_3X^3+\beta_4h(X,\xi_1)+\cdots+\beta_{K+3}h(X,\xi_K),"/> </center>
where

<center><img src="https://latex.codecogs.com/png.latex?h(x,\xi)=\max((x-\xi)^3,0)."/> </center>
This cubic spline function 

* is a cubic polynomal between every pair of knots <img src="https://latex.codecogs.com/png.latex?\xi_i,\xi_{i+1}"/> ;
* is continuous at each knot;
* has continuous first and second order derivative at each knot.

One disadvantage of cubic spline is that it's not stable for extreme values of X (cubic over X). 

<span style="font-weight:bold"><u>Natural cubic splines</u></span> are cubic splines that are linear instead of cubic for <img src="https://latex.codecogs.com/png.latex?X\leq\xi_1,X\geq\xi_K"/>. We can select the parameter K using a validation set or cross-validation.



<span style="font-weight:bold;font-size:32px">1.2 Smooth  spline</span>

<span style="font-weight:bold"><u>Smooth spline</u></span> is a functions f which minimizes

<center><img src="https://latex.codecogs.com/png.latex?\sum_{i=1}^n(y_i-f(x_i))^2+\lambda\int f''(x)^2dx"/>,</center>
where the first half is the residual sum of squares (a measure of error) and the second half is a penalty term that is similar to Ridge regression (a measure of smoothness).

This smooth spline can overfit quite hard, and it can actually fit any training sample. By tuning the parameter <img src="https://latex.codecogs.com/png.latex?\lambda"/>, we can get a model that does not overfit. We can tune <img src="https://latex.codecogs.com/png.latex?\lambda"/> using a validation set or cross-validation.

---

<span style="font-weight:bold;font-size:36px">2. Examples</span>

<span style="font-weight:bold;font-size:32px">2.1 LA Ozone Data (natural cubic spline and smooth spline)</span>

We will use LA Ozone dataset from [here](https://web.stanford.edu/~hastie/ElemStatLearn//datasets/LAozone.data). This data set contains 330 examples.

<center><img src="https://dingma129.github.io/assets/figures/ESL/ozone_head.png" width="500"></center>
The task is to predict `ozone`(Upland Maximum Ozone) using the `dpg` feature only.

<span style="font-weight:bold;font-size:28px">2.1.1 Natural cubic spline</span>

```python
import statsmodels.api as sm
aic_list=[]
bic_list=[]
for k in range(3,15):
    model61 = sm.OLS.from_formula('ozone ~ cr(dpg, df = {})'.format(k), data=ozone_train).fit()
    aic_list.append(model61.aic)
    bic_list.append(model61.bic)
    
# best k = 3 or k = 6
fig,axes = plt.subplots(1,2,figsize=(12,4))
axes[0].plot(range(3,15),aic_list,marker='.')
axes[1].plot(range(3,15),bic_list,marker='.')
axes[0].set_xticks(ticks=range(3,15))
axes[0].set_title("AIC")
axes[1].set_xticks(ticks=range(3,15))
axes[1].set_title("BIC");
```

<center><img src="https://dingma129.github.io/assets/figures/ESL/ozone_aic.png" width="800"></center>
Now let us compare `df = 3` and `df = 6`.

```python
model3 = sm.OLS.from_formula('ozone ~ cr(dpg, df = 3)'.format(k), data=ozone_train).fit()
model6 = sm.OLS.from_formula('ozone ~ cr(dpg, df = 6)'.format(k), data=ozone_train).fit()
    
# using F-statistics, df=6 is not significantly better, so choose df=3
sm.stats.anova_lm(model3,model6)
```

<center><img src="https://dingma129.github.io/assets/figures/ESL/ozone_anova.png" width="500"></center>
With `p-value = 0.081816 > 0.05`, the model with `df = 6` is not significantly better than the model with `df=3`, so we should just choose `df=3`. 

<span style="font-weight:bold;font-size:28px">2.1.2 Smooth spline</span>

We can use scipy's implementation of smooth spline: `scipy.interpolate.UnivariateSpline`.

```python
import scipy as sp
error = []
for s in range(1000,5000,10):
    cs= sp.interpolate.UnivariateSpline(ozone_train_sorted['dpg'], ozone_train_sorted['ozone'],s=s)
    # validation error
    error.append(np.sqrt(mean_squared_error(ozone_val['ozone'],cs(ozone_val['dpg']))))
# best s = 3000
plt.figure(figsize=(16,8))
plt.plot(range(1000,5000,10),error,marker='.')
plt.ylabel("RMSE")
plt.xlabel("smoothing factor s");
```

<center><img src="https://dingma129.github.io/assets/figures/ESL/ozone_smooth_spline.png" width="800"></center>
By using validation set, we should choose `s=3000`.

<span style="font-weight:bold;font-size:28px">2.1.3 Comparison</span>

We can now plot 3 models above, natural cubic spline model with `df=3,6` and smooth spline model with `s=3000`. We can clearly see that natural cubic spline model with `df=6` slightly overfits, while the other two models fit pretty well to the dataset.

<center><img src="https://dingma129.github.io/assets/figures/ESL/ozone_compare.png" width="800"></center>