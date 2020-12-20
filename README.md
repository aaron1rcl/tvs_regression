# TVS Regression

Experiments with Time Varying Stochastic Regression

## Installation 

1. Create `conda`  environment.
```bash
conda create -n tvsr python=3.8
conda activate tvsr
```
2. Install requirements
```bash
pip install -r requirements.txt
```
TODO: add more information about how to install the requirements. 

---
<h3><b>Motivating Problem</b></h3>
Consider a typical regression problem of the form y=f(x) + c, where y is the dependent variable and x is the independent regressor. The intention is to find
the relationship between y and x, defined by some function f(x). The form of the function f(x) isnt particularly important, it could be one of any 
number of shapes. These regressions are well defined and effective for a very large class of problems.
<br>
<br>
When these regression problems are extended to time series tasks, a number of time specific complexities arise. As an example, lets imagine the analysis of
the management of blood sugar level in diabetes patients. In people with diabetes, the pancreas can't effectively regulate blood sugar levels. Therefore 
these levels must be controlled by insulin injections and a special diet. The challenge for many people, is that the relationship between the input (insulin) and
output(blood sugar) is extremely complex. The effects of the insulin can be observed after 15, 20  or "t" minutes depending on a number of factors, 
many of which are unknown. Because of the varying time delays, the actual effect can't be easily determined. Its hard to differentiate the effect from other factors and accurately determine how much one should take.
<br>
<br>
Doing inference on this type of problem can be really challenging. Typical regression models require a fixed alignment between cause and effect.
To model this problem, we'd need to assume that the effect occurs after some fixed time 't' which can be inferred from the data. There are a number of models that
allow us to do this. However, there is a significant issue with these models. If there is any uncertainty in the parameter 't' (t changes or is noisy) the effect will be significantly attenuated.
<br>
<br>
Consider the simple example below where the effect of the input is 1, we can see this clearly in the picture. The observed input is given by the red line, the blue line is when the effect actually occurs. 
The first effect happens 1 time point after the input. The second effect happens at the same time as the input. A fixed time delay isn't valid in this case because the time shifts differ.
<br>
<br>
-- image placeholder --
<br>
<br>
If we are to model this case using a fixed time delay, we would estimate the effect of the input to be 0.5. This is because only one of the points is aligned. Obviously, this isn't ideal, we want the parameter estimates to be as close the real values as possible regardless of any noise in the lag structures. There are a number of tricks to mitigate this problem e.g aggregation, however in complicated multivariate cases, this is just not feasible.
For this project, we experiment with regression models which can handle these stochastic time delay structures. Firstly, basic univariate linear regression, then multivariate,
linear and eventually non-linear methods.
<br>
<br>

<h3><b>Methodology</b></h3>
We consider the problem in a similar way to the typical error-in-variables (EIV) regression. Standard regression analyses (and machine learning models) define the loss function
with respect to errors in the y axis only. For EIV, errors are considered in both the y-axis and the x-axis. This often arises when there are measurement errors in the 
independent variable eg. because your physical measurements have some degree of random error.
<br>
<br>
A visualiation of EIV regression from the wikipedia page:
(https://en.wikipedia.org/wiki/Total_least_squares#/media/File:Total_least_squares.svg)
<br>
<br>
For our problem we assume that we have errors in the y-axis and the t-axis. That is, there are random prediction errors and random errors in the time domain. 
<br>
<br>
Assume we have a series x(t) where we want to determine to the functional relationship f to some other variable y.
There is a true x(t), in principle it could be anything. But we observe x(t + tau) where tau is some shift in time for each non-zero impulse in the series. 
We also make the assumption that the value of tau is not constant, but rather a random draw from some distribution (eg discrete gaussian, poisson). 
<br>
<br>
<b>Tau ~ N(u,tsd) or tau ~ Pois(lambda) </b>
<br>
and
<br>
<b>e ~ N(u, sd) </b>
<br>
<br>
Firstly, we take the input series x(t) and decompose it into its constituent non-zero components.
So, if x(t) is a vector given by [0,0,1,0,1,0] then we decompose the vector into X(t) = [[0,0,1,0,0,0],[0,0,0,0,1,0]] where each impulse is treated separately.
<br>
<br>
Then we define the relationship in the following form:
<br>
<br>
<b>y(t)= f(X(t + Tau)) + e</b>
<br>
<br>
We want to find the function f() which maximises the joint likelihood of both Tau and e, the time-domain likelihood and the error likelihood respectively.
For simplicity, assume that the tau and error distributions are independent. To begin with the values of each individual time shift (tau) are not known to us.  In addition, the size of prediction error e can only be determined if each tau is known (because for each time shift there is 
a different prediction and prediction error). This means that we need to estimate all the individual time shifts from the data.
<br>
<br>
<h3><b>Algorithm</b></h3>
<br>
<br>
Firstly, we define some initial parameters to be estimated for the function f. For example, lets start with the simple univariate linear model where the f(x(t)) is 
parameterised by B, error mean and sd. We define some initial starting values for each of these parameters.
<br>
<br>
Now we want to find the best possible time shift (tau) for each input impulse in X(t). It stands to reason that the best possible time shift would be the one that is not too far away from the observed impulse 
and also gives the best possible prediction. In this example, we can get the prediction y by simply multiplying the shifted value by its parameter B. From there we can calculate the likelihood estimate for time shift + prediction error.
In principle, we can then try a number of values of tau (i.e. optimise) to maximise the likelihood for this impulse.
However, we must also consider that the impulses in X(t) are not independent from each other. After shifting, its possible that two or more effects can occur simultaneously.
This could be particularly problematic if there are multiple impulses within a short period of time, or the impulses have a distributed effect over multiple time points.
As an example consider the series x=[0,0,1,1,0,0] with Tau = [1,0] and B = 1. For this case, X(t + tau) = [[0,0,0,1,0,0],[0,0,0,1,0,0]] and the effect is therefore y = [0,0,0,2,0,0]. 
We need to consider the effects at the same time to accurately calculate the likelihood.
<br>
<br>
As shown above, if the impulses are not independent, then we need to consider time shifts (taus) at the same time. Therefore, we treat the problem of finding the best time shifts as a discrete optimisation problem with constraints.
In principle, a number of methods would would work for this step. In the code, we use derivative-free optimization provided by the library Rbfopt,
https://github.com/coin-or/rbfopt. Constraints are calculated based on sequence length and the second moment of the time shift distribution (larger time shifts are highly unlikely and so we apply some reasonable bounds on them). Very low likelihoods for time shifts are remoed from the solution space.
Once the optimal set of time shifts is found for a given B, we then save the maximum likelihood for this parameter set.
<br>
<br>
Lastly, we optimise over the set of parameters B, error mean and sd, iteratively alternating between the time shift optimisation and the parameter optimisation. For the parameter optimisation,
typical methods can be used such as gradient descent, genetic algorithms or annealing. Through optimisation, we find the best fit for the values of B, error mean and sd. The accuracy of the final parameter estimate is based on the ratio of the y-axis error and the effect size B*X(t). As e/B*X(t) tends to infinity, the parameter estimate B tends to the standard linear regression coefficient.
On the other hand as e/B*X(t) tends to zero, the estimated value of B tends to the correct value. Therefore the result is bounded in the worst case by standard regression estimates. (** not proven, only observed in experiments**)
This also means there is no gaurantee on recovering the exact time shifts, only that we obtain a better estimate than standard regression.
<br>
<br>

<h3><b>Optimisation</b></h3>
As the length of x(t) increases and more and more impulses are introduced, the size of the decomposed matrix X(t) becomes huge. Therefore we make the assumption that 
impulses further away from each other are functionally independent. That is, because they are far away from each other, the likelihood of an overlap in their effects is very very small. Therefore, the matrix X(t) is broken into a number of segments, where each segment is 
composed of a smaller number of dependent impulses. Each of these segments can optimised independently, in parallel, with little impact on the final function estimate.  This makes
the inner optimisation procedure tractable for longer length sequences.
<br>
<br>
<h3><b>Additional Notes</b></h3>
<br>
<br>
<h5>Global vs Local Maxima</h5>
The overall likelihood estimates are based on the inner optimisation procedure, in which the global maximum is not guaranteed to be found. This is especially true for
sequences where there are a large number of impulses within a short period of time. For example consider the sequence x(t) = [0.5,0.6,0.4,1,2,1,0.3,0.2,0.5,0.8], where each impulse can shift between -2 and +2 positions.
Ignoring shifts past the edges, we have  5^t different combinations for tau. So the inner optimisation must find the optimum in a space of nearly a million potential combinations. 
In these cases, the resulting maximum likelihood estimate is likely to be close to the real value but not exactly equal to it and the outer optimisation loop will be noisy.
Its clear as the density (in time) of impulses increases, the accuracy will also decrease. This method works when the problem has a limited number of impulses within a given timeframe.
<br>
<br>
<h5>Bayesian Methods</h5>
Bayesian regression variations would be especially useful for these problems. Firstly, as indicated above as e/BX(t) tends to infinity our coefficient estimates tend to the standard regression values.
If we have some physical intuition about the time shift process, we can introduce a prior on the shift distribution and get better estimates. Bayesian priors
could also help to mitigate any overfitting issues. In addition, knowledge of the full posterior would be useful when analysing effect sizes. Consider the case of two parameters, with opposite effect directions situated closely in time. Because of the time shifts and noise, we can imagine scenarios where, with high likelihood, the estimates have flipped signs (collinearity in time?). Also, for a series with a low number of input impulses, the standard regression likelihood would be close to the time shifted likelihood. The posterior could have a couple of maxima, one local maxima near the standard regression estimate and one at the global time shifted maxima.
<br>
<br>
<h5>New Prediction Mechanisms</h5>
Essentially, existing prediction mechanisms predict the average across time shifts. Even though the loss function is equivalent, sometimes it just doesnt make logical sense.
Eg. do a regression to predict how much coffee I will drink between 9am and 9:30am at work. I set off every day in my car at exactly 8:30 and it takes me about 30 minutes (+- tau) to get to work. When I get to work, I have a coffee. A typical regression might predict 0.5 coffees between 8:30 and 9 and 0.5 coffees between 9 and 9:30. Alternative prediction forms such as: 'When he gets to work, he will have 1 coffee' might be more useful.


