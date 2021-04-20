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

## Example
An example of a univariate linear TVS regression can be found at:
notebooks/1_univariate_example.ipynb

## TVS Regression Article
---
<h3><b>Abstract</b></h3>
Systems with stochastic time delay between the input and output present a number of unique challenges. 
Time domain noise leads to irregular alignments, obfuscates relationships and attenuates inferred coefficients. 
To handle these challenges, we introduce a maximum likelihood regression model that regards stochastic time delay as an 'error' in the time domain. For a certain subset of problems, by modelling both prediction \emph{and} time errors it is possible to outperform traditional models.
Through a simulated experiment of a univariate problem, we demonstrate results that significantly improve upon Ordinary Least Squares (OLS) regression.
<br>
<br>

The full article can be found at:
documentation/article.pdf
