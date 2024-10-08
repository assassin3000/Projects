Regression modelling
================
Kryštof Tomsa
2024-10-06

This project aims to analyze student performance in math exam. The data
was downloaded from Kaggle:
<https://www.kaggle.com/datasets/spscientist/students-performance-in-exams>

# Exploratory Data Analysis

Our dataset has 1000 observations and 8 variables (including our
dependent variable). We can inspect our data below:

    ##     gender          race.ethnicity     parental.level.of.education
    ##  Length:1000        Length:1000        Length:1000                
    ##  Class :character   Class :character   Class :character           
    ##  Mode  :character   Mode  :character   Mode  :character           
    ##                                                                   
    ##                                                                   
    ##                                                                   
    ##     lunch           test.preparation.course   math.score     reading.score   
    ##  Length:1000        Length:1000             Min.   :  0.00   Min.   : 17.00  
    ##  Class :character   Class :character        1st Qu.: 57.00   1st Qu.: 59.00  
    ##  Mode  :character   Mode  :character        Median : 66.00   Median : 70.00  
    ##                                             Mean   : 66.09   Mean   : 69.17  
    ##                                             3rd Qu.: 77.00   3rd Qu.: 79.00  
    ##                                             Max.   :100.00   Max.   :100.00  
    ##  writing.score   
    ##  Min.   : 10.00  
    ##  1st Qu.: 57.75  
    ##  Median : 69.00  
    ##  Mean   : 68.05  
    ##  3rd Qu.: 79.00  
    ##  Max.   :100.00

We have 3 continuous variables (again including the dependent variable)
and 5 nominal variables. Let’s now look at the variables in more detail.

![](student_regression_git_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

The independent variable (math score) looks more or less normally
distributed, but there are some outliers. For now, we will leave them in
our data.

![](student_regression_git_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

This graphs showd us boxplots of math score for both females and males.
Although, the males seem to perform better on average, the difference is
not that big.

![](student_regression_git_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Now have have a graph of boxplots for every race/ethnicity. In this case
there definitely are differences, for example the difference in medians
between group A and E is about 15 points.

![](student_regression_git_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

This graph shows the distribution of math scores for different levels of
parental education. Again we can observe differences in medians, but
this time there are also some differences in ditributions.

![](student_regression_git_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

This graph is focused on lunch prices. The student (or parents) can
either buy the lunch for a full price, have the price reduced or have
the lunch entirely for free. This will probably have a correlation with
the family’s income.

![](student_regression_git_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

The last nominal attribute we will look at is whether the student
finished the test preparation course. Unsurprisingly the median score
for the students that finished the course is higher.

Now we will inspect the correlation between the math and reading score:

![](student_regression_git_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

The correlation is pretty high. What about the math score versus writing
score?

![](student_regression_git_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

Again very high correlation. That could mean, that writing and reading
scores are also correlated.

![](student_regression_git_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

And they are! The correlation is 0.955. However, this means, that we
have a problem, because high correlation could cause our estimates to
have a high variance and therefore not be reliable. We can fix this by
creating a new attribute that will combine the information from the two
attributes and we will use in our model instead of them. In this case,
we decided this feature will be the mean of the scores.

# Modelling - 1

We will estimate our model as:
$$math.score = \beta_{0} + \beta_{1}*gender + \beta_{2}*race.ethnicity + \beta_{3}*parental.level.of.education + \beta_{4}*lunch + $$
$$\beta_{5}*test.preparation.course + \beta_{6}*read.write.mean + \beta_{7}*gender*read.write.mean + u$$

    ## 
    ## Call:
    ## lm(formula = math.score ~ gender + race.ethnicity + parental.level.of.education + 
    ##     lunch + test.preparation.course + read.write.mean + gender:read.write.mean, 
    ##     data = data)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -17.6658  -3.8033   0.0636   3.3815  14.5149 
    ## 
    ## Coefficients:
    ##                                               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                  -12.72120    1.49606  -8.503  < 2e-16 ***
    ## gendermale                                    16.40227    1.70216   9.636  < 2e-16 ***
    ## race.ethnicitygroup B                          0.84078    0.70000   1.201   0.2300    
    ## race.ethnicitygroup C                          0.26295    0.65630   0.401   0.6888    
    ## race.ethnicitygroup D                          0.56226    0.67223   0.836   0.4031    
    ## race.ethnicitygroup E                          5.09643    0.74575   6.834 1.45e-11 ***
    ## parental.level.of.educationbachelor's degree  -0.76505    0.61968  -1.235   0.2173    
    ## parental.level.of.educationhigh school         0.31712    0.53906   0.588   0.5565    
    ## parental.level.of.educationmaster's degree    -1.59044    0.80055  -1.987   0.0472 *  
    ## parental.level.of.educationsome college        0.44308    0.51372   0.862   0.3886    
    ## parental.level.of.educationsome high school    0.26369    0.55270   0.477   0.6334    
    ## lunchstandard                                  3.44898    0.37498   9.198  < 2e-16 ***
    ## test.preparation.coursenone                    2.83704    0.37991   7.468 1.80e-13 ***
    ## read.write.mean                                0.98231    0.01782  55.126  < 2e-16 ***
    ## gendermale:read.write.mean                    -0.05378    0.02438  -2.205   0.0277 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 5.421 on 985 degrees of freedom
    ## Multiple R-squared:  0.874,  Adjusted R-squared:  0.8722 
    ## F-statistic: 487.9 on 14 and 985 DF,  p-value: < 2.2e-16

Now that we have our model, let’s interpret the results above. First,
look at the $adjusted R^2$. We can see, that our model explained 87.2%
of total variance in our data. That’s pretty good.

Now we can interpret the individual coefficients (estimates). For
example the coefficient for test preparation course variable tell us,
that other variables being held equal the students without the course
score on average 2.84 more points than those with the course. This is an
interesting finding, because the boxplots in previous sections told us
that the math score is higher for students with the course.

We also included interaction term in our model. This has different
interpretation than other coefficients. In the term we included our
gender variable and variable of mean of writing and reading scores, this
basically means that we expect that genders have different effect on the
mean of writing and reading scores. If we want to get the math score for
women we just use the $\beta_{13}$ coefficient, for men we however need
to use $\beta_{1} + (\beta_{13} + \beta_{14}) * read.write.mean$.

Before we move forward though, we should check for some assumptions,
that make our model possible. Let’s try to look at the issue of
correlation between the independent variables, also called
multicollinearity. Although we probably dealt with this problem by
creating the new variable, it is still better to check.

    ##                                  GVIF Df GVIF^(1/(2*Df))
    ## gender                      24.613829  1        4.961233
    ## race.ethnicity               1.073862  4        1.008948
    ## parental.level.of.education  1.113295  5        1.010790
    ## lunch                        1.095486  1        1.046655
    ## test.preparation.course      1.128726  1        1.062415
    ## read.write.mean              2.341144  1        1.530080
    ## gender:read.write.mean      22.809045  1        4.775882

These are the variance inflation factors. The problem is, that there is
no high or low value (sometimes we can find in articles that value 10 or
5 is high enough to conclude we have a problem with multicollinearity,
but there is no scientific proof for that). Nevertheless, the values are
pretty low and the high value between the gender and interaction term is
expected and should not cause any problem.

Now we can look at the problem of heteroskedasticity. This is important
for hypothesis testing.

![](student_regression_git_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

    ## 
    ##  studentized Breusch-Pagan test
    ## 
    ## data:  model
    ## BP = 19.598, df = 14, p-value = 0.1433

From the graph it seems that the errors are homoskedastic. We can also
use Breusch-Pagan (or White) test to test for homoskedasticity. This
test has following null and alternative hypothesis:
$$H_{0} = homoskedasticity$$ $$H_{1} = heteroskedasticity$$ From the
p-value we cannot reject the null hypothesis. However, the p-value is
pretty small, so we will behave as we have the heteroskedasticity in our
model nevertheless. We deal with this by adjusting the variance of our
coefficients.

    ## Call: lm(formula = math.score ~ gender + race.ethnicity + parental.level.of.education + lunch + test.preparation.course + read.write.mean + gender:read.write.mean, data = data)
    ## Standard errors computed by vcm 
    ## 
    ## Coefficients:
    ##                                               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                  -12.72120    1.47316  -8.635  < 2e-16 ***
    ## gendermale                                    16.40227    1.68755   9.720  < 2e-16 ***
    ## race.ethnicitygroup B                          0.84078    0.73337   1.146   0.2519    
    ## race.ethnicitygroup C                          0.26295    0.69378   0.379   0.7048    
    ## race.ethnicitygroup D                          0.56226    0.71268   0.789   0.4303    
    ## race.ethnicitygroup E                          5.09643    0.74749   6.818 1.61e-11 ***
    ## parental.level.of.educationbachelor's degree  -0.76505    0.64546  -1.185   0.2362    
    ## parental.level.of.educationhigh school         0.31712    0.54949   0.577   0.5640    
    ## parental.level.of.educationmaster's degree    -1.59044    0.66468  -2.393   0.0169 *  
    ## parental.level.of.educationsome college        0.44308    0.52352   0.846   0.3976    
    ## parental.level.of.educationsome high school    0.26369    0.54671   0.482   0.6297    
    ## lunchstandard                                  3.44898    0.37316   9.243  < 2e-16 ***
    ## test.preparation.coursenone                    2.83704    0.38298   7.408 2.76e-13 ***
    ## read.write.mean                                0.98231    0.01644  59.749  < 2e-16 ***
    ## gendermale:read.write.mean                    -0.05378    0.02402  -2.239   0.0254 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard deviation: 5.421 on 985 degrees of freedom
    ## Multiple R-squared: 0.874
    ## F-statistic: 537.8 on 14 and 985 DF,  p-value: < 2.2e-16 
    ##     AIC     BIC 
    ## 6235.42 6313.94

This is our new model summary with the adjusted with variances. Now we
can test the hypotheses. Reader can read the p-values from the above
table and the stars on the right (1 or more stars means we reject the
null hypothesis on 95% significance level, meaning the variable is
statistically significant). We might be also interested in testing the
joint significance using F-tests for our nominal variables.

    ## Linear hypothesis test
    ## 
    ## Hypothesis:
    ## race.ethnicitygroup B = 0
    ## race.ethnicitygroup C = 0
    ## race.ethnicitygroup D = 0
    ## race.ethnicitygroup E = 0
    ## 
    ## Model 1: restricted model
    ## Model 2: math.score ~ gender + race.ethnicity + parental.level.of.education + 
    ##     lunch + test.preparation.course + read.write.mean + gender:read.write.mean
    ## 
    ## Note: Coefficient covariance matrix supplied.
    ## 
    ##   Res.Df Df      F    Pr(>F)    
    ## 1    989                        
    ## 2    985  4 26.761 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

We see that the race/ethnicity is significant. Let’s try test for
parental level of education.

    ## Linear hypothesis test
    ## 
    ## Hypothesis:
    ## parental.level.of.educationbachelor's degree = 0
    ## parental.level.of.educationhigh school = 0
    ## parental.level.of.educationmaster's degree = 0
    ## parental.level.of.educationsome college = 0
    ## parental.level.of.educationsome high school = 0
    ## 
    ## Model 1: restricted model
    ## Model 2: math.score ~ gender + race.ethnicity + parental.level.of.education + 
    ##     lunch + test.preparation.course + read.write.mean + gender:read.write.mean
    ## 
    ## Note: Coefficient covariance matrix supplied.
    ## 
    ##   Res.Df Df      F  Pr(>F)  
    ## 1    990                    
    ## 2    985  5 2.5465 0.02667 *
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Again we can conclude, that the variable is statistically significant.

Lastly, instead of point estimates we might want to present confidence
interval. Confidence intervals offer more information than tests (which
we can still do with confidence intervals).

    ##                                                  Estimate       2.5 %       97.5 %
    ## (Intercept)                                  -12.72120415 -15.6570436 -9.785364734
    ## gendermale                                    16.40227217  13.0619904 19.742553923
    ## race.ethnicitygroup B                          0.84078385  -0.5328806  2.214448264
    ## race.ethnicitygroup C                          0.26295103  -1.0249642  1.550866259
    ## race.ethnicitygroup D                          0.56226038  -0.7569101  1.881430831
    ## race.ethnicitygroup E                          5.09643221   3.6329981  6.559866366
    ## parental.level.of.educationbachelor's degree  -0.76505177  -1.9810994  0.450995897
    ## parental.level.of.educationhigh school         0.31711657  -0.7407221  1.374955209
    ## parental.level.of.educationmaster's degree    -1.59044210  -3.1614283 -0.019455868
    ## parental.level.of.educationsome college        0.44308432  -0.5650358  1.451204468
    ## parental.level.of.educationsome high school    0.26369099  -0.8209074  1.348289401
    ## lunchstandard                                  3.44897818   2.7131248  4.184831524
    ## test.preparation.coursenone                    2.83704105   2.0915075  3.582574601
    ## read.write.mean                                0.98230730   0.9473392  1.017275370
    ## gendermale:read.write.mean                    -0.05377655  -0.1016262 -0.005926897

# Modelling - 2

In articles it is more common to use percentages for quantifying the
effect rather than original values. We can do this by logarithmic
transformation of the dependent variable. For this model we will get rid
of outliers and present the model’s summary with adjusted variances.

    ## Call: lm(formula = log(math.score) ~ gender + race.ethnicity + parental.level.of.education + lunch + test.preparation.course + log(read.write.mean), data = data_cleaned)
    ## Standard errors computed by vcm_2 
    ## 
    ## Coefficients:
    ##                                               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                   0.023495   0.085326   0.275   0.7831    
    ## gendermale                                    0.195037   0.006079  32.086  < 2e-16 ***
    ## race.ethnicitygroup B                         0.016443   0.013256   1.240   0.2151    
    ## race.ethnicitygroup C                         0.001900   0.012499   0.152   0.8792    
    ## race.ethnicitygroup D                         0.008870   0.012857   0.690   0.4904    
    ## race.ethnicitygroup E                         0.071972   0.012729   5.654 2.05e-08 ***
    ## parental.level.of.educationbachelor's degree -0.008262   0.010846  -0.762   0.4464    
    ## parental.level.of.educationhigh school        0.006353   0.009430   0.674   0.5007    
    ## parental.level.of.educationmaster's degree   -0.020395   0.011118  -1.834   0.0669 .  
    ## parental.level.of.educationsome college       0.013455   0.008562   1.571   0.1164    
    ## parental.level.of.educationsome high school   0.003263   0.009461   0.345   0.7302    
    ## lunchstandard                                 0.061639   0.006580   9.368  < 2e-16 ***
    ## test.preparation.coursenone                   0.039541   0.006370   6.207 7.98e-10 ***
    ## log(read.write.mean)                          0.942981   0.019523  48.300  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard deviation: 0.09256 on 976 degrees of freedom
    ## Multiple R-squared: 0.8425
    ## F-statistic:   258 on 13 and 976 DF,  p-value: < 2.2e-16 
    ##      AIC      BIC 
    ## -1886.74 -1813.27

Now if we want to get the effect of a variable on math score quantified
in percentages we need to calculate $[exp(\beta_{j})-1]*100$. Also, in
this model we did not include the interactions term, but we transformed
the mean of writing and reading scores to have logarithmic scale. This
coefficient is directly interpreted as $1\%$ change in the mean of the
reading and writing scores is $x\%$ change in math score.

It is worth noting that the more transformations we use, the harder it
is to interpret the results.

# Conclussion

The model shows us how different variable have different effects on
student’s performance in math test. All of our attributes were
statistically significant. From the coefficients we can see that gender
has a huge difference on the student’s performance together with
race/ethnicity group E. Lunch price and test preparation course had a
medium effect on the performance. The mean of the writing and reading
scores had a big effect on math score, meaning if student is good at
writing and reading, he is also good at math. Although the parental
level of education was statistically significant, the effect on
performance was very small.

We can (or actually we should) of course question the causal relations
between the dependent and independent variables. For example, the mean
of reading and writing score in this case kind of measure the student’s
intelligence and consistence in learning. The lunch price is definitely
based on some economic well-being of the student’s family and there are
economic factors that could also influence our race/ethnicity variable.
Also it would be interesting to find why males have high score in math
than women. This of course does not immediately that females are worse
in math. For example, it could be caused by different interests between
genders. However, any of these factors were not present in our dataset,
so we can only speculate.
