# MSCS 634 – Lab 4: Regression Analysis with Regularization Techniques

**Author:** Ruchitha Musku  
**Course:** MSCS 634 – Advanced Big Data and Data Mining

## Purpose

This lab looks at how model complexity and regularization affect regression performance. Using the **scikit-learn Diabetes dataset** (442 patients, 10 baseline health measurements, target = disease progression after one year), I:

1. Explored and checked the data (distributions, correlations, missing values, outliers).
2. Built **Simple Linear**, **Multiple Linear** and **Polynomial** regression models.
3. Applied **Ridge (L2)** and **Lasso (L1)** regularization to linear and degree-2 polynomial features, and studied how the penalty strength `alpha` changes the models.
4. Evaluated every model with **MAE, MSE, RMSE and R²** on the same 80/20 train/test split and confirmed the results with 5-fold cross-validation.
5. Visualized fits, predicted-vs-actual plots, residuals, error-vs-complexity curves, R²-vs-alpha curves and coefficient paths.

## Repository Contents

| File | Description |
|---|---|
| `Lab4_Regression_Regularization.ipynb` | Full notebook with code, outputs, plots and analysis (Steps 1–6) |
| `README.md` | This summary |

**To run:** `pip install numpy pandas matplotlib seaborn scikit-learn jupyter`, then open the notebook and select *Run All*. The dataset ships with scikit-learn, so no download is needed.

## Results Summary (test set)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Lasso (poly deg 2)** | 41.34 | **50.75** | **0.514** |
| Ridge (poly deg 2) | **41.08** | 51.25 | 0.504 |
| Lasso (linear) | 42.84 | 53.43 | 0.461 |
| Ridge (linear) | 42.81 | 53.78 | 0.454 |
| Multiple Linear (10 features) | 42.79 | 53.85 | 0.453 |
| Polynomial, all features (deg 2) | 43.58 | 55.64 | 0.416 |
| Polynomial, BMI only (deg 6) | 51.12 | 63.08 | 0.249 |
| Simple Linear (BMI) | 52.26 | 63.73 | 0.233 |
| Polynomial, all features (deg 3) | 162.32 | 284.57 | −14.28 |

## Key Insights

- **Using more information matters most.** Moving from BMI alone (R² = 0.23) to all 10 features (R² = 0.45) gave the largest single improvement in the lab.
- **Polynomial features show the bias-variance trade-off clearly.** Degree-2 features raised training R² but lowered test R². Degree 3 (285 terms for 353 training rows) overfit severely, with test R² of −14. High-degree BMI polynomials became unstable at the edges of the data.
- **Regularization turns an overfitting model into the best one.** Ridge and Lasso applied to the *same* degree-2 features raised test R² from 0.42 to about 0.50–0.51. On the plain linear model the effect was small, because that model wasn't overfitting much to begin with.
- **Alpha controls the trade-off.** Small α behaves like ordinary least squares. Moderate α reduces variance and improves test performance. Large α underfits (Lasso eventually zeroes every coefficient and predicts the mean).
- **Ridge vs Lasso.** Ridge shrinks all coefficients but keeps every feature. Lasso performs feature selection: it kept only 15 of 65 polynomial terms and dropped `s2` from the linear model.
- **Dataset insights.** BMI, s5 (log triglycerides) and blood pressure are the strongest, most consistent predictors. s1 and s2 (total cholesterol and LDL, r ≈ 0.90) are strongly collinear, which makes their unregularized coefficients large and opposite-signed. Even the best model explains only about half of the variance, so disease progression clearly depends on factors outside these 10 measurements.

## Challenges and Decisions

- **Data cleaning.** The dataset has no missing values or duplicates. A few IQR outliers exist, but they are plausible clinical values, so I kept them rather than lose data from an already small dataset.
- **Pre-scaled features.** The features come mean-centered and scaled. I still placed a `StandardScaler` inside every pipeline (and again after `PolynomialFeatures`) so that the Ridge/Lasso penalties treat all terms equally.
- **Choosing the simple-regression feature.** I used BMI because it has the highest correlation with the target (r = 0.59).
- **Choosing alpha without data leakage.** The alpha sweeps use the test set only to *illustrate* behavior. The final Ridge/Lasso models choose α with `RidgeCV`/`LassoCV` using 5-fold CV on the training set only.
- **Small test set.** With only 89 test samples, single-split scores are noisy (for example, Ridge-poly beats plain linear on the test split but not in CV). I added 5-fold cross-validation to support the conclusions.
- **Numerical instability.** Single-feature polynomials of degree ≥ 13 produced nearly collinear, extreme-valued columns. I treated this as part of the overfitting story rather than hiding it.
- **Lasso convergence.** I raised `max_iter` to 50,000 so that Lasso converges at small α values.
