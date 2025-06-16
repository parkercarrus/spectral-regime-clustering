# ProbaSpectra: Cluster-Based Prediction of ETF Returns Using Spectral Covariance Features

## 📈 Overview

**ProbaSpectra** is a machine learning-driven algorithmic trading pipeline that forecasts ETF returns by learning from the covariance structure of its underlying equity constituents. It uses spectral features (eigenvalues of rolling covariance matrices) to identify patterns in the market regime, clusters historical conditions using Gaussian Mixture Models (GMMs), and predicts ETF performance using cluster-wise learned targets.

The strategy then simulates a simple long-short portfolio allocation based on predicted directionality and magnitude.

---

## 🔧 Pipeline Steps

1. **Data Retrieval**  
   Load historical equity and ETF price data.

2. **Feature Construction**  
   For each sliding window of days:
   - Compute covariance matrix of equity returns.
   - Extract top eigenvalues as features.

3. **Model Training**  
   Train a **GMM-based ClusterModel** on these spectral features. Each cluster stores its historical return profile.

4. **Prediction**  
   Use cluster probabilities to estimate expected ETF return.

5. **Simulation**  
   - Long if predicted return > `threshold`
   - Short if < `-threshold`
   - Hold otherwise  
   Calculate equity curve, CAGR, and Sharpe ratio.
