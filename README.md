### Kaggle Competition | [Porto Seguro’s Safe Driver Prediction](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)


### 1. My Conclusion Analysis Report - Jupyter Notebook
* [Porto Seguro’s Safe Driver Analysis](https://nbviewer.jupyter.org/github/miedlev/kaggle-Porto-Seguro-s-Safe-Driver-Prediction/blob/main/Porto%20Seguro%E2%80%99s%20Safe%20Driver%20Prediction.ipynb)
* [Deleted picture 1('ps_reg_01', 'ps_reg_02', 'ps_reg_03')](https://github.com/miedlev/kaggle-Porto-Seguro-Safe-Driver-Prediction/blob/main/scatterplot1.png)
* [Deleted picture 1('ps_calc_01', 'ps_calc_02', 'ps_calc_03')](https://github.com/miedlev/kaggle-Porto-Seguro-Safe-Driver-Prediction/blob/main/scatterplot2.png)


* [Data Visualization Image](https://github.com/miedlev/kaggle---San-Francisco-Crime-Classfication/tree/main/Image)

Deleted picture 1('ps_reg_01', 'ps_reg_02', 'ps_reg_03')()


### 2. About Data :
* In this competition, you will predict the probability that an auto insurance policy holder files a claim.

In the train and test data, features that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation. The target columns signifies whether or not a claim was filed for that policy holder.


### 3. Process Introduction :
It is a competition that can be said to be Kaggle's introductory period and conducts a Python-based analysis. 

**[My focusing was on]** 
1. EDA - Focusing on dependent variable
2. Data_type finding & division
3. Missing Data arrangement & deletion
4. VarianceThreshold - finding varience & balance
5. Correlation & Density Analysis
6. One - Hot - Encoding
7. Feature engineering(Address, Datetime)
8. sparse matrix(csr_matrix)
9. Boosting Model Selection(Catboost)
10.Metrics - model_selection(Normalized Gini Coefficient )


**[Dependencies & Tech]:**
* [IPython](http://ipython.org/)
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org/)
* [SciKit-Learn](http://scikit-learn.org/stable/)
* [SciPy](http://www.scipy.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Matplotlib](http://matplotlib.org/)
* [Plotly](https://plotly.com/python/)
* [Folium](https://pypi.org/project/folium/)
* [StatsModels](http://statsmodels.sourceforge.net/)
* [LightGBM](https://lightgbm.readthedocs.io/en/latest/)
* [Catboost](https://catboost.ai/docs/concepts/python-installation.html)


### 4. Porto Seguro’s Safe Driver Prediction
Nothing ruins the thrill of buying a brand new car more quickly than seeing your new insurance bill. The sting’s even more painful when you know you’re a good driver. It doesn’t seem fair that you have to pay so much if you’ve been cautious on the road for years.

Porto Seguro, one of Brazil’s largest auto and homeowner insurance companies, completely agrees. Inaccuracies in car insurance company’s claim predictions raise the cost of insurance for good drivers and reduce the price for bad ones.

In this competition, you’re challenged to build a model that predicts the probability that a driver will initiate an auto insurance claim in the next year. While Porto Seguro has used machine learning for the past 20 years, they’re looking to Kaggle’s machine learning community to explore new, more powerful methods. A more accurate prediction will allow them to further tailor their prices, and hopefully make auto insurance coverage more accessible to more drivers.
