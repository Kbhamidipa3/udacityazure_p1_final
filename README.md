# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree. In this project, we built and optimized an Azure ML pipeline using the Python SDK and the provided Scikit-learn model. This model was tuned using Hyperparameter Tuning and is then compared to an Azure AutoML run.

## Summary
**The bankmarketing_train.csv dataset contains various metrics about potential customers that could eventually be utilizing the services of the bank. The objective of the project is to determine who is likely to end up becoming a customer based on the information available from the database.**

**Both Hyperparameter Tuning with HyperDrive and Automated ML methods were evaluated. While the difference was relatively smaller, the Automated ML method showed the best performance between the two with a “Best Accuracy” of 0.91709.**

## Scikit-learn Pipeline
**The high-level Scikit-learn Pipeline used to setup the pipeline is shown in the image below:**

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/Figure%201.jpg)

In this project, two different models were used to train and compared as shown in the image below:
![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/Figure%202.jpg)

### train.py module:
There are several steps involved in setting up and running the trained model, which is done in train.py for the HyperDrive model.
* Tabular data is first accessed from the following link and dataset is created:
https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv

* Then the data is converted to dataframe using Pandas and missing values are dropped.

* Subsequently the data is cleaned to get it ready for training the model. This involves translating categorical data into numerical data before fitting a machine learning model. 
* One hot encoding is one such method used where each category under a given parameter is split into it’s own column and assigned values ‘0’ or ‘1’ depending on whether an item doesn’t belong or belong under the given category respectively. Then all the individual columns are added back to the dataframe.

* Another method used to generate numerical data from categorical values is using key-value pairs from Dictionaries.


* Other columns that have binary values like “married” or “not married” are assigned values 1 and 0 respectively.

* A separate column is created for label values that we want to predict by popping it out from the dataframe.


* Then the data is split into train and test values using a certain split such that more data is assigned under “train” data to get accurate model fits. In the following code, 67% of the data is used for training and remaining 33% is used for testing. Setting random_state (42 in the current model) to a specified value can ensure that the data will return same results after each execution. 

* Logistic Regression model is used to fit the train data. And two parameters were chosen – Inverse of Regularization Strength (denoted by C) and Maximum Iterations (denoted by max_iter). These are tuned as explained in the next subsection.

 
* The accuracy of the model is then determined using the score method.



 ### Hyperparameter Tuning 
**Hyperparameter tuning method is used to tune the two parameters defined in train.py to achieve the best prediction accuracy- Inverse of Regularization Strength (denoted by C) and Maximum Iterations (denoted by max_iter). The former parameter, “C”, helps to avoid overfitting. The parameter max_iter dictates the maximum number of iterations for the regression model so that the model doesn't run for too long resulting in diminishing returns. These parameters are randomly sampled using the RandomParameterSampling method to understand the impact of the parameters on the output prediction accuracy. **

**Specifying an early stopping policy improves computational efficiency by terminating poorly performing runs. BanditPolicy was chosen as the early stopping policy in this project with slack factor and evaluation interval as the parameters. Every run is compared to the Best performing run at the end of the specified evaluation interval and in combination with the slack factor (allowed slack value compared to the best performing model) determines whether the run should be continued or terminated. **

## AutoML
**Unlike the Hypertuning method used earlier, Automated ML method automates the iterative tasks associated with the machine learning models thereby improving efficiency and productivity. Accuracy is used as the primary metric similar to the first method. Number of cross-validations is set to 5, meaning each training uses 4/5th of the data, while the remaining 1/5th of the data is used for validation. The final metric reported out is then the average of the five individual metrics.**

## Pipeline comparison
### ** Pipeline and accuracy differences between Hyperparameter tuning and Automated ML:
#### **Hyperparameter Tuning:
**As specified in Figure 1, in the Hyperparameter tuning method, the tabular data is split into test/train data using the train.py model and Scikit-learn is used to perform Logistic Regression. This is subsequently called in the Hyperparameter tuning code and the parameters are randomly sampled. The parameters seemed to however have no impact on the final accuracy as all the runs performed using different combinations of parameters yielded the same accuracy of 0.9105265.**

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/HP%201.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/HP%202.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/HP%203.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/HP%204.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/HP%205.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/HP%206.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/HP%207.jpg)

#### **Automated ML:
**On the other hand, in the case of Automated ML, cleaned data obtained from the train.py module was split into train and test data within the Automated ML code itself and the data was uploaded to a datastore. The Automated ML method evaluated 57 different runs and chose "VotingEnsemble" as the best performing model with an accuracy of 0.9170851. While the difference in accuracy is relatively smaller, Automated ML method definitely showed a higher accuracy than the Hyperparameter tuning method. This could be attributed to the superiority of the Automated ML method in sweeping through a more optimum space to find the best fit.**

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/AML%201.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/AML%202.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/AML%203.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/AML%204.jpg)

![GitHub Logo](https://github.com/Kbhamidipa3/udacityazure_p1_final/blob/main/images/AML%205.jpg)


**In summary comparing the two models, the accuracies are as follows:**

**Hyperparameter Tuning Accuracy | Automated ML Accuracy
------------ | -------------
0.9105265|0.9170851



## Future work
**In the future, the following improvements can be made to the models to potentially improve accuracy:**
* Try different train/test split ratios
* Explore other sampling methods
* Sample other parameters that haven't been tested in this project

## Proof of cluster clean up
**Cluster cleanup is included in the code.**
