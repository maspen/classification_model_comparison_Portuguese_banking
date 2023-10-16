# Comparison of 4 ML Classification Models using Portuguese Banking Promotion

This project compares 4 ML models (Logistic Regression, K Nearest Neighbor, Decision Trees, and Support Vector Machines) using a Portuguese banking's promotional campaign data. The bank used these campaigns to increase its financial presence by offering long-term deposit accounts to targeted clients; accounnts that would yield favorable return interest rates for the client. Furthermore, since this data contains direct promotional data (collected via agent-client phone contact or web), the desired outcome is to determine an efficient way to identify viable clients who can qualify for a promotion.

Section 1 is derived from ![](/prompt_III_1.ipynb) and section 2 from ![](/prompt_III_2.ipynb)

# Section 1

## The Data

The original data contains 41,118 entries with 21 features, the last of thich is a 'y/n' categorical colum identifying if (a) the client received the promotion or (b) did not receive the promotion due to a lack of qualifying factors OR has not been contacted yet. The data contains a mix of numeric and categorical features. Below is the 'y/n' comparison of customers:

![Promotion Customer Comparison](/images/customer_comparison.jpg)

Roughly 88% of the customers did not receive a promotion therefore this is an in balanced data set.

## Business Objective

The business objective is to increase the efficiency of bank's direct campaigns (phone, web) for long-term deposit subscriptions & reduce the number of contact.

## Engineering Features

The first task is to take the first 7 features and engineer them for use in the 4 ML models mentioned above. These features are:

```
 #   Column     Non-Null Count  Dtype 
---  ------     --------------  ----- 
 0   age        41188 non-null  Int64 
 1   job        41188 non-null  string
 2   marital    41188 non-null  string
 3   education  41188 non-null  string
 4   default    41188 non-null  string
 5   housing    41188 non-null  string
 6   loan       41188 non-null  string
```
The first step is to convert the nominal (`string`) columns/features into numeric equivalents. For the non binary features (eg. job, marital, education), One Hot Encoding was used. For the binary features (eg. default, housing, loan) `no` values were converted to 0, `yes` to 1 and `unknown` to -1. The result is a dataset with 29 columns:

```
#   Column                         Non-Null Count  Dtype 
---  ------                         --------------  ----- 
 0   age                            41188 non-null  Int64 
 1   default                        41188 non-null  int64 
 2   housing                        41188 non-null  int64 
 3   loan                           41188 non-null  int64 
 4   job_admin.                     41188 non-null  uint8 
 5   job_blue-collar                41188 non-null  uint8 
 6   job_entrepreneur               41188 non-null  uint8 
 7   job_housemaid                  41188 non-null  uint8 
 8   job_management                 41188 non-null  uint8 
 9   job_retired                    41188 non-null  uint8 
 10  job_self-employed              41188 non-null  uint8 
 11  job_services                   41188 non-null  uint8 
 12  job_student                    41188 non-null  uint8 
 13  job_technician                 41188 non-null  uint8 
 14  job_unemployed                 41188 non-null  uint8 
 15  job_unknown                    41188 non-null  uint8 
 16  marital_divorced               41188 non-null  uint8 
 17  marital_married                41188 non-null  uint8 
 18  marital_single                 41188 non-null  uint8 
 19  marital_unknown                41188 non-null  uint8 
 20  education_basic.4y             41188 non-null  uint8 
 21  education_basic.6y             41188 non-null  uint8 
 22  education_basic.9y             41188 non-null  uint8 
 23  education_high.school          41188 non-null  uint8 
 24  education_illiterate           41188 non-null  uint8 
 25  education_professional.course  41188 non-null  uint8 
 26  education_university.degree    41188 non-null  uint8 
 27  education_unknown              41188 non-null  uint8 
 28  y                              41188 non-null  string
```

## Train/Test Split

The train/test split was carried out using 0.33 as the `test_size`, `random_state` of 42 and `stratify` on feature/column `y`.

## Baseline

Two baselines were explored. The first rounded and normalized the `value_counts` of column y, yielding 89%. The second used the `DummyClassifier` which yielded:

```
'accuracy': 0.8873684984918708, 'precision': 0.8873684984918708, 'recall': 1.0, 'f1': 0.9403235236795946
```

### Simple Models

The first pass at comparing the performance of the 4 ML models was to train, fit and score them with default (no) parameters. Below is a summary table illustrating their performance:

![4 Simple ML Model Comparison](/images/1_problem_10_model_comparison.png)

Here we see that the Support Vector Machines (SVC) model had the worst train(ing) time, KNeighborsClassifier the best, Decision Tree Classifier had the best train(ing) accuracy and the best test(ing) accuracy went to LogisticRegression.

## ROC Plot - Simple Models

The ROC plot identifies LogisticRegression as having the most sensitivity or recall for the dataset.

![ROC Curve of the 4 Simple ML Models](/images/roc_untuned.jpg)

## More Feature Engineering

In the ![accompanying paper](/CRISP-DM-BANK.pdf), we are told that the data set includes `Sex` and that the ration of male and female is who did and did not receive the proportion is negligible:

![Male to Female Promotions](/images/sex-male-to-female.jpg)

There is not enough variance in this feature therefore, it can be removed. This is already true in the dataset provided.

### contact

```
cellular 26144 
telephone 15044
```
These are the value counts for how the client was contacted in regards to the promotion. This may not be a very significant feature except to indicate that customers tend to use (or receive; meaning are more likely to take a call on their cell phone vs. at home) their cell phones rather than their land lines. This could also indicate that it is common for people to only have a cell phone.

### poutcome

```
nonexistent    35563
failure         4252
success         1373
```
"outcome of the previous marketing campaign". Suggests that the campaigns have not been very facorable. However, 'nonexistent' suggests that either good record-keeping is not in place & should be improved and/or that these are potential clients who have not been contacted and need reaching out to.

### default (on a loan)

```
no         32588
unknown     8597
yes            3
```

Is a crucial metric. Only 3 have but there are ~8.5K where this information is unknown. Perhaps before putting in the effort of contacting them (assuming that they have not and 'default' is a deal-breaker for the bank), it would be helpful to get this information before making efforts to contact the clients.

### loan (has personal loan?)

```
no         33950
yes         6248
unknown      990
```
Probably one of the most important factors to consider. Why do most customers not have a loan (what factors play
into this? Do they have bad credit, defaulted on a loan, etc.) or have they not been contacted yet? If the later,
and all bank rubrics are met, these customers should be focused on.

# Section 2

In this section, the 4 ML models were used but were trained after determining their best, respective hyperparameters and trained, tested and validated on the entire set of features.

In contrast to the first section, in this section, ordinal features were encoded using the `OrdinalEncoder`. The `y` column was encoded using the `LabelEncoder`.

### Feature Selection

Although not a very large data set, it was still interesting to investigate feature importance. `SequentialFeatureSelector` was used for this purpose and selected all columns except `nr.employed`. If we plot this column using the count of occurances, we see that it is a left-skewed distributions:

![Number Employees Value Counts](/images/num_employees_value_counts.jpg)

This feature was removed and the data was split similarly as before except that the `y` column was not stratified.

Two methods were created. The first (`run_grid_search_return_cs_results`) was used to run various combinations of hyperparameters per ML  model. The second (`score_model`) was used to extract fit time, training accuracy, test accuracy, precision, recall and F1 scores. The former also calculated the training score, test score, average fit time and returned the best hyperparameters for each ML model.

### Best Hyperparameters

LogisticRegression: `max_iter` = 100, `solver` = 'lbfgs'
KNeighborsClassifier: `n_neighbors` = 9
DecisionTreeClassifier: `criterion` = 'gini', `max_depth` = 3
SVC: `C` = 0.8600000000000001, `kernel` = 'rbf', `gamma` = 'scale'

### Tuned Model Performance

Using the above-mentioned hyperparameters, the 4 ML models were added to a pipeline, prefaced by the `StandardScaler`, fitted and scores. Below are the results:

![Fitted Model Results](/images/tuned_model_comparison_DF.png)

Below is the ROC curve for the tuned ML models:

![ROC for Tuned ML Models](/images/roc_tuned.jpg)

We see a marked inprovement in the models' performance when using the optimal hyperparameters. Although SVC starts off exceeding LogisticRegression, it is the later that appears to be better-performant. Another drawback of SVC is the 'fit time' which in other runs exceeded 1 minute.

Below are the confusion matrices for the 4 models:

![Confusion Matrices for 4 ML Models](/images/4_model_confusion_matrices.jpg)

At this point, we need to ask what is the most desirable characteristic of the model. Since this is a classification problem (should the customer receive the promotion: y/n). Measures like mean squared error for LogisticRegression, for example, would not suffice. Put in tabular form, below are the accuracy, precision, recall, and f1 scores from the confusion matrices:

![Accuracy, Precision, Recall, F1](/images/accuracy_precision_recall_f1.jpg)

Aside from 'accuracy', these scores are significantly different than those calculated after fitting the parametrized models. 'Precision', 'Recall' and 'F1' do appear more realistic (unfortunately worse). Returning to the earlier question (for this business goal, what is the best measure?), 'accuracy' does not seem to paint the entire picture. We need a model that is the best at classifying a bank customer that is best-suited to receive a promotion. Depending on what sort of risk the bank is willing to take (eg. give someone who has previously defaulted in a loan - altough there are many nuances associated with such an event ...), I suggest that the F1 score be selected. It is a balance between 'precision' and 'recall' and is well-suited for datasets containing imbalanced proportions (larger proportions) of negative data. In this case, the 'y' feature containing ~80% 'no'.
