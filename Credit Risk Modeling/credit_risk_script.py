# Load libraries
#-------------------------------------------
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from optbinning import BinningProcess
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import copy
import re

matplotlib.use('TkAgg')

# Load data and prepare sample
#-------------------------------------------
data = pd.read_csv("C:/Users/kryst/OneDrive/Documents/Škola/Vysoká/5. semestr/Aplikovaná ekonometrie v bankovnictví/mortgage_sample.csv", index_col=0)

data_first_time = copy.copy(data[data["first_time"] == data["time"]])
data_first_time.drop(columns=["time", "mat_time", "first_time", "orig_time", "default_time", "payoff_time",
                              "status_time", "sample", "TARGET", "hpi_time", "interest_rate_time", "LTV_time"], inplace=True)
data_first_time.drop(columns=data_first_time.filter(like="orig_time", axis=1).columns, axis=1, inplace=True)

for i in data_first_time.columns:
    if "time" in i:
        data_first_time.rename(columns={i: re.sub(r"time", "first_time", i)}, inplace=True)

data = data.merge(data_first_time, on="id", how="left")

data = data[(data["sample"] == "public") & ((data["time"] - data["first_time"]) % 12 == 0)]

# converting target variable to integer
data["TARGET"] = data["TARGET"].astype(int)

# dropping unnecessary columns or giveaway features
data.drop(columns=["id", "default_time", "payoff_time", "status_time", "sample"], inplace=True)

# checking for missing values and data types
data.info()

# EDA
#-------------------------------------------
sns.countplot(x="TARGET", data=data)
plt.title("Distribution of defaults")
plt.xlabel("Default")
plt.ylabel("Count")
plt.show()

fig, ax = plt.subplots(1, 3)
sns.barplot(x="TARGET", y="balance_time", data=data[data["balance_time"]<=1500000], ax=ax[0])
ax[0].set_title("Average balance by default")
ax[0].set(xlabel="Default", ylabel="Average balance")
sns.histplot(data[data["balance_time"]<=1500000], x="balance_time", hue="TARGET", bins=30, ax=ax[1])
ax[1].set_title("Distribution of balance by default")
ax[1].set(xlabel="Balance", ylabel="Count")
sns.histplot(x="balance_time", hue="TARGET", data=data[data["balance_time"]<=1500000], bins=30, stat="percent", multiple="fill")
ax[2].set_title("Distribution of balance by default")
ax[2].set(xlabel="Balance", ylabel="% of bin count")
plt.show()

fig, ax = plt.subplots(1, 3)
sns.barplot(x="TARGET", y="LTV_time", data=data[data["LTV_time"]<=200], ax=ax[0])
ax[0].set_title("Average LTV by default")
ax[0].set(xlabel="Default", ylabel="Average LTV")
sns.histplot(data[data["LTV_time"]<=200], x="LTV_time", hue="TARGET", bins=30, ax=ax[1])
ax[1].set_title("Distribution of LTV by default")
ax[1].set(xlabel="LTV", ylabel="Count")
sns.histplot(x="LTV_time", hue="TARGET", data=data[data["LTV_time"]<=200], bins=30, stat="percent", multiple="fill")
ax[2].set_title("Distribution of LTV by default")
ax[2].set(xlabel="LTV", ylabel="% of bin count")
plt.show()

fig, ax = plt.subplots(1, 3)
sns.barplot(x="TARGET", y="hpi_time", data=data, ax=ax[0])
ax[0].set_title("Average HPI by default")
ax[0].set(xlabel="Default", ylabel="Average HPI")
sns.histplot(data, x="hpi_time", hue="TARGET", bins=15, ax=ax[1])
ax[1].set_title("Distribution of HPI by default")
ax[1].set(xlabel="HPI", ylabel="Count")
sns.histplot(x="hpi_time", hue="TARGET", data=data, bins=15, stat="percent", multiple="fill")
ax[2].set_title("Distribution of HPI by default")
ax[2].set(xlabel="HPI", ylabel="% of bin count")
plt.show()

fig, ax = plt.subplots(1, 3)
sns.barplot(x="TARGET", y="interest_rate_time", data=data[data["interest_rate_time"]<=15], ax=ax[0])
ax[0].set_title("Average interest rate by default")
ax[0].set(xlabel="Default", ylabel="Average interest rate")
sns.histplot(data[data["interest_rate_time"]<=15], x="interest_rate_time", hue="TARGET", bins=30, ax=ax[1])
ax[1].set_title("Distribution of interest rate by default")
ax[1].set(xlabel="Interest rate", ylabel="Count")
sns.histplot(x="interest_rate_time", hue="TARGET", data=data[data["interest_rate_time"]<=15], bins=30, stat="percent", multiple="fill")
ax[2].set_title("Distribution of interest rate by default")
ax[2].set(xlabel="Interest rate", ylabel="% of bin count")
plt.show()

fig, ax = plt.subplots(1, 3)
sns.barplot(x="TARGET", y="Interest_Rate_orig_time", data=data[data["Interest_Rate_orig_time"]<=15], ax=ax[0])
ax[0].set_title("Average interest rate by default")
ax[0].set(xlabel="Default", ylabel="Average interest rate")
sns.histplot(data[data["Interest_Rate_orig_time"]<=15], x="Interest_Rate_orig_time", hue="TARGET", bins=30, ax=ax[1])
ax[1].set_title("Distribution of interest rate by default")
ax[1].set(xlabel="Interest rate", ylabel="Count")
sns.histplot(x="Interest_Rate_orig_time", hue="TARGET", data=data[data["Interest_Rate_orig_time"]<=15], bins=30, stat="percent", multiple="fill")
ax[2].set_title("Distribution of interest rate by default")
ax[2].set(xlabel="Interest rate", ylabel="% of bin count")
plt.show()

fig, ax = plt.subplots(1, 3)
sns.barplot(x="TARGET", y="uer_time", data=data, ax=ax[0])
ax[0].set_title("Average UER by default")
ax[0].set(xlabel="Default", ylabel="Average UER")
sns.histplot(data, x="uer_time", hue="TARGET", bins=15, ax=ax[1])
ax[1].set_title("Distribution of UER by default")
ax[1].set(xlabel="UER", ylabel="Count")
sns.histplot(x="uer_time", hue="TARGET", data=data, bins=15, stat="percent", multiple="fill")
ax[2].set_title("Distribution of UER by default")
ax[2].set(xlabel="UER", ylabel="% of bin count")
plt.show()

fig, ax = plt.subplots(1, 3)
sns.barplot(x="TARGET", y="FICO_orig_time", data=data, ax=ax[0])
ax[0].set_title("Average FICO score by default")
ax[0].set(xlabel="Default", ylabel="Average FICO score")
sns.histplot(data, x="FICO_orig_time", hue="TARGET", bins=30, ax=ax[1])
ax[1].set_title("Distribution of FICO score by default")
ax[1].set(xlabel="FICO score", ylabel="Count")
sns.histplot(x="FICO_orig_time", hue="TARGET", data=data, bins=30, stat="percent", multiple="fill")
ax[2].set_title("Distribution of FICO score by default")
ax[2].set(xlabel="FICO score", ylabel="% of bin count")
plt.show()

fig, ax = plt.subplots(1, 3)
sns.barplot(x="TARGET", y="gdp_time", data=data, ax=ax[0])
ax[0].set_title("Average GDP by default")
ax[0].set(xlabel="Default", ylabel="Average GDP")
sns.histplot(data, x="gdp_time", hue="TARGET", bins=15, ax=ax[1])
ax[1].set_title("Distribution of GDP by default")
ax[1].set(xlabel="GDP", ylabel="Count")
sns.histplot(x="gdp_time", hue="TARGET", data=data, bins=15, stat="percent", multiple="fill")
ax[2].set_title("Distribution of GDP by default")
ax[2].set(xlabel="GDP", ylabel="% of bin count")
plt.show()

sns.countplot(x="investor_orig_time", hue="TARGET", data=data)
plt.title("Number of investor borrowers by default")
plt.xlabel("Investor borrower")
plt.ylabel("Count")
plt.show()

# plot correlation matrix on unbinned data
corr_matrix = data.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
ax.set_title('Correlation Matrix Heatmap')
plt.show()

# plot scatter plot of uer_time and hpi_time
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data["uer_time"], data["hpi_time"], alpha=0.5)
ax.set_xlabel("UER")
ax.set_ylabel("HPI")
plt.show()

# Data preprocessing
#-------------------------------------------

# train test split
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=["TARGET"]), data["TARGET"], test_size=0.2, stratify=data["TARGET"], random_state=42)

# imputing missing values (multiple imputation)
imputer = IterativeImputer(max_iter=10, random_state=42)
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

# deleting outliers with boxplot method
def delete_outliers(X, y, columns):
    Q1 = X[columns].quantile(0.25)
    Q3 = X[columns].quantile(0.75)
    IQR = Q3 - Q1
    mask = ((X[columns] < (Q1 - 1.5 * IQR)) | (X[columns] > (Q3 + 1.5 * IQR))).any(axis=1)
    X = X[~mask]
    y = y[~mask]
    return X, y

# columns = ["balance_time", "LTV_time", "balance_orig_time", "FICO_orig_time", "LTV_orig_time"]
columns = ["balance_time", "LTV_time", "balance_orig_time", "FICO_orig_time", "balance_first_time", "interest_rate_time"]
X_train, y_train = delete_outliers(X_train, y_train, columns)

# creating new variables
X_train["time_to_maturity"] = X_train["mat_time"] - X_train["time"]
X_train["balance_diff"] = X_train["balance_time"] - X_train["balance_orig_time"]
X_train["balance_ratio"] = X_train["balance_time"] / X_train["balance_first_time"] * 100
X_train["hpi_diff"] = X_train["hpi_time"] - X_train["hpi_orig_time"]
X_train["interest_rate_diff"] = X_train["interest_rate_time"] - X_train["Interest_Rate_orig_time"]
X_train["FICO_interest_rate_int"] = X_train["FICO_orig_time"] * X_train["Interest_Rate_orig_time"]
X_train["uer_diff"] = X_train["uer_time"] - X_train["uer_first_time"]
X_train["gdp_diff"] = X_train["gdp_time"] - X_train["gdp_first_time"]
X_train["balance_uer_int"] = X_train["balance_time"] * X_train["uer_time"]
X_train["balance_uer_int"] = X_train["balance_time"] * X_train["uer_time"]
X_train["gdp_hpi_first_int"] = X_train["gdp_first_time"] * X_train["hpi_orig_time"]
X_train["interest_rate_int"] = X_train["interest_rate_time"] * X_train["Interest_Rate_orig_time"]
X_train["hpi_uer_int"] = X_train["hpi_time"] * X_train["uer_time"]

# X_train["time_to_maturity"] = X_train["mat_time"] - X_train["time"]
# X_train["balance_diff"] = X_train["balance_time"] - X_train["balance_first_time"]
# X_train["balance_ratio"] = X_train["balance_time"] / X_train["balance_first_time"] * 100
# X_train["hpi_diff"] = X_train["hpi_time"] - X_train["hpi_first_time"]
# X_train["interest_rate_diff"] = X_train["interest_rate_time"] - X_train["interest_rate_first_time"]
# X_train["FICO_interest_rate_int"] = X_train["FICO_orig_time"] * X_train["interest_rate_first_time"]
# X_train["uer_diff"] = X_train["uer_time"] - X_train["uer_first_time"]
# X_train["gdp_diff"] = X_train["gdp_time"] - X_train["gdp_first_time"]
# X_train["balance_uer_int"] = X_train["balance_time"] * X_train["uer_time"]

X_test["time_to_maturity"] = X_test["mat_time"] - X_test["time"]
X_test["balance_diff"] = X_test["balance_time"] - X_test["balance_orig_time"]
X_test["balance_ratio"] = X_test["balance_time"] / X_test["balance_first_time"] * 100
X_test["hpi_diff"] = X_test["hpi_time"] - X_test["hpi_orig_time"]
X_test["interest_rate_diff"] = X_test["interest_rate_time"] - X_test["Interest_Rate_orig_time"]
X_test["FICO_interest_rate_int"] = X_test["FICO_orig_time"] * X_test["Interest_Rate_orig_time"]
X_test["uer_diff"] = X_test["uer_time"] - X_test["uer_first_time"]
X_test["gdp_diff"] = X_test["gdp_time"] - X_test["gdp_first_time"]
X_test["balance_uer_int"] = X_test["balance_time"] * X_test["uer_time"]
X_test["gdp_hpi_first_int"] = X_test["gdp_first_time"] * X_test["hpi_orig_time"]
X_test["interest_rate_int"] = X_test["interest_rate_time"] * X_test["Interest_Rate_orig_time"]
X_test["hpi_uer_int"] = X_test["hpi_time"] * X_test["uer_time"]

# X_test["time_to_maturity"] = X_test["mat_time"] - X_test["time"]
# X_test["balance_diff"] = X_test["balance_time"] - X_test["balance_first_time"]
# X_test["balance_ratio"] = X_test["balance_time"] / X_test["balance_first_time"] * 100
# X_test["hpi_diff"] = X_test["hpi_time"] - X_test["hpi_first_time"]
# X_test["interest_rate_diff"] = X_test["interest_rate_time"] - X_test["interest_rate_first_time"]
# X_test["FICO_interest_rate_int"] = X_test["FICO_orig_time"] * X_test["interest_rate_first_time"]
# X_test["uer_diff"] = X_test["uer_time"] - X_test["uer_first_time"]
# X_test["gdp_diff"] = X_test["gdp_time"] - X_test["gdp_first_time"]
# X_test["balance_uer_int"] = X_test["balance_time"] * X_test["uer_time"]

# dropping time columns
X_train.drop(columns=["time", "mat_time", "first_time", "orig_time"], inplace=True)
X_test.drop(columns=["time", "mat_time", "first_time", "orig_time"], inplace=True)

# drop investor column
# X_train.drop(columns=["investor_orig_time"], inplace=True)
# X_test.drop(columns=["investor_orig_time"], inplace=True)

# Binning process
X_train_unbinned = X_train.copy()
X_test_unbinned = X_test.copy()
y_train_unbinned = y_train.copy()

train_binary_cols = X_train[['REtype_CO_orig_time', 'REtype_PU_orig_time', 'REtype_SF_orig_time', 'investor_orig_time']]
test_binary_cols = X_test[['REtype_CO_orig_time', 'REtype_PU_orig_time', 'REtype_SF_orig_time', 'investor_orig_time']]
X_train.drop(columns=['REtype_CO_orig_time', 'REtype_PU_orig_time', 'REtype_SF_orig_time', 'investor_orig_time'], inplace=True)
X_test.drop(columns=['REtype_CO_orig_time', 'REtype_PU_orig_time', 'REtype_SF_orig_time', 'investor_orig_time'], inplace=True)

binning_process = BinningProcess(variable_names=X_train.columns.tolist())

binning_process.fit(X_train, y_train)

bin_summary = binning_process.summary()

X_train = binning_process.transform(X_train)
X_test = binning_process.transform(X_test)

X_train = pd.concat([X_train, train_binary_cols], axis=1)
X_test = pd.concat([X_test, test_binary_cols], axis=1)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
X_train_unbinned = sm.add_constant(X_train_unbinned)
X_test_unbinned = sm.add_constant(X_test_unbinned)

# oversampling
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)
X_train_unbinned, y_train_unbinned = ros.fit_resample(X_train_unbinned, y_train_unbinned)

# feature selection
f_selector = SelectKBest(k="all")
f_selector.fit(X_train, y_train)
sel_feat = pd.DataFrame({"feature": f_selector.feature_names_in_,
                         "score": f_selector.scores_})
sel_feat = sel_feat.sort_values(by="score").reset_index(drop=True)
for i in range(len(f_selector.scores_)):
    print(f"{sel_feat['feature'][i]:<30s} | {sel_feat['score'][i]}")

# plot correlation matrix
corr_matrix = X_train.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
ax.set_title('Correlation Matrix Heatmap')
plt.show()

# dropping balance features because of multicollinearity and low prediction power
X_train.drop(columns=["balance_first_time", "balance_orig_time", "uer_diff", "uer_first_time",
                      "Interest_Rate_orig_time", "interest_rate_int", "hpi_uer_int", "gdp_first_time"], inplace=True)
X_test.drop(columns=["balance_first_time", "balance_orig_time", "uer_diff", "uer_first_time",
                     "Interest_Rate_orig_time", "interest_rate_int", "hpi_uer_int", "gdp_first_time"], inplace=True)

# Modelling
#-------------------------------------------
logit_mod = sm.Logit(y_train, X_train)
estimated_model = logit_mod.fit(disp=0)

print(classification_report(y_test, estimated_model.predict(X_test) > 0.5))

estimated_model.summary()

# drop insignificant features
X_train.drop(columns=['REtype_CO_orig_time', 'REtype_PU_orig_time', 'REtype_SF_orig_time', 'hpi_time'], inplace=True)
X_test.drop(columns=['REtype_CO_orig_time', 'REtype_PU_orig_time', 'REtype_SF_orig_time', 'hpi_time'], inplace=True)

# final model
logit_mod = sm.Logit(y_train, X_train)
estimated_model = logit_mod.fit(disp=0)

# classification report
print(classification_report(y_test, estimated_model.predict(X_test) > 0.5))

# model summary
estimated_model.summary()
model_summary = pd.DataFrame(estimated_model.params)
model_summary["std_err"] = estimated_model.bse
model_summary["interval_low"] = estimated_model.conf_int()[0]
model_summary["interval_high"] = estimated_model.conf_int()[1]
model_summary.reset_index(inplace=True)
model_summary.columns = ["feature", "coef", "std_err", "interval_low", "interval_high"]
# model_summary.to_csv("model_summary.csv", index=False)

# VIF for every feature
vif = pd.DataFrame()
vif["feature"] = X_train.columns
vif["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
# vif.to_csv("vif.csv", index=False)

# print VIF
print(vif)

# plot ROC curve and (add information about AUC into plot)
fpr, tpr, thresholds = roc_curve(y_test, estimated_model.predict(X_test))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve")
plt.text(0.6, 0.1, f"AUC = {roc_auc_score(y_test, estimated_model.predict(X_test)):.4f}")
plt.show()

# Gini coefficient
gini = 2 * roc_auc_score(y_test, estimated_model.predict(X_test)) - 1
print(f"Gini coefficient: {gini}")

# plot confusion matrix
conf_matrix = confusion_matrix(y_test, estimated_model.predict(X_test) > 0.5)
sns.heatmap(conf_matrix, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion matrix")
plt.show()

# calculate MCC
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
print(f"MCC: {mcc}")

# dataframe of true and predicted values
pred_df = pd.DataFrame({"true": y_test, "pred": (estimated_model.predict(X_test) > 0.5).astype(int)})
pred_df["pred_prob"] = estimated_model.predict(X_test)
pred_df["interest_rate_time"] = X_test_unbinned["interest_rate_time"]
pred_df["FICO_orig_time"] = X_test_unbinned["FICO_orig_time"]
pred_df["LTV_time"] = X_test_unbinned["LTV_time"]

# plot the comparison between distribution of true and predicted values with respect to different independent variables
fig, ax = plt.subplots(1,2)
sns.histplot(x="interest_rate_time", hue="true", data=pred_df[pred_df["interest_rate_time"] <= 15], stat="percent", multiple="fill", ax=ax[0], bins=20)
ax[0].set_title("Distribution of interest rate by true values")
ax[0].set(xlabel="Interest rate", ylabel="% of bin count")
sns.histplot(x="interest_rate_time", hue="pred", data=pred_df[pred_df["interest_rate_time"] <= 15], stat="percent", multiple="fill", ax=ax[1], bins=20)
ax[1].set_title("Distribution of interest rate by predicted values")
ax[1].set(xlabel="Interest rate", ylabel="% of bin count")
plt.show()

fig, ax = plt.subplots(1,2)
sns.histplot(x="FICO_orig_time", hue="true", data=pred_df, stat="percent", multiple="fill", ax=ax[0], bins=15)
ax[0].set_title("Distribution of FICO score by true values")
ax[0].set(xlabel="FICO score", ylabel="% of bin count")
sns.histplot(x="FICO_orig_time", hue="pred", data=pred_df, stat="percent", multiple="fill", ax=ax[1], bins=15)
ax[1].set_title("Distribution of FICO score by predicted values")
ax[1].set(xlabel="FICO score", ylabel="% of bin count")
plt.show()

fig, ax = plt.subplots(1,2)
sns.histplot(x="LTV_time", hue="true", data=pred_df[pred_df["LTV_time"] <= 200], stat="percent", multiple="fill", ax=ax[0], bins=15)
ax[0].set_title("Distribution of LTV by true values")
ax[0].set(xlabel="LTV", ylabel="% of bin count")
sns.histplot(x="LTV_time", hue="pred", data=pred_df[pred_df["LTV_time"] <= 200], stat="percent", multiple="fill", ax=ax[1], bins=15)
ax[1].set_title("Distribution of LTV by predicted values")
ax[1].set(xlabel="LTV", ylabel="% of bin count")
plt.show()

# plot the distribution of predicted values
sns.histplot(data=pred_df, x="pred_prob", hue="true", bins=20)
plt.title("Distribution of predicted values with respect to true values")
plt.xlabel("Predicted value")
plt.ylabel("Count")
plt.show()

# find the best threshold by maximizing difference between true positive rate and false positive rate
dec_thresholds = np.linspace(0, 1, 100)
tpr = []
fpr = []
best_threshold = 0
max_diff = 0
for threshold in dec_thresholds:
    conf_matrix = confusion_matrix(y_test, estimated_model.predict(X_test) > threshold)
    tpr.append(conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]))
    fpr.append(conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[0, 0]))
    if tpr[-1] - fpr[-1] > max_diff:
        max_diff = tpr[-1] - fpr[-1]
        best_threshold = threshold

conf_matrix = confusion_matrix(y_test, estimated_model.predict(X_test) > best_threshold)
sns.heatmap(conf_matrix, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion matrix")
plt.show()

TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
print(f"MCC: {mcc}")

print(classification_report(y_test, estimated_model.predict(X_test) > best_threshold))

# dataframe of true and predicted values
pred_df = pd.DataFrame({"true": y_test, "pred": (estimated_model.predict(X_test) > best_threshold).astype(int)})
pred_df["pred_prob"] = estimated_model.predict(X_test)
pred_df["interest_rate_time"] = X_test_unbinned["interest_rate_time"]
pred_df["FICO_orig_time"] = X_test_unbinned["FICO_orig_time"]
pred_df["LTV_time"] = X_test_unbinned["LTV_time"]

# plot the comparison between distribution of true and predicted values with respect to different independent variables
fig, ax = plt.subplots(1,2)
sns.histplot(x="interest_rate_time", hue="true", data=pred_df[pred_df["interest_rate_time"] <= 15], stat="percent", multiple="fill", ax=ax[0], bins=20)
ax[0].set_title("Distribution of interest rate by true values")
ax[0].set(xlabel="Interest rate", ylabel="% of bin count")
sns.histplot(x="interest_rate_time", hue="pred", data=pred_df[pred_df["interest_rate_time"] <= 15], stat="percent", multiple="fill", ax=ax[1], bins=20)
ax[1].set_title("Distribution of interest rate by predicted values")
ax[1].set(xlabel="Interest rate", ylabel="% of bin count")
plt.show()

fig, ax = plt.subplots(1,2)
sns.histplot(x="FICO_orig_time", hue="true", data=pred_df, stat="percent", multiple="fill", ax=ax[0], bins=15)
ax[0].set_title("Distribution of FICO score by true values")
ax[0].set(xlabel="FICO score", ylabel="% of bin count")
sns.histplot(x="FICO_orig_time", hue="pred", data=pred_df, stat="percent", multiple="fill", ax=ax[1], bins=15)
ax[1].set_title("Distribution of FICO score by predicted values")
ax[1].set(xlabel="FICO score", ylabel="% of bin count")
plt.show()

fig, ax = plt.subplots(1,2)
sns.histplot(x="LTV_time", hue="true", data=pred_df[pred_df["LTV_time"] <= 200], stat="percent", multiple="fill", ax=ax[0], bins=15)
ax[0].set_title("Distribution of LTV by true values")
ax[0].set(xlabel="LTV", ylabel="% of bin count")
sns.histplot(x="LTV_time", hue="pred", data=pred_df[pred_df["LTV_time"] <= 200], stat="percent", multiple="fill", ax=ax[1], bins=15)
ax[1].set_title("Distribution of LTV by predicted values")
ax[1].set(xlabel="LTV", ylabel="% of bin count")
plt.show()

# plot the distribution of predicted values
sns.histplot(data=pred_df, x="pred_prob", hue="true", bins=20)
plt.title("Distribution of predicted values with respect to true values")
plt.xlabel("Predicted value")
plt.ylabel("Count")
plt.show()

# final correlation matrix
corr_matrix = X_train.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
ax.set_title('Correlation Matrix Heatmap')
plt.show()