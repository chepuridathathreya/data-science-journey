# Customer Clustering — Personal Notes

This file is for my own learning and revision.
It explains the reasoning behind Project 3, not just the code.

---

# 1. Project Goal

The goal of this project is to divide customers into groups based on their
credit-card usage and purchasing behaviour.

This is an example of:

- Unsupervised Learning
- Clustering
- Customer Segmentation
- K-Means Clustering

There is NO target column.

The model has to discover hidden groups in the customer data itself.

---

# 2. Dataset

The dataset contains:

- 8,950 customers
- 18 original columns

Important columns:

| Column | Meaning |
|---|---|
| CUST_ID | Customer identifier |
| BALANCE | Current account/card balance |
| BALANCE_FREQUENCY | Frequency of balance updates |
| PURCHASES | Total purchase amount |
| ONEOFF_PURCHASES | Amount spent in one-off purchases |
| INSTALLMENTS_PURCHASES | Amount spent through installments |
| CASH_ADVANCE | Total cash advance amount |
| PURCHASES_FREQUENCY | How frequently purchases are made |
| ONEOFF_PURCHASES_FREQUENCY | Frequency of one-off purchases |
| PURCHASES_INSTALLMENTS_FREQUENCY | Frequency of installment purchases |
| CASH_ADVANCE_FREQUENCY | Frequency of cash advances |
| CASH_ADVANCE_TRX | Number of cash advance transactions |
| PURCHASES_TRX | Number of purchase transactions |
| CREDIT_LIMIT | Credit limit |
| PAYMENTS | Total payments |
| MINIMUM_PAYMENTS | Minimum payment amount |
| PRC_FULL_PAYMENT | Proportion/percentage of full payments |
| TENURE | Length of customer relationship |

---

# 3. Why CUST_ID Was Removed

CUST_ID identifies the customer.

It does not describe customer behaviour.

For example:

C10001 is not behaviourally closer to C10002 simply because the numbers
are close.

Therefore:

CUST_ID → keep for identification
CUST_ID → do NOT use as a clustering feature

---

# 4. Initial Data Checks

## Shape

8,950 rows × 18 columns

## Duplicates

There were:

0 duplicate rows

## Missing values

Two columns contained missing values:

- CREDIT_LIMIT → 1 missing
- MINIMUM_PAYMENTS → 313 missing

---

# 5. Handling Missing Values

We checked the distributions before deciding how to fill the missing values.

For CREDIT_LIMIT:

- Mean ≈ 4494
- Median ≈ 3000
- Maximum = 30000

For MINIMUM_PAYMENTS:

- Mean ≈ 864
- Median ≈ 312
- Maximum ≈ 76406

The mean was much higher than the median because of extreme values.

Therefore, median imputation was chosen.

Why?

The median is less affected by extreme values than the mean.

Example:

Values:

10, 20, 30, 40, 1000

Mean → strongly pulled upward by 1000

Median → still represents the middle of the data

---

# 6. Understanding EDA

EDA = Exploratory Data Analysis.

The purpose is not just to run commands.

The purpose is to ask:

"What is happening inside my data?"

Useful questions include:

- Are there missing values?
- Are there duplicates?
- What are the distributions?
- Are there outliers?
- Are features strongly related?
- Are some features redundant?
- What does each feature actually mean?

---

# 7. Correlation

Correlation tells us how two numerical variables tend to move together.

Positive correlation:

When one variable tends to increase, the other also tends to increase.

Negative correlation:

When one variable tends to increase, the other tends to decrease.

Correlation does NOT automatically mean causation.

Example:

Ice cream sales and swimming activity can increase together,
but ice cream does not cause people to swim.

---

# 8. Why We Checked Correlation

Some customer features describe related behaviour.

For example:

PURCHASES
ONEOFF_PURCHASES
INSTALLMENTS_PURCHASES

These can naturally be related because different types of purchases
contribute to overall purchasing behaviour.

Strong correlation does NOT automatically mean that one feature should
be deleted.

We have to ask whether the features provide different useful information.

---

# 9. Distribution and Skewness

Histograms showed that many financial features were heavily right-skewed.

Examples:

- MINIMUM_PAYMENTS
- ONEOFF_PURCHASES
- PURCHASES
- INSTALLMENTS_PURCHASES
- PAYMENTS
- CASH_ADVANCE
- PURCHASES_TRX

Many customers had relatively small values, while a smaller number had
very large values.

This created long right-side tails.

Some columns also contained many zeros.

For example, many customers did not use cash advances at all.

This is not necessarily bad data.
It can represent real customer behaviour.

---

# 10. What Is Skewness?

Skewness describes the asymmetry of a distribution.

Rough idea:

Skewness ≈ 0
→ relatively balanced

Positive skewness
→ longer tail toward larger values

Negative skewness
→ longer tail toward smaller values

Large absolute skewness
→ more strongly skewed distribution

---

# 11. Why Did We Use log1p?

Some features were extremely right-skewed.

For example:

MINIMUM_PAYMENTS had skewness ≈ 13.85

PURCHASES had skewness ≈ 8.14

Large values could have too much influence.

Log transformation compresses large values.

We used:

np.log1p(x)

which is equivalent to:

log(1 + x)

The +1 is important because our data contains zeros.

log(0) is undefined.

But:

log(1 + 0) = log(1) = 0

So log1p can safely handle zero values.

The goal is NOT to make skewness exactly zero.

The goal is to reduce extreme skewness and prevent very large values
from dominating too strongly.

---

# 12. Before vs After Log Transformation

Examples:

| Feature | Before | After |
|---|---:|---:|
| MINIMUM_PAYMENTS | 13.85 | 0.27 |
| ONEOFF_PURCHASES | 10.05 | 0.19 |
| PURCHASES | 8.14 | -0.76 |
| INSTALLMENTS_PURCHASES | 7.30 | -0.02 |
| PAYMENTS | 5.91 | -1.78 |
| CASH_ADVANCE | 5.17 | 0.26 |
| PURCHASES_TRX | 4.63 | 0.03 |

This showed that the transformation substantially reduced the extreme
right-skewness of several features.

Important:

A negative skewness value does NOT mean the actual customer values
became negative.

It only describes the shape of the distribution.

---

# 13. Why We Did NOT Automatically Delete Outliers

An unusual customer is not necessarily bad data.

Example:

A customer with extremely high purchases could simply be a
high-value customer.

That customer may actually be important for customer segmentation.

Therefore:

Extreme value found
→ investigate it

If it is an error/impossible value
→ correct/remove if justified

If it is a genuine customer behaviour
→ don't blindly delete it

---

# 14. Why We Did Not Use Train-Test Split

This is an UNSUPERVISED learning problem.

There is no target column.

In supervised learning:

Features → Target

Example:

Review text → Truthful/Deceptive

So we need train/test data.

Here:

Customer behaviour → ???

There is no known answer.

Finding the groups is the model's job.

Therefore, we did not use a normal train-test split.

---

# 15. Feature Scaling

K-Means is based on distance.

Our features had very different scales.

For example:

PURCHASES → thousands

CREDIT_LIMIT → thousands

PURCHASES_FREQUENCY → 0 to 1

If we used the raw values directly, large-number features could dominate
the distance calculations.

Therefore, we used StandardScaler.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_log)


# 16. What Is K-Means?

K-Means is an **unsupervised clustering algorithm**.

Its job is to divide data into **K groups**, called clusters.

There is no target column telling the model what the correct groups are.

Instead, K-Means looks at the features and tries to find customers that
are similar to each other.

## Basic idea

1. Choose the number of clusters, K.
2. K-Means creates initial cluster centers.
3. Each customer is assigned to the nearest cluster center.
4. The cluster centers are recalculated.
5. Customers are reassigned based on the new centers.
6. This process repeats until the clusters stabilize.

Simple idea:

Customer data
→ find similar customers
→ group them together
→ repeat until the groups become stable

---

# 17. What Does K Mean?

`K` represents the **number of clusters** we want.

For example:

```python
KMeans(n_clusters=5)



