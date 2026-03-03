# 📊 Customer Segmentation with RFM Analysis | E-Commerce Retail | Python

**Business Question:** How can we segment customers based on their purchasing behavior to enable targeted marketing and retention strategies?

**Domain:** E-Commerce / Retail / Customer Analytics

**Tools Used:** Python (Pandas, NumPy, Matplotlib, Seaborn)

> 📌 This project applies the **RFM (Recency – Frequency – Monetary)** model to a real-world e-commerce transaction dataset to classify customers into meaningful segments — enabling the business to prioritize high-value customers, re-engage at-risk ones, and allocate marketing resources more effectively.

**Author:** Phan Trung Hiếu 
**Date:** 2024

---

## 📑 Table of Contents

1. [📌 Background & Overview](#-background--overview)
2. [📂 Dataset Description & Data Structure](#-dataset-description--data-structure)
3. [⚒️ Main Process](#️-main-process)
4. [🔎 Final Conclusion & Recommendations](#-final-conclusion--recommendations)

---

## 📌 Background & Overview

### 📖 What is this project about? What Business Question will it solve?

This project uses **Python** to analyze transaction data from an online retail store to:

✔️ Calculate RFM scores (Recency, Frequency, Monetary) for each customer to quantify purchasing behavior.

✔️ Segment customers into 11 distinct groups (Champions, Loyal, At Risk, Lost, etc.) using a standardized RFM segmentation model.

✔️ Identify high-value customers worth retaining and at-risk customers worth re-engaging.

✔️ Analyze revenue distribution between UK and international customers.

✔️ Measure the financial impact of cancelled orders on overall revenue.

### 👤 Who is this project for?

✔️ **Marketing & CRM teams** — to design targeted campaigns per segment (VIP rewards for Champions, win-back offers for Lost customers).

✔️ **E-commerce managers** — to understand customer lifetime value distribution and churn risk.

✔️ **Data analysts** — to learn how to implement the RFM model end-to-end in Python.

---

## 📂 Dataset Description & Data Structure

### 📌 Data Source

- **Source:** `ecommerce retail.xlsx` (UCI Online Retail Dataset)
- **Size:** 541,909 rows × 8 columns
- **Format:** `.xlsx` — 2 sheets: `ecommerce retail` (transactions) + `Segmentation` (RFM mapping table)
- **Time Range:** December 2010 – December 2011

### 📊 Data Structure & Relationships

#### 1️⃣ Tables Used:

- **Sheet 1: `ecommerce retail`** — Raw transaction-level data (1 row = 1 product line per invoice)
- **Sheet 2: `Segmentation`** — RFM score-to-segment mapping reference (e.g., score `555` → `Champions`)

#### 2️⃣ Table Schema & Data Snapshot

**Sheet 1: ecommerce retail**

<img width="788" height="430" alt="image" src="https://github.com/user-attachments/assets/38864514-7794-475b-a214-4245da4680f4" />

| Column Name | Data Type | Description |
|---|---|---|
| `InvoiceNo` | STRING | Invoice number; starts with `C` if cancelled |
| `StockCode` | STRING | Product stock code |
| `Quantity` | INTEGER | Quantity per transaction line |
| `InvoiceDate` | DATETIME | Date and time of invoice |
| `UnitPrice` | FLOAT | Price per unit (£) |
| `CustomerID` | FLOAT | Unique customer ID (nullable) |
| `Country` | STRING | Customer's country |

**Sheet 2: Segmentation**

| Column Name | Data Type | Description |
|---|---|---|
| `RFM Score` | STRING | Comma-separated RFM codes (e.g., `555, 554, 553`) |
| `Segment` | STRING | Segment label (e.g., `Champions`, `At Risk`) |

---

## ⚒️ Main Process

### 1️⃣ Data Cleaning & Preprocessing

Before computing RFM scores, several steps were applied to ensure data integrity:

- **Dropped `Description`** — not relevant to RFM analysis.
- **Flagged unidentified customers** (`Identifine_Cus`) via `CustomerID.notna()` — customers without ID cannot be tracked.
- **Identified cancelled orders** (`Cancelled_InvoiceNo`) — `InvoiceNo` starting with `'C'` are cancellations and are excluded from revenue.
- **Classified geography** (`is_UK`) — `'UK'` or `'Other'` for geographic revenue analysis.
- **Calculated `Total_Price`** — `Quantity × UnitPrice`, with cancelled orders forced to `0`.

```python
df['Cancelled_InvoiceNo'] = df['InvoiceNo'].astype(str).str.startswith('C')
df['is_UK'] = np.where(df['Country'] == 'United Kingdom', 'UK', 'Other')
df['Total_Price'] = np.where(df['Cancelled_InvoiceNo'], 0, df['Quantity'] * df['UnitPrice'])
```

**Key findings from EDA:**

| Metric | Value |
|---|---|
| Total revenue (all invoices) | £9,747,747.93 |
| Revenue from cancelled orders | £896,812.49 |
| Cancellation rate (% of revenue) | **9.20%** |

<img width="487" height="491" alt="image" src="https://github.com/user-attachments/assets/daa39c32-202c-48f7-b7a5-76336a766f7b" />

---

### 2️⃣ RFM Metric Calculation

RFM was computed at customer level using **December 31, 2011** as the reference date.

```python
ref_date = pd.to_datetime('2011-12-31')

rfm = (df.groupby('CustomerID')
          .agg({
              'InvoiceDate': lambda x: (ref_date - x.max()).days,  # Recency
              'InvoiceNo'  : 'nunique',                            # Frequency
              'Total_Price': 'sum'                                 # Monetary
          }).reset_index())

rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'Total_Price': 'Monetary'}, inplace=True)
```

| Metric | Definition | Scoring Direction |
|---|---|---|
| **Recency (R)** | Days since last purchase from 2011-12-31 | Lower = Better |
| **Frequency (F)** | Number of unique invoices | Higher = Better |
| **Monetary (M)** | Total revenue generated (excl. cancellations) | Higher = Better |

<img width="374" height="408" alt="image" src="https://github.com/user-attachments/assets/9b228cfa-fe16-4876-8ee7-caefcccb3110" />

---

### 3️⃣ RFM Scoring (1–5 Scale)

Each metric was binned into 5 quantile groups. Customers with no `CustomerID` were assigned `9999` as a placeholder.

```python
# Recency: lower days = higher score
rfm_full['R_Score'] = pd.qcut(rfm_full['Recency'], 5, labels=[5,4,3,2,1]).astype(int)

# Frequency & Monetary: higher = higher score
rfm_full['F_Score'] = pd.qcut(rfm_full['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm_full['M_Score'] = pd.qcut(rfm_full['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
```

> 📌 Use the quintile method in statistics.

---

### 4️⃣ Segment Mapping

RFM codes were combined in **F–R–M order** and mapped against the Segmentation sheet.

```python
rfm_full['RFM_Score'] = (
    rfm_full['F_Score'].astype(str) +
    rfm_full['R_Score'].astype(str) +
    rfm_full['M_Score'].astype(str)
)

segment_dict = {}
for i, row in segmentation.iterrows():
    scores = str(row['RFM Score']).replace(' ', '').split(',')
    for s in scores:
        segment_dict[s.strip()] = row['Segment']

rfm_full['Segment'] = rfm_full['RFM_Score'].map(segment_dict).fillna('Unclassified')
```

---

### 5️⃣ Visualization & Insights

#### Customer Segment Distribution

<img width="988" height="489" alt="image" src="https://github.com/user-attachments/assets/76a40e89-3890-477b-9c32-c8b23db3d66f" />

#### Average RFM Metrics by Segment

| Segment | Avg Recency (days) | Avg Frequency | Avg Monetary (£) |
|---|---|---|---|
| **Champions** | 31.1 | 14.4 | 6,567.91 |
| **Loyal** | 45.5 | 4.2 | 2,432.80 |
| **Promising** | 251.2 | 87.9 | 2,397.05 |
| **Need Attention** | 68.1 | 5.5 | 1,892.40 |
| **Cannot Lose Them** | 291.1 | 1.2 | 1,823.86 |
| **At Risk** | 62.2 | 1.4 | 876.84 |
| **Potential Loyalist** | 66.9 | 3.3 | 575.84 |
| **Hibernating customers** | 141.6 | 1.7 | 411.56 |
| **New Customers** | 196.1 | 4.2 | 309.33 |
| **About To Sleep** | 106.7 | 1.6 | 228.50 |
| **Lost customers** | 266.1 | 1.0 | 187.74 |

> 📌 **Champions** have the lowest recency (most recent buyers), highest frequency, and highest monetary — they are the store's most valuable customers.
> **Cannot Lose Them** show very high historical spend (~£1,823) but haven't purchased in ~291 days — a critical re-engagement priority.

#### RFM Correlation Heatmap

<img width="486" height="372" alt="image" src="https://github.com/user-attachments/assets/455cdd37-2a38-4b71-8155-a3b4fce46244" />

- Frequency and Monetary show a **moderate positive correlation** — customers who order more also tend to spend more.
- Recency has a **weak negative correlation** with both — more recent buyers tend to be more active and higher-spending.

#### Revenue by Geography (UK vs. Other)

<img width="981" height="590" alt="image" src="https://github.com/user-attachments/assets/4ebe9222-04ce-4a61-ab0c-5571f02a32cd" />

---

### 6️⃣ Export Results

```python
rfm_full.to_excel("RFM_Segmentation_Result.xlsx", index=False)
```

Output file **`RFM_Segmentation_Result.xlsx`** contains all 4,374 customers with their RFM scores, codes, and segment labels.

---

## 🔎 Final Conclusion & Recommendations

Based on the RFM segmentation results, we would recommend the **Marketing & CRM team** to consider the following:

✔️ **Retain Champions** — Averaging 14.4 purchases and £6,567 revenue each, Champions are the store's engine. Invest in VIP loyalty programs, early product access, and personalized rewards.

✔️ **Re-engage "Cannot Lose Them"** — High historical spend (£1,823 avg) but ~291 days since last purchase. Launch immediate win-back campaigns with exclusive offers before they become permanently lost.

✔️ **Nurture Potential Loyalists** — Moderate recency and frequency signal growth potential. Targeted email flows and membership incentives can push them toward the Loyal or Champions tier.

✔️ **Investigate Cancellation Rate** — At **9.20% of total revenue (£896,812)**, cancellations represent significant value leakage. A root cause analysis (stockouts, delivery failures, pricing issues) is recommended.

✔️ **Expand Internationally** — While the UK dominates, international customers represent an untapped growth opportunity. Localized promotions and regional campaigns could diversify revenue.

✔️ **Capture Unidentified Customers** — A subset of transactions lack `CustomerID`, making them untrackable for retention. Incentivizing account registration (e.g., discount on first login) would improve coverage over time.

