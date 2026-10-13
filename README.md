# 🛡️ Real-Time Credit Card Fraud Detection Using Cost-Sensitive Machine Learning and Deep Autoencoders

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Framework-FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Engine-LightGBM](https://img.shields.io/badge/Model-Cost--Sensitive%20LightGBM-orange.svg)](https://lightgbm.readthedocs.io/)
[![Engine-PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%20Autoencoder-EE4C2C.svg)](https://pytorch.org/)
[![License-Academic](https://img.shields.io/badge/Purpose-VTU%20Academic%20Project-purple.svg)]()

> **Institution:** CMR Institute of Technology (CMRIT), Bengaluru  
> **Department:** Department of Artificial Intelligence and Machine Learning / Computer Science and Engineering  
> **Affiliation:** Visvesvaraya Technological University (VTU), Belagavi  
> **Course / Outcome:** Machine Learning Laboratory / Mini-Project (CO2 | L4: Analysis & Evaluation)  
> **Authors:** K Yashwanth Kumar (`1CR23CI019`), Faizan Raza (`1CR23CI014`), MD Zaid Ibrahim (`1CR23CI027`) 
> **Project Guide:** Prof. Santhiya S (Asst. Professor, Dept. of AIML, CMRIT)  

---

## 📌 Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Why This Project Was Built (Motivation & Problem Statement)](#2-why-this-project-was-built)
3. [For What It Was Built (Goals, Scope & Real-World Use Cases)](#3-for-what-it-was-built)
4. [How It Was Built (System Architecture & Pipeline)](#4-how-it-was-built)
5. [Core Concepts Needed to Understand This Project](#5-core-concepts-needed-to-understand-this-project)
6. [Repository Structure & File Inventory](#6-repository-structure--file-inventory)
7. [Step-by-Step Execution Guide](#7-step-by-step-execution-guide)
8. [Dashboard Walkthrough & Demonstration](#8-dashboard-walkthrough--demonstration)
9. [Viva & Examiner Q&A Defense Guide](#9-viva--examiner-qa-defense-guide)

---

## 1. Executive Summary

This project delivers an end-to-end, enterprise-grade **Dual-Engine Fraud Detection Architecture** designed for high-throughput, low-latency financial transaction networks. 

Benchmarked on the standard **ULB Credit Card Fraud Dataset (284,807 transactions, 492 fraud instances, 0.172% skewness)**, the system solves two fundamental challenges that cripple standard classifiers in production banking:
1. **Severe Class Imbalance & High False Alarms:** Solved using **algorithmic cost-sensitive gradient boosting (`LightGBM` with `scale_pos_weight = 577.88`)**, penalizing missed fraud directly inside the loss function without synthesizing noisy boundary samples.
2. **Zero-Day / Novel Fraud Exploits:** Solved using an **unsupervised symmetrical bottleneck Deep Autoencoder (`PyTorch`)**, trained strictly on legitimate cardholder behavior. Novel attacks produce an **80x spike in Mean Squared Reconstruction Error ($L_2$ norm)**, flagging zero-day fraud without requiring prior attack labels.

Additionally, the system optimizes its decision cutoff against a real **Banking Loss Matrix** ($\text{Cost} = \$120 \times \text{FN} + \$2.50 \times \text{FP}$), proving mathematically that calibrating the threshold from $0.50 \to 0.35$ reduces direct financial loss by **32.6%**. The pipeline is packaged into an interactive **Streamlit Security Dashboard** and an asynchronous **FastAPI REST Microservice** with sub-5ms latency.

---

## 2. Why This Project Was Built

### 2.1 The Failure of Rule-Based Legacy Systems
Historically, financial institutions relied on manual, heuristic rule engines (e.g., `IF amount > $5000 AND location != home THEN flag`). These engines suffer from fatal weaknesses:
* **High Customer Friction:** They generate thousands of False Positives daily, blocking legitimate cardholders and causing high abandonment rates.
* **Vulnerability to Coordinated Syndicates:** Fraud rings easily discover static thresholds and stay just under them (e.g., splitting stolen cards across \$49 purchases).
* **Engineering Bottlenecks:** Rules require manual rule updates by security analysts, lagging days behind new threat vectors.

### 2.2 The Trap of Deceptive "99.83% Accuracy"
In credit card datasets, **99.827%** of transactions are legitimate and only **0.173%** are fraudulent (a class ratio of 578 : 1).
$$\text{Naive Classifier Accuracy} = \frac{284,315 \text{ (Normal)}}{284,807 \text{ (Total)}} = 99.827\%$$
A dumb model that blindly guesses "Legitimate" for every transaction achieves an **A+ grade accuracy of 99.83%** while catching **0 out of 492 thieves**. Therefore, accuracy is discarded in favor of **Precision-Recall AUC (PR-AUC)**, **ROC-AUC**, and **Recall**.

### 2.3 The Asymmetric Financial Cost Reality
In banking, errors are never equal:
* **Missing a Fraud (False Negative):** Direct chargeback loss, lost merchandise, and regulatory fines. Average cost: **~$120.00**.
* **Flagging a Normal Customer (False Positive):** Automated OTP / SMS challenge or brief verification call. Average cost: **~$2.50**.
Because a False Negative is nearly **50 times more expensive** than a False Positive, standard symmetric loss functions fail.

### 2.4 Strict Payment Gateway Latency SLAs
Card networks (Visa, Mastercard, UPI, Stripe) require real-time fraud scoring within **sub-20 millisecond SLAs**. Heavy, multi-billion parameter neural networks cannot meet these SLAs on standard banking server clusters. Our LightGBM scoring engine achieves an inference latency of **< 0.005 ms**, leaving ample headroom for network roundtrips.

---

## 3. For What It Was Built

1. **University Curriculum Fulfillment (CO2 | L4):**
   * Fulfills VTU Course Outcome 2 at Bloom's Level 4 (Analyze & Evaluate) for the Machine Learning Laboratory by addressing data cleaning, class imbalance, model training, confusion matrix benchmarking, and feature importance.
2. **Enterprise Banking Infrastructure:**
   * Designed as a deployable microservice architecture that payment gateways can query synchronously during transaction authorization.
3. **Risk Investigation Desk:**
   * Empowers fraud analysts with plain-English risk categorization (*"The Catch"*, *"Costly Miss"*, *"False Alarm"*), real-time threshold simulations, and **Explainable AI (TreeSHAP)** for regulatory compliance (GDPR Article 22, US Fair Credit Reporting Act).

---

## 4. How It Was Built

### 4.1 System Architecture

The complete system pipeline is illustrated below (also exported as high-resolution 300 DPI artifact [system_architecture.png](file:///c:/Users/LENOVO/Documents/ml2/system_architecture.png)):

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1. INGESTION LAYER                              │
│   ULB Benchmark Dataset (284,807 transactions | 492 frauds : 0.172%)   │
│   Features: Time, Amount, V1-V28 Latent PCA behavioral components      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    2. PREPROCESSING PIPELINE                           │
│   • Stratified 80/20 Train-Test Split (Zero Data Leakage)              │
│   • RobustScaler (Amount, Time): Median & IQR Outlier Resistance       │
│   • Pre-fit Scalers serialized into `scalers.pkl`                      │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│  3A. SUPERVISED ENGINE (LightGBM)    │  │  3B. UNSUPERVISED ENGINE     │
│  • Algorithmic Cost-Sensitive Loss   │  │      (PyTorch Autoencoder)   │
│  • scale_pos_weight = 577.88         │  │  • Symmetrical Bottleneck:   │
│  • Scoring Latency: < 0.005 ms       │  │    [30 -> 20 -> 10 -> 20 -> 30]
│  • High sensitivity to known fraud   │  │  • Trained ONLY on Class 0   │
└───────────────────┬──────────────────┘  │  • Reconstruction MSE (L2)   │
                    │                     │  • Zero-Day Attack Anomaly   │
                    │                     └──────────────┬───────────────┘
                    │                                    │
                    └─────────────────┬──────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│               4. BUSINESS LOSS MATRIX & THRESHOLD OPTIMIZATION         │
│   • Total Cost Equation: Cost = ($120 * FN) + ($2.50 * FP)             │
│   • Optimal Cutoff: tau = 0.35 (Reduces financial losses by >32.6%)    │
└─────────────────────────────────────┬──────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│             5. SERVING, EXPLAINABILITY (SHAP) & DASHBOARDS             │
│   • Streamlit UI (`app.py`): Auto-Pilot vs. Manual Simulator           │
│   • FastAPI Service (`api.py`): Real-time asynchronous sub-5ms REST API│
│   • Explainability: TreeSHAP local & global risk feature attributions │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 4.2 Technical Component Deep-Dive

#### A. Data Preprocessing & Leakage Elimination
* **Failure Mode Prevented:** Fitting scalers on the entire dataset before splitting causes *data leakage* (test set statistics bleed into the training pipeline).
* **Our Solution:** A strict stratified 80/20 split was executed first. `RobustScaler` was fit **strictly on `X_train`** and persisted to `scalers.pkl`. During test scoring or live UI uploads, the pipeline only calls `.transform()`.
* **Why RobustScaler:** Standard Z-score scalers calculate mean $\mu$ and standard deviation $\sigma$. Because transaction amounts feature extreme heavy-tailed outliers (from \$0.00 up to \$25,691.16), the mean and variance become severely biased. `RobustScaler` scales using the median and Interquartile Range (IQR = $Q_3 - Q_1$), keeping outliers bounded cleanly.

#### B. Supervised Engine: Cost-Sensitive LightGBM
* Instead of synthesizing artificial data points with SMOTE (which frequently creates noisy fraud instances in overlapping boundary regions), we modified the **binary log-loss objective** directly:
  $$\text{scale\_pos\_weight} = \frac{N_{\text{legitimate}}}{N_{\text{fraud}}} = \frac{284,315}{492} \approx 577.88$$
* This forces gradient descent updates to treat missing a single fraud instance as 578 times more severe than an accidental false positive.

#### C. Deep Learning Anomaly Engine: PyTorch Autoencoder
* **Network Geometry:**
  * Encoder: $\text{Linear}(30, 20) \to \text{BatchNorm1d} \to \text{LeakyReLU}(0.1) \to \text{Linear}(20, 10) \to \text{BatchNorm1d} \to \text{LeakyReLU}(0.1)$
  * Decoder: $\text{Linear}(10, 20) \to \text{BatchNorm1d} \to \text{LeakyReLU}(0.1) \to \text{Linear}(20, 30)$
* **Training Strategy:** The network is trained using Mean Squared Error (MSE) loss for 6 epochs exclusively on legitimate transactions ($N = 227,451$).
* **Separation Empirical Benchmark:**
  * Legitimate Transactions Mean Reconstruction MSE: **0.2320**
  * Fraudulent Transactions Mean Reconstruction MSE: **18.4676**
  * **Result:** Over **80x divergence in reconstruction loss**, creating an unambiguous separation boundary for zero-day fraud.

#### D. Banking Loss Matrix & Threshold Optimization
* Models typically classify fraud if probability $p \ge 0.50$. However, $0.50$ is an arbitrary mathematical median, not a business policy.
* We evaluated a realistic banking cost function:
  $$\text{Loss} = (FN \times \$120.00) + (FP \times \$2.50)$$
* **Trade-off Analysis:**
  | Threshold ($\tau$) | Strategy | False Positives | False Negatives | Total Banking Cost | Outcome |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | $\tau = 0.80$ | Conservative | 5 | 42 | \$5,052.50 | Massive fraud leakage |
  | $\tau = 0.50$ | Default Standard | 28 | 21 | \$2,590.00 | Standard tutorial baseline |
  | **$\tau = 0.35$** | **Cost-Optimal** | **74** | **13** | **\$1,745.00** | **32.6% Direct Loss Cut** |
  | $\tau = 0.10$ | Hyper-Sensitive | 342 | 8 | \$1,815.00 | Excess customer friction |

Moving from $\tau = 0.50 \to 0.35$ causes 46 additional false alarms (costing \$115 in SMS verification), but catches 8 additional fraudsters (saving \$960 in direct fraud), netting **\$845 in net savings per test batch**.

---

## 5. Core Concepts Needed to Understand This Project

| Concept | What It Means in Plain English | Why It Is Crucial Here |
| :--- | :--- | :--- |
| **Class Imbalance** | One class overwhelmingly outnumbers another (99.83% vs 0.17%). | Prevents standard ML models from lazily predicting "Legitimate" 100% of the time. |
| **Data Leakage** | Information from the test set leaking into the training pipeline. | Guarantees test results reflect real-world performance without inflated optimism. |
| **Cost-Sensitive Learning** | Weighting the penalty of errors asymmetrically. | Penalizes False Negatives 578x more than False Positives. |
| **Deep Autoencoder** | A neural network that compresses data to a bottleneck and rebuilds it. | Detects anomalies without needing prior fraud labels (Unsupervised Anomaly Detection). |
| **$L_2$ Reconstruction Error** | The difference squared between original input and network output. | Spikes dramatically when unexpected attack patterns pass through the Autoencoder. |
| **Decision Threshold ($\tau$)** | The probability cutoff for triggering an alarm. | Shifting $\tau$ balances bank financial loss against customer verification friction. |
| **TreeSHAP** | Game-theoretic metric showing which features moved the probability. | Fulfills regulatory compliance (GDPR/FCRA) by explaining why an alert was triggered. |

---

## 6. Repository Structure & File Inventory

```
ml2/
│
├── creditcard.csv                 # Benchmark Kaggle dataset (284,807 rows, 150 MB)
├── train_models.py                # End-to-end training pipeline for LightGBM & Autoencoder
├── scalers.pkl                    # Serialized RobustScaler objects (avoids data leakage)
├── fraud_model.pkl                # Serialized Cost-Sensitive LightGBM model
├── autoencoder_weights.pth        # Serialized PyTorch Deep Autoencoder weights
├── model_config.json              # System configuration, optimal threshold & cost benchmarks
├── demo_feed_transactions.csv     # 100 balanced sample transactions (with 20 fraud cases)
├── unlabeled_live_test.csv        # 100 blind production transactions for testing
│
├── app.py                         # Streamlit interactive security operations dashboard
├── api.py                         # Asynchronous FastAPI REST scoring microservice
├── generate_architecture_diagram.py # Script generating Figure 3.3 for the PDF report
├── system_architecture.png        # Generated high-resolution 300 DPI architecture diagram
│
├── Untitled.ipynb                 # Experimental exploratory data analysis notebook
└── README.md                      # Comprehensive project documentation (this file)
```

---

## 7. Step-by-Step Execution Guide

### 7.1 Prerequisites
Ensure Python 3.10+ is installed. Install all required dependencies:
```powershell
pip install pandas numpy scikit-learn lightgbm xgboost torch shap streamlit fastapi uvicorn matplotlib joblib
```

### 7.2 Step 1: Train the Dual-Engine Models
Run the training script to process the dataset, calibrate scalers, train LightGBM and the PyTorch Autoencoder, and generate serialized artifacts:
```powershell
python train_models.py
```
*Outputs generated:* `scalers.pkl`, `fraud_model.pkl`, `autoencoder_weights.pth`, `model_config.json`, and `demo_feed_transactions.csv`.

### 7.3 Step 2: Launch the Interactive Dashboard
Launch the web interface for live demonstrations:
```powershell
streamlit run app.py
```
Open your browser to: **`http://localhost:8501`**

### 7.4 Step 3: (Optional) Run the Production REST API
To demonstrate the enterprise backend microservice:
```powershell
python -m uvicorn api:app --reload --port 8000
```
Open interactive Swagger API documentation in your browser: **`http://127.0.0.1:8000/docs`**

---

## 8. Dashboard Walkthrough & Demonstration

The dashboard ([app.py](file:///c:/Users/LENOVO/Documents/ml2/app.py)) is engineered with executive styling and includes:

1. **Operating Mode Toggle (Sidebar):**
   * **🤖 Auto-Pilot (Default):** Locks in the bank-proven optimal settings ($\tau = 0.35$, $\$120$ FN, $\$2.50$ FP). Zero manual configuration needed.
   * **🛠️ Manual Simulator:** Exposes the threshold slider and economic inputs to demonstrate policy trade-offs live to examiners.
2. **Persistent Session Memory & Clear Button:**
   * When you click **"📥 Load 100 Demo Txns"**, the dataset stays in memory while you switch tabs, toggle modes, or move sliders.
   * Click **"🗑️ Clear Data"** anytime to reset the memory.
3. **Four Core Diagnostic Tabs:**
   * **Tab 1 (Real-Time Security Scan):** Live threat table, latency SLA counter ($< 20\text{ms}$), and plain-English confusion matrix (*The Catch*, *Smooth Approvals*, *Costly Misses*, *False Alarms*).
   * **Tab 2 (Deep Autoencoder Anomaly):** Visualizes the $L_2$ norm reconstruction error distribution and flags out-of-distribution zero-day threats.
   * **Tab 3 (Financial Cost Loss Optimization):** Direct dollar comparison showing how much money the bank saves compared to the standard $0.50$ threshold.
   * **Tab 4 (Explainability / SHAP):** Horizontal bar chart detailing the top behavioral vectors ($V_{14}, V_{10}, V_{12}$, Amount) driving risk scores.

---

## 9. Viva & Examiner Q&A Defense Guide

Use these concise, high-impact answers during your project review:

#### Q1: "Why did you choose LightGBM over deep neural networks for tabular data?"
> *"Academic research consistently demonstrates that gradient-boosted decision trees outperform deep learning on tabular data with continuous numerical features. LightGBM handles non-linear feature interactions naturally and achieves single-transaction inference in under 0.005 milliseconds, satisfying strict sub-20ms payment gateway authorization SLAs."*

#### Q2: "If LightGBM is so good, why did you add a Deep Learning Autoencoder?"
> *"Supervised models can only detect patterns they have seen in training data. If a fraud syndicate develops a brand new attack vector, supervised models suffer False Negatives. Our PyTorch Deep Autoencoder is trained strictly on legitimate behavior; when an unseen zero-day attack arrives, its reconstruction error spikes by over 80x (from 0.23 to 18.4), catching anomalies without historical labels."*

#### Q3: "Why did you use cost-sensitive weighting (`scale_pos_weight`) instead of SMOTE?"
> *"SMOTE creates synthetic minority points via linear interpolation. In high-dimensional spaces, this frequently places synthetic fraud points into legitimate decision spaces, inflating false alarms. Cost-sensitive weighting preserves the true data distribution while penalizing missed fraud 578 times more during gradient updates."*

#### Q4: "Why did you use RobustScaler instead of StandardScaler?"
> *"Transaction amounts have extreme positive outliers (up to $25,000+), which heavily distort the mean and standard deviation in StandardScaler. RobustScaler scales using the median and Interquartile Range (IQR), making the scaling impervious to extreme dollar spikes."*

#### Q5: "How does this reduce costs for a bank in the real market?"
> *"Standard models optimize for mathematical accuracy at threshold 0.50. We optimized against a banking loss matrix where missed fraud costs $120 and verification SMS costs $2.50. Shifting the threshold to 0.35 incurs minor SMS costs but catches high-ticket frauds, cutting total business losses by 32.6%."*

---

## 📄 License & Attribution
* Built for academic submission at **CMR Institute of Technology (CMRIT)** under **Visvesvaraya Technological University (VTU)** regulations.
* Benchmark dataset provided by the **Machine Learning Group (ULB - Université Libre de Bruxelles)**.
