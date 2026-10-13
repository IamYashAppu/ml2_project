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
2. [Why This Project Was Built (Problem & Motivation)](#2-why-this-project-was-built)
3. [Technology Selection: Why We Chose These Technologies Over Alternatives](#3-technology-selection-why-we-chose-these-technologies-over-alternatives)
4. [For What It Was Built (Goals, Scope & Real-World Use Cases)](#4-for-what-it-was-built)
5. [How It Was Built (System Architecture & Pipeline)](#5-how-it-was-built)
6. [Core Concepts Needed to Understand This Project](#6-core-concepts-needed-to-understand-this-project)
7. [Repository Structure & File Inventory](#7-repository-structure--file-inventory)
8. [Step-by-Step Execution Guide](#8-step-by-step-execution-guide)
9. [Dashboard Walkthrough & Demonstration](#9-dashboard-walkthrough--demonstration)
10. [Viva & Examiner Q&A Defense Guide](#10-viva--examiner-qa-defense-guide)

---

## 1. Executive Summary

This project delivers an end-to-end, enterprise-grade **Dual-Engine Fraud Detection Architecture** designed for high-throughput, low-latency financial transaction networks. 

Benchmarked on the standard **ULB Credit Card Fraud Dataset (284,807 transactions, 492 fraud instances, 0.172% skewness)**, the system solves two fundamental challenges that cripple standard classifiers in production banking:

1. **Severe Class Imbalance & High False Alarms:** Solved using **algorithmic cost-sensitive gradient boosting (LightGBM with scale_pos_weight = 577.88)**, penalizing missed fraud directly inside the loss function without creating artificial noisy synthetic points.
2. **Zero-Day / Novel Fraud Exploits:** Solved using an **unsupervised symmetrical bottleneck Deep Autoencoder (PyTorch)**, trained strictly on legitimate cardholder behavior. Novel attacks produce an **80x spike in Mean Squared Reconstruction Error (L2 norm)**, flagging zero-day fraud without requiring prior attack labels.

Additionally, the system optimizes its decision cutoff against a real **Banking Loss Matrix**:

```text
Total Business Cost = (False Negatives × $120.00) + (False Positives × $2.50)
```

This proves mathematically that calibrating the threshold from **0.50 down to 0.35** cuts direct bank losses by **32.6%**. The complete pipeline is deployed with an interactive **Streamlit Security Dashboard** and an asynchronous **FastAPI REST Microservice** achieving sub-5ms scoring latency.

---

## 2. Why This Project Was Built

### 2.1 The Failure of Legacy Rule-Based Systems
Historically, financial institutions relied on manual, heuristic rule engines (e.g., `IF amount > $5000 AND location != home THEN flag`). These engines exhibit severe limitations:
* **High Customer Friction:** They generate high volumes of false alarms (False Positives), blocking legitimate cardholders and hurting the customer experience.
* **Vulnerability to Coordinated Syndicates:** Fraud rings easily discover static thresholds and stay just under them (e.g., splitting stolen cards across \$49 purchases).
* **Engineering Overhead:** Rule engines require manual rule updates by analysts, lagging days or weeks behind new threat tactics.

### 2.2 The Trap of Deceptive "99.83% Accuracy"
In credit card datasets, **99.827%** of transactions are legitimate and only **0.173%** are fraudulent (a class ratio of 578 : 1).

```text
Naive Classifier Accuracy = 284,315 (Normal) / 284,807 (Total) = 99.827%
```

A naive model that blindly predicts "Legitimate" for every transaction achieves an **A+ accuracy of 99.83%** while completely failing to catch a single criminal. Consequently, accuracy is discarded in favor of **Precision-Recall AUC (PR-AUC)**, **ROC-AUC**, and **Recall**.

### 2.3 The Asymmetric Financial Cost Reality
In banking, errors are never equal:
* **Missing a Fraud (False Negative):** Direct chargeback loss, lost merchandise, and regulatory fines. Average cost: **~$120.00**.
* **Flagging a Normal Customer (False Positive):** Automated verification SMS or OTP challenge. Average cost: **~$2.50**.

Because missing a fraud is **almost 50 times more expensive** than a verification SMS, standard machine learning objective functions that treat all errors equally fail in production.

### 2.4 Strict Payment Gateway Latency SLAs
Payment networks (Visa, Mastercard, UPI, Stripe) mandate transaction authorization scoring in **under 20 milliseconds**. Heavy neural networks or slow ensemble models cannot meet these constraints on standard hardware. Our LightGBM scoring engine evaluates transactions in **< 0.005 milliseconds**, leaving ample time for network communication.

---

## 3. Technology Selection: Why We Chose These Technologies Over Alternatives

This section clearly explains why each technology was chosen over commonly used alternatives:

### 3.1 Why Cost-Sensitive LightGBM over Other Classifiers?

| Feature / Model | Logistic Regression | Random Forest | XGBoost | LightGBM (Our Choice) |
| :--- | :--- | :--- | :--- | :--- |
| **Handling Non-Linearity** | ❌ Fails on complex feature interactions | ✅ High | ✅ High | ✅ **Superior non-linear splits** |
| **Training Speed** | ⚡ Fast (~1s) | ⏳ Slow (15–30s) | ⏳ Moderate (8–12s) | ⚡ **Ultra-Fast (2.06s on 284k rows)** |
| **Scoring Latency** | ⚡ 0.001 ms | ⚠️ High (14.8 ms) | ⚡ 0.010 ms | ⚡ **< 0.005 ms (Sub-millisecond)** |
| **Memory Efficiency** | ✅ Low | ❌ High RAM usage | ⚠️ Moderate | ✅ **Exclusive Feature Bundling (EFB)** |
| **Class Imbalance Support** | ⚠️ Basic `class_weight` | ⚠️ Slow with SMOTE | ✅ `scale_pos_weight` | ✅ **Native `scale_pos_weight` with GOSS** |

* **Why not Logistic Regression?** Financial fraud patterns involve complex, non-linear relationships across multiple behavioral vectors that linear decision boundaries cannot separate.
* **Why not Random Forest?** Random Forest builds hundreds of deep trees that must all be traversed sequentially during scoring, increasing latency to ~15ms and violating banking SLAs.
* **Why LightGBM?** LightGBM utilizes **Gradient-Based One-Side Sampling (GOSS)** and **histogram binning**. It trains in ~2 seconds, uses minimal memory, and evaluates transactions in under 0.005ms.

---

### 3.2 Why Algorithmic Cost Weighting (`scale_pos_weight`) over SMOTE?

* **The Problem with SMOTE (Synthetic Minority Over-sampling):**
  * SMOTE generates synthetic fraud points by drawing lines between existing minority points and picking random points along those lines.
  * In high-dimensional latent space ($V_1$ to $V_{28}$), this frequently synthesizes fake fraud points inside legitimate transaction clusters, creating high False Positive rates and customer friction.
  * SMOTE expands dataset size artificially, multiplying training memory and training time.
* **Why Cost-Sensitive Weighting is Superior:**
  * It modifies the loss function directly:
    ```text
    scale_pos_weight = Number of Legitimate (284,315) / Number of Frauds (492) ≈ 577.88
    ```
  * During gradient descent, missing a single fraud instance incurs **578 times the gradient penalty** of an accidental false alarm.
  * **Zero fake data is synthesized**, preserving the authentic financial data distribution.

---

### 3.3 Why PyTorch Deep Autoencoders alongside Tree Models?

* **The Blind Spot of Supervised Learning (The "Zero-Day" Threat):**
  * Supervised models (LightGBM, Random Forest) learn strictly from past labeled fraud patterns.
  * When a fraud syndicate invents an entirely new exploit (new bot technique, credential-stuffing variation), supervised models treat it as normal because they have never seen it before.
* **Why the Deep Autoencoder Solves This:**
  * It is an **unsupervised neural network** trained **only on legitimate customer transactions**.
  * It learns the normal baseline manifold of regular card spending.
  * When an unseen zero-day attack vector passes through the network, the autoencoder fails to reconstruct it, creating a **massive spike in Mean Squared Reconstruction Error (0.23 normal vs. 18.4 fraud — an 80x surge!)**.
  * This raises an immediate alert without needing any prior fraud labels.

---

### 3.4 Why `RobustScaler` over `StandardScaler` or `MinMaxScaler`?

* **The Problem with `StandardScaler`:** It calculates the mean ($\mu$) and standard deviation ($\sigma$). Transaction amounts range from \$0.00 to \$25,691.16. Extreme high-dollar transactions pull the mean and standard deviation outward, compressing 99% of normal transactions into a tight, indistinguishable cluster.
* **The Problem with `MinMaxScaler`:** It divides by `(max - min)`. A single \$25,000 transaction crushes all normal \$20–\$100 transactions into a range between `0.0001` and `0.004`.
* **Why `RobustScaler` Wins:** It scales features using the **Median** and the **Interquartile Range (IQR = Q3 - Q1)**:
  ```text
  Scaled Value = (Value - Median) / (75th Percentile - 25th Percentile)
  ```
  Because median and IQR ignore extreme percentiles, outliers cannot distort the scaling pipeline.

---

### 3.5 Why Streamlit & FastAPI?

* **Streamlit (`app.py`):** Provides a visual, interactive dashboard for security analysts and live project reviews. Allows non-technical stakeholders to toggle between Auto-Pilot and Manual Simulator modes, inspect confusion matrices, and review feature importance charts.
* **FastAPI (`api.py`):** Real-world payment gateways communicate via asynchronous REST APIs, not web browsers. FastAPI provides high-performance asynchronous request handling, automated Pydantic schema validation, and sub-5ms JSON scoring responses.

---

### 3.6 Why TreeSHAP for Explainability?

* **The Black-Box Problem:** Regulators (e.g., GDPR Article 22, US Fair Credit Reporting Act) legally forbid financial institutions from rejecting payments using unexplained black-box algorithms.
* **The Solution:** **TreeSHAP (SHapley Additive exPlanations)** computes exact cooperative game-theoretic Shapley values, showing exactly which features ($V_{14}, V_{10}, V_{12}$, Amount) increased the risk score for any given transaction.

---

## 4. For What It Was Built

1. **Academic Excellence (VTU CO2 | L4):**
   * Fulfills VTU Course Outcome 2 at Bloom's Level 4 (Analyze & Evaluate) for Machine Learning by implementing outlier-resistant preprocessing, solving class imbalance, benchmarking dual models, and evaluating cost tradeoffs.
2. **Real-World Banking Authorization Engines:**
   * Built as an enterprise microservice ready to integrate with payment gateways (Visa, Stripe, UPI) for instant authorization scoring.
3. **Fraud Investigation Cockpit:**
   * Gives fraud analysts plain-English translations (*The Catch*, *Smooth Approvals*, *False Alarms*, *Costly Misses*) to make fast operational decisions.

---

## 5. How It Was Built

### 5.1 System Architecture Flow

The complete system pipeline is illustrated below (also exported as a high-resolution 300 DPI artifact [system_architecture.png](file:///c:/Users/LENOVO/Documents/ml2/system_architecture.png)):

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

### 5.2 Technical Pipeline Breakdown

#### A. Zero-Leakage Preprocessing
* To prevent test set statistics from leaking into training folds, a stratified 80/20 split was performed first.
* `RobustScaler` was fit strictly on the training set (`X_train`) and saved to `scalers.pkl`. During testing or live scoring, only `.transform()` is called.

#### B. Supervised LightGBM Engine
* Configured with `scale_pos_weight = 577.88` to penalize minority fraud errors 578 times more during gradient boosting.
* Achieved an empirical scoring latency of **< 0.005 milliseconds** per transaction.

#### C. PyTorch Deep Autoencoder Engine
* Symmetrical architecture: `Input(30) -> Linear(20) -> BatchNorm -> LeakyReLU -> Linear(10) -> Linear(20) -> Linear(30)`.
* Trained strictly on legitimate transactions ($N = 227,451$) for 6 epochs.
* **Empirical Separation:**
  * Normal Transactions Mean Reconstruction MSE: **0.2320**
  * Fraud Transactions Mean Reconstruction MSE: **18.4676**
  * **Result:** An **80x divergence in reconstruction loss**, enabling clear anomaly detection.

#### D. Business Loss Matrix Optimization
Standard ML models pick a default threshold of $\tau = 0.50$. In financial risk management, the threshold must minimize actual dollars lost:

```text
Total Loss = (False Negatives × $120.00) + (False Positives × $2.50)
```

**Empirical Threshold Benchmarks:**

| Threshold (τ) | Strategy | False Positives | False Negatives | Total Banking Cost | Financial Impact |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **τ = 0.80** | Conservative | 5 | 42 | \$5,052.50 | High fraud leakage (Costly) |
| **τ = 0.50** | Default Standard | 28 | 21 | \$2,590.00 | Standard tutorial baseline |
| **τ = 0.35** | **Cost-Optimal** | **74** | **13** | **\$1,745.00** | **32.6% Direct Loss Cut** |
| **τ = 0.10** | Hyper-Sensitive | 342 | 8 | \$1,815.00 | High customer verification friction |

Moving from $\tau = 0.50$ to $\tau = 0.35$ triggers 46 additional verification checks (costing \$115 in automated SMS charges), but intercepts 8 additional fraudulent transactions (saving \$960 in direct fraud), resulting in net savings of **\$845 per test batch**.

---

## 6. Core Concepts Needed to Understand This Project

| Concept | Plain-English Definition | Why It Matters Here |
| :--- | :--- | :--- |
| **Class Imbalance** | When normal transactions vastly outnumber fraud (99.83% vs 0.17%). | Prevents models from predicting "Legitimate" 100% of the time. |
| **Data Leakage** | Accidentally using test set information during model training. | Eliminates artificially inflated metrics; ensures real-world generalization. |
| **Cost-Sensitive Weighting** | Applying unequal penalties to classification errors in the loss function. | Penalizes False Negatives 578x more than False Positives. |
| **Deep Autoencoder** | A neural network that compresses data to a bottleneck and reconstructs it. | Detects anomalies without needing prior fraud labels (Unsupervised Anomaly Detection). |
| **Reconstruction Error** | The difference squared between original input and autoencoder output. | Spikes dramatically when novel zero-day attack patterns are encountered. |
| **Decision Threshold (τ)** | The risk probability cutoff used to flag a transaction. | Calibrating τ balances bank financial loss against customer verification friction. |
| **TreeSHAP** | A game-theoretic algorithm measuring how each feature affected the score. | Provides transparent audit trails for regulatory compliance (GDPR/FCRA). |

---

## 7. Repository Structure & File Inventory

```
ml2/
│
├── creditcard.csv                 # Kaggle benchmark dataset (284,807 transactions, 150 MB)
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
├── .gitignore                     # Git configuration ignoring large files (creditcard.csv)
├── Untitled.ipynb                 # Experimental exploratory data analysis notebook
└── README.md                      # Comprehensive project documentation (this file)
```

---

## 8. Step-by-Step Execution Guide

### 8.1 Prerequisites
Ensure Python 3.10+ is installed. Install all dependencies:
```powershell
pip install pandas numpy scikit-learn lightgbm xgboost torch shap streamlit fastapi uvicorn matplotlib joblib
```

### 8.2 Step 1: Run the Training Pipeline
To retrain the models, calibrate scalers, and export serialized model weights:
```powershell
python train_models.py
```
*Outputs generated:* `scalers.pkl`, `fraud_model.pkl`, `autoencoder_weights.pth`, `model_config.json`, and `demo_feed_transactions.csv`.

### 8.3 Step 2: Launch the Interactive Dashboard
Launch the web interface for demonstrations:
```powershell
streamlit run app.py
```
Open your browser to: **`http://localhost:8501`**

### 8.4 Step 3: Run the Production REST API
To launch the backend scoring microservice:
```powershell
python -m uvicorn api:app --reload --port 8000
```
Open interactive Swagger documentation in your browser: **`http://127.0.0.1:8000/docs`**

---

## 9. Dashboard Walkthrough & Demonstration

The dashboard ([app.py](file:///c:/Users/LENOVO/Documents/ml2/app.py)) provides a professional operations interface:

1. **System Mode Toggle (Sidebar):**
   * **🤖 Auto-Pilot (Default):** Locks in the optimal policy ($\tau = 0.35$, $\$120$ FN, $\$2.50$ FP). Zero manual configuration needed.
   * **🛠️ Manual Simulator:** Exposes the threshold slider and economic inputs to demonstrate trade-offs live to evaluators.
2. **Persistent Session Memory & Clear Button:**
   * Clicking **"📥 Load 100 Demo Txns"** keeps transactions loaded in memory across tab switches and mode toggles.
   * Clicking **"🗑️ Clear Data"** resets session memory.
3. **Four Diagnostic Tabs:**
   * **Tab 1 (Real-Time Security Scan):** Live threat table, latency SLA counter ($< 20\text{ms}$), and plain-English confusion matrix (*The Catch*, *Smooth Approvals*, *Costly Misses*, *False Alarms*).
   * **Tab 2 (Deep Autoencoder Anomaly):** Visualizes the reconstruction error histogram and flags zero-day threats exceeding the anomaly threshold.
   * **Tab 3 (Financial Cost Loss Optimization):** Direct dollar loss comparison showing how much cash the bank saves compared to the standard 0.50 threshold.
   * **Tab 4 (Explainability / SHAP):** Horizontal bar chart detailing the top behavioral vectors ($V_{14}, V_{10}, V_{12}$, Amount) driving risk classifications.

---

## 10. Viva & Examiner Q&A Defense Guide

#### Q1: "Why did you choose LightGBM over deep neural networks for tabular data?"
> *"Research consistently shows that gradient-boosted decision trees outperform deep learning on tabular data with continuous numerical features. LightGBM handles non-linear feature interactions naturally and scores transactions in under 0.005 milliseconds, easily satisfying payment gateway authorization SLAs of sub-20ms."*

#### Q2: "If LightGBM is so effective, why did you add a Deep Learning Autoencoder?"
> *"Supervised models can only detect patterns they have seen in training data. If a fraud syndicate develops a brand new attack vector, supervised models suffer False Negatives. Our PyTorch Deep Autoencoder is trained strictly on legitimate behavior; when an unseen zero-day attack arrives, its reconstruction error spikes by over 80x (from 0.23 to 18.4), catching anomalies without historical labels."*

#### Q3: "Why did you use cost-sensitive weighting (`scale_pos_weight`) instead of SMOTE?"
> *"SMOTE creates synthetic minority points via linear interpolation. In high-dimensional spaces, this frequently places synthetic fraud points into legitimate decision spaces, inflating false alarms. Cost-sensitive weighting preserves the true data distribution while penalizing missed fraud 578 times more during gradient updates."*

#### Q4: "Why did you use RobustScaler instead of StandardScaler?"
> *"Transaction amounts have extreme positive outliers (up to $25,000+), which heavily distort the mean and standard deviation in StandardScaler. RobustScaler scales using the median and Interquartile Range (IQR), making the scaling impervious to extreme dollar spikes."*

#### Q5: "How does this reduce costs for a bank in the real market?"
> *"Standard models optimize for mathematical accuracy at threshold 0.50. We optimized against a banking loss matrix where missed fraud costs $120 and verification SMS costs $2.50. Shifting the threshold to 0.35 incurs minor SMS costs but catches high-ticket frauds, cutting total business losses by 32.6%."*

---

## 📄 License & Attribution
* Built for academic evaluation at **CMR Institute of Technology (CMRIT)** under **Visvesvaraya Technological University (VTU)** regulations.
* Benchmark dataset provided by the **Machine Learning Group (ULB - Université Libre de Bruxelles)**.
