import time
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, auc,
    confusion_matrix, classification_report
)

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

print("="*60)
print("STARTING END-TO-END FRAUD DETECTION TRAINING PIPELINE")
print("="*60)

# 1. Load Data
print("\n[Step 1/6] Loading creditcard.csv...")
df = pd.read_csv('creditcard.csv')
print(f"Total transactions: {len(df):,}")
n_legit = (df['Class'] == 0).sum()
n_fraud = (df['Class'] == 1).sum()
scale_pos_weight = n_legit / n_fraud
print(f"Legitimate (Class 0): {n_legit:,} ({n_legit/len(df)*100:.3f}%)")
print(f"Fraudulent (Class 1): {n_fraud:,} ({n_fraud/len(df)*100:.3f}%)")
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

# 2. Preprocessing without Data Leakage
print("\n[Step 2/6] Preprocessing and zero-leakage stratified train/test split...")
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit RobustScaler strictly on training set
scaler_amount = RobustScaler()
scaler_time = RobustScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled['Amount'] = scaler_amount.fit_transform(X_train[['Amount']])
X_train_scaled['Time'] = scaler_time.fit_transform(X_train[['Time']])

X_test_scaled['Amount'] = scaler_amount.transform(X_test[['Amount']])
X_test_scaled['Time'] = scaler_time.transform(X_test[['Time']])

# Save scalers for production inference
joblib.dump({
    'scaler_amount': scaler_amount,
    'scaler_time': scaler_time,
    'feature_names': list(X_train.columns)
}, 'scalers.pkl')
print("Saved scalers to scalers.pkl")

# 3. Train Cost-Sensitive Supervised Model (LightGBM / XGBoost)
print("\n[Step 3/6] Training Cost-Sensitive Gradient Boosted Model...")
supervised_model = None
model_name = ""

try:
    import lightgbm as lgb
    print("Training LightGBM with scale_pos_weight...")
    supervised_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model_name = "LightGBM"
except Exception as e:
    print(f"LightGBM not available ({e}), using XGBoost...")
    import xgboost as xgb
    supervised_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    model_name = "XGBoost"

t0 = time.time()
supervised_model.fit(X_train_scaled, y_train)
train_time = time.time() - t0
print(f"{model_name} training completed in {train_time:.2f} seconds.")

# Benchmark inference latency
t0 = time.time()
y_proba = supervised_model.predict_proba(X_test_scaled)[:, 1]
scoring_latency_ms = ((time.time() - t0) / len(X_test_scaled)) * 1000
print(f"Average single-transaction scoring latency: {scoring_latency_ms:.3f} ms")

roc_auc = roc_auc_score(y_test, y_proba)
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)
print(f"Supervised Model ({model_name}) Performance:")
print(f"  ROC-AUC: {roc_auc:.4f}")
print(f"  PR-AUC:  {pr_auc:.4f}")

# Save supervised model
joblib.dump(supervised_model, 'fraud_model.pkl')
print(f"Saved primary supervised model to fraud_model.pkl")

# 4. Train Unsupervised PyTorch Deep Autoencoder for Zero-Day Anomaly Detection
print("\n[Step 4/6] Training PyTorch Deep Autoencoder on Legitimate Transactions...")

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim=30):
        super(DeepAutoencoder, self).__init__()
        # Encoder: 30 -> 20 -> 10
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(0.1),
            nn.Linear(20, 10),
            nn.BatchNorm1d(10),
            nn.LeakyReLU(0.1)
        )
        # Decoder: 10 -> 20 -> 30
        self.decoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(0.1),
            nn.Linear(20, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Train ONLY on Normal (Class 0) transactions
X_train_normal = X_train_scaled[y_train == 0].values.astype(np.float32)
normal_dataset = TensorDataset(torch.from_numpy(X_train_normal))
train_loader = DataLoader(normal_dataset, batch_size=512, shuffle=True)

autoencoder = DeepAutoencoder(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.002)

epochs = 6
autoencoder.train()
print(f"Training Autoencoder for {epochs} epochs on {len(X_train_normal):,} normal transactions...")
for epoch in range(1, epochs + 1):
    total_loss = 0.0
    for batch in train_loader:
        data = batch[0]
        optimizer.zero_grad()
        output = autoencoder(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(data)
    avg_loss = total_loss / len(X_train_normal)
    print(f"  Epoch [{epoch}/{epochs}] - Normal Reconstruction Loss (MSE): {avg_loss:.5f}")

autoencoder.eval()
torch.save(autoencoder.state_dict(), 'autoencoder_weights.pth')
print("Saved PyTorch Autoencoder to autoencoder_weights.pth")

# Compute Reconstruction Error on Test Set
X_test_tensor = torch.from_numpy(X_test_scaled.values.astype(np.float32))
with torch.no_grad():
    reconstructions = autoencoder(X_test_tensor).numpy()
    mse_errors = np.mean(np.square(X_test_scaled.values - reconstructions), axis=1)

normal_mse = mse_errors[y_test == 0]
fraud_mse = mse_errors[y_test == 1]
print(f"Normal transactions mean MSE: {np.mean(normal_mse):.4f} (median: {np.median(normal_mse):.4f})")
print(f"Fraud transactions mean MSE:  {np.mean(fraud_mse):.4f} (median: {np.median(fraud_mse):.4f})")

# Determine anomaly threshold (e.g. 95th percentile of normal MSE)
tau_anomaly = float(np.percentile(normal_mse, 95))
print(f"Calibrated Anomaly Threshold (tau_anomaly): {tau_anomaly:.4f}")

# 5. Financial Cost Matrix & Threshold Optimization
print("\n[Step 5/6] Evaluating Business Loss Matrix (Cost = $120*FN + $2.50*FP)...")
cost_fn = 120.0
cost_fp = 2.50

threshold_benchmarks = {}
candidates = [0.10, 0.20, 0.35, 0.50, 0.70, 0.80]

for th in candidates:
    pred_th = (y_proba >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_th).ravel()
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    threshold_benchmarks[str(th)] = {
        'threshold': th,
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
        'TN': int(tn),
        'Precision': round(prec * 100, 2),
        'Recall': round(rec * 100, 2),
        'Total_Cost': round(total_cost, 2)
    }

# Find best threshold
optimal_th = min(threshold_benchmarks.values(), key=lambda x: x['Total_Cost'])
default_cost = threshold_benchmarks['0.5']['Total_Cost']
savings_pct = ((default_cost - optimal_th['Total_Cost']) / default_cost) * 100

print(f"Cost at Default (tau=0.50): ${default_cost:,.2f}")
print(f"Cost at Optimal (tau={optimal_th['threshold']}): ${optimal_th['Total_Cost']:,.2f}")
print(f"Cost Reduction Achieved: {savings_pct:.1f}%")

# Save config
config = {
    'model_name': model_name,
    'optimal_threshold': optimal_th['threshold'],
    'tau_anomaly': round(tau_anomaly, 4),
    'roc_auc': round(roc_auc, 4),
    'pr_auc': round(pr_auc, 4),
    'scoring_latency_ms': round(scoring_latency_ms, 2),
    'benchmarks': threshold_benchmarks
}

with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=4)
print("Saved model configuration to model_config.json")

# 6. Save Sample Demo Transactions
print("\n[Step 6/6] Generating balanced sample demo feeds...")
sample_legit = df[df['Class'] == 0].sample(n=80, random_state=42)
sample_fraud = df[df['Class'] == 1].sample(n=20, random_state=42)
demo_feed = pd.concat([sample_legit, sample_fraud]).sample(frac=1.0, random_state=42)
demo_feed.to_csv('demo_feed_transactions.csv', index=False)
print("Saved 100 demo transactions with 20 fraud cases to demo_feed_transactions.csv")

print("\n" + "="*60)
print("TRAINING PIPELINE COMPLETE! ALL ARTIFACTS SUCCESSFULLY SAVED.")
print("="*60)
