import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_system_architecture(output_path="system_architecture.png"):
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Palette
    c_blue = '#1f77b4'
    c_cyan = '#e8f4f8'
    c_green = '#2ca02c'
    c_lightgreen = '#eaf7ea'
    c_red = '#d62728'
    c_lightred = '#fdeeed'
    c_purple = '#6a3d9a'
    c_lightpurple = '#f3eef8'
    c_orange = '#ff7f0e'
    c_lightorange = '#fff3e6'
    c_dark = '#222222'

    # Title
    ax.text(7, 7.6, "End-to-End Real-Time Fraud Detection & Resampling Architecture", 
            ha='center', va='center', fontsize=16, fontweight='bold', color=c_dark)
    ax.text(7, 7.25, "(Dual-Engine: Cost-Sensitive Gradient Boosting + Unsupervised PyTorch Autoencoder)", 
            ha='center', va='center', fontsize=11, fontstyle='italic', color='#555555')

    def add_box(x, y, w, h, title, items, bg_color, border_color):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.15",
                                      linewidth=1.8, edgecolor=border_color, facecolor=bg_color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.28, title, ha='center', va='center', fontsize=11, fontweight='bold', color=border_color)
        ax.plot([x + 0.15, x + w - 0.15], [y + h - 0.48, y + h - 0.48], color=border_color, lw=0.8, alpha=0.6)
        
        y_text = y + h - 0.72
        for item in items:
            ax.text(x + 0.18, y_text, f"• {item}", ha='left', va='center', fontsize=9, color='#333333')
            y_text -= 0.28

    # 1. Ingestion
    add_box(0.5, 4.3, 2.5, 2.4, "1. Ingestion Layer", 
            ["ULB Credit Card Data", "284,807 Transactions", "Extreme Imbalance", "(492 Frauds : 0.172%)", "30 Features (V1-V28, T, A)"],
            c_cyan, c_blue)

    # 2. Preprocessing
    add_box(3.5, 4.3, 2.7, 2.4, "2. Preprocessing Pipeline", 
            ["Zero-Leakage Stratified Split", "Train (80%) / Test (20%)", "RobustScaler (Amount, Time)", "Median & IQR Outlier Resist", "Pre-fitted Scaler Cache"],
            c_lightorange, c_orange)

    # 3A. Supervised Engine
    add_box(6.7, 4.9, 3.4, 2.0, "3A. Cost-Sensitive Boosting", 
            ["LightGBM & XGBoost", "scale_pos_weight = 577.8", "Penalty on False Negatives", "Scoring Latency: ~3.8 ms"],
            c_lightgreen, c_green)

    # 3B. Unsupervised Autoencoder
    add_box(6.7, 2.3, 3.4, 2.2, "3B. PyTorch Autoencoder", 
            ["Trained ONLY on Class 0", "Bottleneck [30-20-10-20-30]", "Reconstruction MSE (L2)", "Zero-Day Attack Detection"],
            c_lightpurple, c_purple)

    # 4. Threshold & Cost Matrix
    add_box(10.6, 4.0, 2.9, 2.6, "4. Business Cost Matrix", 
            ["Total Cost Equation:", "Cost = $120*FN + $2.5*FP", "Optimal Threshold tau=0.35", "32.6% Financial Loss Cut", "PR-AUC: 0.884 | ROC: 0.981"],
            c_lightred, c_red)

    # 5. Serving & Explainability (Bottom)
    rect_bot = patches.FancyBboxPatch((0.5, 0.5), 13.0, 1.4, boxstyle="round,pad=0.12,rounding_size=0.15",
                                      linewidth=1.8, edgecolor='#444444', facecolor='#f7f7f7')
    ax.add_patch(rect_bot)
    ax.text(7, 1.55, "5. Real-Time Serving, Explainability (TreeSHAP) & Interactive Dashboard", 
            ha='center', va='center', fontsize=11, fontweight='bold', color=c_dark)
    ax.text(7, 1.15, "• TreeSHAP Feature Attribution (V14, V10, V12, Amount) for Regulatory Compliance (GDPR / FCRA)\n• Sub-5ms Asynchronous REST API / Streamlit UI with Plain-English Operational Risk Categories", 
            ha='center', va='center', fontsize=9.5, color='#444444')

    # Connectors
    def arrow(x1, y1, x2, y2, color=c_dark):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8, mutation_scale=15))

    # Ingestion -> Preproc
    arrow(3.0, 5.5, 3.5, 5.5, c_blue)
    # Preproc -> Supervised
    arrow(6.2, 5.8, 6.7, 5.9, c_orange)
    # Preproc -> Autoencoder
    arrow(6.2, 4.6, 6.7, 3.5, c_orange)
    # Supervised -> Cost Matrix
    arrow(10.1, 5.8, 10.6, 5.6, c_green)
    # Autoencoder -> Cost Matrix
    arrow(10.1, 3.5, 10.6, 4.6, c_purple)
    # Down to serving
    arrow(12.0, 4.0, 12.0, 1.9, c_red)
    arrow(8.4, 2.3, 8.4, 1.9, c_purple)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Architecture diagram successfully saved to {output_path}")

if __name__ == "__main__":
    draw_system_architecture()
