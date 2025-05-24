import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score

import numpy as np
import torch
import matplotlib.pyplot as plt

def visualise_reconstruction(trainer, dataset, idx=0):
    # --- load one slice + mask ---
    x, y = dataset[idx]
    x = x.unsqueeze(0).to(trainer.device)   # (1,4,128,128) or (1,1,128,128)
    y_np = y.squeeze().numpy()              # (128,128) ground truth mask

    # --- forward pass ---
    trainer.model.eval()
    with torch.no_grad():
        recon = trainer.model(x)

    # --- move to numpy (channel 0 for display) ---
    x_np    = x.cpu().squeeze().numpy()[0]     # (128,128)
    recon_np= recon.cpu().squeeze().numpy()[0] # (128,128)
    diff    = np.abs(x_np - recon_np)          # raw error map

    # --- mask out background (optional) ---
    brain_mask = (x_np > 0)                    # or use dataset foreground mask
    diff_masked = diff * brain_mask

    # --- rescale for display ---
    # clamp at 99th percentile to avoid outliers dominating
    vmax = np.percentile(diff_masked, 99)
    vmin = 0.0

    # --- tumor presence info ---
    has_tumor   = y_np.sum() > 0
    label_info  = "Tumor Present" if has_tumor else "Healthy Slice"
    print(f"[INFO] Slice {idx}: {label_info} (mask pixels = {y_np.sum()})")

    # --- plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(x_np,    cmap='gray')
    axes[0].set_title(f"Original ({label_info})")
    axes[0].axis('off')

    axes[1].imshow(recon_np, cmap='gray')
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')

    im = axes[2].imshow(
        diff_masked,
        cmap='magma',
        vmin=0, vmax=vmax
    )
    axes[2].set_title("Anomaly Map")
    axes[2].axis('off')

    axes[3].imshow(y_np, cmap='gray')
    axes[3].set_title("Ground Truth Mask")
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()


def plot_pr_curve(y_true, y_scores):
    y_true = y_true.cpu().numpy().astype(bool)
    y_scores = y_scores.cpu().numpy().astype(float)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    