import matplotlib.pyplot as plt
import os

# Mean results from your summary table
methods = ["One-Hot", "Label"]

accuracy = [0.809, 0.803]
precision = [0.822, 0.822]
recall = [0.833, 0.818]
f1 = [0.825, 0.818]

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1
}

# Ensure figures folder exists
os.makedirs("figures", exist_ok=True)

# Generate and save plots
for metric_name, values in metrics.items():
    plt.figure()
    plt.bar(methods, values)
    plt.title(f"{metric_name} Comparison")
    plt.ylabel(metric_name)
    plt.ylim(0.6, 0.95)

    filename = f"{metric_name.lower().replace('-', '').replace(' ', '')}_comparison.png"
    plt.savefig(f"figures/{filename}", dpi=200, bbox_inches="tight")
    plt.close()

print("✅ Plots generated successfully inside /figures folder")
