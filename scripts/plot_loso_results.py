from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Location of the ASJDA results
results_file = Path("runs/deap_valence_loso/loso_summary.csv")

# Load results
df = pd.read_csv(results_file)

# Convert accuracy to percentage
df["accuracy_percent"] = df["accuracy"] * 100

# Calculate mean accuracy
mean_accuracy = df["accuracy"].mean() * 100

# Sort subjects naturally: s01, s02, ..., s32
df = df.sort_values("target_subject")

# Create the graph
plt.figure(figsize=(14, 6))

plt.bar(
    df["target_subject"],
    df["accuracy_percent"]
)

# Mean accuracy line
plt.axhline(
    mean_accuracy,
    linestyle="--",
    linewidth=2,
    label=f"Mean accuracy = {mean_accuracy:.2f}%"
)

# Labels and title
plt.title("ASJDA – DEAP Valence Classification (LOSO)", fontsize=16)
plt.xlabel("Target Subject")
plt.ylabel("Accuracy (%)")

# Make the graph easier to read
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

# Save high-quality image
output_file = Path("runs/deap_valence_loso/ASJDA_DEAP_LOSO_accuracy.png")
plt.savefig(output_file, dpi=300, bbox_inches="tight")

print(f"Mean accuracy: {mean_accuracy:.2f}%")
print(f"Graph saved to: {output_file}")

plt.show()