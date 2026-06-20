print("---- Python Shortcuts ----")

features = ["Mean Radius", "Mean Texture", "Mean Perimeter"]
importances = [0.423, 0.158, 0.419]

summary = [f"{feat}: {imp*100:.1f}%" for feat, imp in zip(features, importances)]
print(summary)

for i, item in enumerate(summary):
    print(f"Rank {i+1} -> {item}")