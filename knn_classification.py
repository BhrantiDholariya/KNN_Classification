# ============================================================
#  Project 2: Data Classification Using AI
#  Algorithm : K-Nearest Neighbors (KNN)
#  Dataset   : Iris Dataset (sklearn)
#  Batch     : 2026 | Powered by DecodeLabs
# ============================================================

# ── Imports ──────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ── Step 1: Load & Understand the Dataset ────────────────────
dataset = load_iris()

print("=" * 50)
print("         STEP 1: DATASET OVERVIEW")
print("=" * 50)
print("Keys          :", dataset.keys())
print("Number of rows:", dataset.data.shape[0])
print("Number of cols:", dataset.data.shape[1])
print("Shape         :", dataset.data.shape)
print("Feature Names :", dataset.feature_names)
print("\nTop 5 rows:\n", dataset.data[:5])

# ── Step 2: Prepare Features and Labels ──────────────────────
X = dataset.data
y = dataset.target

# ── Step 3: Split Data into Training and Testing Sets ────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("=" * 50)
print("         STEP 2: TRAIN-TEST SPLIT")
print("=" * 50)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)

# ── Step 4: Train the KNN Model ──────────────────────────────
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# ── Step 5: Make Predictions ─────────────────────────────────
y_pred = knn.predict(X_test)

# ── Step 6: Evaluate the Model ───────────────────────────────
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("         STEP 3: MODEL EVALUATION")
print("=" * 50)
print(f"Accuracy: {accuracy:.2f}")
print("=" * 50)
