import pandas as pd
import numpy as np

# Load original dataset
df = pd.read_csv("hand_landmarks.csv")  # or .xlsx
print("✅ Original data loaded:", df.shape)

# Check and assign proper column names
label_col = 'label'  # ✅ Update if your column is named differently
landmark_cols = [col for col in df.columns if col != label_col]

X = df[landmark_cols].values
y = df[label_col].values

def augment_landmarks(landmarks, noise_std=0.01, scale_range=0.05, rot_range=10, trans_range=0.02):
    # Convert to NumPy and reshape to 21 x 3
    lm = np.array(landmarks, dtype=np.float32).reshape(-1, 3)

    # Apply random scale
    scale = 1 + np.random.uniform(-scale_range, scale_range)
    lm *= scale

    # Apply 2D rotation (Z-axis)
    angle = np.radians(np.random.uniform(-rot_range, rot_range))
    cos_val, sin_val = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_val, -sin_val], [sin_val, cos_val]])
    lm[:, :2] = np.dot(lm[:, :2], rotation_matrix)

    # Apply translation
    lm[:, 0] += np.random.uniform(-trans_range, trans_range)
    lm[:, 1] += np.random.uniform(-trans_range, trans_range)

    # Apply Gaussian noise
    lm += np.random.normal(0, noise_std, lm.shape)

    return lm.flatten()

# Number of augmented samples per original
AUG_PER_SAMPLE = 3

augmented_X, augmented_y = [], []

for i in range(len(X)):
    for _ in range(AUG_PER_SAMPLE):
        aug = augment_landmarks(X[i])
        augmented_X.append(aug)
        augmented_y.append(y[i])

# Combine original + augmented
X_aug = np.vstack([X, augmented_X])
y_aug = np.concatenate([y, augmented_y])

# Save to new Excel
aug_df = pd.DataFrame(X_aug, columns=landmark_cols)
aug_df[label_col] = y_aug
aug_df.to_csv("augmented_landmarks.csv", index=False)

print("✅ Augmented data saved:", aug_df.shape)