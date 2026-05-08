import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


digits = datasets.load_digits()
n_samples = len(digits.images)
image_shape = digits.images[0].shape
classes = np.unique(digits.target)

print(f"数据集中图像的数量: {n_samples}")
print(f"每张图像的大小: {image_shape[0]} × {image_shape[1]}")
print(f"类别标签: {classes}")

fig, axes = plt.subplots(1, 10, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Label: {label}")
plt.suptitle("Sample Images")
plt.tight_layout()
plt.savefig('1.png')
plt.show()

X = digits.data  
y = digits.target
print(f"单张图像特征向量维度: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")

models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "SVM": SVC(kernel='rbf', gamma='scale'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results.append({"模型": name, "测试准确率": f"{acc:.4f}"})
    
results_df = pd.DataFrame(results)
print("\n模型准确率比较表格：")
print(results_df.to_string(index=False))

best_model_name = "SVM" 
best_model = trained_models[best_model_name]
y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix ({best_model_name})")
plt.tight_layout()
plt.savefig('2.png')
plt.show()

misclassified_idx = np.where(y_test != y_pred_best)[0]
print(f"{best_model_name} 模型共误判了 {len(misclassified_idx)} 个样本。")

if len(misclassified_idx) > 0:
    num_to_show = min(5, len(misclassified_idx))
    fig, axes = plt.subplots(1, num_to_show, figsize=(12, 3))
    
    if num_to_show == 1:
        axes = [axes]
        
    for i, ax in enumerate(axes):
        idx = misclassified_idx[i]
        image = X_test[idx].reshape(8, 8)
        true_label = y_test[idx]
        pred_label = y_pred_best[idx]
        
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color="red")
        
    plt.suptitle("Misclassified Samples")
    plt.tight_layout()
    plt.savefig('3.png')
    plt.show()

