import json
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train decision tree
from sklearn.model_selection import train_test_split

# Match reference: split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Match reference: max_depth=3, random_state=42
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

tree = clf.tree_

# Helper to get samples at each node
def get_node_samples(tree, X):
    node_indicator = clf.decision_path(X)
    node_samples = {}
    for node_id in range(tree.node_count):
        # Samples that pass through this node
        sample_indices = np.where(node_indicator[:, node_id].toarray().ravel())[0]
        node_samples[node_id] = sample_indices.tolist()
    return node_samples

# Helper to calculate gini and its formula
def gini_formula(counts):
    total = sum(counts)
    if total == 0:
        return 0.0, "0"
    probs = [c / total for c in counts]
    gini = 1 - sum(p ** 2 for p in probs)
    formula = f"1 - (" + " + ".join([f"({c}/{total})^2" for c in counts]) + ")"
    numbers = f"1 - (" + " + ".join([f"{p:.3f}^2" for p in probs]) + ")"
    calc = f"{formula} = {numbers} = {gini:.3f}"
    return gini, calc

# Traverse tree and collect info
def extract_tree_info(tree, y, node_samples):
    def recurse(node_id):
        samples = node_samples[node_id]
        class_counts = [int(np.sum(y[samples] == i)) for i in range(int(tree.n_classes[0]))]
        gini, gini_calc = gini_formula(class_counts)
        node_info = {
            'id': int(node_id),
            'gini': float(gini),
            'gini_calc': gini_calc,
            'samples': [int(idx) for idx in samples],
            'class_counts': [int(c) for c in class_counts]
        }
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            node_info['left'] = recurse(int(tree.children_left[node_id]))
            node_info['right'] = recurse(int(tree.children_right[node_id]))
        return node_info
    return recurse(0)

node_samples = get_node_samples(tree, X)
tree_info = extract_tree_info(tree, y, node_samples)

node_samples = get_node_samples(tree, X_train)
tree_info = extract_tree_info(tree, y_train, node_samples)

# Also export test samples for future use
test_samples = []
for i in range(len(X_test)):
    test_samples.append({
        'features': [float(x) for x in X_test[i]],
        'label': int(y_test[i])
    })


# Also export y_train so the frontend can use true class labels for the sample grid
with open('tree_data.json', 'w') as f:
    json.dump({'tree': tree_info, 'test_samples': test_samples, 'train_labels': [int(l) for l in y_train]}, f, indent=2)

print("Exported tree and test samples to tree_data.json")
