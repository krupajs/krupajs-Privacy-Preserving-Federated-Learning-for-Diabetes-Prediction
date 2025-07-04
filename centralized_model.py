# centralized_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from utils import load_data  # Make sure utils.py is correct!

# Neural Network Model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ----------------------------
# Functions to train and evaluate
# ----------------------------

def train_and_evaluate_nn(X_train, y_train, X_test, y_test):
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train.to_numpy()).float()
        y_train = torch.tensor(y_train.to_numpy()).long()

    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    # Evaluation
    model.eval()
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test.to_numpy()).float()
        y_test = torch.tensor(y_test.to_numpy()).long()

    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)

    print(f"Neural Network Accuracy: {accuracy:.4f}")
    return model, accuracy  # returning model as well


def train_and_evaluate_lr(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {acc:.4f}")
    return model, acc  # returning model as well


def train_and_evaluate_rf(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {acc:.4f}")
    return model, acc


# ----------------------------
# Wrappers for Federated Learning usage
# ----------------------------

def train_centralized_nn(X_train, y_train):
    """Train NN model (without evaluating on test set)."""
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train.to_numpy()).float()
        y_train = torch.tensor(y_train.to_numpy()).long()

    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model


def train_centralized_lr(X_train, y_train):
    """Train Logistic Regression (no evaluation)."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


# ----------------------------
# Main centralized training and evaluation
# ----------------------------

def train_and_evaluate_models():
    X_train, y_train, X_test, y_test = load_data()

    print("\nTraining and Evaluating Neural Network:")
    _, nn_acc = train_and_evaluate_nn(X_train, y_train, X_test, y_test)

    print("\nTraining and Evaluating Logistic Regression:")
    _, lr_acc = train_and_evaluate_lr(X_train, y_train, X_test, y_test)

    # print("\nTraining and Evaluating Random Forest:")
    # _, rf_acc = train_and_evaluate_rf(X_train, y_train, X_test, y_test)

    print("\nPerformance Comparison of Centralized Models:")
    print(f"Neural Network Accuracy: {nn_acc:.4f}")
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    # print(f"Random Forest Accuracy: {rf_acc:.4f}")


if __name__ == "__main__":
    train_and_evaluate_models()
