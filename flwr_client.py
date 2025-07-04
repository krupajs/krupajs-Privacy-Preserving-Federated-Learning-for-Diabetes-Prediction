import flwr as fl
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score
from utils import load_data
from centralized_model import SimpleNet, train_centralized_nn, train_centralized_lr

# Load data
X_train, y_train, X_test, y_test = load_data()

# Neural Network Client
class NNClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = SimpleNet()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.local_epochs = 10  # Increase the number of local epochs
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        # Training loop for each epoch
        for epoch in range(self.local_epochs):
            outputs = self.model(X_train)
            loss = self.loss_fn(outputs, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print loss every 2 epochs
            if (epoch + 1) % 2 == 0:
                print(f"[Client] Epoch {epoch + 1}/{self.local_epochs}, Loss: {loss.item():.4f}")

        # Step the learning rate scheduler
        self.scheduler.step()

        return self.get_parameters(config), len(X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        outputs = self.model(X_test)
        loss = self.loss_fn(outputs, y_test)
        
        # Convert outputs to predictions
        preds = outputs.argmax(1).cpu().numpy()
        labels = y_test.cpu().numpy()

        acc = (preds == labels).sum().item() / len(labels)
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')

        print(f"[Client] Federated NN Evaluation -> Loss: {loss.item():.4f}, Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        return loss.item(), len(X_test), {"accuracy": acc, "precision": precision, "recall": recall}


# Logistic Regression Client
class LRClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def get_parameters(self, config):
        if hasattr(self.model, 'coef_'):
            return [self.model.coef_, self.model.intercept_]
        else:
            n_features = X_train.shape[1]
            n_classes = len(np.unique(y_train))
            return [np.zeros((n_classes, n_features)), np.zeros(n_classes)]

    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]
        self.model.classes_ = np.unique(y_train)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(X_train, y_train)
        return self.get_parameters(config), len(X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        y_pred_probs = self.model.predict_proba(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_pred_probs)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        print(f"[Client] Federated LR Evaluation -> Loss: {loss:.4f}, Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        return loss, len(X_test), {"accuracy": acc, "precision": precision, "recall": recall}


# Choose client
def get_client(model_type="nn"):
    if model_type == "nn":
        return NNClient()
    elif model_type == "lr":
        return LRClient()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# Compare Centralized and Federated model
def compare_performance(model_type="nn"):
    client = get_client(model_type)

    # Evaluate Federated Model
    print(f"\n--- Evaluating Federated {model_type.upper()} Model ---")
    fed_loss, fed_len, fed_metrics = client.evaluate(client.get_parameters({}), {})
    fed_acc = fed_metrics["accuracy"]

    # Evaluate Centralized Model
    # print(f"\n--- Evaluating Centralized {model_type.upper()} Model ---")
    # if model_type == "nn":
    #     centralized_acc = train_centralized_nn()
    # elif model_type == "lr":
    #     centralized_acc = train_centralized_lr()

    # Final Comparison
    print("\n==============================")
    print(f"Federated Accuracy   : {fed_acc:.4f}")
    # print(f"Centralized Accuracy : {centralized_acc:.4f}")
    print("==============================")


if __name__ == "__main__":
    model_type = "nn"  # Change to "lr" for Logistic Regression

    client = get_client(model_type)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())

    # After federated learning completes, compare performance
    compare_performance(model_type)
