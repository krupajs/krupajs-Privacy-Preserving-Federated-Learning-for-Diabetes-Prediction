import flwr as fl
from flwr.server.server import ServerConfig
from flwr.server.strategy import FedAvg

# Global list to store aggregated accuracies
aggregated_accuracies = []

# Define custom evaluation function
def get_evaluate_fn():
    def evaluate(server_round, parameters, config):
        global aggregated_accuracies
        
        # Collect evaluation results from all clients
        results, failures = fl.server.strategy.aggregate_evaluate(server_round, results=None, failures=None)

        if results is not None:
            client_metrics = []
            for _, metrics in results:
                acc = metrics.get("accuracy", 0)
                client_metrics.append(acc)

            if client_metrics:
                aggregated_acc = sum(client_metrics) / len(client_metrics)
                aggregated_accuracies.append(aggregated_acc)

                print(f"\n✅ [Round {server_round}] Aggregated Federated Accuracy: {aggregated_acc:.4f}\n")
        
        return None, {}

    return evaluate

def start_server():
    # Attach the custom evaluation function
    strategy = FedAvg(evaluate_fn=get_evaluate_fn())
    
    server_config = ServerConfig(num_rounds=5)

    fl.server.start_server(
        server_address="localhost:8080",
        config=server_config,
        strategy=strategy
    )

    # After server stops, print final results
    print("\n==== Final Aggregated Accuracies per Round ====")
    for i, acc in enumerate(aggregated_accuracies, 1):
        print(f"Round {i}: {acc:.4f}")

    if aggregated_accuracies:
        final_avg = sum(aggregated_accuracies) / len(aggregated_accuracies)
        print(f"\n==== Final Aggregated Accuracy After 5 Rounds: {final_avg:.4f} ====")

if __name__ == "__main__":
    start_server()
