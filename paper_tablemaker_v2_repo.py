import json
import csv
import argparse
import os


def load_data(file_path):
    """Load experiment data from JSON file."""
    print(f"Loading data from {file_path}...")
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def extract_final_metrics(data, models_to_extract, metrics_to_extract):
    """
    Extract final metric values for selected models.

    Parameters:
    -----------
    data : dict
        The loaded JSON data containing experiment results
    models_to_extract : list
        List of model keys to extract metrics from
    metrics_to_extract : list
        List of metrics to extract

    Returns:
    --------
    dict: Dictionary with model names as keys and dictionaries of metric values as values
    """
    results = {}

    metric_mapping = {
        "accuracy": "online_accuracy",
        "f1": "online_f1",
        "precision": "online_precision",
        "recall": "online_recall",
        "macrof1": "online_macrof1",
        "cohenkappa": "online_cohenkappa",
    }

    full_metric_keys = [metric_mapping.get(m.lower(), m) for m in metrics_to_extract]

    for model_key in models_to_extract:
        if model_key in data["models"]:
            model_data = data["models"][model_key]
            model_name = model_data.get("name", model_key)

            results[model_name] = {}

            for metric, full_metric_key in zip(metrics_to_extract, full_metric_keys):
                if full_metric_key in model_data:
                    if model_data[full_metric_key]:
                        results[model_name][metric] = round(
                            model_data[full_metric_key][-1], 3
                        )
                    else:
                        results[model_name][metric] = "N/A"
                else:
                    print(
                        f"Warning: Metric '{metric}' not found for model '{model_key}'"
                    )
                    results[model_name][metric] = "N/A"
        else:
            print(f"Warning: Model '{model_key}' not found in data")

    return results


def save_to_csv(results, metrics, output_file):
    """Save extracted metrics to a CSV file."""
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["Model"] + metrics
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for model_name, model_metrics in results.items():
            row = {"Model": model_name}
            row.update(model_metrics)
            writer.writerow(row)

    print(f"Results saved to {output_file}")


def list_available_models(data):
    """Print all available models in the data."""
    print("\nAvailable models in the dataset:")
    for model_key, model_data in data["models"].items():
        model_name = model_data.get("name", model_key)
        print(f"  - {model_key} ({model_name})")


def list_available_metrics(data):
    """Print all available metrics in the first model of the data."""
    if not data["models"]:
        print("No models found in the data")
        return

    first_model_key = next(iter(data["models"]))
    model_data = data["models"][first_model_key]

    metrics = [key for key in model_data.keys() if key.startswith("online_")]

    print("\nAvailable metrics in the dataset:")
    for metric in metrics:
        display_name = metric.replace("online_", "")
        print(f"  - {display_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract final metrics for models and save to CSV"
    )
    parser.add_argument(
        "--file", "-f", type=str, required=True, help="Path to the JSON data file"
    )
    parser.add_argument(
        "--models", "-m", type=str, nargs="+", help="Models to extract metrics for"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["accuracy"],
        help="Metrics to extract (default: accuracy)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="model_metrics.csv",
        help="Output CSV file path (default: model_metrics.csv)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models in the data file",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List all available metrics in the data file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found")
        return

    data = load_data(args.file)

    if args.list_models:
        list_available_models(data)
        return

    if args.list_metrics:
        list_available_metrics(data)
        return

    if not args.models:
        print(
            "Error: No models specified. Use --models to specify which models to extract metrics for."
        )
        list_available_models(data)
        return

    results = extract_final_metrics(data, args.models, args.metrics)

    save_to_csv(results, args.metrics, args.output)


if __name__ == "__main__":
    main()
