import json
import matplotlib.pyplot as plt
import argparse
import os


def load_data(file_path):
    """Load experiment data from JSON file."""
    print(f"Loading data from {file_path}...")
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def plot_experiments(
    data, models_to_plot, metric="accuracy", add_title=False, output_file=None
):
    """
    Plot selected metrics for chosen models across iterations.

    Parameters:
    -----------
    data : dict
        The loaded JSON data containing experiment results
    models_to_plot : list
        List of model keys to plot from the data
    metric : str
        Which metric to plot ('accuracy', 'f1', 'precision', 'recall', etc.)
    add_title : bool
        Whether to add a title to the plot
    output_file : str
        If provided, save the plot to this file path
    """
    plt.figure(figsize=(14, 10))

    iterations = data.get("iterations", [])
    if not iterations:
        print("Error: No iterations found in data")
        return

    metric_mapping = {
        "accuracy": "online_accuracy",
        "f1": "online_f1",
        "precision": "online_precision",
        "recall": "online_recall",
        "macrof1": "online_macrof1",
        "cohenkappa": "online_cohenkappa",
    }

    metric_key = metric_mapping.get(metric.lower(), metric)

    if add_title:
        plt.title(f"{metric.capitalize()} over Iterations", fontsize=16)

    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "x", "+"]
    i = 0
    for model_key in models_to_plot:
        if model_key in data["models"]:
            model_data = data["models"][model_key]
            model_name = model_data.get("name", model_key)

            if metric_key in model_data:
                metric_values = model_data[metric_key]
                iterations_to_plot = iterations[: len(metric_values)]
                color = plt.cm.tab10(i % 10)  # Cycle through colors
                markerv = markers[i % len(markers)]
                linestyle = line_styles[i % len(line_styles)]
                plt.plot(
                    iterations_to_plot,
                    metric_values,
                    color=color,
                    linestyle=linestyle,
                    markevery=max(1, len(iterations) // 20),
                    marker=markerv,
                    markersize=4,
                    label=model_name,
                )
                i += 1
            else:
                print(f"Warning: Metric '{metric}' not found for model '{model_key}'")
        else:
            print(f"Warning: Model '{model_key}' not found in data")

    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel(f"{metric.capitalize()}", fontsize=14)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize="large",
        frameon=True,
        fancybox=True,
        shadow=True,
    )
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.ylim(0.77, 0.785)
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_file}")

    plt.show()


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
        description="Plot experiment metrics across iterations"
    )
    parser.add_argument(
        "--file", "-f", type=str, required=True, help="Path to the JSON data file"
    )
    parser.add_argument("--models", "-m", type=str, nargs="+", help="Models to plot")
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        help="Metric to plot (default: accuracy)",
    )
    parser.add_argument("--title", action="store_true", help="Add title to the plot")
    parser.add_argument(
        "--output", "-o", type=str, help="Output file path to save the plot"
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

    print("file", args.file)
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
            "Error: No models specified. Use --models to specify which models to plot."
        )
        list_available_models(data)
        return

    plot_experiments(data, args.models, args.metric, args.title, args.output)


if __name__ == "__main__":
    main()
