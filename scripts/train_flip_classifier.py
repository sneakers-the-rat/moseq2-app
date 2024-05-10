import click
from pathlib import Path
from moseq2_app.flip.train import (
    create_training_dataset,
    train_classifier,
    save_classifier,
)


@click.command()
@click.argument("data_index_path", type=click.Path(exists=True))
@click.option("--prefix", default="", type=str, help="Prefix for classifier file name")
@click.option(
    "--classifier-type",
    default="svm",
    type=click.Choice(["svm", "rf"]),
    help="Classifier model to use. Either a support vector machine (svm) or random forest (rf)",
)
def main(data_index_path, prefix, classifier_type):
    print("Creating training dataset")
    dataset_path = create_training_dataset(data_index_path)
    dataset_path = Path(dataset_path)

    if len(prefix) > 0 and not prefix.endswith("_"):
        prefix += "_"

    print(f"Training {classifier_type} classifier")
    classifier = train_classifier(dataset_path, classifier=classifier_type.upper())
    save_classifier(
        classifier, dataset_path.with_name(f"{prefix}{classifier}_flip_classifier.p")
    )


if __name__ == "__main__":
    main()
