# Delta <!-- omit in toc --> ![build](https://img.shields.io/github/actions/workflow/status/delta-rs/delta/core.yml?branch=master)

[//]: # (![crates.io]&#40;https://img.shields.io/crates/v/delta.svg&#41;)
[//]: # ([![documentation]&#40;https://img.shields.io/badge/docs-delta-blue?logo=rust&#41;]&#40;https://docs.rs/delta/latest/&#41;)

An open source Machine Learning Framework in Rust Δ.

## Table of Contents <!-- omit in toc -->

- [Desired API Usage](#desired-api-usage)
- [Getting Help](#getting-help)
- [Reporting Issues](#reporting-issues)
- [Contributing](#contributing)
- [Contributors](#contributors)
- [License](#license)

## Desired API Usage

This is just a rough idea of what the API could look like. This is not the final API.

Example 1:

```rust
use delta_tensor::Tensor;
use delta_nn::layers::{Dense, Relu};
use delta_nn::models::Sequential;
use delta_optimizers::Adam;

fn main() {
    // Create a neural network
    let mut model = Sequential::new()
        .add(Dense::new(784, 128))   // Input: 784, Output: 128
        .add(Relu::new())            // Activation: ReLU
        .add(Dense::new(128, 10));   // Output: 10 classes

    // Define an optimizer
    let optimizer = Adam::new(learning_rate: 0.001);

    // Compile the model
    model.compile(optimizer);

    // Train the model
    let train_data = delta_datasets::mnist::load_train();
    let test_data = delta_datasets::mnist::load_test();

    model.fit(train_data, epochs: 10, batch_size: 32);

    // Evaluate the model
    let accuracy = model.evaluate(test_data);
    println!("Test Accuracy: {:.2}%", accuracy * 100.0);

    // Save the model
    model.save("model_path").unwrap();
}
```

Example two:

```rust
use delta_tensor::Tensor;
use delta_nn::layers::{Dense, Dropout, Relu, Softmax};
use delta_nn::models::Sequential;
use delta_optimizers::{Adam, LearningRateScheduler};
use delta_datasets::mnist::{self, Dataset};
use delta_callbacks::{EarlyStopping, ModelCheckpoint};

fn main() {
    // Load and preprocess the dataset
    let mut train_data = mnist::load_train();
    let mut test_data = mnist::load_test();

    // Data augmentation (example: normalize and add noise)
    train_data.normalize(0.0, 1.0); // Normalize to [0, 1]
    train_data.add_noise(0.05);     // Add Gaussian noise
    test_data.normalize(0.0, 1.0);

    // Create a neural network
    let mut model = Sequential::new()
        .add(Dense::new(784, 128))     // Input: 784, Output: 128
        .add(Relu::new())              // Activation: ReLU
        .add(Dropout::new(0.2))        // Dropout with 20% probability
        .add(Dense::new(128, 64))      // Intermediate layer
        .add(Relu::new())
        .add(Dense::new(64, 10))       // Output: 10 classes
        .add(Softmax::new());          // Output probabilities

    // Define an advanced optimizer with learning rate scheduling
    let mut optimizer = Adam::new(0.001);
    let scheduler = LearningRateScheduler::new(
        |epoch| if epoch < 5 { 0.001 } else { 0.0001 }
    );
    optimizer.set_scheduler(scheduler);

    // Compile the model
    model.compile(optimizer);

    // Define callbacks
    let early_stopping = EarlyStopping::new()
        .patience(3) // Stop training if no improvement for 3 epochs
        .monitor("val_accuracy");
    let checkpoint = ModelCheckpoint::new("best_model_path")
        .save_best_only(true) // Save only the best model
        .monitor("val_accuracy");

    // Train the model using a custom loop
    for epoch in 1..=10 {
        println!("Epoch {}/10", epoch);

        // Training step
        model.train(&train_data, 32);

        // Validation step
        let val_accuracy = model.validate(&test_data);
        println!("Validation Accuracy: {:.2}%", val_accuracy * 100.0);

        // Trigger callbacks
        early_stopping.step(val_accuracy);
        checkpoint.step(&model, val_accuracy);

        // Check for early stopping
        if early_stopping.should_stop() {
            println!("Early stopping triggered.");
            break;
        }
    }

    // Evaluate the final model
    let test_accuracy = model.evaluate(&test_data);
    println!("Final Test Accuracy: {:.2}%", test_accuracy * 100.0);

    // Save the final model
    model.save("final_model_path").unwrap();
}
```

## Getting Help

Are you having trouble with Delta? We want to help!

[//]: # (- Read through the documentation on our [docs]&#40;https://docs.rs/delta/latest/delta/&#41;.)

- If you are upgrading, read the release notes for upgrade instructions and "new and noteworthy" features.

- Ask a question we monitor stackoverflow.com for questions tagged with `delta-rs`.

- Report bugs with Delta at https://github.com/delta-rs/delta/issues.

## Reporting Issues

Delta uses GitHub’s integrated issue tracking system to record bugs and feature requests. If you want to raise an issue, please follow the recommendations below:

- Before you log a bug, please search the issue tracker to see if someone has already reported the problem.

- If the issue doesn’t already exist, create a new issue.

- Please provide as much information as possible with the issue report. We like to know the Delta version, operating system, and Rust version version you’re using.

- If you need to paste code or include a stack trace, use Markdown. ``` escapes before and after your text.

- If possible, try to create a test case or project that replicates the problem and attach it to the issue.

## Contributing

Before contributing, please read the [contribution](https://github.com/delta-rs/Delta/blob/master/CONTRIBUTING.md) guide for useful information how to get started with Delta as well as what should be included when submitting a contribution to the project.

## Contributors

The following contributors have either helped to start this project, have contributed
code, are actively maintaining it (including documentation), or in other ways
being awesome contributors to this project. **We'd like to take a moment to recognize them.**

[<img src="https://github.com/mjovanc.png?size=72" alt="mjovanc" width="72">](https://github.com/mjovanc)

## License

The BSD 3-Clause License.
