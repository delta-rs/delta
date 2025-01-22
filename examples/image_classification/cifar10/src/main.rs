use deltaml::activations::{ReluActivation, SoftmaxActivation};
use deltaml::dataset::Cifar10Dataset;
use deltaml::dataset::base::ImageDatasetOps;
use deltaml::losses::MeanSquaredLoss;
use deltaml::neuralnet::{Dense, Flatten, Sequential};
use deltaml::optimizers::Adam;

#[tokio::main]
async fn main() {
    // Create a neural network
    let mut model = Sequential::new()
        .add(Flatten::new(&[32, 32, 3])) // CIFAR-10: 32x32x3 -> 3072
        .add(Dense::new(128, Some(ReluActivation::new()), true)) // Input: 3072, Output: 128
        .add(Dense::new(10, Some(SoftmaxActivation::new()), false)); // Output: 10 classes

    // Display the model summary
    model.summary();

    // Chose either CPU or GPU
    model.use_optimized_device();

    // Define an optimizer
    let optimizer = Adam::new(0.001);

    // Compile the model
    model.compile(optimizer, MeanSquaredLoss::new());

    // Loading the train and test dataset
    let mut train_data = Cifar10Dataset::load_train().await;
    let mut test_data = Cifar10Dataset::load_test().await;
    let mut val_data = Cifar10Dataset::load_val().await;

    println!("Training the model...");
    println!("Train dataset size: {}", train_data.len());

    let epoch = 10;
    let batch_size = 32;

    match model.fit(&mut train_data, epoch, batch_size) {
        Ok(_) => println!("Model trained successfully"),
        Err(e) => println!("Failed to train model: {}", e),
    }

    // Validate the model
    match model.validate(&mut val_data, batch_size) {
        Ok(validation_loss) => println!("Validation Loss: {:.6}", validation_loss),
        Err(e) => println!("Failed to validate model: {}", e),
    }

    // Evaluate the model
    let accuracy =
        model.evaluate(&mut test_data, batch_size).expect("Failed to evaluate the model");
    println!("Test Accuracy: {:.2}%", accuracy * 100.0);

    // Save the model
    model.save("model_path").unwrap();
}
