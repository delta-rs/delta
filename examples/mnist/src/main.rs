use delta_common::Shape;
use delta_data::mnist::MnistDataset;
use delta_data::DatasetOps;
use delta_losses::MeanSquaredLoss;
use delta_nn::layers::{Dense, Flatten, Relu};
use delta_nn::models::Sequential;
use delta_optimizers::Adam;

#[tokio::main]
async fn main() {
    // Create a neural network
    let mut model = Sequential::new()
        .add(Flatten::new(Shape::new(vec![28, 28]))) // Input: 28x28, Output: 784
        .add(Dense::new(784, 128)) // Input: 784, Output: 128
        .add(Relu::new()) // Activation: ReLU
        .add(Dense::new(128, 10)); // Output: 10 classes

    // Define an optimizer
    let optimizer = Adam::new(0.001);

    // Compile the model
    model.compile(optimizer, MeanSquaredLoss::new());

    // Train the model
    println!("Training...");
    let train_data = MnistDataset::load_train().await;
    let test_data = MnistDataset::load_test().await;

    println!("Training the model...");
    println!("Train data size: {}", train_data.len());

    let epoch = 10;
    let batch_size = 32;

    model.fit(&train_data, epoch, batch_size);

    // Evaluate the model
    let accuracy = model.evaluate(&test_data);
    println!("Test Accuracy: {:.2}%", accuracy * 100.0);

    // Save the model
    model.save("model_path").unwrap();
}
