use deltaml::activations::relu::ReluActivation;
use deltaml::activations::softmax::SoftmaxActivation;
use deltaml::common::ndarray::{IxDyn, Shape};
use deltaml::dataset::ImageNetV2Dataset;
use deltaml::dataset::base::ImageDatasetOps;
use deltaml::losses::SparseCategoricalCrossEntropyLoss;
use deltaml::neuralnet::{Dense, Flatten, Sequential};
use deltaml::optimizers::Adam;

#[tokio::main]
async fn main() {
    // Create a neural network
    let mut model = Sequential::new()
        .add(Flatten::new(Shape::from(IxDyn(&[224, 224, 3])))) // Input: 224x224x3 (ImageNet images)
        .add(Dense::new(512, Some(ReluActivation::new()), true)) // Hidden layer: 512 units
        .add(Dense::new(256, Some(ReluActivation::new()), true)) // Hidden layer: 256 units
        .add(Dense::new(1000, None::<SoftmaxActivation>, false)); // Output: 1000 classes (ImageNet categories)

    // Display the model summary
    model.summary();

    // Chose either CPU or GPU
    model.use_optimized_device();

    // Define an optimizer
    let optimizer = Adam::new(0.001);

    // Compile the model
    model.compile(optimizer, SparseCategoricalCrossEntropyLoss::new());

    // Load the train and test dataset
    let mut train_data = ImageNetV2Dataset::load_train().await;
    let mut test_data = ImageNetV2Dataset::load_test().await;
    let mut val_data = ImageNetV2Dataset::load_val().await;

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
    println!("Evaluating the model...");
    let accuracy = model.evaluate(&mut test_data, batch_size).expect("Failed to evaluate model");
    println!("Test Accuracy: {:.2} %", accuracy * 100.0);

    // Save the model
    let model_path = ".cache/models/imagenetv2/imagenetv2_model";
    println!("Saving the model to {}...", model_path);
    model.save(model_path).unwrap();

    println!("Model training and evaluation complete.");
}
