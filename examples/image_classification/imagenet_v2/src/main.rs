use deltaml::activations::relu::ReluActivation;
use deltaml::activations::softmax::SoftmaxActivation;
use deltaml::common::shape::Shape;
use deltaml::dataset::base::ImageDatasetOps;
use deltaml::dataset::ImageNetV2Dataset;
use deltaml::losses::SparseCategoricalCrossEntropyLoss;
use deltaml::neuralnet::Sequential;
use deltaml::neuralnet::{Dense, Flatten};
use deltaml::optimizers::Adam;

#[tokio::main]
async fn main() {
    // Create a neural network
    let mut model = Sequential::new()
        .add(Flatten::new(Shape::new(vec![224, 224, 3]))) // Input: 224x224x3 (ImageNet images)
        .add(Dense::new(512, Some(ReluActivation::new()), true)) // Hidden layer: 512 units
        .add(Dense::new(256, Some(ReluActivation::new()), true)) // Hidden layer: 256 units
        .add(Dense::new(1000, None::<SoftmaxActivation>, false)); // Output: 1000 classes (ImageNet categories)

    // Display the model summary
    model.summary();

    // Define an optimizer
    let optimizer = Adam::new(0.001);

    // Compile the model
    model.compile(optimizer, SparseCategoricalCrossEntropyLoss::new());

    // Load the train and test dataset
    println!("Loading ImageNetV2 training dataset...");
    let mut train_data = ImageNetV2Dataset::load_train().await;

    println!("Loading ImageNetV2 test dataset...");
    let test_data = ImageNetV2Dataset::load_test().await;

    println!("Training the model...");
    println!("Train dataset size: {}", train_data.len());

    let epochs = 10;
    let batch_size = 32;

    model.fit(&mut train_data, epochs, batch_size);

    // Evaluate the model
    println!("Evaluating the model...");
    let accuracy = model.evaluate(&test_data, batch_size);
    println!("Test Accuracy: {:.2} %", accuracy * 100.0);

    // Save the model
    let model_path = ".cache/models/imagenetv2/imagenetv2_model";
    println!("Saving the model to {}...", model_path);
    model.save(model_path).unwrap();

    println!("Model training and evaluation complete.");
}

#[cfg(test)]
mod test {
    use super::*;
    
    #[tokio::test]
    async fn test_model_creation() {
        // Test if the model can be created successfully
        let model = Sequential::new()
            .add(Flatten::new(Shape::new(vec![224, 224, 3]))) // Input: 224x224x3 (ImageNet images)
            .add(Dense::new(512, Some(ReluActivation::new()), true)) // Hidden layer: 512 units
            .add(Dense::new(256, Some(ReluActivation::new()), true)) // Hidden layer: 256 units
            .add(Dense::new(1000, None::<SoftmaxActivation>, false)); // Output: 1000 classes (ImageNet categories)
        
        // Check if the model creation is successful
        assert!(model.is_ok(), "Failed to create the model");
    }

    #[tokio::test]
    async fn test_dataset_loading() {
        // Test if ImageNetV2 train and test datasets load correctly
        let train_data = ImageNetV2Dataset::load_train().await;
        let test_data = ImageNetV2Dataset::load_test().await;

        // Ensure the datasets are not empty
        assert!(train_data.len() > 0, "Train dataset should not be empty");
        assert!(test_data.len() > 0, "Test dataset should not be empty");
    }

    #[tokio::test]
    async fn test_model_training() {
        let mut model = Sequential::new()
            .add(Flatten::new(Shape::new(vec![224, 224, 3]))) // Input: 224x224x3 (ImageNet images)
            .add(Dense::new(512, Some(ReluActivation::new()), true)) // Hidden layer: 512 units
            .add(Dense::new(256, Some(ReluActivation::new()), true)) // Hidden layer: 256 units
            .add(Dense::new(1000, None::<SoftmaxActivation>, false)); // Output: 1000 classes (ImageNet categories)

        let optimizer = Adam::new(0.001);
        model.compile(optimizer, SparseCategoricalCrossEntropyLoss::new());

        let mut train_data = ImageNetV2Dataset::load_train().await;

        // Ensure model can train without errors
        let epochs = 1;
        let batch_size = 32;

        let result = std::panic::catch_unwind(|| {
            model.fit(&mut train_data, epochs, batch_size);
        });

        assert!(result.is_ok(), "Model training should not panic");
    }

    #[tokio::test]
    async fn test_model_evaluation() {
        let mut model = Sequential::new()
            .add(Flatten::new(Shape::new(vec![224, 224, 3]))) // Input: 224x224x3 (ImageNet images)
            .add(Dense::new(512, Some(ReluActivation::new()), true)) // Hidden layer: 512 units
            .add(Dense::new(256, Some(ReluActivation::new()), true)) // Hidden layer: 256 units
            .add(Dense::new(1000, None::<SoftmaxActivation>, false)); // Output: 1000 classes (ImageNet categories)

        let optimizer = Adam::new(0.001);
        model.compile(optimizer, SparseCategoricalCrossEntropyLoss::new());

        let test_data = ImageNetV2Dataset::load_test().await;

        // Ensure the model can be evaluated without errors
        let batch_size = 32;
        let accuracy = model.evaluate(&test_data, batch_size);

        // Check if accuracy is within a reasonable range (0 to 1)
        assert!(
            accuracy >= 0.0 && accuracy <= 1.0,
            "Accuracy should be between 0 and 1"
        );
    }
}

