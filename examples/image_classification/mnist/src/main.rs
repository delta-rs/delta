use deltaml::activations::ReluActivation;
use deltaml::activations::SoftmaxActivation;
use deltaml::common::Shape;
use deltaml::dataset::{ImageDatasetOps, MnistDataset};
use deltaml::losses::SparseCategoricalCrossEntropyLoss;
use deltaml::neuralnet::{Dense, Flatten, Sequential};
use deltaml::neuralnet::layers::{Conv2D, MaxPooling2D};
use deltaml::optimizers::Adam;

#[tokio::main]
async fn main() {
    // Create a neural network
    let mut model = Sequential::new()
        .add(Conv2D::new(32, 3, 1, 1, Some(Box::new(ReluActivation::new())), true)) // Conv2D layer with 32 filters, kernel size 3x3
        .add(MaxPooling2D::new(2, 2)) // MaxPooling2D layer with pool size 2x2
        .add(Flatten::new(Shape::new(vec![28, 28, 32]))) // Flatten layer
        .add(Dense::new(128, Some(ReluActivation::new()), true)) // Dense layer with 128 units
        .add(Dense::new(10, None::<SoftmaxActivation>, false)); // Output layer with 10 classes

    // Display the model summary
    model.summary();

    // Define an optimizer
    let optimizer = Adam::new(0.001);

    // Compile the model
    // model.compile(optimizer, CrossEntropyLoss::new());
    model.compile(optimizer, SparseCategoricalCrossEntropyLoss::new());

    // Loading the train and test dataset
    let mut train_data = MnistDataset::load_train().await;
    let test_data = MnistDataset::load_test().await;

    println!("Training the model...");
    println!("Train dataset size: {}", train_data.len());

    let epoch = 1;
    let batch_size = 32;

    model.fit(&mut train_data, epoch, batch_size);

    // Evaluate the model
    let accuracy = model.evaluate(&test_data, batch_size);
    println!("Test Accuracy: {:.2}%", accuracy * 100.0);

    // Save the model
    model.save(".cache/models/mnist/mnist").unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use deltaml::dataset::MnistDataset;

    #[tokio::test]
    async fn test_model_creation(){
        //testing for model test_model_creation
        let model = Sequential::new()
            .add(Conv2D::new(32, 3, 1, 1, Some(Box::new(ReluActivation::new())), true))
            .add(MaxPooling2D::new(2, 2))
            .add(Flatten::new(Shape::new(vec![28, 28, 32])))
            .add(Dense::new(128, Some(ReluActivation::new()), true))
            .add(Dense::new(10, None::<SoftmaxActivation>, false));

        assert!(model.is_ok(), "Failed to create the model");
    }
    
     #[tokio::test]
    async fn test_model_evaluation() {
        let mut model = Sequential::new()
            .add(Conv2D::new(32, 3, 1, 1, Some(Box::new(ReluActivation::new())), true))
            .add(MaxPooling2D::new(2, 2))
            .add(Flatten::new(Shape::new(vec![28, 28, 32])))
            .add(Dense::new(128, Some(ReluActivation::new()), true))
            .add(Dense::new(10, None::<SoftmaxActivation>, false));

        let optimizer = Adam::new(0.001);
        model.compile(optimizer, SparseCategoricalCrossEntropyLoss::new());

        let test_data = MnistDataset::load_test().await;

        let batch_size = 32;
        let accuracy = model.evaluate(&test_data, batch_size);

        assert!(
            accuracy >= 0.0 && accuracy <= 1.0,
            "Accuracy should be between 0 and 1"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deltaml::dataset::MnistDataset;

    #[tokio::test]
    async fn test_model_creation() {
        // Test if the model can be created without errors
        let model = Sequential::new()
            .add(Conv2D::new(32, 3, 1, 1, Some(Box::new(ReluActivation::new())), true))
            .add(MaxPooling2D::new(2, 2))
            .add(Flatten::new(Shape::new(vec![28, 28, 32])))
            .add(Dense::new(128, Some(ReluActivation::new()), true))
            .add(Dense::new(10, None::<SoftmaxActivation>, false));

        assert!(model.is_ok(), "Failed to create the model");
    }

    #[tokio::test]
    async fn test_dataset_loading() {
        // Test if MNIST train and test datasets load correctly
        let train_data = MnistDataset::load_train().await;
        let test_data = MnistDataset::load_test().await;

        assert!(train_data.len() > 0, "Train dataset should not be empty");
        assert!(test_data.len() > 0, "Test dataset should not be empty");
    }

    #[tokio::test]
    async fn test_model_training() {
        let mut model = Sequential::new()
            .add(Conv2D::new(32, 3, 1, 1, Some(Box::new(ReluActivation::new())), true))
            .add(MaxPooling2D::new(2, 2))
            .add(Flatten::new(Shape::new(vec![28, 28, 32])))
            .add(Dense::new(128, Some(ReluActivation::new()), true))
            .add(Dense::new(10, None::<SoftmaxActivation>, false));

        let optimizer = Adam::new(0.001);
        model.compile(optimizer, SparseCategoricalCrossEntropyLoss::new());

        let mut train_data = MnistDataset::load_train().await;

        // Ensure model can train without errors
        let epoch = 1;
        let batch_size = 32;

        let result = std::panic::catch_unwind(|| {
            model.fit(&mut train_data, epoch, batch_size);
        });

        assert!(result.is_ok(), "Model training should not panic");
    }

    #[tokio::test]
    async fn test_model_evaluation() {
        let mut model = Sequential::new()
            .add(Conv2D::new(32, 3, 1, 1, Some(Box::new(ReluActivation::new())), true))
            .add(MaxPooling2D::new(2, 2))
            .add(Flatten::new(Shape::new(vec![28, 28, 32])))
            .add(Dense::new(128, Some(ReluActivation::new()), true))
            .add(Dense::new(10, None::<SoftmaxActivation>, false));

        let optimizer = Adam::new(0.001);
        model.compile(optimizer, SparseCategoricalCrossEntropyLoss::new());

        let test_data = MnistDataset::load_test().await;

        let batch_size = 32;
        let accuracy = model.evaluate(&test_data, batch_size);

        assert!(
            accuracy >= 0.0 && accuracy <= 1.0,
            "Accuracy should be between 0 and 1"
        );
    }
}

