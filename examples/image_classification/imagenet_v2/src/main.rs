// use deltaml::activations::relu::ReluActivation;
// use deltaml::activations::softmax::SoftmaxActivation;
// use deltaml::dataset::base::ImageDatasetOps;
// use deltaml::dataset::ImageNetV2Dataset;
// use deltaml::losses::SparseCategoricalCrossEntropyLoss;
// use deltaml::neuralnet::Sequential;
// use deltaml::neuralnet::{Dense, Flatten};
// use deltaml::optimizers::Adam;
//
// #[tokio::main]
// async fn main() {
//     // Create a neural network
//     let mut model = Sequential::new()
//         .add(Flatten::new(Shape::new(vec![224, 224, 3]))) // Input: 224x224x3 (ImageNet images)
//         .add(Dense::new(512, Some(ReluActivation::new()), true)) // Hidden layer: 512 units
//         .add(Dense::new(256, Some(ReluActivation::new()), true)) // Hidden layer: 256 units
//         .add(Dense::new(1000, None::<SoftmaxActivation>, false)); // Output: 1000 classes (ImageNet categories)
//
//     // Display the model summary
//     model.summary();
//
//     // Define an optimizer
//     let optimizer = Adam::new(0.001);
//
//     // Compile the model
//     model.compile(optimizer, SparseCategoricalCrossEntropyLoss::new());
//
//     // Load the train and test dataset
//     println!("Loading ImageNetV2 training dataset...");
//     let mut train_data = ImageNetV2Dataset::load_train().await;
//
//     println!("Loading ImageNetV2 test dataset...");
//     let test_data = ImageNetV2Dataset::load_test().await;
//
//     println!("Training the model...");
//     println!("Train dataset size: {}", train_data.len());
//
//     let epochs = 10;
//     let batch_size = 32;
//
//     model
//         .fit(&mut train_data, epochs, batch_size)
//         .expect("Failed to train model");
//
//     // Evaluate the model
//     println!("Evaluating the model...");
//     let accuracy = model
//         .evaluate(&test_data, batch_size)
//         .expect("Failed to evaluate model");
//     println!("Test Accuracy: {:.2} %", accuracy * 100.0);
//
//     // Save the model
//     let model_path = ".cache/models/imagenetv2/imagenetv2_model";
//     println!("Saving the model to {}...", model_path);
//     model.save(model_path).unwrap();
//
//     println!("Model training and evaluation complete.");
// }
