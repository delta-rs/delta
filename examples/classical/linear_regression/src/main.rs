use deltaml::{
    classical::{Classical, LinearRegression, calculate_mse_loss},
    common::ndarray::{Array1, Array2},
};
use plotters::{
    chart::ChartBuilder,
    prelude::{BitMapBackend, Circle, IntoDrawingArea},
    series::LineSeries,
    style::{BLUE, Color, IntoFont, RED, WHITE},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create some dummy data for example purposes
    let x_data = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y_data = Array1::from_vec(vec![2.0, 4.0, 5.0, 4.0, 5.0]);

    // Instantiate the model
    let mut model = LinearRegression::new();

    // Train the model
    let learning_rate = 0.01;
    let epochs = 1000;
    model.fit(&x_data, &y_data, learning_rate, epochs);

    // Make predictions with the trained model
    let new_data = Array2::from_shape_vec((3, 1), vec![2.5, 3.5, 4.5]).unwrap();
    let predictions = model.predict(&new_data);

    println!("Predictions for new data: {:?}", predictions);

    // Calculate accuracy or loss for the test data for demonstration
    let test_loss = calculate_mse_loss(&model.predict(&x_data), &y_data);
    println!("Test Loss after training: {:.6}", test_loss);

    // Plot the regression
    plot_regression(&x_data, &y_data, &model)?;

    Ok(())
}

fn plot_regression(
    x_data: &Array2<f64>,
    y_data: &Array1<f64>,
    model: &LinearRegression,
) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new("regression_plot.png", (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let x_min = x_data[[0, 0]];
    let x_max = x_data[[x_data.nrows() - 1, 0]];
    let y_min = y_data.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Linear Regression", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    // Plot the original data points
    chart.draw_series(
        x_data
            .column(0)
            .iter()
            .zip(y_data.iter())
            .map(|(&x, &y)| Circle::new((x, y), 5, BLUE.filled())),
    )?;

    // Plot the regression line
    let regression_points: Vec<(f64, f64)> = (0..=100)
        .map(|i| {
            let x = x_min + (x_max - x_min) * i as f64 / 100.0;
            let y = model.predict(&Array2::from_shape_vec((1, 1), vec![x]).unwrap())[0];
            (x, y)
        })
        .collect();

    chart.draw_series(LineSeries::new(regression_points, &RED))?;

    Ok(())
}
