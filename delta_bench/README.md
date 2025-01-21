# Delta Bench

To run the various benchmarks for Delta, execute `cargo bench`. However, it’s important to note that the current benchmarks are more of a template to
show the structure of how we should approach benchmarking, rather than an optimized or effective test of our code.

There are major improvement opportunities to be discovered and acted upon, and this guide is designed to help you properly evaluate Delta’s performance.

## Current State of the Benchmarks

The existing benchmarks are in their infancy and are structured as starting points. They are not ideal representations of real-world workloads,
and they don’t fully capture the performance characteristics of the system. Our goal is to evolve them into effective tools for evaluating realistic performance.

**Currently, the benchmarks:**

-	Use small synthetic data.
- Lack measures for system resource utilization (e.g., memory, CPU).
- Do not simulate production-like loads or variability in workloads.
- Focus on individual operations but don’t consider the broader context of performance goals such as throughput, latency, and scalability.

While these are useful for early-stage development, they need to be expanded and refined to make Delta ready for real-world use cases and heavy workloads.

## Benchmarking Guidelines

### 1. Define Clear Performance Goals

Before you dive into improving the benchmarks, start by defining what you are measuring. Are you focused on:

- **Throughput:** The rate at which the system performs operations (e.g., SGD updates per second).
- **Latency:** The time it takes to complete a single operation (e.g., time to process one batch of data).
- **Percentile Performance:** Measuring performance at different percentiles (e.g., ensuring the 90th or 95th percentile of operations meet latency goals).
- **Worst-case Performance:** Evaluating the system under extreme load conditions (e.g., the maximum number of operations before performance degrades unacceptably).

These performance goals will shape how the benchmarks are structured and how the data is measured. Without clear goals, the benchmarks will not provide actionable insights.

### 2. Simulate Realistic Workloads

The key to effective benchmarking is ensuring that the workload closely reflects how Delta will be used in production. This means:

- **Data Size & Shape:** Ensure the data you’re using in the benchmarks is representative of what the model will actually encounter. For example, training data sizes, tensor shapes, and complexities in your models should mirror real-world scenarios.
- **Workload Variability:** The current benchmarks assume static and predictable loads. To improve this, we need to simulate dynamic loads, where the data and training conditions vary in ways that resemble real-world applications.
- **Longer Runs:** Instead of short and quick runs, consider longer benchmark tests to observe performance over time, especially if memory management or other resources need to be monitored under prolonged load.

### 3. Measure System Resource Utilization

Benchmarking isn’t just about how fast an operation runs—it’s about how efficiently resources (like CPU and memory) are used.
To get a more comprehensive view of performance, incorporate the following:

- **CPU/GPU Usage:** How much CPU/GPU is consumed during operations, especially during parallel training steps or large tensor operations.
- **Memory Usage:** Monitor memory usage as your tensors grow or as the optimizer works on larger datasets. Memory bottlenecks can have significant performance impacts that are not immediately visible with raw execution time.
- **Disk I/O:** If applicable, measure how often the system performs disk reads/writes (especially if data loading is a bottleneck).

### 4. Incorporate Parallelism and Concurrency

Delta is designed to benefit from parallelism and concurrency, so it’s crucial that benchmarks reflect this aspect of the framework. Some recommendations:

- **Multi-core Utilization:** Ensure your benchmarks take advantage of multi-threading, especially for operations like matrix multiplication, gradient updates, etc.
- **Parallel Optimizer Steps:** Test how well Delta can perform optimizer steps concurrently on multiple data batches or models.

### 5. Realistic Load Testing

Arguably the hardest but most important aspect of benchmarking is testing under realistic load conditions. Here’s how to approach this:

- **Record and Replay:** Ideally, we want to capture real production data and replay it under various conditions. This will help simulate realistic performance scenarios without needing to constantly gather new data.
- **Simulate Production Stress:** Add load to the system progressively until it reaches realistic stress points. This will help us understand how Delta scales and where bottlenecks appear.
- **Vary the Load:** Rather than running the system at a constant rate, introduce dynamic load variations—some data points may be heavier than others, or the optimizer might be tested on more complex datasets.

### 6. Refining the Benchmark Structure

While the existing benchmarks are a good starting point, they are limited in scope. Here are some improvements to consider:

- **Workload Diversity:** Add more test cases that reflect different scenarios (e.g., small and large datasets, different tensor shapes, multi-model optimizations).
- **Latency & Throughput Metrics:** Capture latency for individual operations and throughput for batches of operations, and analyze how these metrics behave as the system is loaded.
- **Error Handling:** Include tests for how well the system handles edge cases (e.g., memory exhaustion, data corruption, interrupted operations).

### 7. Iterative Refinement and Continuous Monitoring

Finally, remember that benchmarking is an iterative process. As we add features and refine the system, it’s important to revisit benchmarks and update them to reflect any changes in system behavior or performance goals.

- **Continuous Monitoring:** Once the improved benchmarks are in place, use them as a part of continuous integration (CI) pipelines to monitor ongoing performance.
- **Refinement Over Time:** As we gather more data and insights, continually refine the benchmarks to better reflect real-world conditions.

## Next Steps

The current state of the Delta benchmarks is a starting point. However, there are significant improvements to be made. By following the guidelines above, we can create meaningful, real-world benchmarks that provide insight into the true performance of our system.

If you’re contributing to the Delta project, start by enhancing the existing benchmarks with these principles in mind. Focus on making the benchmarks:

- More representative of real-world data and loads.
- More comprehensive, measuring not just speed but also resource usage and system behavior under stress.
- More scalable, testing how well Delta performs as it grows.

Let’s make sure that our benchmarks not only test performance but give us valuable insights into how Delta behaves in production-like conditions.
