

# Prompts

We're making a scheduler for placement decisions of services (containers, etc). 
As part of this an essential analysis is understanding the workload characteristics of the services
Thus, we're wanting to analyse the timeseries of resource consumption (CPU, mem, network, disk, ...).
To integrate the workload classification into information into the scheduler we need some way to map service IDs -> workload characteristics.
The approach we're initially trying is:
* Timeseries Pre-processing - to be able to classify timeseries data 
  * FFT, CNN, Wavelets, RNN (LSTM / GRU) **add more here**
* Data extraction Pre-processing - compresses the data into a (hopefully) good intermediate representation
  * AutoEncoder, LSTM-VAE, **add more here**
* Classifier - classifies based on the intermediate representation
  * KNN, **Skip, this will wait until later**
  * Classifiers will have to be unsupervised
    * Though we have an emulator that can generate synthetic workloads of clearly distinct resource consumptions. Thus we can benchmark the classification and intermediate representations

Current dataset has the following features:
['CPU cores', 'CPU usage [%]', 'Memory capacity provisioned [KB]', 'Memory usage [KB]', 'Disk read throughput [KB/s]', 'Disk write throughput [KB/s]', 'Network received throughput [KB/s]', 'Network transmitted throughput [KB/s]']

1. Please extend the list of approaches proposed were it's marked "**add more here**"
2. Please motivate the different approaches, benefits and drawbacks
3. Please critizice the general approach

# 3 Extension:
Adding to this:
Note that the classifiers will have to be unsupervised. Though we have an emulator that can generate synthetic workloads of clearly distinct resource consumptions. Thus we can benchmark the classification and intermediate representations.

You make a good point on feature engineering
Derived Features: Ratios (e.g., CPU usage normalized by memory usage), rolling statistics (e.g., moving averages, standard deviations), burstiness metrics, (e.g., statistical summaries, peak analysis), etc
Please provide an extensive list of useful engineered features for the timeserieses.
Also provide any feature engineering that can be generated as an intermediate pre-processing step, e.g. after FFT or Wavelet. 




## Interesting aspects
- [x] Feature Engineering: Time series data can be notoriously tricky. Careful feature engineering could make a substantial difference. While your existing features are a good start, consider these enhancements:
- [x] Derived Features: Ratios (e.g., CPU usage normalized by memory usage), rolling statistics (e.g., moving averages, standard deviations), burstiness metrics.
- [x] Beyond pre-processing, carefully engineering features from your timeseries data (e.g., statistical summaries, peak analysis) can significantly enhance model performance.

Timeseries Pre-Processing

CNNs: CNNs can be effective with time series, but might require adaptations. Explore 1D convolutions to match the nature of your data.
Wavelets: Consider wavelet-based analysis instead of (or in addition to) FFT. Wavelets better handle non-stationarity often found in resource consumption patterns.
Recurrence Plots: Can help visualize inherent cyclical or repetitive behaviors in workloads
RNN (Recurrent Neural Networks): Specifically designed to handle sequential data, RNNs (and their variants like LSTM and GRU) could better capture temporal dependencies in timeseries data compared to CNNs.

Data Extraction Pre-Processing

LSTMs & Variants: Definitely a strong choice! Look into bi-directional LSTMs and attention-based LSTMs to potentially capture both forward and backward temporal dependencies.
Transformer-Based Models: Transformers are proving very powerful in sequence modeling. While computationally heavier, investigate if workload characterization might benefit from them.

Scalability and Efficiency: Given the potentially large volume of data from monitoring resources, consider the computational efficiency of your chosen methods, especially for real-time or near-real-time scheduling decisions.