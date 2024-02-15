# Prompts

## Implementation pre-processing
You're a data-scientist expert focused on timeseries analysis. Please help me with the following:
I have a folder containing multiple CSVs of timeseries data with the following features: 
['CPU cores', 'CPU usage [%]', 'Memory capacity provisioned [KB]', 'Memory usage [KB]', 'Disk read throughput [KB/s]', 'Disk write throughput [KB/s]', 'Network received throughput [KB/s]', 'Network transmitted throughput [KB/s]']

Please provide only the following python code for pre-processing the data for these functions:

* **Spectral Density**: Power of the signal across different frequencies.
* **Dominant Frequencies**: Identify the frequency components with the most significant amplitudes.
* **Spectral Entropy**: Measures the "complexity" or irregularity of the frequency spectrum.
* **Wavelet Coefficients**: Wavelet analysis yields coefficients across different timescales. These can themselves be features, or you can compute statistics on them.
  * **Coefficient Statistics**: Statistical measures (mean, variance, etc.) of the wavelet coefficients within each scale.
  * **Cross-Scale Coefficient Correlation**: Measures of how coefficients across different scales relate to each other, capturing hierarchical dependencies.
* **Band Power**: The sum of spectral power within specific frequency bands, useful for identifying dominant modes of variability.
* **Energy**: The total energy of the signal in the frequency domain, useful for identifying signals with high activity levels.

## Implementation Analysis
Given the potentially large number of features, feature selection or reduction techniques become crucial:

Thus, please add the following methods: 
* Principal Component Analysis (PCA): For reducing dimensionality while preserving variance.
* t-Distributed Stochastic Neighbor Embedding (t-SNE) and Uniform Manifold Approximation and Projection (UMAP): For non-linear dimensionality reduction, particularly useful for visualization and exploring the structure of high-dimensional data.
Both plot and extract all the useful results of them to improve the decision making (make sure to make the printing each result informative as to its usefullness in analysis)



# Info

### 1. Basic Statistical Features

* **Mean, Median, Mode**: Basic statistical measures of central tendency.
* **Standard Deviation**, Variance: Measures of data dispersion or spread.
* **Min, Max, range**: Extreme values in your data series.
* **Percentiles (median, 25th, 75th), interquartile range**: Quantiles.
* **Skewness, Kurtosis**: Measures of the asymmetry and peakedness of the distribution of values.
* **Sum**: Total value, which can be especially relevant for resource consumption over time

### 2. Rolling / Window-Based Features

* **Rolling Averages**: Simple moving averages (SMA) with varied window sizes.
* **Rolling Volatility**: Rolling standard deviation.
* **Rolling Quantiles**: To capture how percentile metrics shift over time.
* **Bollinger Bands**: Combine a moving average with bands based on rolling volatility.
* **Change Rates**: The rate of change between consecutive measurements or over specified intervals, capturing acceleration or deceleration in resource usage.

### 3. Trend & Seasonality

* **Decomposition**: Break down your time series into trend, seasonal, and residual components (e.g., STL decomposition). Features can then be engineered from each component.
* **Differencing**: Calculate first-order differencing (differences between consecutive points) to de-trend data. You can apply higher-order differencing as well.
* **Autocorrelation Features**: Measures of how the signal correlates with itself at different lags, useful for identifying repeating patterns or periodicity.

### 4. Frequency Domain Features (based on FFT, Wavelets, etc.)

* **Spectral Density**: Power of the signal across different frequencies.
* **Dominant Frequencies**: Identify the frequency components with the most significant amplitudes.
* **Spectral Entropy**: Measures the "complexity" or irregularity of the frequency spectrum.
* **Wavelet Coefficients**: Wavelet analysis yields coefficients across different timescales. These can themselves be features, or you can compute statistics on them.
  * **Coefficient Statistics**: Statistical measures (mean, variance, etc.) of the wavelet coefficients within each scale.
  * **Cross-Scale Coefficient Correlation**: Measures of how coefficients across different scales relate to each other, capturing hierarchical dependencies.
* **Band Power**: The sum of spectral power within specific frequency bands, useful for identifying dominant modes of variability.
* **Energy**: The total energy of the signal in the frequency domain, useful for identifying signals with high activity levels.

### 5. Burstiness and Distributional Characteristics

* **Inter-arrival Times**: Times between events exceeding a defined threshold.
* **Time Above/Below Thresholds**: Proportion of time spent exceeding or falling below set levels.
* **Distributional Fit**: If you have theoretical notions of workload distributions, test for similarity (e.g., with the Kolmogorov-Smirnov test).

### 6. Cross-Feature Relationships

* **Ratios**: CPU usage / memory usage, network transmitted / network received, combinations of your existing features.
* **Correlations**: Pearson, Spearman, or other correlation metrics between resource metrics over rolling windows.
* **Lagged Correlations**: Look for correlations with time offsets (one resource usage leading/lagging another).


# Feature Selection and Reduction
Given the potentially large number of features, feature selection or reduction techniques become crucial:

Principal Component Analysis (PCA): For reducing dimensionality while preserving variance.
t-Distributed Stochastic Neighbor Embedding (t-SNE) or Uniform Manifold Approximation and Projection (UMAP): For non-linear dimensionality reduction, particularly useful for visualization and exploring the structure of high-dimensional data.


