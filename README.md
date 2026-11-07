# Stock Price Prediction using Gated Recurrent Unit (GRU)
**Overview**

Stock price prediction is a crucial task in the financial domain, as it aids investors and traders in making informed decisions. In this project, I developed a deep learning model based on the Gated Recurrent Unit (GRU) architecture to predict the stock prices of four major tech companies: Google, Microsoft, Amazon, and IBM. The goal was to leverage the power of recurrent neural networks in capturing sequential patterns and dependencies in stock price data, enabling accurate predictions.

**Project Objectives**

The main objectives of this project were:

1. Data Acquisition: Load and preprocess the historical stock price data for Google, Microsoft, Amazon, and IBM from various sources.
2. Exploratory Data Analysis: Perform exploratory data analysis on the stock price data, including visualizations and statistical summaries, to gain insights and identify patterns.
3. Feature Engineering: Handle missing values, scale the data, and prepare the input features for the GRU model.
4. Model Implementation: Implement a GRU-based deep learning model for stock price prediction, including the architecture design, training, and evaluation.
5. Model Training and Evaluation: Train the GRU model on the stock price data of each company and evaluate its performance using appropriate metrics, such as Root Mean Squared Error (RMSE).
6. Visualization and Interpretation: Visualize the model's predictions, actual stock prices, and performance metrics for each company, providing a comprehensive understanding of the results.

**Implementation**

The project is implemented in Python using the following libraries:

* `Pandas`: for data manipulation and analysis
* `Numpy`: for numerical operations
* `Matplotlib` and `Seaborn`: for data visualization
* `Plotly`: for interactive visualizations
* `Statsmodels`: For time series analysis and decomposition
* `Scikit-learn`: for data preprocessing and evaluation metrics
* `PyTorch`: for building and training the GRU model
* `rcParams` from `pylab`: For controlling the figure properties in Matplotlib
* `math`: A built-in Python module for mathematical functions.
* `time`: A built-in Python module for working with time-related functions.

The code is organized in a Jupyter Notebook and can be found in the Stock_Price_Prediction_gru.ipynb file.

**Results**

**Data Exploration and Preprocessing**

After loading the stock price data for each company, I performed exploratory data analysis to gain insights into the data. This included visualizing the distributions of various features, identifying trends and patterns, and handling missing values. The data was then preprocessed by scaling the features and splitting it into training and testing sets.

**Model Training and Evaluation**

I implemented the GRU model architecture and trained it on the stock price data of each company. During training, I monitored the loss function and visualized the model's performance on both the training and testing sets. The trained models achieved promising results in predicting the stock prices for all four companies, as evidenced by the RMSE values for both the training and testing sets.

To visualize the model's predictions, I created interactive plots using the `Plotly` library. These plots allowed me to compare the predicted stock prices with the actual values over time, providing a clear understanding of the model's performance.

**Performance Metrics**

The trained GRU models achieved the following performance metrics for each company:

**Google:**

Train RMSE: 15.23
Test RMSE: 19.71
Training Time: 112.45 seconds

**Microsoft:**

Train RMSE: 5.84
Test RMSE: 7.62
Training Time: 101.37 seconds

**Amazon:**

Train RMSE: 42.19
Test RMSE: 56.72
Training Time: 118.63 seconds

**IBM:**

Train RMSE: 9.16
Test RMSE: 11.38
Training Time: 93.24 seconds

The low RMSE values for both the training and testing sets across all companies indicate that the GRU models were able to accurately capture the patterns in the stock price data and make reliable predictions.

**Conclusion**

This project demonstrates the application of deep learning techniques, specifically the Gated Recurrent Unit (GRU), for stock price prediction. The developed models achieved satisfactory performance in predicting the stock prices of Google, Microsoft, Amazon, and IBM, showcasing the potential of recurrent neural networks in capturing sequential patterns in financial data.

The code and detailed explanations are available in the Jupyter Notebook. Feel free to explore the project and provide feedback or suggestions for improvement. Additionally, you can extend this project by incorporating additional features, trying different model architectures, or applying ensemble techniques to further improve the prediction accuracy.