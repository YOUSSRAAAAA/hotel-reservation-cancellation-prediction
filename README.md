# 🌟 **Hotel Reservation Cancellation Analysis and Prediction** 🌟

This project focuses on predicting hotel reservation cancellations using **machine learning** techniques. The objective is to assist hotel managers in optimizing revenue, minimizing booking uncertainties, and enhancing customer satisfaction.



## 📋 **Overview**

This project utilizes a rich dataset containing **119,390 samples** and **32 variables**, such as booking status, lead time, arrival date, and customer demographics. The project workflow involves **data exploration**, **preprocessing**, and the application of advanced **machine learning models** to achieve high prediction accuracy.



## 🚀 **Features**

### 🔍 **Exploratory Data Analysis (EDA)**  
- **Univariate Analysis**: Distribution of key variables like `lead time` and `ADR`.  
- **Bivariate Analysis**: Relationship analysis between booking status (`is_canceled`) and other features like `market_segment`.  
- **Geographical Visualization**: Insights into the origins of customers.  

### 🛠️ **Data Preprocessing**  
- Handled **missing values**, **duplicates**, and **outliers**.  
- Engineered features to enhance model predictions.  

### 🤖 **Machine Learning Models**  
Implemented and evaluated the following models:  
- **Logistic Regression**  
- **K-Nearest Neighbors (KNN)**  
- **Decision Tree**  
- **Random Forest**  
- **XGBoost**  
- **CatBoost**  
- **LightGBM**



## 📂 **Dataset**

- **Source**: [Kaggle - Hotel Booking Dataset](https://www.kaggle.com/code/swetarajsinha/hotel-bookings)  
- **Details**:  
  - **H1 (Resort Hotel)**: 40,060 observations.  
  - **H2 (City Hotel)**: 79,330 observations.  
  - **Timeframe**: July 1, 2015 – August 31, 2017.  
- **Key Features**:  
  - `is_canceled`: Booking status (canceled or not).  
  - `lead_time`: Days between booking and check-in.  
  - `adr`: Average daily rate.  



## ⚙️ **Methodology**

1. **📊 Data Collection**: Ensured data integrity by timestamping reservations the day before arrival.  
2. **🧹 Data Cleaning**:  
   - Removed inconsistencies and anomalies.  
   - Standardized data types.  
3. **🔀 Data Splitting**:  
   - Train-test split: **80/20** ratio.  
   - Preserved the natural distribution of classes (`Canceled` vs. `Not Canceled`).  
4. **📈 Model Training**:  
   - Compared models using metrics like **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **AUC**.  



## 🏆 **Results**

- **Top Models**:  
  - **Random Forest**: Test Accuracy of **83.11%**, AUC of **0.9681**.  
  - **XGBoost**: AUC of **0.9166**.  
  - **LightGBM**: AUC of **0.9100**.  

- **Insights**:  
  - Seasonal trends and customer demographics strongly influence cancellations.  
  - Advanced boosting models outperformed simpler algorithms.


## 🔧 **Usage**

1. Explore the data using Jupyter notebooks.  
2. Train and evaluate models using Python scripts.  
3. Visualize results with built-in plots and metrics.  



## 🌟 **Future Work**

- **Real-Time Data**: Incorporate weather and local event data.  
- **Advanced Techniques**: Use ensemble methods like Voting Classifiers.  
- **Temporal Analysis**: Capture seasonal trends more effectively.  
- **Customer Segmentation**: Refine clustering for tailored strategies.  



## ✍️ **Contributors**

- **Yousra Wakrim**   



## 📜 **License**

This project is licensed under the **MIT License**. Feel free to use and adapt it for educational or research purposes.

