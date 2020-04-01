# NEW YORK GREEN TAXI DATASET

**Abstract**
The Taxi Limousine Commission(TLC) provided yellow taxi service for the city of New York. But these taxi services were concentrated to the Manhattan region of the city and did not ride to the outskirts. The green taxi service was hence initiated to cater to the demands outside of Manhattan which started in 2013. The data is vast and there are various conclusions we can draw from it. The aim is to maximize the revenue of the company by implementing various machine learning techniques of supervised and unsupervised algorithms to identify the ideal model to the data. 

**About Data**
The dataset is about TLC green taxi service for years 2016 to 2018. The data is being consolidated from NYC Taxi Limousine Commission for each of the above years. The data set consists of 19 columns, which consist of numerical and categorical values. There are total of 22.5 million instances overall. The domain of the dataset is Public Transportation. The description of each column of the data is given as follows: 
![alt text](https://github.com/madhura42/ML-Revenue-Prediction-using-TLC-Green-Taxi-Data/blob/master/Images/Capture1.PNG)
![alt text](https://github.com/madhura42/ML-Revenue-Prediction-using-TLC-Green-Taxi-Data/blob/master/Images/Capture2.PNG)

**EDA:**
**Given EDA is for the year 2018. Similar EDA was done for the year 2017 and 2019.**

![alt text](https://github.com/madhura42/ML-Revenue-Prediction-using-TLC-Green-Taxi-Data/blob/master/Images/Capture3.PNG)

![alt text](https://github.com/madhura42/ML-Revenue-Prediction-using-TLC-Green-Taxi-Data/blob/master/Images/Capture4.PNG)

![alt text](https://github.com/madhura42/ML-Revenue-Prediction-using-TLC-Green-Taxi-Data/blob/master/Images/Capture5.PNG)

![alt text](https://github.com/madhura42/ML-Revenue-Prediction-using-TLC-Green-Taxi-Data/blob/master/Images/Capture6.PNG)

![alt text](https://github.com/madhura42/ML-Revenue-Prediction-using-TLC-Green-Taxi-Data/blob/master/Images/Capture7.PNG)

![alt text](https://github.com/madhura42/ML-Revenue-Prediction-using-TLC-Green-Taxi-Data/blob/master/Images/Capture8.PNG)

![alt text](https://github.com/madhura42/ML-Revenue-Prediction-using-TLC-Green-Taxi-Data/blob/master/Images/Capture9.PNG)



Research Questions

 Which rides are the most profitable for the drivers on the basis of location, type of ride and time?
Data Preprocessing:
 Raw data is a monthly data which I consolidated by year.  
 Next, I combine all 3 years (June 2016 – June 2019) dataset into a single dataset which results into 22.5 million instances. 
 Valid Data:  
 ‘tolls_amount’ – Since toll amount cannot be negative, I am filtering positive values for toll amounts. 
 ‘fare_amount’ – Since the initial base fare charge is $2.5, I have taken fare amount values greater than or equal to $2.5. 
 ‘passenger_count’ – Since an XL ride can take up to 6 passengers, I have limited the maximum passenger count to 6. 
 ‘RatecodeID’ –  Valid Rate code ID range from 1-6. Hence, a constraint has been put to remove any invalid category values 
 ‘trip_distance’ – Rides whose trip distance lesser than or equal to zero are considered as cancelled rides. Hence, trip distances which are greater than zero are considered. 
 
 Date and Time format: ‘lpep_pickup_datetime’ and ‘lpep_dropoff_datetime’ store pickup and drop off date and time of every unique ride. Each of these columns were separated according to year, month, day, hour and minutes. 
 Dropped columns:  
 ‘ehail_fee’ – This column consists entirely of zeros values which can be ignored. 
 ‘lpep_pickup_datetime’ – Since date and time of pickup of customer has been incorporated by the separate columns created for year, month, date and time, this column for removed. 
 ‘lpep_dropoff_datetime’ -- Since date and time of drop off of the customer has been incorporated by the separate columns created for year, month, date and time, this column for removed. 
 ‘store_and_fwd_flag’ -- The store and forward flag indicates if the record was initially held in vehicle memory due to some connection issue. This column does not add any value to our research and hence it is dropped.  
Normalization: 
The continuous variables were normalized using the formula:
Xnorm = (X-Xmin)/(Xmax-Xmin)

Model used: 
The model chosen to fit the model is Random Forest Tree Regressor. The aim of this research question is to explain the prediction of the total amount (profit) by analysing how each feature affects the output instance. We judge the feature importance using the SHapley Additive exPlanations (SHAP) that uses game theory to interpret the model chosen. SHAP has two estimation approaches KernalSHAP and TreeSHAP. TreeSHAP is the estimation approach used to predict the Shapley values here as Tree based models as that would help us correctly estimate SHAP values when features are dependent. Also it os computationally less expensive when compared against KernalSHAP . The features can be interpreted on a global as well as a local level. 
The model chosen to fit the model is Random Forest Tree Regressor, where our input features are 'VendorID', 'RatecodeID', 'PULocationID', 'DOLocationID', 'passenger_count', 'trip_distance',’duration’ , 'payment_type', 'trip_type', 'pickup_year', 'pickup_month', 'pickup_day', 'pickup_hour','pickup_minutes', 'dropOff_year','dropOff_month','dropOff_day','dropOff_hour','dropOff_minutes',’speed’.‘total_amount’ is the output variable as is constituted of approximately the sum of the input features given below. Hence these features, 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', and 'improvement_surcharge', have been removed to build the model.

Conclusions:
The model gives 89.55% accuracy when trained on data from 2018 and tested on 2019 data hence this model was chosen. 

1.1.Shap feature importance
The features are given importance based on their individual effect on the output variable, ‘total_amount’. This summary plot depicts the global importance of every feature on the output. The y-axis has the features and the x-axis has the mean absolute Shapley values per feature.
Ij= ∑_(j=1)^(j=n)▒〖| θ〗_(j )^i |
Where n = the number of features and θ = shapley values


1.2.Shap dependence plot
A dependency plot is also displayed which shows the effect of a single feature on the predictions of the model. Since our summary plot indicates that ‘trip_distance’ effects the total amount the greatest, a dependency plot of ‘trip_distance’ is created. 

 Predict the trend in the revenue during holiday season like Christmas
Data Preprocessing:
 1.Pre-processing: For this question, we had to do a different pre-processing of data since we wanted the date in year-month-date format. For this question our focus is mainly on two columns of the dataset namely ‘total_amount’ and ‘pickup_date’. 
2.For date to be categorized as a holiday or not, we first imported the US Federal calendar and weekend calendar. We then marked these holidays and weekends as Boolean value 1 and the rest as 0.  
3.We grouped the data yearly. The training data consists of the year 2017, 2018 and 6 months for 2019 while we have predict the forecast values for the year 2019.
4.The rest of data pre-processing will be the same as stated earlier.

Model used:
We have used ARIMAX model which stands for Autoregressive Integrated Moving Average. ARIMA model basically explains the time series based on its own past values and lagged forecast errors. It creates an equation based upon the average of the past values to forecast the future or the predicted values.  ARIMA models are denoted by ARIMA(p,d,q) which stands for seasonality, trend and noise respectively.
 P denoted the number of lagged values that have to be added or subtracted from the target which captures the “autoregressive” nature of ARIMA. This results in improved predictions on local growth or decline in our data.
 D is basically the degree to which our data is going up or down. So if d=0 that means out data does not go up or down, d=1 means that our data trends linearly, d=2 means our data trends exponentially. To summarize, d denotes the number of times that the data have to be difference to produce a signal (which has constant mean over time).
 Q captures the moving average part of ARIMA. It represents the number of prior or lagged values for the error term that are added or subtracted to Y
 The main aim is to select the best (ie. optimal) set of parameters that yields best performance for our model. In simpler terms we take the lowest AIC value. 

Conclusions: 

2.1.Revenue for each day (Jan 2017-Apr 2019)
Some distinguishable patterns appear when we plot the data.  The time series has seasonality pattern such as during holiday swason such has Christmas and New Years there’s a downward trend. One more such trend is seen in all the three years in the month of March end and beginning of April. We can see a distinct trend that the revenue for the year 2017 is more than for the year 2018 and 2019.


2.2.Time Series Decomposition
We can also visualize our data using a method called time-series decomposition that allows us to decompose our time series into three distinct components: trend, seasonality, and noise. The plot above states that the total revenue for all the years 2017,2018 and 2019 are unstable. 

2.3. Revenue Diagostics
From the second plot (Histogram plus estimated density) we can conclude that the total amount seasonality is approximately normally distributed. The green curve signifies what the actual distribution of the data should be like. While the yellow curve signifies the true distribution which is close to the actual distribution which very few outliers. 
The normal QQ plot is a linear line which clearly states that the data is normally distributed. 


2.4.Observed values v/s forecast predictions
The line plot is showing the observed values compared to the rolling forecast predictions. Overall, our forecasts align with the true values very well. The blue line shows the observed values of the data. The one step ahead values are the predicted values which are denoted by red. The grey area denotes the confidence interval of the forecasted values

2.5. Visualizing Forecast
Our model clearly captured total amount (revenue) seasonality. As we forecast further out into the future, it is natural for us to become less confident in our values. This is reflected by the confidence intervals generated by our model, which grow larger as we move further out into the future. 

 Analyzing distribution of rides during the day according to trip distance
Data Preprocessing: 
Data preprocessing was done similar to Data processing in research question 1.

Model used:
We have used Quantile-Regression model for this research question. 
Linear Regression depicts the relationship between the dependent and independent variables, providing a mean estimate for the independent variable as a depiction of the strength of the model. While this has been a classic approach to understanding the relationship between the predictor and response, a mean estimate does not depict what is truly depict the what occurs in different ranges of data. Given our vast number of data points, dividing and analysing the data in different ranges was imperative.  This is what quantile regression allows—we estimate coefficients of our model to estimate and conditional median. 
Through this research question we explored the relation between the trip distance and the trip duration, where the basic assumption is that a greater trip duration and lesser distances indicates greater traffic in the area. We see how shorter and longer trip durations are affected by trip distance and total amount (cab fare). 

Conclusions:
For initial analysis, the 25th, 50th and 75th quantiles regression models were created.  
(The categorical variables were not one hot-encoded, instead considered label encoded as that they did not prove to contribute heavily to the model, and would add too many features considering the large number of groups with each categorical feature.) 

3.1. 0.25 quantile summary for model


3.2. 0.50 quantile summary for model


3.3. 0.75 quantile summary for model

Considering the fluctuating values of the coefficient of trip_distance and most instinctive relationship between trip distance and duration, we further analysed the relationship between trip distance and different ranges in duration of ride. We compare the quantile regressions with the linear regression to draw conclusions.


We also see that the slopes low incline because of the majority of the trip durations are between 0-150.
The analysis indicates that the estimated mean and median (0.5 quantile) coincide. The different slopes with respect their quantiles indicate that different ranges of trip distances affect the trip distances differently. However we may conclude that for this data, linear regression maybe a suitable model in this case considering the coinciding of mean and median.  









[https://slundberg.github.io/shap/notebooks/plots/dependence_plot.html]
https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b
https://towardsdatascience.com/deep-quantile-regression-c85481548b5a
http://inversionlabs.com/2016/02/07/using-quantile-regression.html

