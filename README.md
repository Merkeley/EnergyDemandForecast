# Electricity Demand Forecasting

## Demand Forecasting

Demand forecasting is one of the more challenging operations task for any business and yet it is critically important input for most the the operations planning.  Short term forecasts are used for labor management and supply chain planning.  Long term forecasts are used for capital planning and long term budgeting decissions.

Historically, regression based and causal models provided better accuracy than statistical models but were expensive to implement.  With recent advances in AI and Machine Learning this may no longer be the case.

>[Sound predictions of demands and trends are no longer luxury items, but a necessity, if managers are to cope with seasonality, sudden changes in demand levels, price-cutting maneuvers of the competition, strikes, and large swings of the economy. - *[Harvard Business Review, 'How to Choose The Right Forecasting Technique'](https://hbr.org/1971/07/how-to-choose-the-right-forecasting-technique)*

However, time series modeling is a different animal than most Machine Learning modeling techniques.  Many machine learning models assume that samples are independent from each other.  Most time dependent processes are autoregressive, meaning the state of the process at any time is a function of what came before it.  For physical processes, changes in state are limited by the laws of physics.  While general business processes aren't subject to the laws of physics they rarely make radical abrupt changes.

Intuitively, we might propose projecting the current state of business into the future for planning.  We might also consider trends, seasonaly averages and marketing activities in our projections.  Fortunately, there are several Machine Learning models that can incorporate all these features in an automated way.

SARIMAX is a commonly used time series modeling technique.  SARIMAX stands for Seasonal Autoregressive Integrating Moving Average with Exogenous data (yes, that's a mouthful).  If you're interested in the math for this model you can find the basic equations described [here](./Notes/SARIMAX\ Equation\ Notes.md).  In short, SARIMAX uses a combination of the process state at previous time steps, the modeling error at those time steps, past seasonal components and inputs that are related to the main variable of interest (exogenous data) to predict future values.  SARIMAX is a general time series forecasting model and any business cycle 'awareness' must be included in the model configuration by the analyst or data scientist.

Recently, engineers at Facebook developed a forecasting Machine Learn model that was designed specifically to forecast for normal business cycles.  A publication describing the model development can be found [here](
https://peerj.com/preprints/3190/).  Acknowledging that there are typically weekly, monthly and yearly business cycles the model uses mathmatical transformations to identify those fluctuations.  The model also accomodates exogenous data and has a built in system for handling semi-periodic events like holidays.

In this project I will compare the results of these two common forecasting techniques.

## Electricity Demand in California

California currently has nearly 200 gas fired power plants.  These power plants provide up to 33% of the total electricity needed in the state.  Many of these power plant are what is known as Peaker plants.  They are a critical part of the energy infrastructure because they can be brought online or taken offline quickly to meet rapid changes in demand.  While the plants can change their generation state quickly, accurate demand planning can be used to improve operations.  Accurate short-term demand forecasts can help improve facility management and gas supply planning.  Accurate long-term forecasting can be used when scheduling down time for maintenance and planning for capital improvements.


## Data

Electricity demand data for this exploration was collected from [EIA.gov](https://www.eia.gov/) for the state of California.  The EIA API provides hourly demand values for a 4 year time period.  High temperatures for 6 major cities in the state was added to the electricity demand using the Dark Sky API.  Finally, state household counts were added using data from [The 
American Community Survey](https://www.census.gov/programs-surveys/acs).

The hourly data was aggregated to provide daily demand totals and weekly demand totals that could be used as alternative inputs into the two modeling techniques.

## Modeling

Each of the modeling techniques was applied to hourly, daily and weekly demand data.  The best two models were then evaluated using walk forward validation.  Finally, the best model (Prophet applied to daily demand) was used to forecast electricity future electricity demand.

## Repository Resources
This repository contains the following resource for this project.

- **[Notebooks](./Notebooks)**
    - **[EF_EDA_CleanSplitAgg](Notebooks/EF_EDA_CleanSplitAgg.ipynb)** - This notebook contains some data cleaning and feature engineering.  It also creates the initial training and test datasets.  After the data is split the cells in the notebook contain some initial data analysis and finally do data aggregation for model inputs.
    - **[EF_Modeling_SARIMAX_hour](./Notebooks/EF_Modeling_SARIMAX_hour.ipynb)** - SARIMAX modeling using hourly demand data.
    - **[EF_Modeling_SARIMAX_day](./Notebooks/EF_Modeling_SARIMAX_day.ipynb)** - SARIMAX modeling using daily demand data.
    - **[EF_Modeling_SARIMAX_week](./Notebooks/EF_Modeling_SARIMAX_week.ipynb)** - SARIMAX modeling using weekly demand data.
    - **[EF_Modeling_Prophet_hour](./Notebooks/EF_Modeling_Prophet_hour.ipynb)** - Prophet modeling using hourly demand data.
    - **[EF_Modeling_Prophet_day](./Notebooks/EF_Modeling_Prophet_day.ipynb)** - Prophet modeling using hourly demand data.
    - **[EF_Modeling_Prophet_week](./Notebooks/EF_Modeling_Prophet_week.ipynb)** - Prophet modeling using weekly demand data.
    - **[EF_Walk_Validation](./Notebooks/EF_Walk_Validation.ipynb)** - Walk forward validation for the best two models.
    - **[EF_Model_Evaluation](./Notebooks/EF_Model_Evaluation.ipynb)** - Final model development and forecasting.
    
- **[src](./src)**
    - **[eia_query.py](./src/eia_query.py)** - Script to queary the EIA API.
    - **[dsky_query.py](./src/dsky_query.py)** - Functions to access Dark Sky for weather data.
    - **[add_census.py](./src/add_census.py)** - Functions to add census data to the electricity demand dataset.
    - **[data_merge.py](./src/data_merge.py)** - Functions to merge data from multiple files.
    - **[EF_Tools.py](./src/EF_Tools.py)** - Functions used in multiple notebooks for data aggregation and validation.

