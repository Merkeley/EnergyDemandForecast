# EnergyDemandForecast
Project Exploring Time Series Modeling with ARIMA and LSTM RNN

This project is currently a work in process.

## Data

Electricity demand data is pulled from eia.gov.  The first iteration is using just electricity demand 
for the state of California.  The api provides hourly demand values for a 4 year time period.  The Dark Sky api is
used to collect weather data in 6 major cities in CA for the same time period.  For the first pass I'm just 
collecting temperature values.  The third piece of data I've added to this data set is population data.  The 
American Community Survey provides census data between the primary census surveys. This data is only provided annually
up to 2018.  I will extrapolate to estimate population data for 2019 & 2020.

