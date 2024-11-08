import requests
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Create a list of all UN areas/countries
geoAreaResponse = requests.get("https://unstats.un.org/sdgs/UNSDGAPIV5/v1/sdg/GeoArea/List")
geoAreaList = geoAreaResponse.json()

# Print the 10 first geo areas
i = 0
for area in geoAreaList:
    if (i == 10):
        break
    print(area)
    i += 1

# Get SDG indicator codes and names (called Series in UNSDG API language)
seriesCodesResponse = requests.get("https://unstats.un.org/sdgs/UNSDGAPIV5/v1/sdg/SDMXMetadata/GetSeries")
seriesCodesList = seriesCodesResponse.json()

# Print the 10 first series codes and indicator names
i = 0
for series in seriesCodesList:
    if (i == 10):
        break
    print("Series code: " + series["code"] + "\t Indicator name: " + series["description"])
    i += 1

# Get time series for one country. Called "Series/Data" in UNSDG API language
traffickingResponse = requests.get(
    "https://unstats.un.org/sdgs/UNSDGAPIV5/v1/sdg/Series/Data",
    params={"seriesCode": "VC_HTF_DETV", "areaCode": 8}
)
traffickingDict = traffickingResponse.json()

# The response has a "data" key to a list of dictionaries for each time period. This list is the value of the "data" key.
# Inside each dictionary is a new key called "value" which holds the number of children who are victims of trafficking
# "timePeriodStart" is the key of the year which contains the current time period

# Print the number of children who are victims of trafficking in Albania from first to latest year in data
# Fill the data in two numpy arrays. These are needed for plotting and for machine learning

# Initalize data
sum = 0
N = len(traffickingDict["data"])
numYears = 7
years = np.zeros(numYears)
numChild = np.zeros(numYears)
previousYear = 0
newYear = 0
currentNumberOfYears = 0
childSumPerYear = 0

for i in range(N):
    # Store the current dictionary of data
    timePeriod = traffickingDict["data"][i]

    # Store new year
    newYear = int(timePeriod["timePeriodStart"])
    
    if (i == 0): # First year
        print("\nFirst year of data: " + str(timePeriod["timePeriodStart"]))
        previousYear = int(timePeriod["timePeriodStart"])

        # Store first year
        years[currentNumberOfYears] = int(timePeriod["timePeriodStart"])
    
    if (previousYear == newYear):
        # If the year is the same as last: store the sum of children in a temporal variable
        # This is also happening when i == 0
        childSumPerYear += int(timePeriod["value"])
    else:
        currentNumberOfYears += 1
        previousYear = newYear
    
        # New year
        # Store the previous sum of children into previous position (i-1): Store new year in position i 
        # Start a new temporal sum of children for the new year
        years[currentNumberOfYears] = int(timePeriod["timePeriodStart"])
        numChild[currentNumberOfYears - 1] = childSumPerYear
        childSumPerYear = 0
        childSumPerYear = int(timePeriod["value"])

    # Last year
    if (i == (N - 1)):
        print("\nLast year of data: " + str(timePeriod["timePeriodStart"]) + "\n")
        numChild[currentNumberOfYears] = childSumPerYear
    
    # Compute total sum
    sum += int(timePeriod["value"])

print(f"Total sum of children who are victims of trafficking in Albania from 2008 to 2014: {sum}\n")

print(f"Total sum of victims stored in the numpy array: {np.sum(numChild)}\n")

# Print the numpy arrays
print(years)
print(numChild)

# Plot the time series
plt.figure(1)
plt.title("Trafficking victims in Albania from 2008 to 2014 and below 18 years old")
plt.xlabel("Years")
plt.ylabel("Number of children victims")
plt.plot(years, numChild)

# Use third degree regression on the time series
poly = PolynomialFeatures(degree=3, include_bias=False)
poly_features = poly.fit_transform(years.reshape(-1, 1))
poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, numChild)
childrenPredicted = poly_reg_model.predict(poly_features)

# Plot the regression results
plt.figure(2)
plt.title("Regression model on the trafficking data")
plt.scatter(years, numChild)
plt.plot(years, childrenPredicted, c="red")

plt.show()

