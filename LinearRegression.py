import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import pylab as pl
import numpy as np

#Storing NC GDPC information into a pandas dataframe
gdpc_df = pd.read_csv('GDPC.csv')


# Print the table columbs for each CSV file using the .columns function and count
# the number of rows using .count function
print("This is a column for the GDPC table : " + gdpc_df.columns)
shape2=gdpc_df.shape
print("The numbner of rows and columns for the GDPC table is " + str(shape2))

# Take only the rows that do NOT have any null values.
cdf = gdpc_df[pd.notnull(gdpc_df[['State Industry GDPS ($Millions)','Data Period']])]
cdf1 = cdf[['State Industry GDPS ($Millions)','Data Period']]
print(cdf1.head(9))

# Generate and print a descriptive statistic that summirizes the two columns seperately
print(cdf1['State Industry GDPS ($Millions)'].describe())
print(cdf1['Data Period'].describe())

# Rename the columns
cdf2 = cdf1.rename(columns={'State Industry GDPS ($Millions)': 'GDP',
'Data Period': 'Year'})
print(cdf2.head(1))

# Remove the $ signs and commas
cdf2['GDP'] = cdf2['GDP'].replace({'\$': '', ',': ''}, regex=True)

print(cdf2.head(1))

# Convert the column data types to float
cdf2['Year'] = cdf2['Year'].astype(float)
cdf2['GDP']= cdf2['GDP'].astype(float)


# Plot
plt.scatter(cdf2.Year, cdf2.GDP, color='blue')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.show()

# Only use values where the GDP is less than 200,000
cdf2 = cdf2[~(cdf2['GDP'] <= 200000)]

# Look at the new plot
plt.scatter(cdf2.Year, cdf2.GDP, color='blue')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.show()


# split the data set into train and test test, 80% of the entire data
# for training and 20% for testing using np.random.rand() function
msk = np.random.rand(len(cdf2)) < 0.8
train = cdf2[msk]
test = cdf2[~msk]

# Train data distribution
plt.scatter(train.Year, train.GDP,  color='blue')
plt.xlabel("Year")
plt.ylabel("GDP")
plt.show()

# use sklearn to model data
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Year']])
train_y = np.asanyarray(train[['GDP']])
regr.fit (train_x, train_y)

# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# Plot fit line over the data
plt.scatter(train.Year, train.GDP,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Year")
plt.ylabel("GDP")
plt.show()

# Use Mean Squared Error (MSE) here to calculate the accuracy of our
# model based on the test set. The higher R-squared, the better the model
# fits our data

test_x = np.asanyarray(test[['Year']])
test_y = np.asanyarray(test[['GDP']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )
