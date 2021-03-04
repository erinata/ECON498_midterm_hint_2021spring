import pandas
from sklearn import linear_model
from datetime import datetime
import numpy

dataset = pandas.read_csv("Crimes_-_2001_to_Present.csv")   # load the dataset

dataset = dataset[dataset['Year'] == 2018]       # Use only the 2018 part of dataset  
dataset.dropna(inplace=True)	                 # Drop observations with missing data



dataset['Date'] = pandas.to_datetime(dataset['Date'])     # Turn the Date column into datetime
dataset['hour'] = dataset['Date'].dt.hour          # Get the hour

dataset['hour_slot'] = numpy.select([
	(dataset['hour'] < 4),
	(dataset['hour'] < 8),
	(dataset['hour'] < 12),
	(dataset['hour'] < 16),
	(dataset['hour'] < 20),
	(dataset['hour'] < 24)]
	, [0,1,2,3,4,5])

dataset['hour_slot_0'] = numpy.where(dataset['hour_slot']==0, 1, 0)  # Generate a column which is equal to 1 if the hour_slot is 0
dataset['hour_slot_1'] = numpy.where(dataset['hour_slot']==1, 1, 0)  # Generate a column which is equal to 1 if the hour_slot is 1
dataset['hour_slot_2'] = numpy.where(dataset['hour_slot']==2, 1, 0)  # Generate a column which is equal to 1 if the hour_slot is 2
dataset['hour_slot_3'] = numpy.where(dataset['hour_slot']==3, 1, 0)  # Generate a column which is equal to 1 if the hour_slot is 3
dataset['hour_slot_4'] = numpy.where(dataset['hour_slot']==4, 1, 0)  # Generate a column which is equal to 1 if the hour_slot is 4
dataset['hour_slot_5'] = numpy.where(dataset['hour_slot']==5, 1, 0)  # Generate a column which is equal to 1 if the hour_slot is 5


dataset['theft'] = numpy.where(dataset['Primary Type']=='THEFT', 1, 0)  # Generate a column which is equal to 1 if the reported crime is theft



target = dataset.iloc[:,30].values     # Get the column 'theft' 

data = dataset.iloc[:,24:30].values    # Get the columns 'hour_slot_0','hour_slot_1' ......


machine = linear_model.LogisticRegression()   # Construct the machine
machine.fit(data, target)   # Fit the data and the target

new_data = [
	[1,0,0,0,0,0],
	[0,1,0,0,0,0],
	[0,0,1,0,0,0],
	[0,0,0,1,0,0],
	[0,0,0,0,1,0],
	[0,0,0,0,0,1]
]

new_target = machine.predict_proba(new_data)  
print(new_target)
