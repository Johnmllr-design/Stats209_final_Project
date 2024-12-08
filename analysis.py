import pandas as pd
import matplotlib.pyplot as plt
import numpy
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class dataVisualizations():
    def __init__(self, givendata: list[int]) -> None:
        self.data = givendata

    def getData(self, csvfile):
        df = pd.read_excel(csvfile)

        # Convert the DataFrame to a 2D NumPy array
        array_2d = df.values

        # the following is to show the distribution of mileage to xc times
        averageMileage = []
        averageTimes = []
        for row in array_2d:
            averageMileage.append(row[41])
            averageTimes.append(row[54])
        plt.scatter(averageMileage, averageTimes)
        plt.xlabel('averageMileage')
        plt.ylabel('averageTimes')
        plt.title('Plot of X vs Y')
        plt.show()

        # the following is to show the distribution of proportion of cross training miles to times

        averageProportions = []
        for i in range(0, len(array_2d)):
            curRow = array_2d[i]
            totalMiles = 0
            XTmiles = 0
            for i in range(2, 40, 3):
                if not numpy.isnan(curRow[i]):
                    XTmiles += curRow[i]
                    totalMiles += curRow[i + 1]
            averageProportions.append(XTmiles/totalMiles)
        plt.scatter(averageProportions, averageTimes)
        plt.xlabel('average proportions')
        plt.ylabel('averageTimes')
        plt.title('Plot of X vs Y')
        plt.show()

        # perform a linear regression on the first data set

        # Create a linear regression model
        model = LinearRegression()

        # Fit the model to the training data
        avgMiles = np.array(averageMileage).reshape((-1, 1))
        avgTimes = np.array(averageTimes)
        model = LinearRegression().fit(avgMiles, avgTimes)
        r_sq = model.score(avgMiles, avgTimes)
        print(f"coefficient of determination: {r_sq}")
        y_pred = model.predict(avgMiles)
        plt.plot(averageMileage, y_pred)
        plt.scatter(averageMileage, averageTimes)
        plt.xlabel('average proportions')
        plt.ylabel('prediction')
        plt.title('Plot of X vs Y')
        plt.show()


obj = dataVisualizations([1, 2, 3])
obj.getData("dataa.xlsx")
