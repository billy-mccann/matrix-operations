import csv
import src.matrix_operations as mo 
from src.linear_regression import LinearRegression

path = "/Users/bill/.cache/kagglehub/datasets/nikhil7280/student-performance-multiple-linear-regression/versions/1/Student_Performance.csv"


print("Path to dataset files:", path)

with open(path) as file:
    reader = csv.reader(file)
    header = next(reader)

    matrix = [row for row in reader]
    print(len(matrix))

    max_row = 2000
    # for i in range(max_row):
    #     print(matrix[i])

    smaller = matrix[:max_row]

    for row in smaller:
        row[2] = 1.0 if row[2] == 'Yes' else 0.0

    smaller = [[float(entry) for entry in row] for row in smaller]

    # print("==============") 
    # for row in smaller:
    #     print(row)

    model = LinearRegression.train_least_squares(smaller)

    test = [[7.0, 99.0, 1.0, 9.0, 1.0],
            [4.0, 82.0, 0.0, 4.0, 2.0]]
    
    print("Model trained, here are some results:")
    for entry in test:
        prediction = model.predict(entry)
        rounded = round(prediction, 2)
        print(rounded)



