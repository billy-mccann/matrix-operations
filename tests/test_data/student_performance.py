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

    end_training_range = 7000
    end_validation_range = 8500

    for row in matrix:
        row[2] = '1.0' if row[2] == 'Yes' else '0.0'

    training_data= [[float(entry) for entry in row] for row in matrix[:end_training_range]]
    validation_data= [[float(entry) for entry in row] for row in matrix[end_training_range:end_validation_range]]
    testing_data= [[float(entry) for entry in row] for row in matrix[end_validation_range:]]

    print(f"training_data: {len(training_data)}")
    print(f"validation_data: {len(validation_data)}")
    print(f"testing_data: {len(testing_data)}")
    
    model = LinearRegression.train_least_squares(training_data)
    test = [[7.0, 99.0, 1.0, 9.0, 1.0],
            [4.0, 82.0, 0.0, 4.0, 2.0]]
    
    print("Model trained, here are some results:")
    for entry in test:
        prediction = model.predict(entry)
        rounded = round(prediction, 2)
        print(rounded)

