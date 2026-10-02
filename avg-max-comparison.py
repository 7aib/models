import csv
import sys


def calculate_average(values):
    total = 0
    for value in values:
        total = total + value
    return total / len(values)


def calculate_max(values):
    maximum = values[0]
    for value in values:
        if value > maximum:
            maximum = value
    return maximum


def main():
    file_name = input("Enter the name of the data file (example: sample_data.csv): ")

    try:
        file = open(file_name, "r")
    except FileNotFoundError:
        print("File not found.")
        sys.exit(1)

    reader = csv.reader(file)
    header = next(reader)

    rows = []
    for row in reader:
        rows.append(row)

    file.close()

    for i in range(len(header)):
        column_name = header[i]

        values = []
        for row in rows:
            if i >= len(row):
                continue
            try:
                value = float(row[i])
                values.append(value)
            except ValueError:
                pass

        if len(values) == 0:
            print(column_name + ": no numbers found in this column")
            print()
            continue

        average = calculate_average(values)
        maximum = calculate_max(values)

        print("Column: " + column_name)
        print("Average value: " + str(average))
        print("Max value: " + str(maximum))

        print()


main()