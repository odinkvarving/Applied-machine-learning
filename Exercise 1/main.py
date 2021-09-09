import linear_regression
import non_linear_regression

import csv
import os.path

debug = False


def loadData(filePath):
    csvFile = open(filePath)
    csvData = csv.reader(csvFile)

    dataList = []

    for row in csvData:
        dataList.append(row)
    csvFile.close()
    return dataList

def checkIfInt(user_input):
    try:
        int(user_input)
        return True
    except ValueError:
        return False

def choose_file():
    print("Select which file to perform vizualization of: ")
    print("1. length_weight.csv")
    print("2. day_length_weight.csv")
    print("3. day_head_circumference.csv")

    file_chosen = False
    while not file_chosen:
        selected_file = input("Select option 1, 2 or 3: ")

        if checkIfInt(selected_file):
            selected_file = int(selected_file)

            if(selected_file == 1):
                return "length_weight.csv"

            if(selected_file == 2):
                return "day_length_weight.csv"

            if(selected_file == 3):
                return "day_head_circumference.csv"

data = []
fileName = choose_file()

fullPath = os.path.dirname(os.path.abspath(__file__))+'/datasets/' + fileName
if os.path.isfile(fullPath) :
    print("loading " + fileName)
    data = loadData(fullPath)

if fileName == "day_head_circumference.csv" :
    non_linear_regression.visualize2D(data)
else :
    linear_regression.runLinearRegression(data)







