import csv
import sys
import itertools
from statistics import mean
from collections import defaultdict

class Results:

    # Read the data from console    
    def __init__(self,c_path=sys.argv[1]):
        with open(c_path,'r') as file:
            csv_file = csv.DictReader(file)
            self.data = []
            for row in csv_file:
                self.data.append(dict(row)) 
                self.list_keys = list(row.keys())  
    
    # Find the Topper in each subject
    def find_topper(self):
        result = defaultdict(list)
        for keys in self.list_keys[1:]:
            mat = []
            for rows in self.data:
                mat.append(rows[keys])
            maximum = max(map(int,mat))                                      # Maximum marks scored by student
            for (index, data) in enumerate(self.data):
                if data[keys] == str(maximum):
                    result[keys].append(data["Name"])           #save the result in dictionary
            mat.clear()
        for key,values in result.items():
            print(f'{" & ".join(values)} Topper in {key}')           # Get Topper in each subject

    # Find the best students of class
    def find_best_students(self):
        avg_dict = {}
        for rows in self.data:
            value_list = list(rows.values())
            best_avg = round(mean(map(int,value_list[1:])),2)       # Get the best average
            avg_dict.update({rows["Name"]:best_avg})
        sorted_dict = sorted(avg_dict.items(), key=lambda x: x[1]) 
        print(f"Best students in the class are: \n 1. {sorted_dict[-1][0]} \n 2. {sorted_dict[-2][0]} \n 3. {sorted_dict[-3][0]}")


if __name__ == "__main__":
    obj = Results() 
    obj.find_topper()
    obj.find_best_students()
        