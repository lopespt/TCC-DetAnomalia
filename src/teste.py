
from dados import read_csv
from classbalance import imbalance 

x = read_csv.load_data("./dados/weather_data_jdt5.txt")

imbalance.printHead(x)