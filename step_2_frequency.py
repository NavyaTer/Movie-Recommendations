import matplotlib.pyplot as plt

map = dict()

def counter(number):
    file_name = f"recommendations/recommendation_{number}.csv"
    with open(file_name) as myfile:
        head = [next(myfile) for x in range(10)]
        for line in head:
            title = line.split('|')[1]
            if (title in map):
                map[title] += 1
            else:
                map[title] = 1


def return_map():
    for x in range(1,11):
        counter(x)
    return map

