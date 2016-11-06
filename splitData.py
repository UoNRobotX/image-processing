import sys, re, random

usage = """Usage: " + sys.argv[0] + " file1
    Splits data file "file1" into training, validation, and test sets.
"""

#process command line arguments
if len(sys.argv) != 2:
    print(usage, file=sys.stderr)
    sys.exit(1)
dataFile = sys.argv[1]

filenames = []
data = []
with open(dataFile) as file:
    for line in file:
        if line[0] != " ":
            filenames.append(line)
            data.append([])
        else:
            data[-1].append(line)
records = [(filenames[i], data[i]) for i in range(len(filenames))]
random.shuffle(records)

PROPORTIONS = (0.7, 0.2, 0.1) #training, validation, test
if len(PROPORTIONS) != 3 or sum(PROPORTIONS) > 1.0:
    raise Exception("Invalid proportions for splitting")

trainingSplit = int(len(records) * PROPORTIONS[0])
validationSplit = int(len(records) * (PROPORTIONS[0] + PROPORTIONS[1]))

trainingFile = re.sub(r"(\.[^.]*)?$", "_train\g<0>", dataFile)
validationFile = re.sub(r"(\.[^.]*)?$", "_validate\g<0>", dataFile)
testingFile = re.sub(r"(\.[^.]*)?$", "_test\g<0>", dataFile)

with open(trainingFile, "w") as file:
    for i in range(trainingSplit):
        file.write(records[i][0])
        for line in records[i][1]:
            file.write(line)
with open(validationFile, "w") as file:
    for i in range(trainingSplit, validationSplit):
        file.write(records[i][0])
        for line in records[i][1]:
            file.write(line)
with open(testingFile, "w") as file:
    for i in range(validationSplit, len(records)):
        file.write(records[i][0])
        for line in records[i][1]:
            file.write(line)
