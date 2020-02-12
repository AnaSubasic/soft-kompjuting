import sys
import os

if len(sys.argv) > 1:
    DATASET_PATH = sys.argv[1]
else:
    DATASET_PATH = '.' + os.path.sep + 'validation' + os.path.sep
# ------------------------------------------------------------------

labeled_samples = dict()

with open(DATASET_PATH+'annotations.csv') as file:
    data = file.read()
    lines = data.split('\n')
    for index, line in enumerate(lines):
        cols = line.split(',')
        if cols and cols[0] == '':
            continue
        cols[0] = cols[0].replace('\r', '')
        cols[1] = cols[1].replace('\r', '')
        labeled_samples[cols[0]] = float(cols[1])

results = dict()

with open('result.csv') as file:
    data = file.read()
    lines = data.split('\n')
    for index, line in enumerate(lines):
        cols = line.split(',')
        if cols and cols[0] == '':
            continue
        cols[0] = cols[0].replace('\r', '')
        cols[1] = cols[1].replace('\r', '')
        results[cols[0]] = float(cols[1])


# evaluate how results file matches the labelled samples
percentage = 0
for labeled_image_name in labeled_samples:
    diff = abs(labeled_samples[labeled_image_name] - results[labeled_image_name])
    if diff == 0:
        percentage += 100.0
    else:
        percentage += 100.0/(diff+1)


percentage /= len(labeled_samples)

print(percentage)
