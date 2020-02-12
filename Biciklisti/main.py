from process import train_or_load_model, count_bicycles
import glob
import sys
import os

if len(sys.argv) > 1:
    TRAIN_DATASET_PATH = sys.argv[1]
else:
    TRAIN_DATASET_PATH = '.' + os.path.sep + 'train' + os.path.sep

TRAIN_DATASET_POSITIVE = TRAIN_DATASET_PATH + 'positive' + os.path.sep
TRAIN_DATASET_NEGATIVE = TRAIN_DATASET_PATH + 'negative' + os.path.sep

if len(sys.argv) > 1:
    VALIDATION_DATASET_PATH = sys.argv[2]
else:
    VALIDATION_DATASET_PATH = '.' + os.path.sep + 'validation' + os.path.sep
# -------------------------------------------------------------------

# priprema skupa podataka za metodu za treniranje
train_positive_image_paths = []
for image_name in os.listdir(TRAIN_DATASET_POSITIVE):
    if '.jpg' in image_name or '.png' in image_name:
        train_positive_image_paths.append(os.path.join(TRAIN_DATASET_POSITIVE, image_name))


train_negative_image_paths = []
for image_name in os.listdir(TRAIN_DATASET_NEGATIVE):
    if '.jpg' in image_name or '.png' in image_name:
        train_negative_image_paths.append(os.path.join(TRAIN_DATASET_NEGATIVE, image_name))

model = train_or_load_model(train_positive_image_paths, train_negative_image_paths)


processed_image_names = []
bicycles_count = []

for image_path in glob.glob(VALIDATION_DATASET_PATH + "*.jpg"):
    image_directory, image_name = os.path.split(image_path)
    processed_image_names.append(image_name)
    bicycles_count.append(count_bicycles(image_path, model))

for image_path in glob.glob(VALIDATION_DATASET_PATH + "*.png"):
    image_directory, image_name = os.path.split(image_path)
    processed_image_names.append(image_name)
    bicycles_count.append(count_bicycles(image_path, model))

# -----------------------------------------------------------------
result_file_contents = ""
for image_index, image_name in enumerate(processed_image_names):
    result_file_contents += "%s,%s\n" % (image_name, bicycles_count[image_index])
# sacuvaj formirane rezultate u csv fajl
with open('result.csv', 'w') as output_file:
    output_file.write(result_file_contents)