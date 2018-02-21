import os
from os.path import join, isfile, isdir
from PIL import Image
from itertools import islice
from csv import DictReader
from urllib.request import urlretrieve
from get_md5 import file_content_hash
from sys import exc_info
from global_config import ONE_IMAGE_SIZE, INPUT_HEIGHT, INPUT_WIDTH
import cv2
from random import seed, choice
from collections import defaultdict
import coco_text

def redact_ticker(activ_D_folder):

    # Ticker removal only pertains to AljazeeraHD and France24 per AcTiV readme
    channels = ["France24", "AljazeeraHD"]
    modes = ["training", "test"]

    for channel in channels:
        print("Blocking out ticker from {0}".format(channel))
        for mode in modes:
            path_to_images = join(activ_D_folder, channel, mode + "Files")
            file_list = os.listdir(path_to_images)
            file_list = [x for x in file_list if x.endswith(".png")]

            for image_filename in file_list:
                arabic_image = Image.open(join(path_to_images, image_filename))
                width, height = arabic_image.size

                # Block out entire ticker with a random color
                r_color = choice(range(0,255))
                g_color = choice(range(0,255))
                b_color = choice(range(0,255))

                # France24 should be 720 x 576
                if channel == "France24" and width == 720 and height == 576:
                    for x in range(60, 665):
                        for y in range(490, 526):
                            arabic_image.putpixel((x, y), (r_color,g_color,b_color))
                #Aljazeera should be 1920 x 1080
                elif channel == "AljazeeraHD" and width == 1920 and height == 1080:
                    for x in range(0, 1700):
                        for y in range(980, 1040):
                            arabic_image.putpixel((x, y), (r_color,g_color,b_color))
                #Aljazeera should be 1920 x 1080 but a few are off
                elif channel == "AljazeeraHD" and width == 1920 and height == 1088:
                    for x in range(0, 1700):
                        for y in range(980+8, 1040+8):
                            arabic_image.putpixel((x, y), (r_color,g_color,b_color))
                else:
                    print("Warning: {0} has unexpected shape {1}".format(image_filename,arabic_image.size))

                arabic_image.save(join(path_to_images, image_filename))


def generate_training_data(activ_D_folder, activ_R_folder, ALIF_folder, filler_images_file, data_generation_limit=1000):

    # Get filler images from openimages dataset
    fieldnames = ['ImageID', 'Subset', 'OriginalURL', 'OriginalLandingURL', 'License', 'AuthorProfileURL', 'Author', \
                  'Title', 'OriginalSize', 'OriginalMD5', 'Thumbnail300KURL']
    downloaded_folder = join(activ_D_folder,"Downloaded")
    if not isdir(downloaded_folder): os.mkdir(downloaded_folder)
    filler_images = []
    counter = 0

    with open(filler_images_file, 'r', encoding='latin-1') as f:
        filler_images_dict = DictReader(f, fieldnames=fieldnames)

        for line in islice(filler_images_dict, 1, data_generation_limit):

            candidate_file_name = join(downloaded_folder,line['OriginalURL'].split('/')[-1])

            try:

                # Check if file has already been downloaded
                if isfile(candidate_file_name) and line['OriginalMD5'] == file_content_hash(candidate_file_name):
                    #print("Verified {0}".format(line['OriginalURL']))
                    filler_images.append(candidate_file_name)
                    counter += 1

                # Download file and test hash
                else:
                    urlretrieve(line['OriginalURL'], candidate_file_name)

                    if line['OriginalMD5'] != file_content_hash(candidate_file_name):
                        os.remove(candidate_file_name)
                        #print("Warning - {0} did not match expected hash; skipping".format(line['OriginalURL']))
                    else:
                        #print("Downloaded {0}".format(line['OriginalURL']))
                        filler_images.append(candidate_file_name)
                        counter += 1

            except:
                print("Error - ", exc_info())
                continue

            if counter % 1000 ==0 and counter !=0:
                print("Downloaded {0} Open Images".format(counter))

    print("Number of Open Images:", len(filler_images))

    arabic_chips = list()

    # Inventory all activ-R files in training and test
    channels = ["France24", "AlJazeeraHD", "RussiyaAl-Yaum", "TunisiaNat1"]
    modes = ["training", "test"]

    for channel in channels:
        for mode in modes:
            path_to_chips = join(activ_R_folder, channel, mode + "Files", "images/")
            if isdir(path_to_chips):
                file_list = os.listdir(path_to_chips)
                file_list = [join(path_to_chips, x) for x in file_list if x.endswith(".png")]
                arabic_chips += file_list

    # Inventory all ALIF files in training and test
    modes = ["alif_train", "alif_test1", "alif_test2", "alif_test3"]

    for mode in modes:
        path_to_chips = join(ALIF_folder, mode)
        if isdir(path_to_chips):
            file_list = os.listdir(path_to_chips)
            file_list = [join(path_to_chips, x) for x in file_list if x.endswith(".jpg")]
            arabic_chips += file_list

    num_arabic_chips = len(arabic_chips)
    print("Number of Arabic chips:", num_arabic_chips)

    # create triplets of arabic_chips to place in filler images
    step = int(num_arabic_chips/3)
    arabic_chip_triplets = []
    for i in range(step):
        # triplets
        arabic_chip_triplets.append([arabic_chips[i],arabic_chips[i+step],arabic_chips[i+2*step]])

    # Create a folder to store generated images
    generated_folder = join(activ_D_folder,"Generated")
    if not isdir(generated_folder):
        os.mkdir(generated_folder)
        if not isdir(join(generated_folder,"trainingFiles")):
            os.mkdir(join(generated_folder,"trainingFiles"))

    seed(41)
    counter = 0
    # Start XML file text
    xml_file_output = '''<?xml version="1.0" encoding="UTF-8"?>\n\n<Protocol4 channel="Generated">\n\n'''

    # Put activR and ALIF chips into openimage candidates
    # for filler_image, arabic_chip in zip(filler_images,arabic_chips):
    for filler_image, arabic_chips in zip(filler_images, arabic_chip_triplets):

        filler = cv2.imread(filler_image)

        if ONE_IMAGE_SIZE:
            # Resize openimage candidates to INPUT_HEIGHT, INPUT_WIDTH to align with AcTiV-D dataset
            resized_filler = cv2.resize(filler, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        else:
            resized_filler = filler

        resized_filler_rows, resized_filler_cols, _ = resized_filler.shape

        xml_file_output += '''<frame source="vd00" id="{0}">'''.format(str(counter))
        pixels_used = defaultdict(bool)
        rectangle_num = 0

        for arabic_chip in arabic_chips:

            rectangle_num +=1
            chip = cv2.imread(arabic_chip)
            chip_rows, chip_cols, _ = chip.shape
            column_placement_list = list(range(0, resized_filler_cols - chip_cols))
            row_placement_list = list(range(0, resized_filler_rows - chip_rows))

            # If chip is too big (likely too long) for image then use as negative training example
            if len(column_placement_list) == 0 or len(row_placement_list) == 0:
                continue

            # Chip fits in Filler Image
            else:
                placed_chip = False
                attempts = 0
                # Look for location to place chip that doesn't overlap with previous chips
                while not placed_chip and attempts < 50:
                    chip_column_start = choice(column_placement_list)
                    chip_row_start = choice(row_placement_list)

                    use_columns = range(chip_column_start, chip_column_start + chip_cols)
                    use_rows = range(chip_row_start, chip_row_start + chip_rows)

                    placed_chip = check_pixels_used(pixels_used,use_columns,use_rows)
                    attempts += 1

                # Continue to next chip if this chip was not placed after max attempts
                if not placed_chip:
                    continue

                # Blend chip into filler background image
                background = resized_filler[chip_row_start:chip_row_start + chip_rows,
                                            chip_column_start:chip_column_start + chip_cols]
                # Give equal weighting to reduce hard edges
                blended = cv2.addWeighted(background, 0.4, chip, 0.6, 0)
                resized_filler[chip_row_start:chip_row_start + chip_rows,
                               chip_column_start:chip_column_start + chip_cols] = blended

                # Record location as xml format
                xml_file_output += '''<rectangle id="{4}" height="{0}" width="{1}" y="{2}" x="{3}"/>\n'''.format(
                    chip_rows, chip_cols, chip_row_start, chip_column_start, rectangle_num)

        xml_file_output += '''</frame>\n'''

        cv2.imwrite(join(generated_folder, "trainingFiles", "Generated_vd00_frame_" + str(counter) + ".png"),
                    resized_filler)

        counter += 1
        if counter % 1000 == 0:
            print("Generated {0} training examples".format(counter))

    # End XML file text
    xml_file_output += "\n</Protocol4>"
    #print(xml_file_output)
    with open(join(generated_folder,"gtraining_Ge.xml"),'w') as f:
        f.write(xml_file_output)

    # Print out final count
    print("Generated {0} training examples".format(counter))


def check_pixels_used(pixels_used, use_columns, use_rows):

    # Check if any pixel in proposed chip placement has been used. If so, return False
    for col in use_columns:
        for row in use_rows:
            if pixels_used[str(col)+"_"+str(row)]:
                return False
    # If no conflicts exist, mark pixels as used and return True
    for col in use_columns:
        for row in use_rows:
            pixels_used[str(col)+"_"+str(row)] = True

    return True


def add_negative_sampling_data(activ_D_folder, total_negative_samples=1000):

    # Create a folder to store negative sampling images
    negative_folder = join(activ_D_folder,"Negative")
    if not isdir(negative_folder):
        os.mkdir(negative_folder)
        if not isdir(join(negative_folder,"trainingFiles")):
            os.mkdir(join(negative_folder,"trainingFiles"))

    # Get list of images with text from COCO-Text dataset
    ct = coco_text.COCO_Text('COCO_Text.json')
    negative_images = ct.loadImgs(ct.getImgIds(imgIds=ct.train,
                             catIds=[('legibility', 'legible'),
                                     #('class', 'machine printed'),
                                     ('language', 'english')])[0:total_negative_samples])

    # Start XML file text
    xml_file_output = '''<?xml version="1.0" encoding="UTF-8"?>\n\n<Protocol4 channel="Negative">\n\n'''

    counter = 0

    for negative_image_dict in negative_images:
        xml_file_output += '''<frame source="vd00" id="{0}"></frame>\n'''.format(str(counter))

        # Make path to COCO train2014 folder to load image
        negative_image = cv2.imread(join(activ_D_folder,"COCO","train2014",negative_image_dict['file_name']))

        if ONE_IMAGE_SIZE:
            # Resize to INPUT_HEIGHT, INPUT_WIDTH to align with AcTiV-D dataset
            resized_image = cv2.resize(negative_image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        else:
            resized_image = negative_image

        # Save in Negative folder under AcTiV-D
        cv2.imwrite(join(negative_folder, "trainingFiles", "Generated_vd00_frame_" + str(counter) + ".png"),
                    resized_image)

        counter += 1
        if counter % 200 == 0:
            print("Added {0} negative training examples".format(counter))

    # End XML file text
    xml_file_output += "\n</Protocol4>"

    #print(xml_file_output)
    with open(join(negative_folder,"gtraining_Ne.xml"),'w') as f:
        f.write(xml_file_output)

    # Print out final count
    print("Added {0} negative training examples".format(total_negative_samples))


def main(remove_ticker, generate_data, data_generation_limit, activ_D_folder, activ_R_folder, ALIF_folder,
         filler_images_file, add_negative_sampling):

    # PART 1 - Block out box over ticker in aljazeera and france24 pictures per readme instructions
    if remove_ticker:
        redact_ticker(activ_D_folder)

    # PART 2 - Generate new training examples by combining openimages data with AcTiV recognition chips
    if generate_data:
        generate_training_data(activ_D_folder, activ_R_folder, ALIF_folder, filler_images_file, data_generation_limit)

    # PART 3 - Add negative sampling images with non-Arabic text
    if add_negative_sampling:
        add_negative_sampling_data(activ_D_folder)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess AcTiV dataset by removing ticker (--remove_ticker) and generating additional training \
                    images (--generate_data')

    # Optional arguments
    parser.add_argument(
        '--remove_ticker',
        default=False,
        action='store_true',
        help='Remove bottom ticker from France24 and AlJazeera channels as suggested in AcTiV readme. Default = True.')

    parser.add_argument(
        '--activ_D_folder',
        type=str,
        default="/arabic_text/AcTiV-D",
        help='Location of AcTiV dataset. Default = /arabic_data/AcTiV-D')

    parser.add_argument(
        '--generate_data',
        default=False,
        action='store_true',
        help='Generate data using AcTiV recognition chips. Default = False')

    parser.add_argument(
        '--data_generation_limit',
        type=int,
        default=1000,
        help='Limit of number of generated data files using AcTiV recognition chips. Default = 1000')

    parser.add_argument(
        '--activ_R_folder',
        type=str,
        default="/arabic_text/AcTiV-R",
        help='Location of AcTiV dataset. Default = /arabic_data/AcTiV-R')

    parser.add_argument(
        '--ALIF_folder',
        type=str,
        default="/arabic_text/ALIF",
        help='Location of ALIF dataset. Default = /arabic_data/ALIF')

    parser.add_argument(
        '--filler_images_file',
        type=str,
        default="/arabic_text/OpenImages/2017_11/train/images.csv",
        help='Location of file containing filler image details in Open Images csv format. Default = /filler_image_data')

    parser.add_argument(
        '--add_negative_sampling',
        default=False,
        action='store_true',
        help='Add negative sampling data for text using COCO-Text images. Default = False')

    args = parser.parse_args()
    print(
        "Parameters set as: \n \
           Remove Ticker = {0} \n \
           Generate Data = {1} \n \
           Data Generation Limit = {2} \n \
           AcTiV-D folder = {3} \n \
           AcTiV-R folder = {4} \n \
           ALIF folder = {5} \n \
           Filler images file = {6} \n \
           Add Negative Samples = {7} "
        .format(
            args.remove_ticker,
            args.generate_data,
            args.data_generation_limit,
            args.activ_D_folder,
            args.activ_R_folder,
            args.ALIF_folder,
            args.filler_images_file,
            args.add_negative_sampling))
    main(args.remove_ticker, args.generate_data, args.data_generation_limit, args.activ_D_folder, args.activ_R_folder,
         args.ALIF_folder, args.filler_images_file, args.add_negative_sampling)
    print("Done")