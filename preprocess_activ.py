import os
from os.path import join, isfile, isdir
from PIL import Image
from itertools import islice
from csv import DictReader
from urllib.request import urlretrieve
from get_md5 import file_content_hash
from sys import exc_info
from config import INPUT_HEIGHT, INPUT_WIDTH
import cv2
from random import seed, choice


def redact_ticker(activ_D_folder):

    # Ticker removal only pertains to AljazeeraHD and France24 per AcTiV readme
    channels = ["France24", "AljazeeraHD"]
    modes = ["training", "test"]

    for channel in channels:
        print("Blacking out ticker from {0}".format(channel))
        for mode in modes:
            path_to_images = join(activ_D_folder, channel, mode + "Files")
            file_list = os.listdir(path_to_images)
            file_list = [x for x in file_list if x.endswith(".png")]

            for image_filename in file_list:
                arabic_image = Image.open(join(path_to_images, image_filename))
                width, height = arabic_image.size
                # France24 should be 720 x 576
                if channel == "France24" and width == 720 and height == 576:
                    for x in range(60, 665):
                        for y in range(490, 526):
                            arabic_image.putpixel((x, y), 0)
                #Aljazeera should be 1920 x 1080
                if channel == "AljazeeraHD" and width == 1920 and height == 1080:
                    for x in range(0, 1700):
                        for y in range(980, 1040):
                            arabic_image.putpixel((x, y), 0)

                arabic_image.save(join(path_to_images, image_filename))


def generate_training_data(activ_D_folder, activ_R_folder, ALIF_folder, filler_images_file, data_generation_limit=1000):

    # Get filler images from openimages dataset
    fieldnames = ['ImageID', 'Subset', 'OriginalURL', 'OriginalLandingURL', 'License', 'AuthorProfileURL', 'Author', \
                  'Title', 'OriginalSize', 'OriginalMD5', 'Thumbnail300KURL']
    downloaded_folder = join(activ_D_folder,"Downloaded")
    if not os.path.isdir(downloaded_folder): os.mkdir(downloaded_folder)
    filler_images = []

    with open(filler_images_file, 'r', encoding='latin-1') as f:
        filler_images_dict = DictReader(f, fieldnames=fieldnames)

        for line in islice(filler_images_dict, 1, data_generation_limit):

            candidate_file_name = join(downloaded_folder,line['OriginalURL'].split('/')[-1])

            try:

                # Check if file has already been downloaded
                if isfile(candidate_file_name) and line['OriginalMD5'] == file_content_hash(candidate_file_name):
                    print("Verified {0}".format(line['OriginalURL']))
                    filler_images.append(candidate_file_name)

                # Download file and test hash
                else:
                    urlretrieve(line['OriginalURL'], candidate_file_name)

                    if line['OriginalMD5'] != file_content_hash(candidate_file_name):
                        os.remove(candidate_file_name)
                        print("Warning - {0} did not match expected hash".format(line['OriginalURL']))
                    else:
                        print("Downloaded {0}".format(line['OriginalURL']))
                        filler_images.append(candidate_file_name)

            except:
                print("Error - ", exc_info())
                continue

    print("Number of Open Images:", len(filler_images))

    arabic_chips = list()

    # Inventory all activ-R files in training and test
    channels = ["France24", "AlJazeeraHD", "RussiyaAl-Yaum", "TunisiaNat1"]
    modes = ["training", "test"]

    for channel in channels:
        for mode in modes:
            path_to_chips = join(activ_R_folder, channel, mode + "Files", "images/")
            if os.path.isdir(path_to_chips):
                file_list = os.listdir(path_to_chips)
                file_list = [join(path_to_chips, x) for x in file_list if x.endswith(".png")]
                arabic_chips += file_list

    # Inventory all ALIF files in training and test
    modes = ["alif_train", "alif_test1", "alif_test2", "alif_test3"]

    for mode in modes:
        path_to_chips = join(ALIF_folder, mode)
        if os.path.isdir(path_to_chips):
            file_list = os.listdir(path_to_chips)
            file_list = [join(path_to_chips, x) for x in file_list if x.endswith(".jpg")]
            arabic_chips += file_list

    print("Number of Arabic chips:", len(arabic_chips))

    # Create a folder to store generated images
    generated_folder = join(activ_D_folder,"Generated")
    if not os.path.isdir(generated_folder):
        os.mkdir(generated_folder)
        if not os.path.isdir(join(generated_folder,"trainingFiles")):
            os.mkdir(join(generated_folder,"trainingFiles"))

    seed(41)
    counter = 0
    # Start XML file text
    xml_file_output = '''<?xml version="1.0" encoding="UTF-8"?>\n\n<Protocol4 channel="Generated">\n\n'''

    # Put activR and ALIF chips into openimage candidates
    for filler_image, arabic_chip in zip(filler_images,arabic_chips):

        chip = cv2.imread(arabic_chip)
        filler = cv2.imread(filler_image)

        # Resize openimage candidates to INPUT_HEIGHT, INPUT_WIDTH to align with AcTiV-D dataset
        resized_filler = cv2.resize(filler, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        #print("resized",resized_filler.shape)

        chip_rows, chip_cols, _ = chip.shape
        resized_filler_rows, resized_filler_cols, _ = resized_filler.shape
        column_placement_list = list(range(0,resized_filler_cols-chip_cols))
        row_placement_list = list(range(0,resized_filler_rows-chip_rows))

        # Skip this iteration if image not big enough for chip
        if len(column_placement_list) == 0 or len(row_placement_list) == 0:
            print("Image {0} could not fit chip {1}; skipping".format(filler_image,arabic_chip))
            continue
        chip_column_start = choice(column_placement_list)
        chip_row_start = choice(row_placement_list)
        #print(chip_row_start,chip_column_start)
        resized_filler[chip_row_start:chip_row_start+chip_rows,chip_column_start:chip_column_start+chip_cols] = chip

        # Save image for future training in Generated folder
        print("Created", join(generated_folder,"trainingFiles","Generated_vd00_frame_" + str(counter) + ".png"))
        cv2.imwrite(join(generated_folder,"trainingFiles","Generated_vd00_frame_" + str(counter) + ".png"), resized_filler)

        # Record location as xml format
        xml_file_output += '''<frame source="vd00" id="{0}">
                <rectangle id="1" height="{1}" width="{2}" y="{3}" x="{4}"/>
                </frame>\n'''.format(str(counter),chip_rows,chip_cols,chip_row_start,chip_column_start)
        counter += 1

    # End XML file text
    xml_file_output += "\n</Protocol4>"
    #print(xml_file_output)
    with open(join(generated_folder,"gtraining_Ge.xml"),'w') as f:
        f.write(xml_file_output)


def main(remove_ticker, generate_data, data_generation_limit, activ_D_folder, activ_R_folder, ALIF_folder,
         filler_images_file):

    # PART 1 - Put black box over ticket in aljazeera and france24 pictures per readme instructions
    if remove_ticker:
        redact_ticker(activ_D_folder)

    # PART 2 - Generate new training examples by combining openimages data with AcTiV recognition chips
    if generate_data:
        generate_training_data(activ_D_folder, activ_R_folder, ALIF_folder, filler_images_file, data_generation_limit)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess AcTiV dataset by removing ticker (--remove_ticker) and generating additional training \
                    images (--generate_data')

    # Optional arguments
    parser.add_argument(
        '--remove_ticker',
        #type=bool,
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
        #type=bool,
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

    args = parser.parse_args()
    print(
        "Parameters set as: \n \
           Remove Ticker = {0} \n \
           Generate Data = {1} \n \
           Data Generation Limit = {2} \n \
           AcTiV-D folder = {3} \n \
           AcTiV-R folder = {4} \n \
           ALIF folder = {5} \n \
           Filler images file = {6} \n "
        .format(
            args.remove_ticker,
            args.generate_data,
            args.data_generation_limit,
            args.activ_D_folder,
            args.activ_R_folder,
            args.ALIF_folder,
            args.filler_images_file))
    main(args.remove_ticker, args.generate_data, args.data_generation_limit, args.activ_D_folder, args.activ_R_folder,
         args.ALIF_folder, args.filler_images_file)
    print("Done")