from lxml import etree
import tensorflow as tf
from object_detection.utils import dataset_util
from os.path import join, isfile
from PIL import Image
import io
from global_config import INPUT_HEIGHT, INPUT_WIDTH, ONE_IMAGE_SIZE, USE_GRAYSCALE


def create_tf_example(example, mode):
    # Some images referenced in the xml aren't in the dataset
    try:
        activ_image = Image.open(join(example['path_to_image'], example['file_name']), mode='r')
    except:
        print("Could not find {0}; skipping".format(join(example['path_to_image'], example['file_name'])))
        return None

    if USE_GRAYSCALE:
        activ_image = activ_image.convert('L').convert('RGB')

    # Normalized x,y coordinates
    width, height = activ_image.size
    xmins = [x / float(width) for x in example['bbox_xmins']]
    xmaxs = [x / float(width) for x in example['bbox_xmaxs']]
    ymins = [y / float(height) for y in example['bbox_ymins']]
    ymaxs = [y / float(height) for y in example['bbox_ymaxs']]

    # Skip the image if it doesn't match INPUT_WIDTH x INPUT_HEIGHT
    if mode != "test" and ONE_IMAGE_SIZE and (height != INPUT_HEIGHT or width != INPUT_WIDTH):
        #print("Input image does not match expected size {0}x{1}; skipping".format(INPUT_WIDTH,INPUT_HEIGHT))
        return None
    
    # If needed, resize now that the normalized box coordinates have been calculated
    if width > 1000 or height > 1000:
        basewidth = 1000
        wpercent = (basewidth/float(width))
        hsize = int((float(height)*float(wpercent)))
        activ_image = activ_image.resize((basewidth,hsize), Image.ANTIALIAS)
        width, height = activ_image.size

    imgByteArr = io.BytesIO()
    if example['extension'] == 'jpg':
        activ_image.save(imgByteArr, format='JPEG')
    else:
        activ_image.save(imgByteArr, format='PNG')
    encoded_image_data = imgByteArr.getvalue()  # Encoded image bytes

    filename = example['file_name'].encode('utf-8')  # Filename of the image. Empty if image is not from file
    image_format = example['image_format']  # b'jpeg' or b'png'

    # List of string class name of bounding box (1 per box)
    classes_text = [example['label'] for i in range(len(xmins))]
    #classes_text = ['arabic'.encode('utf8') for i in range(len(xmins))]

    # List of integer class id of bounding box (1 per box)
    classes = [example['label_num'] for i in range(len(xmins))]
    #classes = [1 for i in range(len(xmins))]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def main(activ_D_folder, program_data_folder):

    # Use some of the test files as training examples and reserve two test batches for evaluation
    modes = ["training", "test"]

    channels = ["AljazeeraHD", "Negative", "France24", "RussiyaAl-Yaum", "TunisiaNat1", "France24", "RussiyaAl-Yaum",
                "TunisiaNat1", "Generated"]
    training_files = ["gtraining_Aj.xml", "gtraining_Ne.xml", "gtraining_Fr.xml", "gtraining_Rt.xml", "gtraining_Tn.xml",
                      "gtest_Fr.xml", "gtest_Rt.xml", "gtest_Tn.xml", "gtraining_Ge.xml"]
    testing_files = ["gtest_Aj.xml","gtest_Ne.xml"]

    for mode in modes:

        # writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
        writer = tf.python_io.TFRecordWriter(join(program_data_folder, mode + ".tfrecord"))
        print("Processing {0} files".format(mode))
        if mode == "training":
            files = training_files
        else:
            files = testing_files

        examples = list()

        for channel, file in zip(channels, files):
            # print("Channel:",channel)
            path_to_image = join(activ_D_folder, channel, mode + "Files")
            path_to_xml = join(activ_D_folder, channel, file)

            # Skip to next entry if file doesn't exist
            if not isfile(path_to_xml):
                print("\t{0} was not located; skipping".format(path_to_xml))
                continue
            tree = etree.parse(path_to_xml)
            xml_contents = tree.getroot()

            counter = 0
            for frame in xml_contents:

                frame_attributes = dict()
                frame_attributes['source'] = frame.attrib['source']
                frame_attributes['frame_num'] = frame.attrib['id']
                
                # Negative sampling and Generated text have file extension in xml
                if 'ext' in frame.attrib:
                    frame_attributes['extension'] = frame.attrib['ext']
                else:
                    frame_attributes['extension'] = 'png'

                frame_attributes['item_id'] = list()
                frame_attributes['bbox_xmins'] = list()
                frame_attributes['bbox_xmaxs'] = list()
                frame_attributes['bbox_ymins'] = list()
                frame_attributes['bbox_ymaxs'] = list()

                for rectangle in frame:
                    frame_attributes['item_id'].append(rectangle.attrib['id'])
                    frame_attributes['bbox_xmins'].append(int(rectangle.attrib['x']))
                    frame_attributes['bbox_xmaxs'].append(
                        int(rectangle.attrib['x']) + int(rectangle.attrib['width']))
                    frame_attributes['bbox_ymins'].append(int(rectangle.attrib['y']))
                    frame_attributes['bbox_ymaxs'].append(
                        int(rectangle.attrib['y']) + int(rectangle.attrib['height']))
                
                #if frame_attributes['extension'] is not None and frame_attributes['extension']!='':
                #    frame_attributes['file_name'] = channel + "_" + frame_attributes['source'] + "_frame_" + \
                #                                    frame_attributes['frame_num'] + "." + frame_attributes['extension']
                #else:
                #    # File name format for ActiV is France24_vd01_frame_11.png
                #    frame_attributes['file_name'] = channel + "_" + frame_attributes['source'] + "_frame_" + \
                #                                    frame_attributes['frame_num'] + ".png"

                frame_attributes['file_name'] = channel + "_" + frame_attributes['source'] + "_frame_" + \
                                                    frame_attributes['frame_num'] + "." + frame_attributes['extension']
                if frame_attributes['extension'] in ['jpg','jpeg']:
                    frame_attributes['image_format'] = b'jpeg' 
                elif frame_attributes['extension'] in ['png']:
                    frame_attributes['image_format'] = b'png'    
                    
                frame_attributes['path_to_image'] = path_to_image

                # Add class label
                if channel == 'Negative':
                    frame_attributes['label'] = 'english'.encode('utf8')
                    frame_attributes['label_num'] = 2
                else:
                    frame_attributes['label'] = 'arabic'.encode('utf8')
                    frame_attributes['label_num'] = 1

                counter += 1
                #if counter == 1: print(frame_attributes)
                examples.append(frame_attributes)

            print("\tProcessed {0} frames for channel {1}".format(counter, channel))

        counter = 0
        len_examples = len(examples)
        for example in examples:
            tf_example = create_tf_example(example, mode)
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())
                counter += 1
            else:
                #print("Error with {0}; skipping".format(example['file_name']))
                continue

        print("Wrote {0} tfrecord entries from {1} examples to {2}".format(counter, len_examples,
                                                                           join(program_data_folder, mode + ".tfrecord")))
        writer.close()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess AcTiV dataset by removing ticker (--remove_ticker) and generating additional training \
                    images (--generate_data')

    # Optional arguments
    parser.add_argument(
        '--activ_D_folder',
        type=str,
        default="/arabic_text/AcTiV-D",
        help='Location of AcTiV dataset. Default = /arabic_data/AcTiV-D')

    parser.add_argument(
        '--program_data_folder',
        type=str,
        default="/prog/data",
        help='Location of scimitar repository data folder. Default = /prog/data')

    args = parser.parse_args()
    print(
        "Parameters set as: \n \
           AcTiV-D folder = {0} \n \
           Program Data folder = {1} \n"
        .format(
            args.activ_D_folder,
            args.program_data_folder))
    main(args.activ_D_folder, args.program_data_folder)
    #tf.app.run()
    print("Done")

