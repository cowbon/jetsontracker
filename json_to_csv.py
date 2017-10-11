import csv
import json
import glob


with open('data/train.csv','wb+') as csv_file:
    f = csv.writer(csv_file)
    f.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])

    for json_file in glob.glob('images/annotations'+'/*.json'):
        with open(json_file, 'r') as data_file:
            data = json.load(data_file)
            for member in data['objects']:
                xmax = member['x_y_w_h'][0] + member['x_y_w_h'][2]
                ymax = member['x_y_w_h'][1] + member['x_y_w_h'][3]
                f.writerow([data['filename'],
                            data['image_w_h'][0],
                            data['image_w_h'][1],
                            member['label'],
                            member['x_y_w_h'][0],
                            member['x_y_w_h'][1],
                            xmax,
                            ymax])
