import os
import pandas as pd
import xml.etree.ElementTree as ET
# from pandas_profiling import ProfileReport

WIDTH_KEY='width'
HEIGHT_KEY='height'
CLASS_NAME_KEY='class_name'
XMIN_KEY='xmin'
YMIN_KEY='ymin'
XMAX_KEY='xmax'
YMAX_KEY='ymax'
XML_PATH_KEY = 'xml_path'
HEADERS = [XML_PATH_KEY, WIDTH_KEY, HEIGHT_KEY, CLASS_NAME_KEY, YMIN_KEY, XMIN_KEY, YMAX_KEY, XMAX_KEY]

def get_xml_tags(xml_dir):
    df_data = []
    for root,dirs,files in os.walk(xml_dir):
        for f in files:
            if f.startswith('.'):
                continue
            xml_path = os.path.join(root,f)
            try:
                xml_obj = open(xml_path)
                doc = ET.parse(xml_obj)
            except:
                print("parsing xml file: %s failed!"%xml_path)
                continue

            width = int(doc.findtext('size/width'))
            height = int(doc.findtext('size/height'))
            for item in doc.iterfind('object'):
                class_name = item.findtext('name')
                xmin = int(item.findtext('bndbox/xmin'))
                ymin = int(item.findtext('bndbox/ymin'))
                xmax = int(item.findtext('bndbox/xmax'))
                ymax = int(item.findtext('bndbox/ymax'))
                df_data.append([xml_path,width,height,class_name,ymin,xmin,ymax,xmax])
    df = pd.DataFrame(data=df_data, columns=HEADERS)
    return df


if __name__ == '__main__':
    xml_dir = r"Annotation"
    df= get_xml_tags(xml_dir)
    print('finish')





