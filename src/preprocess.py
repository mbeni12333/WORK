import argparse
import numpy as np
import pandas as pd
import cv2
import os
import wsiprocess as wp
import glob
import gc
import shutil


def get_tmp_annotation():
    """
    """
    
    xml = "<?xml version=\"1.0\"?>\
            <ASAP_Annotations>\
                <AnnotationGroups>\
                    <Group Name=\"_0\">\
                        <Attributes />\
                    </Group>\
                    <Group Name=\"_1\" >\
                        <Attributes />\
                    </Group>\
                </AnnotationGroups>\
            </ASAP_Annotations>"

    with open("tmp.xml", "w") as file:
        file.write(xml)
    
    return wp.annotation("tmp.xml")


def make_tissue(filepath, xmlfilepath=None, size=1000):

    slide = wp.slide(filepath)
    
    thumb = np.array(slide.get_thumbnail(size))

    print("segmenting the tissue ...")
    thumb_hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    segmented = cv2.pyrMeanShiftFiltering(thumb_hsv, 10, 50)

    notcross= segmented[:, :, 2] >= 60
    notbackground= segmented[:, :, 1] >= 30 # 
    mask = notbackground*notcross
    
    h, w = slide.height, slide.width

    print("Reading annotations ...")
    if((xmlfilepath is not None) and (os.path.exists(xmlfilepath))):
        
        annotation = wp.annotation(xmlfilepath)
        annotation.make_masks(slide, size=size, foreground=False)

        print("Making masks ...")

        if not os.path.exists("tmp"):
            os.mkdir("tmp")
            
        annotation.export_thumb_masks(f"tmp", size=size)

        tumor_mask = 0
        for key in annotation.masks.keys():
            if key in ["_0", "_1"]:
                tumor_mask |= cv2.imread(f"tmp/{key}_thumb.png").max(2)
        tumor_mask = (tumor_mask>0)*1
        
        
        annotation.masks["foreground"] = cv2.resize(mask.astype(np.uint8), [w, h], interpolation=cv2.INTER_NEAREST)
        annotation.masks["tumor"] = cv2.resize(tumor_mask.astype(np.uint8), [w, h], interpolation=cv2.INTER_NEAREST)
        annotation.masks["normal"] = annotation.masks["foreground"] - annotation.masks["tumor"]

        shutil.rmtree('tmp')
        
    else:
        annotation = get_tmp_annotation()
        annotation.masks["normal"] = cv2.resize(mask.astype(np.uint8), [w, h], interpolation=cv2.INTER_NEAREST)
        annotation.masks["foreground"] = annotation.masks["normal"]
        annotation.masks["tumor"] = cv2.resize((mask*0).astype(np.uint8), [w, h], interpolation=cv2.INTER_NEAREST)
    
    annotation.dot_bbox_height = 1
    annotation.dot_bbox_width = 1
    
    
    return slide, annotation



def process_single(image,
                   annotation=None,
                   size=2000,
                   patchWidth=224,
                   patchHeight=224,
                   onForeground=0.3,
                   onAnnotation=0.05,
                   saveTo="processed"):
    """
    """
    
    print("Creating the annotations ...")
    
    slide, annotation = make_tissue(image,
                                    annotation,
                                    size)
    patcher = wp.patcher(slide,
                         "classification",
                         annotation,
                         patch_width=patchWidth,
                         patch_height=patchHeight,
                         on_foreground=onForeground,
                         on_annotation=onAnnotation,
                         save_to=saveTo,
                         verbose=True,
                         overlap_height=20,
                         overlap_width=20)
    
    slidename = os.path.basename(image).split(".")[0]
    annotation.export_thumb_masks(f"{saveTo}/{slidename}", size=size)
    slide.export_thumbnail(f"{saveTo}/{slidename}/slide.jpg", size=size)
    patcher.get_patch_parallel(['tumor', 'normal'])
    
def process_dataset(dataset,
                    size=2000,
                    patchWidth=224,
                    patchHeight=224,
                    onForeground=0.3,
                    onAnnotation=0.05,
                    saveTo="processed",
                    skip=0):
    """
    """
    for file in sorted(glob.glob(f"{dataset}/*.tif"))[skip:]:
        xml = os.path.basename(file).split(".")[0] + ".xml"
        if(not os.path.isfile(xml)):
            xml = None
        process_single(file,
                       xml,
                       size=size,
                       patchHeight=patchHeight,
                       patchWidth=patchWidth,
                       onAnnotation=onAnnotation,
                       onForeground=onForeground)

    dataset_path=saveTo
    dfs= []

    for filepath in glob.glob(dataset_path+"/*/*.json"):
        with open(filepath, "r") as file:
            
            data = json.load(file)
            df = pd.DataFrame(data["result"])
            slide = os.path.basename(data["slide"]).split(".")[0]
            df["local_class"] = df["class"] == "tumor"
            df["global_class"] = slide.split("_")[0] == "tumor"
            df["path"] = os.path.join(os.path.dirname(filepath), "patches") + "/" + df["class"] + "/" + df["x"].astype(str).str.zfill(6)+"_"+df["y"].astype(str).str.zfill(6)+".png"
            df = df.drop(columns=["w", "h", "class"])
            dfs.append(df)

    dataset = pd.concat(dfs)
    dataset.set_index(["path"])
    dataset.to_csv(dataset_path+"data.csv")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess Camelyon dataset into fixed sized patches")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-d", "--dataset", type=str, help="The location of the dataset's root folder, should be organised in 2 folders one for normal slides, and the second for tumor slides")
    group.add_argument("-i", "--image", type=str, help="Slide to process")

    parser.add_argument("-a", "--annotation", type=str,
                        default=None, help="Slide's annotations to process")

    parser.add_argument("-pp", "--processedPath",
                        default="dataset/processed", type=str, help="The destination for the processed patches")

    parser.add_argument("-h", "--patchWidth",
                        default=224, type=int, help="The width of the generated patch")

    parser.add_argument("-w", "--patchHeight",
                        default=224, type=int, help="The width of the generated patch")

    parser.add_argument("-r", "--physicalSpaceRadius",
                        default=30, type=int, help="Physical Space Radius for the Mean Shift segmentation")

    parser.add_argument("-c", "--ColorSpaceRadius",
                        default=30, type=int, help="The radius of the drift color space for the Mean Shift segmentation")

    parser.add_argument("-s", "--saturationThreashold",
                        default=20, type=int, help="Saturation (HSV colorspace) threshold on the segmented image in order to remove background")

    parser.add_argument("-v", "--valueThreshold",
                        default=40, type=int,  help="Value (HSV colorspace) threshold on the segmented image in order to remove markers on the slide")
    
    parser.add_argument("-t", "--thumbSize",
                        default=2000, type=int,  help="Thumb size that will be used to extract tissue")
        
    parser.add_argument("-f", "--onForeground",
                        default=0.3, type=float,  help="Percentage of overlap between foreground mask and patch to accept a patch as a foreground")
            
    parser.add_argument("-A", "--onAnnotation",
                        default=0.1, type=float,  help="Percentage of overlap between annotation mask and patch to accept a patch as a annotation")
    
    parser.add_argument("-S", "--skip",
                        default=0, type=int,  help="skip already processed images")

    args = parser.parse_args()
    
    if(args.dataset is None):
        # process a single image
        process_single(args.image,
                       args.annotation,
                       args.thumbSize,
                       args.patchWidth,
                       args.patchHeight,
                       args.onForeground,
                       args.onAnnotation,
                       args.processedPath)
    else:
        process_dataset(args.dataset,
                        args.thumbSize,
                        args.patchWidth,
                        args.patchHeight,
                        args.onForeground,
                        args.onAnnotation,
                        args.processedPath,
                        args.skip)