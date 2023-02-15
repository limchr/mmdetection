import os

ds_path = '.'


ds = ['Train','Test']

ds_root = '/home/chris/data/okutama_action/'

label_path = os.path.join(ds_root,'[ds]SetFrames/Labels/MultiActionLabels/3840x2160/')
img_paths = os.path.join(ds_root,'[ds]SetFrames/*/*/Extracted-Frames-1280x720/*')

### vanilla the plain okutama action dataset with default classes
out_path = os.path.join(ds_root,'converted_vanilla')
ds_classes = {}

# # ### meta2 for 2 meta classes
# out_path = os.path.join(ds_root,'converted_meta2')
# ds_classes = {
#     'Calling': 'Activity',
#     'Carrying': 'Activity',
#     'Drinking': 'Activity',
#     'Hand': 'Activity',
#     'Hugging': 'Activity',
#     'Lying': 'Activity',
#     'Pushing/Pulling': 'Activity',
#     'Reading': 'Activity',
#     'Running': 'Movement',
#     'Shaking': 'Activity',
#     'Sitting': 'Activity',
#     'Standing': 'Movement',
#     'Walking': 'Movement',
# }

# # ### meta3 for 3 meta classes
# out_path = os.path.join(ds_root,'converted_meta3')
# ds_classes = {
#     'Calling': 'Activity',
#     'Carrying': 'Activity',
#     'Drinking': 'Activity',
#     'Hand': 'Activity',
#     'Hugging': 'Activity',
#     'Lying': 'Resting',
#     'Pushing/Pulling': 'Activity',
#     'Reading': 'Activity',
#     'Running': 'Movement',
#     'Shaking': 'Activity',
#     'Sitting': 'Resting',
#     'Standing': 'Resting',
#     'Walking': 'Movement',
# }


# ### meta1 for some meta classes
# out_path = os.path.join(ds_root,'converted_meta1')
# ds_classes = {
#     'Lying': None, 
#     'Sitting': None,
#     'Walking': 'Moving', 
#     'Running': 'Moving', 
#     'Calling': 'Activity',
#     'Carrying': 'Activity',
#     'Drinking': 'Activity',
#     'Hugging': 'Activity',    
# }

# ### min for testing
# out_path = os.path.join(ds_root,'converted_min')
# ds_classes = {
#     'Lying': None, 
#     'Sitting': None,
# }




import os
import shutil
import glob
import numpy as np
import src.helper as h
import re



def clean_labels(l):
    '''remove the unneccessary labels person and empty list from label array'''
    l = list(l)
    while '' in l:
        l.remove('')
    if 'Person' in l:
        l.remove('Person')
    if len(l) == 0:
        l.append('Standing') # TODO: this workaround might be inaccurate for some examples
    return l


def extract_label_file(pathf):
    with open(pathf,'rt') as f:
        lfl = f.read().splitlines()
        data = np.zeros([len(lfl),9], dtype=int)
        labels = np.zeros([len(lfl),3], dtype=object)
        for i,l in enumerate(lfl):
            v = l.split(' ')
            v[:9] = [int(vv) for vv in v[:9]]
            if len(v) <= 11:
                v = v + ['']
            v[9:] = v[9].replace('"',''), v[10].replace('"',''), v[11].replace('"','')
            data[i,:] = v[:9]
            labels[i,:] = v[9:]
    
    # clear 'lost' bboxes that are outside of image area
    valid_i = data[:,6] == 0
    data = data[valid_i]
    labels = labels[valid_i]

    return data, labels


json_dict = {'images': [], 'annotations': [], 'categories': []}

h.setup_clean_directory(out_path)

h.create_directory_if_not_defined(os.path.join(out_path,'images'))
h.create_directory_if_not_defined(os.path.join(out_path,'labels'))

label_files = []
im_dirs = []

labeldl = [] # list with all label files
imdl = [] # list with all image directories

# Extract all the label and image directory files (in labeldl and imdl)

for d in ds:
    lp = label_path.replace('[ds]',d)
    ip = img_paths.replace('[ds]',d)
    label_files.append(glob.glob(os.path.join(ds_path,lp,'*.txt')))
    im_dirs.append(glob.glob(os.path.join(ds_path,ip)))

    labeld = {}
    imd = {}

    for f in label_files[-1]:
        if os.path.basename(f)[:-4] in labeld.keys():
            exit("ERROR IN KEYS")
        labeld[os.path.basename(f)[:-4]] = {'label_path':f}
    for d in im_dirs[-1]:
        if os.path.basename(d) in imd.keys():
            exit("ERROR IN KEYS")
        imd[os.path.basename(d)] = {'im_path':d}

    assert len(imd) == len(labeld)

    labeldl.append(labeld)
    imdl.append(imd)




#
# Generate list of possible classes
#
# All 13 classes are 
# ['Calling', 'Carrying', 'Drinking', 'Hand', 'Hugging', 'Lying', 
# 'Pushing/Pulling', 'Reading', 'Running', 'Shaking', 'Sitting', 'Standing', 'Walking']
# Dict ds_classes defines the samples to extract (only images containing defined class) 
# and meta labels consolidating several native classes
# extract only images/labelfiles containing these classes (empty dict for all classes, value None for no meta class)


labeldump = []
train_index = ds.index('Train')
for lf in labeldl[train_index]:
    data, labels = extract_label_file(labeldl[train_index][lf]['label_path'])
    labeldump.extend(np.unique(labels.flatten()))
    print('read '+lf)

ds_classlist = np.unique(labeldump)
ds_classlist = clean_labels(ds_classlist)

# make sure that all classes are in the data set
for dc in ds_classes.keys():
    assert(dc in ds_classlist)
    
# if no class selection or meta classes are defined, just use all classes with no meta classes
if len(ds_classes) == 0:
    for cl in ds_classlist:
        ds_classes[cl] = None

# class list containing native and meta labels
class_list = list(np.unique([ds_classes[k] if ds_classes[k] else k for k in ds_classes]))


#
# Write .yaml file for giving information about the dataset
#
with open(os.path.join(out_path,'okutama_action.yaml'),'wt') as f:
    f.write('#download: bash ./scripts/get_okutama_action.sh\n')
    
    f.write('train: '+os.path.join(out_path,'train.txt\n'))
    f.write('test: '+os.path.join(out_path,'test.txt\n'))
    f.write('val: '+os.path.join(out_path,'test.txt\n'))
    
    f.write('nc: '+str(len(class_list))+'\n')    

    f.write('names: ['+(','.join(class_list))+']\n')

#
# filling json dict with class information
#
for cli, cl in enumerate(class_list):
    json_dict['categories'].append({'id': cli, 'name':cl})    

#
# Copy images to image directory, write file with list of images (e.g. train.txt)
# and create a file for each image containing all labels (e.g. 1.2.2.233.txt)
#
for di, dd in enumerate(ds):
    with open(os.path.join(out_path,dd.lower()+'.txt'),'wt') as img_list:
        for lf in labeldl[di]:
            data, labels = extract_label_file(labeldl[di][lf]['label_path'])
            print('read '+lf)
            for imi in np.unique(data[:,5]): # iterate images frames (imi image index)
                data_i = data[data[:,5] == imi]
                labels_i = labels[data[:,5] == imi]

                # find out which labels are present in image
                img_labels = clean_labels(np.unique(labels_i.flatten())) 
                
                copy_sample = np.array([cc in img_labels for cc in ds_classes.keys()]).any()
                
                if not copy_sample:
                    continue
                
                # generate path to image
                imp = os.path.join(imdl[di][lf]['im_path'], str(imi) + '.jpg')

                # copy image to output directory
                slashes = [m.start() for m in re.finditer('/', imp)]
                new_name = imp[slashes[-5]+1:slashes[-4]] + '_' + imp[slashes[-4]+1:slashes[-3]] + '_' + imp[slashes[-2]+1:-4].replace('/','_frame_')

                img_file_name = new_name+'.jpg'
                copied_path = os.path.join(out_path,'images',img_file_name)




                # write label file
                
                img_labels = []
                for d,l in zip(data_i,labels_i):
                    obj_label = clean_labels(l)[0] # todo: maybe also above use ONLY primary labels!
                    
                    # check weather the label is wanted for export
                    if obj_label in ds_classes.keys():
                        # if there is a meta label, use this instead
                        if not ds_classes[obj_label] is None:
                            obj_label = ds_classes[obj_label]
                        img_labels.append([class_list.index(obj_label)] + list(d[1:]))

                # if there is any label associated with the image (of the relevant classes),
                # then write the image path and labels in the files and copy the actual image to
                # the output directory, too                       
                if len(img_labels) > 0:
                    # copy image file to export directory
                    shutil.copy2(imp, copied_path)

                    # write image to train/test file
                    img_list.write('./images/'+new_name+'.jpg\n')
                    # write the label file
                    with open(os.path.join(out_path,'labels',new_name+'.txt'),'wt') as label_txt:
                        for il in img_labels:
                            # coco dataset standard requires class index, normalized x mean, y mean and w and h [newline]
                            xm = (il[1]+il[3])/2/3840
                            ym = (il[2]+il[4])/2/2160
                            w = (il[3]-il[1])/3840
                            h = (il[4]-il[2])/2160

                            label_txt.write(' '.join([str(il[0]), str(xm),str(ym),str(w),str(h)])+'\n')
                    
                    def convertToNumber (s):
                        return int.from_bytes(s.encode(), 'little')
                    import hashlib

                    # write the json file with the information
                    img_id = convertToNumber(hashlib.sha256(str(new_name).encode()).hexdigest()[:12])
                    json_dict['images'].append({'id': img_id,'file_name': img_file_name, 'width': 1280, 'height': 720})
                    for ili, il in enumerate(img_labels):
                        w_mult = 1280.0/3840.0
                        h_mult = 720.0/2160.0
                        assert(il[3]>il[1])
                        assert(il[4]>il[2])
                        json_dict['annotations'].append(
                            {'id': img_id+ili, 
                             'image_id': img_id, 
                             'category_id': il[0], 
                             'bbox': [il[1]*w_mult,il[2]*h_mult,(il[3]-il[1])*w_mult,(il[4]-il[2])*h_mult],
                             'area': abs(il[1]*w_mult-il[3]*w_mult)*abs(il[2]*h_mult-il[4]*h_mult),
                             'iscrowd': False,
                             'ignore': False,
                             'segmentation': None
                             })
    import json
    # write final json file for the dataset
    with open(os.path.join(out_path, dd.lower()+'.json'), 'wt') as f:
        json.dump(json_dict, f, indent=4, separators=(", ", ": "))
        
        
        
# # check for duplicate ids

# asdf = [an['id'] for an in json_dict['annotations']]
# l2 = []

# for a in asdf:
#     if a in l2:
#         print('duplicate id '+str(a))
#     l2.append(a)