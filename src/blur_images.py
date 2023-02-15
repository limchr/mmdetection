import sys
import os
import glob

size = 75
#path = sys.argv[1]
path = '/home/chris/data/okutama_action/converted_vanilla'
out_path = path+'_'+str(size)
target_size = '1280x720'

os.system("cp -r "+path+" "+out_path)


imgs = glob.glob(os.path.join(out_path,'images','*.jpg'))

inv_size = int(100/(size/100.0))

for i in imgs:
    os.system('convert '+i+' -size '+str(size)+'% -size '+target_size+' '+i)
    print(i)