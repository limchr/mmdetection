import os

base_cmd = """

convert \
\( -size 1x250 gradient:mc1-mc2 \) \
\( -size 1x250 gradient:mc2-mc1 \) \
-append -scale 100x500! -rotate 90 xc:  -stroke cs  -fill cf  -strokewidth 5 -draw 'circle p1,50 p2,50' \
outputname.png



"""

class_config = {
    'meta1': {'meta_colors':['blue','red'], 'class_colors':[['green','green'],['yellow','yellow']], 'circle_position': 0.25},
    'meta2': {'meta_colors':['orange','pink'], 'class_colors':[['magenta','magenta'],['cyan','cyan']], 'circle_position': 0.75},
}

for mc in class_config:
    for cls_i, col in enumerate(class_config[mc]['class_colors']):
        cmd = base_cmd.replace('mc1',class_config[mc]['meta_colors'][0])
        cmd = cmd.replace('mc2',class_config[mc]['meta_colors'][1])
        pos = int(class_config[mc]['circle_position'] * 500)
        p1, p2 = pos-5,pos+5
        cmd = cmd.replace('p1',str(p1))
        cmd = cmd.replace('p2',str(p2))
        
        cmd = cmd.replace('cs',col[0])
        cmd = cmd.replace('cf',col[1])

        cmd = cmd.replace('outputname',mc+'_'+str(cls_i))
        print(cmd)

        os.system(cmd)