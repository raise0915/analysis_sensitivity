import glob
import icecream as ic
import os

files = glob.glob('results/pos_1.0_rot_5.0/*.xlsx')

for file in files:
    file_name = file.split('/')[-1]
    file_name = file_name.replace('_1227', '').replace('_0108', '')
    file_name = file_name.replace('pos_1', 'pos_1.0').replace('rot_5', 'rot_5.0')
    new_file = os.path.join(os.path.dirname(file), file_name)
    os.rename(file, new_file)
    ic.ic(file_name)
