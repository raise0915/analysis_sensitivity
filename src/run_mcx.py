
import os
import json
import subprocess
import icecream as ic

class Runmcx():
    def __init__(self, save=0) -> None:
        self.save = 1
        json_open = open("src/inputs/settings.json", "r")
        settings = json.load(json_open)
        self.MCX_PATH = settings.get('MCX_PATH')
        self.radius = settings.get('raidus')
        self.length = settings.get('length')
        self.input_name = settings.get('INPUT_PATH')
        self.cfg = json.load(open("src/inputs/input.json"))
        self.HOME_PATH = settings.get('HOME_PATH')
        self.OUTPUT_PATH = settings.get('OUTPUT_PATH')
        self.model_path = self.cfg['Domain']['VolumeFile']
   
    def set_conditions(self, position, rotation): # , rotation, opt_tumour, opt_normal):
        output_path = os.path.join(self.MCX_PATH, "test.json")
        self.cfg['Session']['ID'] =  self.OUTPUT_PATH
        self.cfg['Optode']['Source']['Type'] = 'gaussian'
        self.cfg['Optode']['Source']['Pos'] = [position[0], position[1], position[2]]
        self.cfg['Optode']['Source']['Param1'] = [self.radius, rotation[0], rotation[1], rotation[2]] # gaussianの回転方向はここで指定する
        self.cfg['Optode']['Source']['Param2'] = [self.length, 0, 0, 0]
        self.cfg['Optode']['Source']['Lambda'] = 664
        """
        self.cfg['Domain']['Media'] = [
        {
            "mua": 0,
            "mus": 0,
            "g": 1,
            "n": 1
        },
        {
        "mua": opt_tumour[0],
        "mus": opt_tumour[1],
        "g": opt_tumour[2],
        "n": opt_tumour[3]
        },
        {
        "mua": opt_normal[0],
        "mus": opt_normal[1],
        "g": opt_normal[2],
        "n": opt_normal[3]
        },
        {
            "mua": opt_tumour[0],
            "mus": opt_tumour[1],
            "g": opt_tumour[2],
            "n": opt_tumour[3]
        }
        ]
        """
        ic.ic(self.cfg)
        
        # save for run
        json_file1 = open(output_path, mode="w")
        json.dump(self.cfg, json_file1, indent=4)
        json_file1.close()

    def run_mcx(self, position, rotation): # , rotation, opt_tumour, opt_normal):
        self.set_conditions(position, rotation) # , rotation, opt_tumour, opt_normal)
        os.chdir(self.MCX_PATH) 
        subprocess.run("./bin/mcx -f test.json", shell=True)

