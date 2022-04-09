from . import rundir
from . import tools
from . import model_prof_tools
import os, shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u

wav_width = {'H':15, 'CaII':25, 'He':10}
del_wav = 20

class nicole(rundir.rundir):
    def __init__(self, model, atom, line_list):
        '''
        Initialize a nicole instance.
        '''
        super(nicole, self).__init__('{}/.pynicole/'.format(os.environ['HOME']), 'nicole')
        self.model = model
        self.atom = atom
        self.line_list = line_list
        self.file_path = '{}/.pynicole/files/'.format(os.environ['HOME'])
        
    def prepare_file(self):
        '''
        Prepare the necessary files for running NICOLE.
        '''
        
        # Save model file to the rundir_path
        tools.save_model(self.model, self.rundir_path + 'inmodel.model')
        
        # Copy the ATOM file to the current directly and read it in to nicole/
        shutil.copy(self.file_path + 'ATOM/atom.{}'.format(self.atom), self.rundir_path + 'ATOM')
        self.atom = tools.read_atom(self.rundir_path + 'ATOM')
        
        # The run may crash if NLIN+NCNT in atom > len(model); raise a warning if this is the case.
        if self.atom['NLIN']+self.atom['NCNT'] > len(self.model):
            print('NLIN({})+NCNT({}) in atom > len(model) ({}); the run may crash'.format(self.atom['NLIN'], self.atom['NCNT'], len(self.model)))
        
        # Read the LINES file 
        self.lines = tools.read_input_file(self.file_path + '/LINES', lines=True)
        
        # Modify Transition index in model atom
        for line_name in self.line_list:
            if tools.lines[line_name]['Excitation potential'][1].lower() in ['ev', '']:
                tar_ecm = (tools.lines[line_name]['Excitation potential'][0] * u.eV).to(u.cm**-1, equivalencies=u.spectral()).value
            elif tools.lines[line_name]['Excitation potential'][1].lower() == 'cm-1':
                tar_ecm = tools.lines[line_name]['Excitation potential'][0]
            else:
                raise ValueError('find_line only accept eV or cm-1 in excitation potential')

            tools.lines[line_name]['Transition index in model atom'] = tools.find_line(tar_ecm, self.atom)
        
        # Save LINES
        tools.save_line_file(tools.lines, file_path=self.rundir_path + 'LINES')
        
        # Create input dictionary.
        input_dict = tools.read_input_file(self.file_path + '/NICOLE.input')
        input_dict['section 1']['Command'] = self.file_path + '/NICOLE/main/nicole'
        input_dict['section 1']['Printout detail'] = 10
        input_dict['section 1']['Gravity'] = 10**4.437

        for i in range(len(self.line_list)):
            input_dict['Line {}'.format(i+1)] = {'Line': self.line_list[i]}

        for i in range(len(self.line_list)):
            input_dict['Region {}'.format(i+1)] = {'First wavelength': tools.lines[self.line_list[i]]['Wavelength'] - wav_width[self.line_list[i].split()[0]], 'Wavelength step': [del_wav, 'mA'], 'Number of wavelengths': wav_width[self.line_list[i].split()[0]]*2 / (del_wav/1000)}
        
        self.input_dict = input_dict
        
    def run_nicole(self, output=False, dry_run=False):
        '''
        Do the other file manupulations and run NICOLE.
        '''
        
        # Save the input file 
        tools.save_input_file(self.input_dict, file_path=self.rundir_path + '/NICOLE.input')
        
        # Copy the main control python file to working directory
        cp_status = subprocess.run(['cp', self.file_path + '/run_nicole.py', self.rundir_path], encoding='UTF-8', stdout=subprocess.PIPE)
        cp_status = subprocess.run(['cp', self.file_path + '/model_prof_tools.py', self.rundir_path], encoding='UTF-8', stdout=subprocess.PIPE)
        
        if not(dry_run):
            # Run NICOLE
            NICOLE_run = subprocess.run(['./run_nicole.py', '>', 'run.log'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=self.rundir_path)
            NICOLE_run_out = str(NICOLE_run.stdout, encoding = "utf-8").split('\n')
            self.output = NICOLE_run_out

            # Handling the printout
            if output == True:
                for i in NICOLE_run_out:
                    print(i)
            elif type(output) == str:
                # Output the stdout to a file.
                NICOLE_run_out = ['{}\n'.format(i) for i in NICOLE_run_out]
                with open(self.rundir_path + '/' + output, 'w') as file:
                    file.writelines(NICOLE_run_out)
                    
    def read_prof(self, profile_name='', remove=True, outmodel=True, pop=True):
        '''
        Read the profile. 
        The profile is assumed to be in the same directly with the input file.
        
        outmodel : bool
            If True, will also read the output model
        '''

        input_content = tools.read_input_file(self.rundir_path + 'NICOLE.input')
        wav_dict = tools.get_wav_array(input_content)

        if profile_name == '':
            profile_file = '/'.join([self.rundir_path, input_content['section 1']['Output profiles']])
        else:
            profile_file = '/'.join([self.rundir_path, profile_name])

        prof_paras = model_prof_tools.check_prof(profile_file)
        flux = model_prof_tools.read_prof(profile_file, *prof_paras, 0, 0)
        flux = np.array(flux).reshape(-1, 4)

        spec_dict = {}
        len_record = 0
        for region in wav_dict.keys():
            spec_array = np.concatenate([wav_dict[region].reshape(-1, 1), flux[len_record:len_record+len(wav_dict[region]), :]], axis=1)
            len_record += len(wav_dict[region])
            spec_dict[region] = pd.DataFrame(spec_array, columns=['wavelength', 'I', 'Q', 'U', 'V'])
        self.spec_dict = spec_dict
        
        if outmodel:
            outmodel_name = input_content['section 1']['Output model']
            outmodel_path = '/'.join([self.rundir_path, outmodel_name])
            self.outmodel = tools.read_model(outmodel_path)['model']
        
        if pop:
            # outmodel must be true if pop is true.
            pop_path = '/'.join([self.rundir_path, 'Populations.dat'])
            self.pop = tools.read_pop(pop_path, self.outmodel)
   
        if remove:
            self.remove()
                   
