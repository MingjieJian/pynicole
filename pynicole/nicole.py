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
                   
def plot_nicole_model(model_df_list, label_list=[], mag_field=False, t_range=[], density_name='Pe', x='log_tau5000'):
    '''
    Plot the nicole format model(s).
    '''
    label_swith = True
    if len(label_list) != 0 and len(label_list) != len(model_df_list):
        raise ValueError('The length of label_list must be the same as that of model_df_list.')
    if len(label_list) == 0:
        label_list = [''] * len(model_df_list)
        label_swith = False
    
    if mag_field:
        fig, axs = plt.subplots(4, 2, figsize=(10, 10), dpi=150)
    else:
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), dpi=150)
    axs = axs.flatten()
    
    count = 0
    for model in model_df_list:
        axs[0].plot(model[x], model['T'], label=label_list[count], marker='.', markersize=5)
        axs[1].plot(model[x], np.log10(model[density_name]), marker='.', markersize=5)
        axs[2].plot(model[x], model['vmic'] / 1e5, marker='.', markersize=5)
        axs[3].plot(model[x], model['Vlos'] / 1e5, marker='.', markersize=5)
        if mag_field:
            try:
                axs[4].plot(model[x], model['b_long'])
                axs[5].plot(model[x], model['b_x'])
                axs[6].plot(model[x], model['b_y'])
            except:
                pass
        count += 1
        
    
    ylabel_list = ['T (K)', 
                   {'Pe':r'$\log{[P_\mathrm{e} (\mathrm{dyn/cm^2})]}$', 
                    'Ne':r'$\log{[N_\mathrm{e} (\mathrm{cm^{-3}})]}$',
                    'Pgas':r'$\log{[P_\mathrm{gas} (\mathrm{dyn/cm^2})]}$',
                    'rho':r'$\log{[\rho (\mathrm{g/cm^3})]}$'}, 
                   '$V_\mathrm{mic}$ (km/s)', '$V_\mathrm{los}$ (km/s)']
    if mag_field:
        ylabel_list += ['$B_\mathrm{long}$', '$B_\mathrm{y}$', '$B_\mathrm{y}$']
    count = 0
    for ylabel in ylabel_list:
        if x.lower() == 'log_tau5000':
            axs[count].set_xlabel(r'$\log{\tau_{5000}}$')
        elif x.lower() == 'z':
            axs[count].set_xlabel('Z')
        if type(ylabel) == str:
            axs[count].set_ylabel(ylabel)
        else:
            axs[count].set_ylabel(ylabel[density_name])
        count += 1 
    
    if len(t_range) == 2:
        axs[0].set_ylim(t_range)
    
    if label_swith:
        axs[0].legend()
    plt.tight_layout()
    
    return axs
    
def interpolate_model(model_old, N_new_grid):
    '''
    Interpolate the model to the given number in equidistance.
    '''
    
    log_interp = ['Pe', 'Pgas', 'Ne']

    reverse_switch = False

    if model_old.iloc[0]['log_tau5000'] > model_old.iloc[1]['log_tau5000']:
        # Assuming the x is decreasing.
        model_inter = model_old.loc[::-1].reset_index(drop=True)
        reverse_switch = True
    else:
        model_inter = model_old

    x_new_grid = np.linspace(model_inter.iloc[0][model_inter.columns[0]], model_inter.iloc[-1][model_inter.columns[0]], N_new_grid)
    model_new_grid = pd.DataFrame({model_inter.columns[0]:x_new_grid})
    # model_inter
    for column in model_old.columns[1:]:
        if column in log_interp:
            model_new_grid[column] = 10**np.interp(x_new_grid, model_inter[model_inter.columns[0]].values, np.log10(model_inter[column].values))
        else:
            model_new_grid[column] = np.interp(x_new_grid, model_inter[model_inter.columns[0]].values, model_inter[column].values)

    if reverse_switch:
        model_new_grid = model_new_grid.loc[::-1].reset_index(drop=True)
        
    return model_new_grid