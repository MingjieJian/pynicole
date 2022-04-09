from . import model_prof_tools
import re
import numpy as np
import pandas as pd
from astropy import constants
from astropy import units as u
from os import path
import subprocess
import rulerwd
import pymoog
import matplotlib.pyplot as plt

para_type_dict = {
    'command': 'str',
    'cycles': 'int',
    'start cycle': 'int',
    'mode': 'str', 
    'input model': 'str', 
    'input model 2': 'str', 
    'output profiles': 'str', 
    'heliocentric angle': 'float', 
    'observed profiles': 'str',
    'restart': 'int',
    'output model': 'str',
    'output model 2': 'str',
    'stray light file': 'str',
    'formal solution method': 'int', 
    'printout detail': 'int', 
    'noise': 'float',
    'acceptable chi-square': 'int',
    'maximum number of inversions': 'int',
    'maximum inversion iterations': 'int',
    'always compute derivatives': 'str',
    'centered derivatives': 'int',
    'gravity': 'float',
    'regularization': 'float',
    'update opacities every': 'int',
    'negligible opacity': 'float',
    'continuum reference': 'int', 
    'continuum value': 'float',
    'impose hydrostatic equilibrium': 'str', 
    'input density': 'str', 
    'keep parameter': 'int',
    'eq of state': 'str',
    'eq of state for h': 'str',
    'pe consistency': 'float',
    'opacity package': 'str',
    'opacity package uv': 'str',
    'start x position': 'int',
    'start irec': 'int',
    'debug mode': 'int', 
    'height scale': 'str', 
    'optimize grid': 'int', 
    'output populations': 'int', 
    'output nlte source function': 'int', 
    'output continuum opacity': 'int', 
    'elim': 'float',
    'isum': 'int',
    'istart': 'int',
    'cper': 'flaot',
    'use collisional switching': 'str',
    'nmu': 'int',
    'qnorm': 'float',
    'formal solution': 'int',
    'linear formal solution': 'int',
    'optically thin': 'float',
    'optically thick': 'float',
    'vel free': 'str',
    'ngacc': 'str',
    'max iters': 'int',
    'lambda iterations': 'int',
    'ltepop': 'str',
    'elements to ignore in backgroudn opacities': 'str',
    'abundance set': 'str',
    'abundance file': 'str',
    
    'first wavelength': 'float',
    'wavelength step': 'float',
    'number of wavelengths': 'int',
    'macroturbulent enhancement': 'int',
    
    'line': 'str',
    
    'temperature': ['int', 'str'], 
    't': ['int', 'str'], 
    'velocity': ['int', 'str'], 
    'microturbulence': ['int', 'str'], 
    'macroturbulence': ['int', 'str'], 
    'bz': ['int', 'str'], 
    'bx': ['int', 'str'], 
    'by': ['int', 'str'], 
    'stray light': ['int', 'str'], 
    'filling factor': ['int', 'str'], 
    'abundances': ['int', 'str'], 
    'sx': ['int', 'str'], 
    'sy': ['int', 'str']
}

line_type_dict = {
    'element':'str',
    'ionization stage':'int',
    'wavelength':'float',
    'excitation potential':['float', 'str'],
    'log(gf)':'float',
    'term (lower)':'str',
    'term (upper)':'str',
    'collisions':['str', 'int'],
    'damping sigma':'float',
    'damping alpha':'float',
    'gamma radiative':'float',
    'gamma stark':'float',
    'gamma van der waals':'float',
    'damping enhancement':'float',
    'width':'float',
    'mode':'str',
    'transition index in model atom': 'int',
    'lower level population ratio': 'float',
    'upper level population ratio': 'float',
    'hyperfine structure': 'str',
    'hyperfine alow': 'float',
    'hyperfine blow': 'float',
    'hyperfine aup': 'float',
    'hyperfine bup': 'float',
    'nuclear spin': 'float'
}

def input2dict(input_string):
    
    # Remove all characters after '#'
    mid_string = input_string.split('#')[0]
    mid_list = [i.strip() for i in mid_string.split('=')]
    try: 
        mid_list[1] = float(mid_list[1])
    except:
        pass
    return mid_list

def read_input_file(filename, lines=False):
    '''
    Read the NICOLE input file to dict. This function can also read LINES file.
    '''

    with open(filename, 'r') as file:
        fdata = file.readlines()
    # Remove all comment lines 
    fdata = [i for i in fdata if i[0] != '#']
    # Remove line break
    fdata = [i.replace('\n', '') for i in fdata]
    # Remove "  [[Override]]" line
    fdata = [i for i in fdata if 'override' not in i.lower()]
    
    # Find the region part of the input
    input_content = {'section 1':{}}
    input_section = 'section 1'
    for line in fdata:
        if line == '':
            continue
        if line[0] != '[':
            paras = input2dict(line)
            if paras[0].lower() in ['wavelength step', 'excitation potential'] and type(paras[1]) == str:
                paras[1] = paras[1].split(' ')
                paras[1][0] = float(paras[1][0])
            input_content[input_section][paras[0]] = paras[1]
        else:
            input_section = line[1:-1]
            input_content[input_section] = {}
    
    if lines:
        del input_content['section 1']
    
    return input_content

def get_wav_array(input_content):
    '''
    Get the wavelength array from NICOLE input dict.
    '''
    wav_dict = {}
    for region in [i for i in input_content.keys() if 'Region' in i]:
        
        # Judge which unit the wavelength is in.
        if input_content[region]['Wavelength step'][1] == 'mA':
            unit_division = 1000
        elif input_content[region]['Wavelength step'][1] == 'A':
            unit_division = 1
        wav_dict[region] = np.linspace(input_content[region]['First wavelength'], 
                                       input_content[region]['First wavelength']+input_content[region]['Number of wavelengths']*input_content[region]['Wavelength step'][0]/unit_division, int(input_content[region]['Number of wavelengths']))
        
    return wav_dict

def save_input_file(input_dict, file_path='temp.input'):
    '''
    Create NICOLE.input file.
    '''
    
    content = []
    for section in input_dict.keys():
        if section.lower() == 'section 1':
            for name in input_dict[section].keys():
                out_val = input_dict[section][name]
                if para_type_dict[name.lower()] == 'float' and out_val < 1e-3 and out_val != 0:
                    content.append('{} = {:.2e}\n'.format(name, eval(para_type_dict[name.lower()])(out_val)))
                else:
                    content.append('{} = {}\n'.format(name, eval(para_type_dict[name.lower()])(out_val)))
        elif 'nlte' in section.lower():
            content.append('[{}]\n'.format(section))
            for name in input_dict[section].keys():
                out_val = input_dict[section][name]
                if para_type_dict[name.lower()] == 'float' and out_val < 1e-3 and out_val != 0:
                    content.append('  {} = {:.2e}\n'.format(name, eval(para_type_dict[name.lower()])(out_val)))
                else:
                    content.append('  {} = {}\n'.format(name, eval(para_type_dict[name.lower()])(out_val)))
        elif 'region' in section.lower():
            content.append('[{}]\n'.format(section))
            for name in input_dict[section].keys():
                out_val = input_dict[section][name]
                if name.lower() == 'wavelength step':
                    content.append('  {} = {} {}\n'.format(name, eval(para_type_dict[name.lower()])(out_val[0]), out_val[1]))
                else:
                    if para_type_dict[name.lower()] == 'float' and out_val < 1e-3:
                        content.append('  {} = {:.2e}\n'.format(name, eval(para_type_dict[name.lower()])(out_val)))
                    else:
                        content.append('  {} = {}\n'.format(name, eval(para_type_dict[name.lower()])(out_val)))
        elif 'line' in section.lower():
            content.append('[{}]\n'.format(section))
            for name in input_dict[section].keys():
                out_val = input_dict[section][name]
                content.append('  {} = {}\n'.format(name, eval(para_type_dict[name.lower()])(out_val)))
        elif section.lower() == 'nodes':
            content.append('[{}]\n'.format(section))
            for name in input_dict[section].keys():
                out_val = input_dict[section][name]
                if type(out_val) != str:
                    content.append('  {} = {}\n'.format(name, int(out_val)))
                else:
                    content.append('  {} = {}\n'.format(name, str(out_val)))
        elif section.lower() == 'abundances':
            content.append('[{}]\n'.format(section))
            for name in ['abundance set', 'abundance file']:
                if name in [i.lower() for i in input_dict[section].keys()]:
                    out_val = input_dict[section][name]
                    content.append('  {} = {}\n'.format(name, str(out_val)))
            content.append('  [[Override]]\n')
            for name in input_dict[section].keys():
                if name in ['abundance set', 'abundance file']:
                    pass
                else:
                    out_val = input_dict[section][name]
                    content.append('    {} = {}\n'.format(name, float(out_val)))
            

    with open(file_path , 'w') as file:
        file.writelines(content)

def save_model(model, save_path, vmac=0, stray_light=0):
    
    model.to_csv(save_path, index=False, header=False, sep=' ', float_format='%.3e')
    with open(save_path, 'r') as file:
        content = file.readlines()
    content = [' {}'.format(line) for line in content]
    content = ['Format version: 1.0\n', '   {:.1f}  {:.1f}\n'.format(vmac, stray_light)] + content

    with open(save_path, 'w') as file:
        file.writelines(content)

def read_prof(run_folder, profile_name=''):
    '''
    Read the profile. 
    The profile is assumed to be in the same directly with the input file.
    '''
    
    input_content = read_input_file(run_folder + 'NICOLE.input')
    wav_dict = get_wav_array(input_content)
    
    if profile_name == '':
        profile_file = '/'.join([run_folder] + [input_content['section 1']['Output profiles']])
    else:
        profile_file = '/'.join([run_folder] + [profile_name])
    
    prof_paras = model_prof_tools.check_prof(profile_file)
    flux = model_prof_tools.read_prof(profile_file, *prof_paras, 0, 0)
    flux = np.array(flux).reshape(-1, 4)
    
    spec_dict = {}
    len_record = 0
    for region in wav_dict.keys():
        spec_array = np.concatenate([wav_dict[region].reshape(-1, 1), flux[len_record:len_record+len(wav_dict[region]), :]], axis=1)
        len_record += len(wav_dict[region])
        spec_dict[region] = pd.DataFrame(spec_array, columns=['wavelength', 'I', 'Q', 'U', 'V'])
        
    return spec_dict

def read_model(file_path):
    '''
    Read the model of NICOLE format. The code will read the file as ascii file fisrt, and as binary file if failed.
    Please note that the binary file contains more information than that in ascii file, so the key of the output dictionary will be different.
    '''
    
    model_dict = {}
    # Check the file type
    paras = model_prof_tools.check_model(file_path)
    file_type = paras[0]
    
    if 'ascii' in file_type.lower(): 
        model_df = pd.read_csv(file_path, skiprows=2, sep=' +', names=['log_tau5000', 'T', 'Pe', 'vmic', 'b_long', 'Vlos', 'b_x', 'b_y'], engine='python')
        model_dict['model'] = model_df
        return model_dict
    
    else: 
        
        t = model_prof_tools.read_model(file_path, *paras, 0, 0)
        # paras
        nz = paras[-1]

        para_name_dict = {
            'z':nz, 'log_tau5000':nz, 'T':nz, 'Pgas':nz, 'rho':nz, 'Pe':nz, 'Vlos':nz, 'vmic':nz, 'b_long':nz, 'b_x':nz, 'b_y':nz, 'b_localx':nz, 'b_localy':nz, 'b_localz':nz, 'v_localx':nz, 'v_localy':nz, 'v_localz':nz, 'nH':nz, 'nHminus':nz, 'nHplus':nz, 'nH2':nz, 'nH2plus':nz, 'vmac':1, 'stray_frac':1, 'ffactor':1, 'keep_Pe':1, 'keep_Pgas':1, 'keep_rho':1, 'keep_nH':1, 'keep_nHminus':1, 'keep_nHplus':1, 'keep_nH2':1, 'keep_nH2plus':1, 'abund':92
        }
        model_df = pd.DataFrame()
        index = 0

        for para_name in para_name_dict.keys():
            if para_name_dict[para_name] not in [1, 92]:
                model_df[para_name] = np.array(t[index:index+para_name_dict[para_name]])
                index += para_name_dict[para_name]
            elif para_name_dict[para_name] == 1:
                model_dict[para_name] = t[index]
                index += para_name_dict[para_name]
            else:
                model_dict[para_name] = np.array(t[-92:])
                index += para_name_dict[para_name]
        
        # Add Ne
        Pe = model_df['Pe'].values * (u.dyn / u.cm**2)
        T = model_df['T'].values * (u.K)
        Ne = (Pe / (constants.k_B * T)).to(u.cm**-3).value
        
        model_df['Ne'] = Ne
        
        model_dict['model'] = model_df
        return model_dict

def save_line_file(line_dict, file_path='temp.lines'):
    '''
    Save line dictonary to file in LINES format.
    '''
    
    content = []
    for section in line_dict.keys():
        content.append('[{}]\n'.format(section))
        for name in line_dict[section].keys():
            out_val = line_dict[section][name]
            if name.lower() == 'excitation potential':
                if type(out_val) != list:
                    content.append('  {} = {}\n'.format(name, eval(line_type_dict[name.lower()][0])(out_val)))
                else:
#                     out_val = out_val.split()
                    content.append('  {} = {} {}\n'.format(name, eval(line_type_dict[name.lower()][0])(out_val[0]), out_val[1]))
            elif name.lower() == 'collisions':
                try:
                    content.append('  {} = {}\n'.format(name, int(out_val)))
                except:
                    content.append('  {} = {}\n'.format(name, str(out_val)))
            else:
                if line_type_dict[name.lower()] == 'float' and out_val < 1e-3 and out_val != 0:
                    content.append('  {} = {:.2e}\n'.format(name, eval(line_type_dict[name.lower()])(out_val)))
                else:
                    content.append('  {} = {}\n'.format(name, eval(line_type_dict[name.lower()])(out_val)))
        content.append('\n')   

    with open(file_path , 'w') as file:
        file.writelines(content)

def read_atom(filename):
    '''
    Read the first 5 parts MULTI-format ATOM file. 
    '''

    with open(filename) as file:
        fdata = file.readlines()

    # Remove all comment lines 
    fdata = [i for i in fdata if i[0] != '!']
    fdata = [i.replace('\n', '') for i in fdata]

    content = {}

    content['atom'] = fdata[0]
    content['abund'], content['awgt'] = list(np.fromstring(fdata[1], sep=' '))
    content['NK'], content['NLIN'], content['NCNT'], content['NFIX'] = list(np.fromstring(fdata[2], sep=' ', dtype=int))

    # Read the level part
    row_record = 3
    level = []
    for i in range(row_record, content['NK']+row_record):
        iter = re.finditer(r"'", fdata[i])
        term_position = [m.start(0) for m in iter]
        level_single = list(np.fromstring(fdata[i][:term_position[0]] + fdata[i][term_position[1]+1:], sep=' '))
        level_single = level_single[:-1] + [fdata[i][term_position[0]:term_position[1]+1]] + level_single[-1:]
        level.append(level_single)

    content['level'] = pd.DataFrame(level, columns=['ecm', 'g', 'label', 'ion'])
    row_record += content['NK']

    # Read the line part
    line = []
    for i in range(row_record, content['NLIN']+row_record):
        line_single = list(np.fromstring(fdata[i], sep=' '))
        line.append(line_single)
    content['line'] = pd.DataFrame(line, columns=['J', 'I', 'F', 'NQ', 'QMAX', 'Q0', 'IW', 'GA', 'GVW', 'GS'])
    row_record += content['NLIN']
    
    return content
        
def find_line(target_ecm, atom, ecm_thres=0.1):
    '''
    Find the corresponding line from line in ATOM dict.
    '''

    # Find low and high level
    level_all = []
    for i in range(len(atom['level'])):
        for j in range(i+1, len(atom['level'])):
            level_all.append([j+1, i+1, atom['level'].loc[j, 'ecm'] - atom['level'].loc[i, 'ecm']])

    level_all = pd.DataFrame(level_all, columns=['J', 'I', 'ecm'])
    level_all['abs_del_ecm'] = np.abs(level_all['ecm'] - target_ecm)
    level_all = level_all.sort_values('abs_del_ecm')

    if len(level_all) == 0 or level_all.iloc[0]['abs_del_ecm'] >= ecm_thres:
        raise ValueError('Find no line.')

    tar_J, tar_I = list(level_all.iloc[0][['J', 'I']].values)
    tar_line_index = atom['line'].loc[(atom['line']['J'] == tar_J) & (atom['line']['I'] == tar_I)].index[0] + 1
    return tar_line_index

def read_pop(filename, model):
    '''
    Read the NICOLE output population file.
    '''
    
    nz = len(model)
    f = open(filename, 'rb')
    pop_all = np.fromfile(f)
    pop_all = pop_all.reshape(-1, nz)
    npop = pop_all.shape[0]

    pop = {'NLTE':pop_all[:int(npop/2)], 'LTE':pop_all[int(npop/2):]}
    
    return pop

lines = read_input_file('/media/disk/py-package/rulerwd/rulerwd/file/nicole/LINES', lines=True)

package_path = path.dirname(__file__)

# Read some atmosphere models to the code



# Read the NLTE lines to the code.

def run_nicole(atom, working_dir='./', output=True):
    
    # Copy the main control python file to working directory
    cp_status = subprocess.run(['cp', package_path + '/file/nicole/run_nicole.py', working_dir], encoding='UTF-8', stdout=subprocess.PIPE)
    cp_status = subprocess.run(['cp', package_path + '/file/nicole/model_prof_tools.py', working_dir], encoding='UTF-8', stdout=subprocess.PIPE)
    cp_status = subprocess.run(['cp', package_path + '/file/nicole/LINES', working_dir], encoding='UTF-8', stdout=subprocess.PIPE)
    if path.exists(package_path + '/file/ATOM/atom.{}'.format(atom)):
        cp_status = subprocess.run(['cp', package_path + '/file/nicole/ATOM/atom.{}'.format(atom), working_dir + '/ATOM'], encoding='UTF-8', stdout=subprocess.PIPE)
    
    # Run NICOLE
    NICOLE_run = subprocess.run(['./run_nicole.py', '>', 'run.log'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=working_dir)
    NICOLE_run_out = str(NICOLE_run.stdout, encoding = "utf-8").split('\n')
    
    if output == True:
        for i in NICOLE_run_out:
            print(i)
    elif type(output) == str:
        # Output the stdout to a file.
        NICOLE_run_out = ['{}\n'.format(i) for i in NICOLE_run_out]
        with open(working_dir + '/' + output, 'w') as file:
            file.writelines(NICOLE_run_out)
            
def vald2nicole():
    '''
    This is the python verison of the IDL function `vald_to_nicole`.
    '''
    
    pass
    
def carry(x, base):
    '''
    Check whether x requires carry, based on the base.
    '''
    if type(x) == list:
        x = np.array(x)
    
    if x[0] == base:
        return x
    for i in range(base+1):
        div = x // base
        mod = x % base
        x = mod + np.concatenate([div[1:], [0]])
        
        if x[0] == base:
            break

    return x

def create_all_grids(rows, columns):
    x = np.array([0]*columns)
    one = np.array([0]*columns)
    one[-1] = 1

    x_all = []

    while x[0] != rows:
        if np.all(rulerwd.tools.calculate_delta(x[x != 0]) >= 0):
    #         print('good')
            x_all.append(x-1)
        x = carry(x + one, rows)
        
    return x_all

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

def get_tau5000_model(teff, logg, m_h):
    '''
    Get the log_tau5000 of a model from MOOG.
    '''
    
    s = pymoog.synth.synth(teff, logg, m_h, 5000-1, 5000+1, 28000, line_list='ges')
    s.prepare_file()
    s.run_moog()
    s.read_model()
    s.model['log_tau5000'] = np.log10(s.model['tauref'])
    s.model['logNe'] = np.log10(s.model['Ne'])
    s.model['logPgas'] = s.model['logPg']
    
    return s.model

def create_chromo_model(model_in, log_tau_control_array, T_control_array, T_min, density_name='Pe', vmic=9*1e5, Vlos=0, plot=False, keep=False):
    
    '''
    Create chromosphere model from a given LTE model.
    
    Input
    --------
    keep : bool, default False
        If True, then the model will be kept the same as input; this can be used to convert the format to NICOLE.
    '''
    
    ylabel_dict = {'Pe':r'$\log{[P_\mathrm{e} (\mathrm{dyn/cm^2})]}$', 
                    'Ne':r'$\log{[N_\mathrm{e} (\mathrm{cm^{-3}})]}$',
                    'Pgas':r'$\log{[P_\mathrm{gas} (\mathrm{dyn/cm^2})]}$'}
    
    log_tau5000 = model_in['log_tau5000']
    T = model_in['T']
    log_density = model_in['log{}'.format(density_name)]
    
    if plot:
        plt.figure(figsize=(14, 4))
        ax1, ax2 = plt.subplot(121), plt.subplot(122)
        ax1.plot(log_tau5000, T, marker='.')
        ax2.plot(log_tau5000, log_density, marker='.')
    
    if not(keep):
    
        # Find the index in T_min
        T_min_index = np.abs(model_in['T'] - T_min).sort_values().index[0]

        log_tau5000 = log_tau5000[T_min_index:].reset_index(drop=True)
        T = T[T_min_index:].reset_index(drop=True)
        log_density = log_density[T_min_index:].reset_index(drop=True)

        if log_tau_control_array[-1] > log_tau5000[0]:
            raise ValueError('log_tau_control_array[-1]:{} is larger than that in T_min:{}; modify log_tau_control_array.'.format(log_tau_control_array[-1], log_tau5000[0]))

        log_density_fit = np.polyfit(log_tau5000[:5], log_density[:5], 1)
        density = np.concatenate([10**np.polyval(log_density_fit, log_tau_control_array), 10**log_density])
        T = np.concatenate([T_control_array, T])
        log_tau5000 = np.concatenate([log_tau_control_array, log_tau5000])
        
    else:
        density = 10**log_density
    
    if plot:
        ax1.plot(log_tau5000, T, marker='.')
        ax2.plot(log_tau5000, np.log10(density), marker='.')
        
        ax1.set_xlabel(r'$\log{\tau_{5000}}$')
        ax2.set_xlabel(r'$\log{\tau_{5000}}$')
        ax1.set_ylabel('$T$ (K)')
        ax2.set_ylabel(ylabel_dict[density_name])
        
    model_out = pd.DataFrame()
    model_out['log_tau5000'] = log_tau5000
    model_out['T'] = T
    model_out['{}'.format(density_name)] = density
    model_out['vmic'] = vmic
    model_out['logM'] = 0
    model_out['Vlos'] = Vlos
    model_out['Mx'] = 0
    model_out['My'] = 0
    
    return model_out

def find_consecutive_int(in_list):
    '''
    Find all consecutive integer in a list. Note that if only one number, it will also be included.
    '''
    
    consecutive_lists = []
    
    s = []  # 空栈
    for i in sorted(set(in_list)):
        if len(s) == 0 or s[-1] + 1 == i:
            s.append(i)  # 入栈
        else:
            if len(s) >= 1:
                consecutive_lists.append(s)
            s = []    # 清空
            s.append(i)  # 入栈

    if len(s) >= 1:
        consecutive_lists.append(s)
        
    return consecutive_lists

def polish_model(model, max_diff_T=800, plot=False, factor=2):
    '''
    Mianly do two things: 
        1. interpolate the model if any of the two grid points have large diff_T, etc
        2. smooth the model if required (not done yet)
    '''
    
    # double the grid when del-T is large
    
    model_out = model
    del_T = np.abs(rulerwd.tools.calculate_delta(model_out['T']))
    
    polish_indices = del_T >= max_diff_T
    indices_list = find_consecutive_int(model_out[polish_indices].index)
    
    if plot:
        ax1 = plt.subplot()
        ax2 = plt.twinx()
        ax1.plot(model_out['log_tau5000'], model_out['T'])
        ax2.plot(model_out['log_tau5000'], del_T, c='C1')
    
#     return indices_list
    iter_n = 0
    while len(model_out[polish_indices]) > 0 and iter_n < 10:
        for indices in indices_list:
            if len(indices) == 1:
                index_start, index_end = indices[0], indices[0]
            else:
                index_start, index_end = indices[0], indices[-1]
            if index_end - index_start <= 2:
                index_start -= 2
                index_end += 2
#             if index_start != 0:
#                 index_start -= 1
#             if index_end != len(model_out)-1:
#                 index_end += 1
            model_out = pd.concat([model_out.loc[:index_start], 
                                   interpolate_model(model_out.loc[index_start:index_end], int(len(model_out.loc[index_start:index_end])*factor)), 
                                   model_out.loc[index_end:]]).reset_index(drop=True)
#             print(interpolate_model(model_out.loc[index_start:index_end], int(len(model_out.loc[index_start:index_end])*factor)))
            

        del_T = np.abs(rulerwd.tools.calculate_delta(model_out['T']))
        polish_indices = del_T >= max_diff_T
        indices_list = find_consecutive_int(model_out[polish_indices].index)
            
        
        if plot:
            ax1.plot(model_out['log_tau5000'], model_out['T'])
            ax2.plot(model_out['log_tau5000'], del_T, ls='--')
        iter_n += 1
        
    return model_out