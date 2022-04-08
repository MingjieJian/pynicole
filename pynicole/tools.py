from . import model_prof_tools
import re
import numpy as np
import pandas as pd
from astropy import constants
from astropy import units as u
from os import path
import subprocess

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