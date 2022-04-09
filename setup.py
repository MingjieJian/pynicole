import subprocess
import os
import setuptools

# Define pynicole_path. This path will store the code of nicole and any other temporary files in the calculation.
pynicole_path = '{}/.pynicole/'.format(os.environ['HOME'])

# Create the folder according to pynicole_path
if not(os.path.isdir(pynicole_path)):
    os.mkdir(pynicole_path)
    
# Copy the files folder into working directory
if not(os.path.isdir(pynicole_path + 'files')):
    os.mkdir(pynicole_path + 'files')
rm_status = subprocess.run(['rm', '-r', pynicole_path + 'files/'], stdout=subprocess.PIPE)
cp_status = subprocess.run(['cp', '-r', 'pynicole/files', pynicole_path + 'files'], stdout=subprocess.PIPE)

# # Copy the NICOLE folder to pynicole_path; if the folder already exist it will be removed first.
# if os.path.isdir(pynicole_path + 'files/NICOLE'):
#     rm_status = subprocess.run(['rm', '-r', pynicole_path + 'files/NICOLE'], stdout=subprocess.PIPE)
# mv_status = subprocess.run(['cp', '-r', 'NICOLE', pynicole_path + 'files/NICOLE'], stdout=subprocess.PIPE)

# Install NICOLE
install = subprocess.run(['./create_makefile.py', '--compiler=gfortran', "--otherflags='-O3 -march=native -fdefault-real-8 -fdefault-double-8 -fallow-argument-mismatch'"],  cwd=pynicole_path + 'files/NICOLE/main', stdout=subprocess.PIPE)
install = subprocess.run(['make', 'clean'],  cwd=pynicole_path + 'files/NICOLE/main', stdout=subprocess.PIPE)
install = subprocess.run(['make', 'nicole'],  cwd=pynicole_path + 'files/NICOLE/main', stdout=subprocess.PIPE)


# # Check the permission of ./install
# chmod_subp = subprocess.run(['chmod', '775', './install.sh'], cwd=pynicole_path+'moog_nosm/moog_nosm_NOV2019/', stdout=subprocess.PIPE)

# Check if NICOLE is in the folder
if not(os.path.isfile(pynicole_path+'files/NICOLE/main/nicole')):
    raise ValueError("NICOLE is not installed correctly!")
else:
    print('Successfully installed NICOLE!')
        
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
      name='pynicole',
      version='0.0.1',
      description='The python wrapper to run NLTE spectra synthesis code NICOLE.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/MingjieJian/pynicole',
      author='Mingjie Jian',
      author_email='ssaajianmingjie@gmail.com',
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Framework :: IPython",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Astronomy"
      ],
      python_requires=">=3.5",
      packages=setuptools.find_packages(),
      install_requires=[
          'numpy >= 1.18.0',
          'pandas >= 1.0.0',
          'matplotlib >= 3.1.0',
          'mendeleev >= 0.6.0',
          'scipy >= 1.4.0',
          'astropy >= 4.0',
          'spectres',
          'tqdm',
          'pymoog',
          'rulerwd'
      ],
      include_package_data=True,  
    #   package_data={'': ['moog_nosm/moog_nosm_FEB2017/']},
      zip_safe=False)