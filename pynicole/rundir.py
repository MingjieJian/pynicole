#!/usr/bin/python
from . import private

class rundir(object):

    def __init__(self, run_math, run_type, prefix=''):
        '''

        '''

        self.run_math = run_math

        if prefix == '':
            self.rundir_path = '{}/{}-{}/'.format(self.run_math, run_type, private.datetime.now().strftime("%H:%M:%S.%f"))
        else:
            self.rundir_path = '{}/{}-{}-{}/'.format(self.run_math, prefix, run_type, private.datetime.now().strftime("%H:%M:%S.%f"))

        private.subprocess.run(['mkdir', '-p', self.rundir_path])
        
    def remove(self):
        private.subprocess.run(['rm', '-r', self.rundir_path])