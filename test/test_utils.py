from unittest import TestCase
import subprocess

class StoreDependentTestCase(TestCase):
    ''' Unit test base class that starts the Limbo plasma server
        for the tests in this case.
    '''

    def setUp(self):
        ''' Start the server
        '''
        self.p = subprocess.Popen(['plasma_store',
                              '-s', '/tmp/store',
                              '-m', str(1000000)],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)

    def tearDown(self):
        ''' Kill the server
        '''

        self.p.kill()
        