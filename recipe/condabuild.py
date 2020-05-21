import argparse
import json
import os
import platform
import sys
from glob import glob
from subprocess import run

import yaml

__version__ = "0.1.19"

platforms = dict([('Linux', 'linux-64'), ('Darwin', 'osx-64'), ('Windows', 'win-64')])
current_platform = platforms[platform.system()]

# ----------------------------------------------------------------------------------------------------------------------
def conda_exec(commands, perror=True):
    commandlist = commands.strip().split(" ")
    isjson = False
    if commandlist[1] not in ['build', 'update', 'upload', 'convert']:
        commandlist.append('--json')
        isjson = True
        
    proc = run(commandlist, text=True, capture_output=True)
    
    if isjson:
        proc = json.loads(proc.stdout)
        if perror and isinstance(proc, dict) and 'error' in proc.keys():
            print(proc['error'])
        return proc
    else:
        if perror and proc.stderr:
            print(proc.stderr)
        return proc.stdout

# ----------------------------------------------------------------------------------------------------------------------
def build(package, upload=False, pypi=False, convert=True, user='spectrocat'):
    
    recipe = os.path.join(os.path.dirname(__file__), package, 'meta.yaml')
    if not os.path.exists(recipe):
        print(f'Sorry, but I cannot find a recipe to build this package: `{recipe}`!')
        exit(1)
    
    if not upload and not pypi:
        
        # building the conda package
        st = f'BUILDING THE {package} CONDA PACKAGE...'
        print(f'\n{st}\n' + '_' * len(st) + '\n')
        print(
            'Get a cup of tea or coffee and be patient: it\'s generally a long process, and no display occurs before termination...')
        
        if not os.path.exists('pkg_folder'):
            os.mkdir('pkg_folder')
        proc = conda_exec(f'conda build {os.path.dirname(recipe)} --output-folder pkg_folder', perror=False)
        print(proc)
        proc = conda_exec('conda build purge')

        upload = True
        pypi = False
        if package == 'spectrochempy':
            pypi == True
            
    
    if convert and not package=='quadprog':
        
        tarbzs = glob(os.path.join('pkg_folder', '*', f'{package}*.*'))
    
        st = f'CONVERT CONDA PACKAGE ...'
        print(f'\n{st}\n' + '_' * len(st) + '\n')
    
        for tar in tarbzs:
            for plf in platforms.values():
                if plf == current_platform:
                    continue
                print( f'convert {tar} to {plf}')
                proc = conda_exec(f'conda convert {tar} -p {plf} -o pkg_folder -f')

    tarbzs = glob(os.path.join('pkg_folder', '*', f'{package}*.*'))
    if upload:
        # upload to conda
        st = f'UPLOAD CONDA PACKAGE ...'
        print(f'\n{st}\n' + '_' * len(st) + '\n')
    
        for tar in tarbzs:
            print( f'uploading {tar} to {user}/label/dev')
            proc = conda_exec(f'anaconda upload {tar} --user {user} --label dev --skip')
    
    elif pypi:
        pass
    
    

if __name__ == '__main__':
    
    # Parse arguments
    # ----------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    
    parser.add_argument("packages", nargs='*', default='', help="name(s) of the conda package(s) to build")
    parser.add_argument("--noconvert", help="do not convert to missing platform (only pure python packages)", action="store_true")
    parser.add_argument("--upload", help="load to anaconda.org", action="store_true")
    parser.add_argument("--pypi", help="upload to Pypi", action="store_true")
    parser.add_argument("--user", help="change user (default spectrocat)", default="spectrocat")
    parser.add_argument("--update", help="update conda package", action="store_true")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        exit(1)
    
    args = parser.parse_args()
    
    if not args.packages:
        parser.print_help(sys.stderr)
        exit(1)
    
    packages = args.packages
    upload = args.upload
    pypi = args.pypi
    convert = not args.noconvert
    user = args.user
    update = args.update
    
    if update:
        # we build or we upload, not both
        st = f'PREPARING CONDA PACKAGE BUILDING ...'
        print(f'\n{st}\n' + '_' * len(st) + '\n')
    
        # update conda
        print('update: conda ...')
        proc = conda_exec("conda update conda -y")
    
        print('check: ~/.condarc ...')
        # make condarc properly
        condarc = yaml.load(open(os.path.expanduser(os.path.join('~', '.condarc')), 'r').read(), Loader=yaml.Loader)
        for channel in ['conda-forge', 'cantera', 'spectrocat']:
            if channel not in condarc['channels']:
                proc = conda_exec(f"conda config --prepend channels {channel}")
            else:
                print(f'Channel "{channel}" is in "~/.condarc" - OK.')
        proc = conda_exec("conda config --set channel_priority strict")
        print('Channel priority is set to Strict.')
    
        # add package needed for the building
        pkgs = []
        for pkg in conda_exec("conda list"):
            pkgs.append(pkg['name'])
        pkg2update = ['pip', 'setuptools', 'wheel', 'twine', 'conda-build', 'conda-verify', 'anaconda-client']
        pup = []
        pin = []
        for pkg in pkg2update:
            pup.append(pkg) if pkg in pkgs else pin.append(pkg)
        if pin:
            pin = " ".join(pin)
            print(f'install: {pin} ...')
            proc = conda_exec(f"conda install {pin} -y")
        if pup:
            pup = " ".join(pup)
            print(f'update: {pup}')
            proc = conda_exec(f"conda update {pup} -y")

    for package in packages:
        convert = True
        if package == 'quadprog':
            convert = False
        build(package, upload, pypi, convert, user)