#m2w.py
from sys import argv
import os
from argparse import ArgumentParser
from pydub import AudioSegment

def convert_to_wav(in_path, out_path):
    '''
    convert_to_wav(in_path, out_path)
    
    in_path:    path to a mp3 file

    Exports WAV audio signal to out_path
    '''
    if '.mp3' not in in_path.split('/')[-1]:
        return

    sound = AudioSegment.from_mp3(in_path)

    try:
        os.makedirs('/'.join(out_path.split('/')[:-1]))
    except FileExistsError:
        pass

    sound.export(out_path, format="wav")

    print(f"Conversion {in_path} to {out_path} successful!")
    return

def list_files_of_tree(path):
    '''
    list_files_of_tree(path)
    
    path: path to a parent directory

    Recursively adds paths to mp3 files to the global variable mp3_list
    '''
    if str(out) == path:
        mp3_list.remove(path)
        print("One of input (sub)directories is equal to output directory. That bro is so cooked LMAO")
        return False

    try:
        ld = os.listdir(path)
        mp3_list.remove(path)
        for item in ld:
            mp3_list.append(path + '/' + item)
            list_files_of_tree(path + '/' + item)
        return True
    except WindowsError:
        return False

parser = ArgumentParser(prog="mp3 2 wav",
                        usage="m2w.py [options] files ...",
                        description="Converts mp3 to wav",
                        epilog="Play Dark Souls III instead of using this thing")
parser.add_argument('files', nargs='+', type=str, help = "directories or files to convert")
parser.add_argument('-o', '--output', help = "Output directory")

args = parser.parse_args(argv[1:])

files = args.files
out = args.output
if out != None:
    out = out.replace('\\', '/').removesuffix('/')
for file in files:
    file = file.replace('\\', '/').removesuffix('/')
    mp3_list = [file]
    if list_files_of_tree(file):
        if out == None:
            out = file + '/wav'
        for mp3 in mp3_list:
            convert_to_wav(mp3, out + mp3.removeprefix(file).removesuffix('.mp3') + '.wav')
    else:
        if out == None:
            out = '/'.join(file.split('/')[:-1]) + '/wav'
        convert_to_wav(file, out + '/' + file.split('/')[-1].removesuffix('.mp3') + '.wav')