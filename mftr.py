#mftr.py

from sys import argv
from os import listdir
import librosa
from numpy import mean, std, array
import pandas as pd
from argparse import ArgumentParser

def Calculate_Channel(wave, sr):
    '''
    Calculate_Channel(wave, sr)
    
    wave : np.ndarray(t)
        Audio signal, np.ndarray from librosa.load(...)
    sr : int
        Sample rate, number from librosa.load(...)

    Returns a tuple, containing:
    rms : np.ndarray(t)
        Root Mean Squared of audio signal 
    zcr : np.ndarray(t)
        Zero Crossing Rate of audio signal
    sce : np.ndarray(t)   
        Spectral Centroid of audio signal  
    sb : np.ndarray(t)    
        Spectral Bandwidth of audio signal
    sro : np.ndarray(t)   
        Spectral Rolloff of audio signal
    sco : np.ndarray(7, t)   
        Spectral Contrast of audio signal  
    mfccs : np.ndarray(n_mfccs, t)
        MFCCs of audio signal             
    '''
    rms =   librosa.feature.rms(y=wave, frame_length=frame_size, hop_length=hop_length)[0]
    zcr =   librosa.feature.zero_crossing_rate(y=wave, frame_length=frame_size,hop_length=hop_length)[0]
    sce =   librosa.feature.spectral_centroid(y=wave, sr=sr, n_fft=frame_size,hop_length=hop_length)[0]
    sb =    librosa.feature.spectral_bandwidth(y=wave, sr=sr, n_fft=frame_size,hop_length=hop_length)[0]
    sro =   librosa.feature.spectral_rolloff(y=wave, sr=sr, n_fft=frame_size,hop_length=hop_length)[0]
    sco =   librosa.feature.spectral_contrast(y=wave, sr=sr, n_fft=frame_size,hop_length=hop_length)
    mfccs = librosa.feature.mfcc(y=wave, n_mfcc=n_mfccs, sr=sr, win_length=frame_size,hop_length=hop_length)
    return  (rms, zcr, sce, sb, sro, sco, mfccs)

def Calculate_Features(wav_file_path):
    '''
    Calculate_Features(wav_file_path)

    wav_file_path : str
        Path to WAV file

    Returns a list, containing:
    rms_mean : float
        Root Mean Squared mean
    rms_std : float
        Root Mean Squared standard deviation
    zcr_mean : float
        Zero Crossing Rate mean
    zcr_std : float
        Zero Crossing Rate standard deviation
    sce_mean : float
        Spectral Centroid mean
    sce_std : float
        Spectral Centroid standard deviation
    sb_mean : float
        Spectral Bandwidt mean
    sb_std : float
        Spectral Bandwidth standard deviation
    sro_mean : float
        Spectral Rolloff mean
    sro_std : float
        Spectral Rolloff standard deviation
    sco_mean0, ..., sco_mean6 : float
        Spectral Contrast mean
    sco_std0, ..., sco_std6 : float
        Spectral Contrast standard deviation
    mfcc_mean0, ...,  mfcc_meanN : float
        MFCCs mean
    mfcc_std0, ...,  mfcc_stdN : float
        MFCCs std
    '''
    features = []

    wave, sr = librosa.load(wav_file_path, mono = False)
    if wave.ndim == 1:
        channels = 1
        start = 0
        while wave[start] == 0:
            start += 1
        end = wave.size - 1
        while wave[end] == 0:
            end -= 1
        wave = wave[start:end]
    else:
        channels = wave.shape[0]
        start = 0
        while wave[0, start] == 0:
            start += 1
        end = wave[0].size - 1
        while wave[0, end] == 0:
            end -= 1
        wave = wave[:, start:end]

    rms, zcr, sce, sb, sco, sro, mfccs = [], [], [], [], [], [], []
    for i in range(channels):
        if channels > 1:
            res = Calculate_Channel(wave[i, :], sr)
        else:
            res = Calculate_Channel(wave, sr)
        rms +=      [res[0]]
        zcr +=      [res[1]]
        sce +=      [res[2]]
        sb +=       [res[3]]
        sro +=      [res[4]]
        sco +=      [res[5]]
        mfccs +=    [res[6]]

    rms = array(rms).mean(axis = 0)
    zcr = array(zcr).mean(axis = 0)
    sce = array(sce).mean(axis = 0)
    sb = array(sb).mean(axis = 0)
    sro = array(sro).mean(axis = 0)
    sco = array(sco).mean(axis = 0)
    mfccs = array(mfccs).mean(axis = 0)

    features.append(mean(rms))
    features.append(std(rms))

    features.append(mean(zcr))
    features.append(std(zcr))

    features.append(mean(sce))
    features.append(std(sce))

    features.append(mean(sb))
    features.append(std(sb))

    features.append(mean(sro))
    features.append(std(sro))

    for i in range(sco.shape[0]):
        features.append(sco[i].mean())
    for i in range(sco.shape[0]):
        features.append(sco[i].std())

    for i in range(n_mfccs):
        features.append(mfccs[i].mean())
    for i in range(n_mfccs):
        features.append(mfccs[i].std())

    return features

parser = ArgumentParser(prog="mftr",
                        usage="mftr.py [options] dirs/files",
                        description="Extracts time-domain, spectral and cepstral features from music files. "
                                    "Specifically mean and standard deviation for "
                                    "Root mean squared, Zero crossing rate, Spectral centroid, "
                                    "Spectral bandwith, Spectral rolloff, Spectral contrast "
                                    "And MFCCs",
                        epilog="Play Elden Ring instead of using this thing")
parser.add_argument('-a', '--append', metavar="CSV_FILE", help = "Append rows to existing csv file")
parser.add_argument('-f', '--frame_size', type = int, default = 1024, help = "Set the frame size for short-time Fourier transform (Default: 1024")
parser.add_argument('-l', '--hop_length', type = int, default = 512, help = "Set the hop length for short-time Fourier transform (Default: 512)")
parser.add_argument('-n', '--n_mfccs', type = int, default = 13, help = "Set the MFCCs quantity (Default: 13)")
parser.add_argument('-o', '--output', help = "Set the output csv file (Default: ./features.csv)")
parser.add_argument('-p', '--parent', action="store_true", help = "Set the arguments to be processed as parental directories with genre-named subdirectories")
parser.add_argument('files', nargs='+', help = "Files OR directories to process")

args = parser.parse_args(argv[1:])

frame_size = args.frame_size
hop_length = args.hop_length
csv_file = args.append
n_mfccs = args.n_mfccs
output = args.output
files = args.files

COLS_NAMES = ["RMS_MEAN", "RMS_STD", "ZCR_MEAN", "ZCR_STD", "SCe_MEAN", "SCe_STD",
"SB_MEAN", "SB_STD", "SRo_MEAN", "SRo_STD"] + ["SCo" + str(i) + "_MEAN" for i in range(7)]
+ ["SCo" + str(i) + "_STD" for i in range(7)] + ["MFCC" + str(i) + "_MEAN" for i in
range(n_mfccs)] + ["MFCC" + str(i) + "_STD" for i in range(n_mfccs)]

dirs_flag = True

if output == None:
    output = "./features.csv"
    if csv_file != None:
        output = csv_file

for item in files:
    item = item.replace('\\', '/').removesuffix('/')
    if ".wav" in item.split('/')[-1] and dirs_flag:
        print("Wave file argument found! Directories will be ignored")
        dirs_flag = False

if dirs_flag:
    if args.parent:
        genre_directories = []
        for directory in files:
            try:
                genre_directories += [directory.removesuffix('/') + '/' + genre for genre in listdir(directory)]
            except:
                print(directory + " is skipped")
        if len(genre_directories) == 0:
            print("Deez arguments are shi.. I mean... incorrect...")
            exit(1)
    else:
        genre_directories = files
    genres = [f.split('/')[-1] for f in genre_directories]
    if csv_file == None:
        g = genres[:]
        for item in genres:
            if ',' in item:
                for i in item.split(','):
                    if i not in g:
                        g.append(i)
                g.remove(item)
        df = pd.DataFrame(columns = COLS_NAMES + g)
    else:
        df = pd.read_csv(csv_file, index_col=0)
    for i in range(len(genre_directories)):
        try:
            items = listdir(genre_directories[i])
        except:
            print(genre_directories[i] + " ain't a valid directory, man... Git gud bruh...")
        for item in items:
            if ".wav" not in item.split('/')[-1]:
                continue
            track_features = Calculate_Features(genre_directories[i] + '/' + item)
            genre = genres[i].split(',')
            track_features = pd.DataFrame([track_features + [1.0 for i in genre]], columns
            = COLS_NAMES + genre, index = [item])
            df = pd.concat((df, track_features), axis = 0)
        print(genre_directories[i] + ' finished!')
    df = df.fillna(0.0)
else:
    if csv_file == None:
        df = pd.DataFrame(columns = COLS_NAMES)
    else:
        df = pd.read_csv(csv_file, index_col=0)
    for item in files:
        if ".wav" not in item.split('/')[-1]:
            continue
        track_features = Calculate_Features(item)
        track_features = pd.DataFrame([track_features], columns = COLS_NAMES, index = [item])
        df = pd.concat((df, track_features), axis = 0)
        print(item + " finished!")
print(df)
df.to_csv(output)
