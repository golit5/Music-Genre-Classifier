# Music Genre Classifier
 
# Music Genre Classification Pipeline

This repository contains a set of Python programs for music genre classification. The pipeline involves three main steps:

1. **Convert MP3 files to WAV**  
2. **Extract audio features from WAV files**  
3. **Train or use a pre-trained model to classify genres based on the extracted features**

Each script is designed for a specific task within this workflow. Below is a brief explanation of each program and how to use them.

---

## Contents

1. **`m2w.py`** - Convert MP3 to WAV files.
2. **`mftr.py`** - Extract audio features from WAV files and store them in a CSV file.
3. **`clsfr.py`** - Train a classification model or predict music genres from a CSV file containing extracted features.

---

## Prerequisites

Before using any of the scripts, make sure you have the following Python libraries installed:

```bash
pip install pydub librosa pandas scikit-learn xgboost matplotlib seaborn
```

Additionally, the `pydub` library requires you to have `ffmpeg` or `libav` installed on your system for MP3 handling.

---

## 1. `m2w.py` - Convert MP3 to WAV

This script recursively or selectively converts MP3 files to WAV files. You can specify the input directory or files, and optionally an output directory.

### Usage:

```bash
python m2w.py [options] files ...
```

### Options:

- `files`: List of MP3 files or directories containing MP3 files to convert.
- `-o`, `--output`: Specify the output directory for the WAV files.

### Example:

```bash
python m2w.py ./music_folder/ -o ./output_folder/
```

This command will convert all MP3 files in the `./music_folder/` and its subdirectories into WAV format and store them in `./output_folder/`.

---

## 2. `mftr.py` - Extract Audio Features

This script extracts a set of audio features from WAV files, including time-domain features (like RMS), spectral features (like spectral centroid, bandwidth, and rolloff), and cepstral features (MFCCs). These features are saved into a CSV file, which can be used for training a classification model.

### Usage:

To create training csv:

```bash
python mftr.py [options] dirs/files
```

### Options:

- `dirs/files`: List of WAV files or genre-named directories with WAV files to extract features from.
- `-a`, `--append`: Append the extracted features to an existing CSV file.
- `-f`, `--frame_size`: Set the frame size for short-time Fourier transform (Default: 1024).
- `-l`, `--hop_length`: Set the hop length for short-time Fourier transform (Default: 512).
- `-n`, `--n_mfccs`: Set the number of MFCCs to extract (Default: 13).
- `-o`, `--output`: Set the output CSV file (Default: `./features.csv`).
- `-p`, `--parent`: Process directories with genre-named subdirectories.

### Example:

```bash
python mftr.py -o features.csv ./music_folder/
```

This command will extract audio features from all WAV files in the `./music_folder/` directory and save them to `./features.csv`. There will be a column with genre name flooded with '1'

```bash
python mftr.py -o features.csv --parent ./music_folder/
```

This command will extract audio features from all WAV files in the `./music_folder/GENRE_NAME` directories and save them to `./features.csv`. There will be a column for each genre name

```bash
python mftr.py -o features.csv  ./file.wav
```

This command will extract audio features from './file.wav' and save them to `./features.csv`. There will be no genre-named colummns

---

## 3. `clsfr.py` - Train or Predict Genres

This script uses the extracted features from the CSV file to either train a new genre classification model or make predictions using an existing model.

### Usage:

```bash
python clsfr.py [options] file
```

### Options:

Either --train or --predict is required! They are mutually exclusive

- `-t`, `--train`: Train a new model using the CSV file and save it.
- `-p`, `--predict`: Predict genres for the input CSV file using an existing model.
- `-m`, `--model`: Path to save/load the model.
- `csv_file`: The CSV file containing extracted features.

### Example (Training a Model):

```bash
python clsfr.py -t -m ./model_folder/ features.csv
```

This command will train a model using the data from `./features.csv` and save it to `./model_folder/`.

### Example (Predicting Genres):

```bash
python clsfr.py -p -m ./model_folder/ features.csv
```

This command will predict the genres for the WAV files described in `./features.csv` using the model stored in `./model_folder/`.

---

## Pipeline Overview

1. **Convert MP3 to WAV**:  
   Run `m2w.py` to convert your MP3 files to WAV format. You can process a directory of MP3 files or individual MP3 files.

2. **Extract Features**:  
   Run `mftr.py` to extract audio features from the converted WAV files and store them in a CSV file. The features extracted include Root Mean Square (RMS), Zero Crossing Rate (ZCR), Spectral Centroid (SCE), Spectral Bandwidth (SB), Spectral Rolloff (SRO), Spectral Contrast (SCO), and Mel-Frequency Cepstral Coefficients (MFCCs).

3. **Train or Predict with Model**:  
   Run `clsfr.py` to either train a new model or predict genres for the music files based on the extracted features. The model used is an XGBoost classifier, and it can be saved for future predictions.

---

## Example Workflow

1. Convert your MP3 files to WAV:
   ```bash
   python m2w.py music_folder/ -o wav_folder/
   ```

2. Extract features from the WAV files:
   ```bash
   python mftr.py -o features.csv --parent wav_folder/
   ```

3. Train a genre classification model:
   ```bash
   python clsfr.py -t -m model/ features.csv
   ```

4. Extract features from the other WAV files:
   ```bash
   python mftr.py -o new_features.csv track1.wav track2.wav track3.wav
   ```

5. Predict genres for new data:
   ```bash
   python clsfr.py -p -m model/ new_features.csv
   ```
