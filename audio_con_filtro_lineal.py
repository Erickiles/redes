import math
import librosa
import librosa.display
import numpy as np
import pandas as pd
import scipy.signal as signal
from os import listdir, makedirs
from os.path import isfile, join, exists

from filtro_lineal import filter_design
from filtro_wavelet import filtrado
import matplotlib.pyplot as plt  # Agregar esta línea para importar pyplot

def Cargar_Audio(filename):
    '''
    Carga el archivo de audio y aplica un filtro pasabandas FIR, filtrando
    entre 100Hz y 2000Hz
    '''
    y, sr = librosa.load(filename)
    fs = sr
    order, lowpass = filter_design(fs, locutoff=0, hicutoff=2000, revfilt=0)
    order, highpass = filter_design(fs, locutoff=100, hicutoff=0, revfilt=1)
    y_hp = signal.filtfilt(highpass, 1, y)
    y_bp = signal.filtfilt(lowpass, 1, y_hp)
    y_bp = np.asfortranarray(y_bp)
    return y_bp, sr

def Save_Spectrogram(y, sr, filename):
    '''
    Genera y guarda un espectrograma de una señal de audio.

    :param y: Señal de audio
    :param sr: Frecuencia de muestreo de la señal
    :param filename: Nombre del archivo de imagen a guardar (puede incluir la ruta)
    '''
    plt.figure(figsize=(8, 4))
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)  # Corregir llamada a la función
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')  # Ocultar ejes x e y
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)  # Ajustar el área de la imagen al contenido
    plt.close()

Directorio = 'C:\\Users\\Erick\\Downloads\\archive\\Respiratory_Sound_Database\\Respiratory_Sound_Database\\audio_and_txt_files'
Output_Directory = 'C:\\Users\\Erick\\Downloads\\archive\\Respiratory_Sound_Database\\Respiratory_Sound_Database\\espec'  # Ruta de la carpeta donde se guardarán las imágenes

if not exists(Output_Directory):
    makedirs(Output_Directory)

Archivos_Audio = [file for file in listdir(Directorio) if file.endswith(".wav") if isfile(join(Directorio, file))]
Archivos_Texto = [file for file in listdir(Directorio) if file.endswith(".txt") if isfile(join(Directorio, file))]

for i in range(len(Archivos_Audio)):
    print('Procesando ' + str(i + 1) + '/' + str(len(Archivos_Audio)) + " " + Archivos_Audio[i])
    Audio_filtrado, sr = Cargar_Audio(join(Directorio, Archivos_Audio[i]))
    Corazon_Filtrado = filtrado(Audio_filtrado, 0, 1, 2)

    # Generar y guardar el espectrograma de la señal filtrada
    Output_File = join(Output_Directory, Archivos_Audio[i].replace('.wav', '.png'))
    Save_Spectrogram(Corazon_Filtrado, sr, Output_File)