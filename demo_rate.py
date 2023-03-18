# Importing the functions from the note_rate file where the code is written
from note_rate import *

# Instructions on how to setup and run the test demo can be found in the README_py file 

# Passing in the files, for this demo we will be using "guitart1.wav" for ideal and "guitart2.wav" for real
# For simplicity, I have commented these commands out so that you will not need to pass in anything in the terminal
# other than running the program demo_rate.py
# ideal_file = str(input('Insert Ideal Audio File: '))
# practice_file = str(input('Insert Practice Audio File: '))

ideal_file = 'guitart1.wav'
practice_file = 'guitart2.wav'

# Selecting parameters for the functions and analysis, as well as loading the audio as a time series
dur = find_dur(ideal_file, practice_file)
yreal1, sreal1 = librosa.load(ideal_file, duration=10)
yreal2, sreal2 = librosa.load(practice_file, duration=10)
hop_real1 = hop_detector(yreal1, sreal1)
hop_real2 = hop_detector(yreal2, sreal2)

# # # Test run of pitch, tempo and correlation matrix functions # # #

# Chroma correlation matrix
stefs_cor(yreal1, sreal1, yreal2, sreal2)

# Tempo Analysis
tempo_score = stefs_beat(yreal1, sreal1, hop_real1, yreal2, sreal2, hop_real2)

# # # Test of note-to-note analysis # # #

segment_real1, segment_real2 = segmenting(yreal1, yreal2, hop_real1, hop_real2)
pitch_score = note_comp(segment_real1, segment_real2, sreal1, sreal2)

# Weighting both scores by 25%
weighting(tempo_score, pitch_score)
