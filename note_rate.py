# 351 Project Algorithm
# Importing all required libraries
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sys

# # # # # # # # # # # # # # # # # # # # All Functions are written below this line # # # # # # # # # # # # # # # # # # # # # # # # # #

# Global variables: Tempo and Pitch score
tempo_score = 0
pitch_score = 0

# Plotting the spectrogram of an audio file
def spectrogram(y, sr):
    fig, ax = plt.subplots()
    D_highres = librosa.stft(y, hop_length=128, n_fft=4096)
    S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
    img = librosa.display.specshow(S_db_hr, hop_length=256, x_axis='time', y_axis='log',
                               ax=ax)
    ax.set(title='Higher time and frequency resolution')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    return plt.show()

# Chromogram of audio file
def chromagram(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    ax.set(title='Chromagram demonstration')
    fig.colorbar(img, ax=ax)
    return plt.show()

# Chroma correlation matrix, not used for our program
def chroma_correlation(y1, sr1, y2, sr2):
    chroma1 = librosa.feature.chroma_cqt(y=y1, sr=sr1)
    chroma2 = librosa.feature.chroma_cqt(y=y2, sr=sr2)
    ccov = np.cov(chroma1, chroma2)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(ccov, y_axis='chroma', x_axis='chroma',
                               key='Eb:maj', ax=ax)
    ax.set(title='Chroma covariance')
    fig.colorbar(img, ax=ax)
    return plt.show()

# Spectral correlation matrix, not used for our program
def correlation_matrix(y1, sr1, y2, sr2):
    chroma1 = librosa.feature.chroma_cqt(y=y1, sr=sr1)
    chroma2 = librosa.feature.chroma_cqt(y=y2, sr=sr2)
    R1 = librosa.segment.recurrence_matrix(chroma1, mode='affinity')
    # R2 = librosa.segment.recurrence_matrix(chroma2, mode='affinity')
    fig, ax = plt.subplots()
    img = librosa.display.specshow(R1, y_axis='time', x_axis='time', ax=ax)
    ax.set(title='Recurrence / self-similarity')
    fig.colorbar(img, ax=ax)
    return plt.show()

# Chroma correlation matrix plotting both audio files in terms of chroma cross similarity
def stefs_cor(y1, sr1, y2, sr2):
    hop = 1024
    chroma_ref = librosa.feature.chroma_cqt(y=y1, sr=sr1, hop_length=hop)
    chroma_comp = librosa.feature.chroma_cqt(y=y2, sr=sr2, hop_length=hop)
    # Use time-delay embedding to get a cleaner recurrence matrix DO WE NEED THIS MAYBE
    x_ref = librosa.feature.stack_memory(chroma_ref, n_steps=10, delay=3)
    x_comp = librosa.feature.stack_memory(chroma_comp, n_steps=10, delay=3)
    xsim_aff = librosa.segment.cross_similarity(x_comp, x_ref, metric='cosine', mode='affinity')
    fig, ax = plt.subplots()
    imgaff = librosa.display.specshow(xsim_aff, x_axis='s', y_axis='s',\
                          cmap='magma_r', hop_length=hop, ax=ax)
    ax.set(title='Affinity recurrence')
    fig.colorbar(imgaff, ax=ax, orientation='horizontal')

    return plt.show()

# Plots Spectrograms and then the main beats in different colours of both audio signals
def stefs_beat(y1, sr1, h1, y2, sr2, h2):
    onset_env1 = librosa.onset.onset_strength(y=y1, sr=sr1, aggregate=np.median)
    tempo1, beats1 = librosa.beat.beat_track(onset_envelope=onset_env1, sr=sr1)
    print("Piece 1 Tempo thru Onset Envelope is: %02d BPM" % (tempo1))
    
    fig, ax = plt.subplots(nrows=2, sharex = True)
    times = librosa.times_like(onset_env1, sr=sr1, hop_length=h1)
    M1 = librosa.feature.melspectrogram(y=y1, sr=sr1, hop_length=h1)
    librosa.display.specshow(librosa.power_to_db(M1, ref=np.max), y_axis='mel', \
                                    x_axis='time', hop_length=h1, ax=ax[0])
    
    onset_env2 = librosa.onset.onset_strength(y=y2, sr=sr2, aggregate=np.median)
    tempo2, beats2 = librosa.beat.beat_track(onset_envelope=onset_env2, sr=sr2)
    print("Piece 2 Tempo thru Onset Envelope is: %02d BPM" % (tempo2))

    M2 = librosa.feature.melspectrogram(y=y2, sr=sr2, hop_length=h2)
    librosa.display.specshow(librosa.power_to_db(M2, ref=np.max), y_axis='mel', \
                                    x_axis='time', hop_length=h2, ax=ax[1])

    ax[0].label_outer()
    ax[0].set(title='Mel spectrogram')
    plt.figure()
    plt.plot(times, librosa.util.normalize(onset_env1), label='Onset strength')
    plt.plot(times, librosa.util.normalize(onset_env2), label='Onset strength')
    plt.vlines(times[beats1], 0, 1, alpha=0.5, color='g', linestyle='--', label='Beats')
    plt.vlines(times[beats2], 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')
    # ax[2].legend(loc='upper right')
    plt.title("Onset Strength (Solid) Beats (Dashed)")
    plt.xlabel("Time (s)")
    plt.ylabel("On Set Strength Magnitude")
    
    # Printing the delta of each beat of the two vectors of beats with respect to time
    scoring = stefs_beat_algo(beats1, beats2, times)
    plt.show()
    return scoring

# Helper that plots the beat delta and reviews the offset
def stefs_beat_algo(lebeat1, lebeat2, letime):
    plt.figure()
    deltabeat = []
    beat_review = []
    xaxis = []
    total_beats = 0
    beats_on_time = 0 # Perfectly synchronized
    half_beats = 0 # Less than 0.02s error
    off_beats = 0 # More than 0.02s error
    if len(letime[lebeat1]) <= len(letime[lebeat2]):
        total_beats = len(letime[lebeat1])
        for i in range(len(letime[lebeat1])):
            delta = letime[lebeat1][i] - letime[lebeat2][i]
            if(abs(delta) <= 0.02 and abs(delta) != 0):
                half_beats += 1
            elif(abs(delta) == 0):
                beats_on_time += 1
            else:
                off_beats += 1
            deltabeat.append(delta)
            xaxis.append(i)
            if (delta < 0):
                beat_review.append('Youre too fast in beat %02d young lad' % (i))
            elif (delta > 0):
                beat_review.append('Youre too slow in beat %02d mate' % (i))
            else:
                beat_review.append('Good job at beat %02d Bonzo' % (i))
    elif len(letime[lebeat2]) < len(letime[lebeat1]):
        total_beats = len(letime[lebeat2])
        for i in range(len(letime[lebeat2])):
            delta = letime[lebeat2][i] - letime[lebeat1][i]
            if(abs(delta) <= 0.02 and abs(delta) != 0):
                half_beats += 1
            elif(abs(delta) == 0):
                beats_on_time += 1
            else:
                off_beats += 1
            deltabeat.append(delta)
            xaxis.append(i)
            if (delta < 0):
                beat_review.append('Youre too fast in beat %02d young lad' % (i))
            elif (delta > 0):
                beat_review.append('Youre too slow in beat %02d mate' % (i))
            else:
                beat_review.append('Good job at beat %02d Bonzo' % (i))
    # Loop to print beat review
    for l in range(len(beat_review)):
        print('%s\n' % (beat_review[l]))
    # Print final score 
    score = ((0.5*half_beats)+beats_on_time) / total_beats
    print("Your weighted tempo score is = {:.2%}".format(score))
    # Plot the delta of each beat in the piece
    plt.title('Beat Offset Delta in Seconds')
    plt.xlabel('# of Beats')
    plt.ylabel('Δt in Seconds')
    plt.stem(xaxis, deltabeat)

    return score

# Segments the audio piece by identifying the silent segments in-between notes
def segmenting(y1, y2, hop1, hop2):
    divyup1 = librosa.effects.split(y1, top_db=15, hop_length=hop1)
    divyup2 = librosa.effects.split(y2, top_db=15, hop_length=hop2)
    # print(len(divyup1))
    # print(len(divyup2))
    siggy1 = []
    siggy2 = []
    # Seems as though the smaller divyup size has better success_rate
    if len(divyup1) <= len(divyup2):
        for i in range(len(divyup1)):
            this1 = y1[slice(divyup1[i, 0], divyup1[i, 1])]
            this2 = y2[slice(divyup1[i, 0], divyup1[i, 1])]
            siggy1.append(this1)
            siggy2.append(this2)
    else:
        for i in range(len(divyup2)):
            this1 = y1[slice(divyup2[i, 0], divyup2[i, 1])]
            this2 = y2[slice(divyup2[i, 0], divyup2[i, 1])]
            siggy1.append(this1)
            siggy2.append(this2)
        # TEST BOTH ABOVE AND BELOW CODES TO EVALUATE WHICH IS MORE ACCURATE
    # for i in range(len(divyup2)):
    #     this2 = y2[slice(divyup2[i, 0], divyup2[i, 1])]
    #     siggy2.append(this2)
    return siggy1, siggy2

# Selects what hop length to use dependent on the bpm of each piece
def hop_detector(y, sr):
    onset_env1 = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env1, sr=sr)
    if(tempo <= 0 or tempo <= 90):
        hop = 512
    else:
        hop = 256
    return hop

# Calculate the frequency of each segment of the piece
def note_at_seg(siggy, sr):
    length = len(siggy)
    w = int(np.floor(length/2))-1 
    freq = np.linspace (0,1,length)*sr
    S = np.abs(np.fft.fft(siggy)/length)[:w]
    notefreq = freq[S.argmax()]
    note = librosa.hz_to_note(notefreq)
    return note

# Creates an array of all the notes for the ideal piece
def get_ideal_notes(siggy, sr):
    notes_ideal = []
    for i in range(len(siggy)):
       note = note_at_seg(siggy[i], sr)
       notes_ideal.append(note) 
    return notes_ideal

# Creates an array of all the notes for the practice piece
def get_practice_notes(siggy, sr):
    notes_real = []
    for i in range(len(siggy)):
       note = note_at_seg(siggy[i], sr)
       notes_real.append(note) 
    return notes_real

# Prints two columns of each note side by side for comparison and calculates the success rate
def note_comp(sig1, sig2, sr1, sr2):
    notes1 = get_ideal_notes(sig1, sr1)
    notes2 = get_practice_notes(sig2, sr2)
    review = {'Ideal Notes': notes1,
            'Practice Notes': notes2}
    for each_row in zip(*([i] + (j)
                      for i, j in review.items())):
        print(*each_row, "                 ")
    # Calculating overall score
    correct_notes = 0
    missed_notes = 0
    total_notes1 = len(notes1)
    # total_notes2 = len(notes1)
    # Fix so that the larger total notes will be the reference signal
    for i in range(total_notes1):
        if (notes1[i] == notes2[i]):
            correct_notes += 1
        elif (notes1[i] != notes2[i]):
            missed_notes +=1
    success_rate = (correct_notes / total_notes1)
    # elif total_notes1 > total_notes2:
    #     for i in range(total_notes2):
    #         if (notes2[i] == notes1[i]):
    #             correct_notes += 1
    #         elif (notes1[i] != notes2[i]):
    #             missed_notes +=1
    #     success_rate = (correct_notes / total_notes2)
    print('Successful Note Rate of {:.2%} Percent'.format(success_rate))
    return success_rate


# Pass in the midi file as notes: NOT USED
def get_em_midis(mid):
    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
    for msg in track:
        print(msg)
    # notes = []
    notes = librosa.midi_to_note(mid)
    # return print(notes)

# If pieces are of different size, selects the shorter one to use as the sample duration for windowing, when passing in the files
def find_dur(f1, f2):
    if len(f1) <= len(f2):
        return len(f1)
    else:
        return len(f2)

# Weighting the score for tempo and pitch
def weighting(tempo, pitch):
    # Both have equal weights:
    weighted_score = 0.25*tempo + 0.25*pitch
    weighted_out = print("The weighted output score for pitch and tempo of this demo is = {:.2%} Percent".format(weighted_score))
    return weighted_out