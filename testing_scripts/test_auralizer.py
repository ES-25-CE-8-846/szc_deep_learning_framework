from data_processing import auralize_sound
import numpy as np 
from scipy.io import wavfile

if __name__ == "__main__":
    dz_rirs = np.load("./testing_scripts/dz_rir.npy").transpose(1,0,2)


    rate, sound = wavfile.read("./testing_scripts/relaxing-guitar-loop-v5-245859.wav")
    print(dz_rirs.shape)
    print(np.array(sound).shape)
    for speaker in dz_rirs:
        sound_auralizer = auralize_sound.AuraFarmer(max_length=10000, rirs = speaker)
        length = speaker.shape[-1]
        print(sound.shape)
        auralized_sound = sound_auralizer.auralize(sound[:length,0])
        print(auralized_sound)
        

    

