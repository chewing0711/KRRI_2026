def sound_main():

    audio_1s = record_audio()
    preprocessed_sound_1s = preprocess_sound(audio_1s)
    sound_model_output = sound_preprocess(preprocessed_sound_1s)
    
    return sound_model_output

