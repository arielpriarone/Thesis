from math import remainder
from gtts import gTTS
import os
from rich import print
import datetime as dt
from mutagen.mp3 import MP3
from pydub import AudioSegment
import io
import numpy as np
from pydub import AudioSegment

def delete_files(file_list):
    for file in file_list:
        try:
            os.remove(file)
            print(f"File '{file}' deleted successfully.")
        except OSError as e:
            print(f"Error deleting file '{file}': {e}")


def join_mp3_files(input_files, output_file):
    # Initialize an empty AudioSegment object to store the combined audio
    combined_audio = AudioSegment.empty()

    # Iterate over each input file
    for file in input_files:
        # Load the MP3 file
        audio_segment = AudioSegment.from_mp3(file)
        # Append the loaded audio to the combined audio
        combined_audio += audio_segment

    # Export the combined audio to an MP3 file, overwriting if it already exists
    combined_audio.export(output_file, format="mp3", codec="libmp3lame")

class time:
    def __init__(self, hh=0, mm=0, ss=0):
        self.ss = ss % 60 # seconds
        self.mm = (mm + ss//60) % 60 # minutes
        self.hh = (hh + mm//60) % 24 # hours
    def __str__(self):
        out = ""
        if self.hh > 0:
            out += str(self.hh) + " hours "
        if self.mm > 0:
            out += str(self.mm) + " minutes "
        if self.ss > 0:
            out += str(self.ss) + " seconds"
        return out
    def __repr__(self):
        return str(self)

def generate_silent_mp3(duration_seconds, output_file_path="silent.mp3"):
    # Create a silent audio segment
    silence = AudioSegment.silent(duration=duration_seconds * 1000)  # duration in milliseconds

    # Save the silent audio segment to a file
    silence.export(output_file_path, format="mp3")

    
def get_mp3_duration(file_path):
    try:
        audio = MP3(file_path)
        duration_in_seconds = audio.info.length
        return duration_in_seconds
    except Exception as e:
        print("Error:", e)
        return None


def text_to_speech(text, filename="speech.mp3"):
    tts = gTTS(text=text, 
               lang='en',
               tld='com',
               slow=False)
    tts.save(filename)
    #os.system(f"start {filename}")

if __name__ == "__main__":
    i=0
    
    print("Please enter the intervals in seconds:")
    try:
        interval = int(input())
    except ValueError:
        print("Please enter a valid number.")
    print("Please enter the number of repetitions:")
    try:
        repetitions = int(input())
    except ValueError:
        print("Please enter a valid number.")
    
    text = "This timer will track " + str(repetitions) + " repetitions of " + str(interval) + " seconds. Starting in 3,2,1,now."
    
    filenames = [f"primer{i}.mp3" for i in range(1, 2*repetitions + 3)]
    text_to_speech(text, filenames[i]); i+=1

    to_say = [time(ss=t) for t in np.array(range(1, repetitions + 1))*interval]
    print(to_say)
    
    # wait for interval seconds
    generate_silent_mp3(interval,filenames[i]); i+=1

    for say in to_say:
        print(say)
        text_to_speech(str(say), filenames[i]); i+=1
        if say != to_say[-1]:
            wait = interval - get_mp3_duration(filenames[i-1])
            if wait < 0:
                delete_files(filenames)
                raise ValueError("There is not enough time to say the time. Please increase the interval or reduce the repetition.")
            print("Wait for", wait, "seconds.")
            generate_silent_mp3(wait, filenames[i]); i+=1

    text_to_speech("Time's up!", filenames[i]); i+=1

    join_mp3_files(filenames[0:i], "timer.mp3")

    delete_files(filenames)
    