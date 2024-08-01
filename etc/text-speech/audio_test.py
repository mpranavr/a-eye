from IPython.display import Audio

# Replace with the path to a local audio file
local_audio_path = "C:/Users/manda/Downloads/amma_parade.mp3"

# Read the local audio file as bytes
try:
    with open(local_audio_path, 'rb') as f:
        local_audio_bytes = f.read()
    print("Audio Playing")
    # Play the local audio file
    Audio(local_audio_bytes, autoplay=True)
    
except FileNotFoundError:
    print(f"File '{local_audio_path}' not found.")
except Exception as e:
    print(f"Error playing audio: {e}")
