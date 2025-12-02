import numpy as np
import pyaudio
import time

class SaoMeoEngine:
    def __init__(self):
        self.sample_rate = 48000
        self.p = pyaudio.PyAudio()
        self.volume = 0.8
        
        self.target_freqs = set()
        self.envelopes = {} 
        self.phases = {}
        self.lfo_phase = 0 
        
        self.attack_samples = int(self.sample_rate * 0.15) 
        self.release_samples = int(self.sample_rate * 0.3) 
        
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            stream_callback=self.callback,
            frames_per_buffer=1024
        )
        self.stream.start_stream()

    def callback(self, in_data, frame_count, time_info, status):
        dt = 1.0 / self.sample_rate
        t = np.arange(frame_count) * dt
        output_signal = np.zeros(frame_count)
        
        processing_freqs = list(self.envelopes.keys())
        
        if not processing_freqs:
            self.lfo_phase = 0
            return (output_signal.astype(np.float32), pyaudio.paContinue)

        for freq in processing_freqs:
            current_env_val = self.envelopes[freq]
            target_env_val = 1.0 if freq in self.target_freqs else 0.0
            
            env_curve = np.zeros(frame_count)
            
            if target_env_val > current_env_val: 
                step = 1.0 / self.attack_samples
                env_curve = current_env_val + np.arange(1, frame_count + 1) * step
                env_curve = np.minimum(env_curve, 1.0)
            else: 
                step = 1.0 / self.release_samples
                env_curve = current_env_val - np.arange(1, frame_count + 1) * step
                env_curve = np.maximum(env_curve, 0.0)
            
            self.envelopes[freq] = env_curve[-1]
            
            if self.envelopes[freq] <= 0.0 and freq not in self.target_freqs:
                del self.envelopes[freq]
                if freq in self.phases: del self.phases[freq]
                continue 
            
            if freq not in self.phases: self.phases[freq] = 0
            
            vibrato_speed = 5.0
            vibrato_depth = 0.01 
            vibrato = 1.0 + vibrato_depth * np.sin(self.lfo_phase + 2 * np.pi * vibrato_speed * t)
            
            self.lfo_phase += 2 * np.pi * vibrato_speed * (frame_count * dt)
            self.lfo_phase %= 2 * np.pi

            phase_inc = 2 * np.pi * freq * dt
            chunk_phases = self.phases[freq] + np.cumsum(np.full(frame_count, phase_inc))
            
            wave = 1.0 * np.sin(chunk_phases)
            wave += 0.5 * np.sin(chunk_phases * 2)
            wave += 0.08 * np.sin(chunk_phases * 3)
            wave += 0.02 * np.sin(chunk_phases * 4)

            wave *= vibrato
            wave *= env_curve
            
            output_signal += wave
            
            self.phases[freq] = chunk_phases[-1] % (2 * np.pi)

        output_signal *= self.volume
        
        output_signal = np.tanh(output_signal)
        
        return (output_signal.astype(np.float32), pyaudio.paContinue)

    def update_notes(self, active_frequencies):
        self.target_freqs = set(active_frequencies)
        for freq in self.target_freqs:
            if freq not in self.envelopes:
                self.envelopes[freq] = 0.0 
        
    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

if __name__ == "__main__":
    notes = {
        'Rest': 0,
        'C2': 65.41, 'D2': 73.42, 'E2': 82.41, 'F2': 87.31, 'G2': 98.00, 'A2': 110.00, 'B2': 123.47,
        'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61, 'G3': 196.00, 'A3': 220.00, 'B3': 246.94,
        'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00, 'A4': 440.00, 'B4': 493.88,
        'C5': 523.25, 'D5': 587.33, 'E5': 659.25, 'F5': 698.46, 'G5': 783.99, 'A5': 880.00, 'B5': 987.77,
        'C6': 1046.50, 'D6': 1174.66, 'E6': 1318.51, 'F6': 1396.91, 'G6': 1567.98, 'A6': 1760.00, 'B6': 1975.53,
        'C7': 2093.00, 'D7': 2349.32, 'E7': 2637.02, 'F7': 2793.83, 'G7': 3135.96, 'A7': 3520.00, 'B7': 3951.07,
        'C8': 4186.01
    }
    
    # Beo Dat May Troi melody
    melody = [
        ('C4', 0.9), ('Rest', 0.1), ('C4', 1), ('G4', 0.25), ('F4', 0.25), ('E4', 0.25), ('F4', 0.25), ('G4', 1), # Beo dat... may troi
        ('A4', 0.5), ('G4', 0.5), ('G4', 1), # Chon xa xoi...
        ('E4', 0.5), ('D4', 0.5), ('E4', 1), # Anh oi
        ('E4', 0.25), ('D4', 0.25), ('E4', 0.25), ('G4', 0.25), ('C4', 0.5), ('E4', 0.5), ('D4', 0.5), ('C4', 0.5), ('G3', 2), # Em van doi...beo dat
        ('G4', 0.5), ('F4', 0.5), ('E4', 0.5), ('F4', 0.5), ('G4', 1), # May... troi
        ('E4', 0.5), ('D4', 0.5), ('E4', 1), ('E4', 0.25), ('D4', 0.25), ('E4', 0.25), ('G4', 0.25), ('C4', 1), # Chim ca, tang tinh tinh...
        ('C4', 0.25), ('E4', 0.25), ('D4', 0.25), ('C4', 0.25), ('G3', 2), # Ca loi...
        ('A3', 0.5), ('C4', 0.5), ('C4', 0.5), ('D4', 0.25), ('E4', 0.25), ('E4', 1.5), # Ngam mot tin trong...
        ('D4', 0.5), ('E4', 0.9), ('Rest', 0.1), ('E4', 0.5), ('E4', 0.25), ('D4', 0.25), ('C4', 0.75), # Hai tin doi...
        ('D4', 0.25), ('E4', 0.25), ('D4', 0.25), ('E4', 0.25), ('G4', 0.25), ('C4', 0.5), ('A3', 1.0), # Ba bon tin cho...
        ('C4', 0.5), ('A3', 0.5), ('C4', 0.5), ('E4', 0.5), ('E4', 0.25), ('D4', 0.25), ('C4', 2) # Sao chang thay dau...
    ]
    
    print("Testing Sao Meo Engine...")
    engine = SaoMeoEngine()
    
    print("Playing Beo Dat May Troi...")
    try:
        for note, duration in melody:
            print(f"Playing: {note}")
            freq = notes.get(note, 0)
            
            if freq > 0:
                engine.update_notes([freq])
            else:
                engine.update_notes([])
            
            time.sleep(duration)
            
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        engine.close()
        print("Engine closed.")
