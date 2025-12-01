import numpy as np
import pyaudio
import time

class SaoMeoEngine:
    def __init__(self):
        self.sample_rate = 44100
        self.p = pyaudio.PyAudio()
        self.volume = 0.6 
        
        self.target_freqs = set()
        self.envelopes = {} 
        self.phases = {}
        self.lfo_phase = 0 
        
        self.attack_samples = int(self.sample_rate * 0.04) 
        self.release_samples = int(self.sample_rate * 0.06) 
        
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
    print("Testing Sao Meo Engine...")
    engine = SaoMeoEngine()
    
    notes = {
        'Rest': 0,
        'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61, 'G3': 196.00, 'A3': 220.00, 'B3': 246.94,
        'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00, 'A4': 440.00, 'B4': 493.88,
        'C5': 523.25, 'D5': 587.33, 'E5': 659.25, 'F5': 698.46, 'G5': 783.99, 'A5': 880.00, 'B5': 987.77,
        'C6': 1046.50, 'D6': 1174.66, 'E6': 1318.51
    }
    
    melody = [
        ('C4', 0.4), ('D4', 0.4), ('E4', 0.4),
        ('F4', 0.4), ('G4', 0.4), 
        ('A4', 0.8),
        ('C5', 0.4), ('A4', 0.4), ('G4', 0.4), ('E4', 0.8),
        ('D4', 0.4), ('C4', 1.0)
    ]
    
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