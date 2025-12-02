from SaoMeoEngine import SaoMeoEngine
import numpy as np
import pyaudio
import time
import threading

"""
    Fixing the "click" issue when transferring notes (melody/chord) in real-time.

    This is a real-time Additive Synthesizer combined with a Digital Mixer.
    It generates audio waveforms mathematically at the exact moment of playback 
    (Real-time DSP), rather than playing pre-recorded audio samples.

    The Fixes:

    1. Anti-click / gain caching mechanism:
       The Issue: Previously, when a note was stopped (removed from the active 
        melody/chord list), its gain would instantly drop to 0.0, truncating 
        the natural decay (Fade Out/Release phase). This caused audible "clicks" 
        or "pops".
        The Solution (cached_gains): The system now caches the last known volume 
        (gain) of a note. When a note is stopped, the system retrieves this 
        cached value to ensure the Fade Out process completes smoothly until 
        total silence.

    2. thread safety
        Implements `threading.Lock()` to synchronize the Main Thread (where 
        logic/UI updates notes) and the Audio Callback Thread (where waves are 
        calculated).
        This prevents race conditions where the note list is modified while 
        being read, ensuring the program never crashes randomly.

    ---------------------------------------------------------------------------
    DSP PIPELINE (Inside Callback):

    B1. Thread Locking: Ensures data integrity during calculation.
    B2. Envelope (ADSR): Calculates volume curves (Fade In/Out) for smooth 
        transitions.
    B3. Gain Logic: Determines channel gain (Melody = 1.0 / Chord = 0.3).
        If the note is fading out, retrieves gain from Cache.
    B4. Synthesis: 
        Wave = Sin(f) + 0.5 * Sin(2f) + 0.08 * Sin(3f) ... 
        (Simulates Sao Meo timbre using Harmonic series).
    B5. Effects: Applies Vibrato (LFO) and Envelope.
    B6. Mixing & Mastering: Sums all signals and applies Soft Clipping (tanh) 
        to add warmth and prevent digital distortion.
"""

class SaoMeoMixer(SaoMeoEngine):
    def __init__(self):
        self.melody_freqs = set()
        self.chord_freqs = set()
        self.chord_volume_ratio = 0.8
        
        self.cached_gains = {} 
        
        self.lock = threading.Lock()
        
        super().__init__() 

    def callback(self, in_data, frame_count, time_info, status):
        dt = 1.0 / self.sample_rate
        t = np.arange(frame_count) * dt
        output_signal = np.zeros(frame_count)
        
        with self.lock:
            processing_freqs = list(self.envelopes.keys())
            all_targets = self.melody_freqs.union(self.chord_freqs)
            
            if not processing_freqs:
                self.lfo_phase = 0
                return (output_signal.astype(np.float32), pyaudio.paContinue)

            for freq in processing_freqs:
                current_env = self.envelopes[freq]
                target_env = 1.0 if freq in all_targets else 0.0
                
                env_curve = np.zeros(frame_count)
                if target_env > current_env:
                    step = 1.0 / self.attack_samples
                    env_curve = np.minimum(current_env + np.arange(1, frame_count+1)*step, 1.0)
                else:
                    step = 1.0 / self.release_samples
                    env_curve = np.maximum(current_env - np.arange(1, frame_count+1)*step, 0.0)
                
                self.envelopes[freq] = env_curve[-1]
                
                if self.envelopes[freq] <= 0.0 and freq not in all_targets:
                    del self.envelopes[freq]
                    if freq in self.phases: del self.phases[freq]
                    if freq in self.cached_gains: del self.cached_gains[freq]
                    continue 
                
                current_gain = 0.0
                
                if freq in self.melody_freqs:
                    current_gain = 1.0
                elif freq in self.chord_freqs:
                    current_gain = max(current_gain, self.chord_volume_ratio)
                else:
                    current_gain = self.cached_gains.get(freq, 1.0)
                
                self.cached_gains[freq] = current_gain
                
                if freq not in self.phases: self.phases[freq] = 0
                
                vibrato = 1.0 + 0.01 * np.sin(self.lfo_phase + 2 * np.pi * 5.0 * t)
                
                phase_inc = 2 * np.pi * freq * dt
                chunk_phases = self.phases[freq] + np.cumsum(np.full(frame_count, phase_inc))
                
                wave = 1.0 * np.sin(chunk_phases)
                wave += 0.5 * np.sin(chunk_phases * 2)
                wave += 0.08 * np.sin(chunk_phases * 3)
                wave += 0.02 * np.sin(chunk_phases * 4)

                wave *= vibrato
                wave *= env_curve
                wave *= current_gain
                
                output_signal += wave
                self.phases[freq] = chunk_phases[-1] % (2 * np.pi)

            self.lfo_phase += 2 * np.pi * 5.0 * (frame_count * dt)
            self.lfo_phase %= 2 * np.pi
            
        output_signal *= self.volume
        output_signal = np.tanh(output_signal)
        
        return (output_signal.astype(np.float32), pyaudio.paContinue)

    def set_melody(self, freqs):
        with self.lock:
            self.melody_freqs = set(freqs)
            self._sync_envelopes_unsafe()
        
    def set_chords(self, freqs):
        with self.lock:
            self.chord_freqs = set(freqs)
            self._sync_envelopes_unsafe()
        
    def _sync_envelopes_unsafe(self):
        all_freqs = self.melody_freqs.union(self.chord_freqs)
        for freq in all_freqs:
            if freq not in self.envelopes:
                self.envelopes[freq] = 0.0

if __name__ == "__main__":
    print("Testing Sao Meo Mixer (Melody + Chords)...")
    mixer = SaoMeoMixer()

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

    song_data = [
        (['C4'], [], 0.95), (['Rest'], [], 0.05),
        (['C4'], ['C3'], 0.5), (['C4'], ['G3'], 0.5), (['G4'], ['C4'], 0.25), (['F4'], ['C4'], 0.25), (['E4'], ['G3'], 0.25), (['F4'], ['G3'], 0.25), # C
        (['G4'], ['C3'], 0.5), (['G4'], ['G3'], 0.5), (['A4'], ['C4'], 0.5), (['G4'], ['G3'], 0.45), (['Rest'], ['G3'], 0.05), # C
        (['G4'], ['B2'], 0.5), (['G4'], ['D3'], 0.5), (['E4'], ['G3'], 0.5), (['E4'], ['D3'], 0.25), (['D4'], ['D3'], 0.20), (['Rest'], ['D3'], 0.05), # G/B
        (['E4'], ['A2'], 0.5), (['E4'], ['E3'], 0.4), (['Rest'], ['E3'], 0.1), (['E4'], ['A3'], 0.25), (['D4'], ['A3'], 0.25), (['E4'], ['E3'], 0.25), (['G4'], ['E3'], 0.25), # Am
        (['C4'], ['F2'], 0.5), (['C4'], ['E3'], 0.25), (['E4'], ['E3'], 0.25), (['D4'], ['F3'], 0.5), (['C4'], ['E3'], 0.5), # F
        (['G3'], ['G2'], 0.5), (['G3'], ['D3'], 0.5), (['G3'], ['G3'], 0.5), (['G3'], ['D3'], 0.5), # G
        (['G4'], ['C3'], 0.5), (['G4'], ['G3'], 0.25), (['F4'], ['G3'], 0.25), (['E4'], ['C4'], 0.5), (['F4'], ['G3'], 0.5), # C
        (['G4'], ['B2'], 0.5), (['G4'], ['D3'], 0.5), (['E4'], ['G3'], 0.5), (['E4'], ['D3'], 0.25), (['D4'], ['D3'], 0.25), # G/B
        (['E4'], ['A2'], 0.5), (['E4'], ['E3'], 0.4), (['Rest'], ['E3'], 0.1), (['E4'], ['A3'], 0.25), (['D4'], ['A3'], 0.25), (['E4'], ['E3'], 0.25), (['G4'], ['E3'], 0.25), # Am
        (['C4'], ['F2'], 0.5), (['C4'], ['E3'], 0.4), (['Rest'], ['E3'], 0.1), (['C4'], ['F3'], 0.25), (['E4'], ['F3'], 0.25), (['D4'], ['E3'], 0.25), (['C4'], ['E3'], 0.25), # F
        (['G3'], ['G2'], 0.5), (['G3'], ['D3'], 0.5), (['G3'], ['G3'], 0.5), (['G3'], ['D3'], 0.5), # G
        (['A3'], ['F2'], 0.5), (['C4'], ['E3'], 0.45), (['Rest'], ['E3'], 0.05), (['C4'], ['F3'], 0.5), (['D4'], ['E3'], 0.25), (['E4'], ['E3'], 0.25), # F
        (['E4'], ['C3'], 0.5), (['E4'], ['G3'], 0.5), (['E4'], ['C4'], 0.5), (['D4'], ['G3'], 0.5), # C
        (['E4'], ['C3'], 0.5), (['E4'], ['G3'], 0.4), (['Rest'], ['G3'], 0.1), (['E4'], ['C4'], 0.45), (['Rest'], ['C4'], 0.05), (['E4'], ['G3'], 0.25), (['D4'], ['G3'], 0.25), # C
        (['C4'], ['A2'], 0.5), (['C4'], ['E3'], 0.25), (['D4'], ['E3'], 0.25), (['E4'], ['A3'], 0.25), (['D4'], ['A3'], 0.25), (['E4'], ['E3'], 0.25), (['G4'], ['E3'], 0.25), # Am
        (['C4'], ['F2'], 0.5), (['G3'], ['E3'], 0.5), (['G3'], ['F3'], 0.5), (['C4'], ['E3'], 0.5), # F
        (['G3'], ['G2'], 0.5), (['C4'], ['D3'], 0.5), (['E4'], ['G3'], 0.45), (['Rest'], ['G3'], 0.05), (['E4'], ['D3'], 0.25), (['D4'], ['D3'], 0.25), # G
        (['C4'], ['C3'], 0.5), (['C4'], ['G3'], 0.5), (['C4'], ['E3'], 0.5), (['C4'], ['G3'], 0.5), # C
        (['E4'], ['C3'], 0.5), (['G4'], ['G3'], 0.5), (['C5'], ['C4'], 0.5), (['D5'], ['G3'], 0.5), # C
        ([], ['E5', 'G5', 'C6'], 2)
    ]

    print("Playing Beo Dat May Troi with chords...")
    try:
        for mel_names, chord_names, duration in song_data:
            mel_freqs = [notes.get(n, 0) for n in mel_names]
            chord_freqs = [notes.get(n, 0) for n in chord_names]
            
            mixer.set_melody([f for f in mel_freqs if f > 0])
            mixer.set_chords([f for f in chord_freqs if f > 0])
            
            print(f"Melody: {mel_names} | Chord: {chord_names}")
            time.sleep(duration)
            
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        mixer.close()
        print("Mixer closed.")