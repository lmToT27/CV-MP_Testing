# 🎵 Sáo Mèo (Hmong Flute) Simulator

<p align="center">
  <b>English</b> &nbsp;|&nbsp; <a href="./README.vi.md">Tiếng Việt</a>
</p>

---

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)](https://google.github.io/mediapipe/)

Recreating the soulful sound of the **Sáo Mèo** (Hmong Flute) using computer vision and hand gestures.

## 🏔 About the Instrument
**Sáo Mèo** is a unique wind instrument characteristic of the H'Mong ethnic group in the mountainous regions of Northern Vietnam. Unlike standard bamboo flutes, the Sáo Mèo features a **metal reed** (lưỡi gà) at the mouthpiece. This mechanism creates a distinctive timbre that is buzzing, warm, resonant, and mimics the human voice. Traditionally, H'Mong men use this flute for courtship ("calling a partner") on moonlit nights.

This project uses **PyAudio** to synthesize the physics of the reed sound and **MediaPipe** to translate traditional fingering techniques into digital controls.

## 🎮 How to Play

### 🖐 Left Hand: Octave & Accidental Control
The left hand acts as the breath control and register keys (switching octaves).

| Component | Gesture | Effect |
| :--- | :--- | :--- |
| **Thumb** | **Open** | Play Flat note (**$\flat$**) |
| | **Closed** | Play Natural note |
| **Other Fingers** | **0 fingers open** | **Octave 2** (Low/Deep) |
| | **1 finger open** | **Octave 3** (Mid) |
| | **2 fingers open** | **Octave 4** (High) |
| | **...** | **...** |

### ✋ Right Hand: Melody (Notes)
The right hand controls the pitch by splitting the octave into two parts, using the thumb as a toggle switch.

#### 1. Thumb (Shift Key)
The thumb acts as a modifier to switch between low and high notes within the scale.
* **Closed:** Play the **first 4 notes** of the scale (Do, Re, Mi, Fa).
* **Open:** **Shift up by a 5th** (+3.5 tones) to play the higher notes (Sol, La, Si...).

#### 2. Finger Map

| Fingers Open | Thumb **CLOSED** (Lower) | Thumb **OPEN** (Higher) |
| :---: | :---: | :---: |
| **1** | **C** (Do) | **G** (Sol) |
| **2** | **D** (Re) | **A** (La) |
| **3** | **E** (Mi) | **B** (Si) |
| **4** | **F** (Fa) | **C** (Do - Next Octave) |

> **Mechanism:** This system allows you to play a full diatonic scale (7 notes) using only 4 fingers by toggling the Thumb position.

## 🛠 Installation & Run

You don't need to install Python or complex libraries. Simply download and run the executable file.

> **⚠️ Important:** Ensure you have **[Git LFS](https://git-lfs.github.com/)** installed to download the `.exe` file correctly. Without Git LFS, you will only download a broken 1KB pointer file.

```bash
# 1. Clone the repository
git clone [https://github.com/lmToT27/CV-MP_Testing.git](https://github.com/lmToT27/CV-MP_Testing.git)

# 2. Navigate to the folder
cd CV-MP_Testing

# 3. Run the application
# (On Windows)
.\dist\main.exe