import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
ax.set_xlim(0, 11)
ax.set_ylim(0, 6.5)
ax.axis("off")

def box(x, y, w, h, text, fc, ec, fs=9.5, fw="bold", tc="#1d1d1d"):
    b = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.6, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, fontweight=fw, color=tc, wrap=True)
    return (x + w/2, y, x + w/2, y + h)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#555555", linewidth=1.6))

# Row 1: Sensing
p1 = box(0.3, 5.0, 2.1, 1.0, "3-Lead ECG\nElectrodes\n(Patient)", "#fde8e8", "#c1121f")
p2 = box(2.8, 5.0, 2.1, 1.0, "AD8232 / ADS1292R\nECG Analog Front-End", "#fde8e8", "#c1121f")

# Row 1b: MCU
p3 = box(5.6, 5.0, 2.3, 1.0, "ESP32 Microcontroller\n(ADC sampling @500Hz)", "#e6f6f4", "#2a9d8f")

# Row 2: transmission + processing
p4 = box(5.6, 3.3, 2.3, 1.0, "WiFi/Serial\nData Transmission", "#e6f6f4", "#2a9d8f")
p5 = box(8.3, 3.3, 2.3, 1.0, "Python ML Backend\nFilter -> Features -> Classifier", "#eaf0fb", "#3a5a9b")

# Row 3: results back + display
p6 = box(5.6, 1.6, 2.3, 1.0, "Prediction Result\n(Normal/Arrhythmia/etc.)", "#eaf0fb", "#3a5a9b")
p7 = box(2.8, 1.6, 2.3, 1.0, "OLED/TFT Display\n+ LED + Buzzer Alert", "#fff4e0", "#e9a941")
p8 = box(0.3, 1.6, 2.1, 1.0, "Patient / Doctor\nView Result", "#f3f0fb", "#7b5ea7")

arrow(2.4, 5.5, 2.8, 5.5)
arrow(4.9, 5.5, 5.6, 5.5)
arrow(6.75, 5.0, 6.75, 4.3)
arrow(7.9, 3.8, 8.3, 3.8)
arrow(8.3, 3.5, 6.75, 3.5)
ax.annotate("", xy=(6.75, 2.6), xytext=(6.75, 3.3),
            arrowprops=dict(arrowstyle="-|>", color="#555555", linewidth=1.6))
arrow(5.6, 2.1, 5.1, 2.1)
arrow(2.8, 2.1, 2.4, 2.1)

ax.text(6.75, 4.55, "raw samples", ha="center", fontsize=7.5, color="#555555", style="italic")
ax.text(8.05, 4.0, "filtered\nsignal", ha="center", fontsize=7.5, color="#555555", style="italic")
ax.text(7.55, 3.15, "class label", ha="center", fontsize=7.5, color="#555555", style="italic")
ax.text(6.75, 2.95, "result", ha="center", fontsize=7.5, color="#555555", style="italic")

ax.set_title("CardioSight — Hardware & Data Flow Block Diagram", fontsize=14, fontweight="bold", pad=15)

plt.tight_layout()
plt.savefig("/home/claude/CardioSight/assets/hardware_block_diagram.png", bbox_inches="tight", facecolor="white")
print("saved hardware_block_diagram.png")
