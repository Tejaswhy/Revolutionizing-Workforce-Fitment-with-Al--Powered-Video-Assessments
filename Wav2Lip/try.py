import os

# =========================
# INPUT FILES
# =========================

face_path = "/Users/tejasy/Desktop/small.mp4"
audio_path = "/Users/tejasy/Desktop/input.wav"

# =========================
# MODEL CHECKPOINT
# =========================

checkpoint = "checkpoints/wav2lip.pth"

# =========================
# OUTPUT PATH
# =========================

output_path = "results/output.mp4"

# Create results folder if missing
os.makedirs("results", exist_ok=True)

# =========================
# WAV2LIP COMMAND
# =========================
command = f"""
python inference.py \
--checkpoint_path "{checkpoint}" \
--face "{face_path}" \
--audio "{audio_path}" \
--outfile "{output_path}" \
--resize_factor 2 \
--wav2lip_batch_size 32
"""

# =========================
# RUN
# =========================

print("Running Wav2Lip...\n")

os.system(command)

print(f"\nDone! Video saved at:\n{output_path}")