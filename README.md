# ASA-dataset  
GitHub repository for **ASA: An Auditory Spatial Attention Dataset with Multiple Speaking Locations**  

🔗 **Dataset Link:** [Zenodo](https://zenodo.org/uploads/11541114)  


🔗 **Paper Link:** [Interspeech2024](https://www.isca-archive.org/interspeech_2024/lin24f_interspeech.pdf)

## 📌 About  
This repository provides an **EEG dataset** and the corresponding **source code** for **Auditory Spatial Attention Decoding (ASAD)** research.  

### 📂 **Contents**  
- **64-channel EEG data**: Responses to **two-speaker** speech stimuli from **10 different locations** (±90°, ±60°, ±45°, ±30°, and ±5°).  
- **Baseline model (CA-CNN)** for ASAD, along with other neural network models.  
- **Preprocessing, training, and testing scripts** for model evaluation.  
- **Visualization script** for dataset analysis.  

## 📖 **Reference**  
For a detailed description of the dataset, refer to the paper:  
📝 *"ASA: An Auditory Spatial Attention Dataset with Multiple Speaking Locations"*  

## 🛠 **Setup Guide**  
1️⃣ **Download & Unzip** all subjects' EEG data into your `asa_data/` directory.  
2️⃣ **Modify Path Settings** in `"main.py"` to align with your local setup.  
3️⃣ **Ensure Python Environment Compatibility**:  
   - Python **3.10.8**  
   - TensorFlow **2.13.0**  
   - MNE **1.5.0**  
   - NumPy **1.23.5**  
   - SciPy **1.10.1**  
   - Matplotlib **3.7.1**  
4️⃣ **Run Analysis** by executing `"main.py"`.  

## 📊 **Results**  
Analysis results will be saved as:  
- `"results_***.txt"` (detailed output)  
- `"averages_***.txt"` (aggregated metrics)  

