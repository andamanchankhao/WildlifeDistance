# Wildlife Distance Tool

Ever wanted to know how far away that deer is in your trail cam photo? **Now you can!**  
Wildlife Distance Tool is a desktop application that lets you train your own AI to estimate the distance to wildlife in your images.

---

## 📖 Table of Contents
- [Introduction](#introduction)  
- [Features](#features)  
- [How It Works](#how-it-works)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Annotation Tool](#annotation-tool)  
  - [Calculator Tool](#calculator-tool)  
- [Saving Results](#saving-results)  
- [Downloads](#downloads)  
- [Troubleshooting](#troubleshooting)  
- [Credits](#credits)  
- [License](#license)  

---

## 🐾 Introduction
This app combines depth perception AI with custom user training to provide accurate wildlife distance measurements. It’s perfect for researchers, wildlife enthusiasts, and anyone using trail cameras.

---

## ✨ Features
- **Super Smart AI** – Uses a **DPT model** to understand the 3D structure of your photos.  
- **One-Click Training** – Train your custom distance model with a single click.  
- **Easy Calculations** – Load your trained model, click on an animal, and instantly get a distance estimate.  
- **Result Export** – Save your distance calculations into a CSV file for further analysis.  
- **Cross-Platform** – Available for **macOS** and **Windows**.

---

## 🧩 How It Works
1. **Step 1: Depth Perception**  
   A **DPT (Dense Prediction Transformer)** generates a *depth map* from your photo—a grayscale image where brightness corresponds to distance.  

2. **Step 2: Real-World Calibration**  
   You annotate your photos with real distances, teaching a smaller AI to translate grayscale depth values into actual meters (or feet).  

3. **Step 3: Instant Distance**  
   When analyzing new photos, the app maps depth values to real-world units, giving you quick and reliable distance estimates.

---

## ⚙️ Installation
1. Head to the [Releases page](https://github.com/**YourUsername**/**YourRepo**/releases).  
2. Download the `.zip` for your operating system (**macOS** or **Windows**).  
3. Unzip the file.  
4. Double-click the app to run it. 🎉  

> **Mac Users**: The first launch may be blocked by Gatekeeper. Right-click the app, choose **Open**, and confirm.

---

## 🚀 Usage

### Annotation Tool
- Load your photos.  
- Draw bounding boxes around wildlife.  
- Enter the real-world distance for each annotation.  
- Save annotations to train your custom distance model.  

### Calculator Tool
- Load your trained model.  
- Open a new photo.  
- Click on an animal, and get the distance instantly.  

---

## 💾 Saving Results
- All distance calculations can be exported to **CSV format**.  
- Use the exported data in spreadsheets or other analytical tools.

---

## 📥 Downloads
Grab the latest release here:  
👉 [Download Wildlife Distance Tool](https://github.com/**YourUsername**/**YourRepo**/releases)  

---

## 🛠️ Troubleshooting
- **Mac app won’t open?** – Right-click the app, select **Open**, and confirm.  
- **No results?** – Make sure your model has been trained with at least a few annotated photos.  
- **Strange values?** – Double-check that your annotations have accurate real-world distances.

---

## 👥 Credits
- **Developer:** Andaman Chankhao  
- **App Help & CI/CD Setup:** Gemini  

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).  
