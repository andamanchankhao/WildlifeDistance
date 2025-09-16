================================ Wildlife Distance Tool
A desktop application for annotating wildlife images, training a distance-prediction model, and calculating the distance to wildlife using computer vision.

This tool is split into two main components:

Annotation & Training Tool: An interface for drawing bounding boxes on images, assigning known distances, and training a custom machine learning model from your data.

Calculator Tool: A simple interface to load a pre-trained model and instantly calculate the distance to an animal by clicking on it in an image.

Key Features
AI-Powered: Uses a state-of-the-art DPT (Dense Prediction Transformer) model for depth estimation, coupled with a regression model for accurate distance prediction.

Automated Training: A simple one-click training process that saves a portable model (.keras) and a data scaler (.joblib) file.

Point-and-Click Calculation: Load your trained model and get instant distance predictions in a user-friendly interface.

Data Export: All predictions from the calculator can be exported to a CSV file for further analysis.

How It Works: The AI Model
The application uses a two-stage AI process to accurately estimate distance:

Stage 1: Depth Estimation with DPT
When an image is processed, it is first analyzed by a powerful, pre-trained model called the Dense Prediction Transformer (DPT). This model creates a detailed depth map of the entire scene—a grayscale image where brighter pixels represent objects that are closer to the camera and darker pixels represent objects that are farther away.

Stage 2: Distance Prediction
The depth map itself doesn't provide a measurement in meters. This is where your custom-trained model comes in.

When you annotate, you provide the real-world distance for a specific area. The system records the corresponding depth value from the DPT map at that spot.

During training, a simple regression model learns the relationship between these DPT depth values and the actual distances you provided.

When you calculate, the application gets the depth value from your clicked point, feeds it to your trained model, which then predicts the final distance in meters.

This two-step process allows the application to leverage the power of a large, general-purpose AI for understanding the scene's structure, while still being customizable with your own data for precise measurements.

Downloads
You can download the latest pre-compiled, standalone versions for macOS and Windows from the official GitHub Releases page.

>> Download Here <<

(Note: Please replace YourUsername/YourRepo with the actual URL of your GitHub repository.)

How to Use
Go to the Downloads link above and find the latest release.

Download the .zip file for your operating system (WildlifeTool-macOS.zip or WildlifeTool-Windows.zip).

Unzip the file.

Double-click the WildlifeTool.app (on Mac) or WildlifeTool.exe (on Windows) to run the launcher. No installation is needed.

On macOS, you may need to right-click the application and select "Open" the first time you run it to bypass the security warning.

Credits
Developer: Andaman Chankhao

CI/CD & Application Structure: Gemini