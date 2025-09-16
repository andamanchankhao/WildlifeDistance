Wildlife Distance Tool
Ever wanted to know how far away that deer is in your trail cam photo? Well, now you can! This is a cool desktop app that lets you train your own AI to figure out the distance to wildlife in your pictures.

The app has two main parts, so you can choose what you need to do:

The Annotation Tool: This is where you teach the AI. You'll draw boxes around animals in your photos and tell the app how far away they are in real life.

The Calculator Tool: Once your AI is trained, you can use this tool to load it up, click on an animal in a new photo, and get an instant distance estimate!

Cool Features
Super Smart AI: It uses a fancy AI model (called a DPT) to understand the 3D layout of your photos, then uses another model that you train to get super accurate distances!

One-Click Training: Seriously, just click a button! The app does all the hard work and spits out a trained model file that you can use anytime.

Easy Peasy Calculations: Once your model is trained, just load it up, click on an animal, and boom! Instant distance.

Save Your Results: You can easily save all your distance calculations into a CSV file to use in a spreadsheet or for other projects.

So, How Does It Work?
It's a neat two-step trick!

1. First, It Sees in 3D!
A big, powerful AI called a DPT looks at your photo. It creates something called a "depth map," which is basically a cool grayscale version of your image where bright spots are close and dark spots are far away. It figures out the 3D structure of the scene!

2. Then, You Teach It About "Meters"
That depth map doesn't know about meters or feet. That's where you come in! When you annotate your photos, you're teaching a second, smaller AI. You're telling it, "Hey, a spot with this shade of gray in the depth map equals this many meters in real life."

So, when you go to calculate a distance later, the app just looks at the depth map's gray value, asks your little AI what it means, and gives you the answer! Pretty cool, right?

Downloads
Ready to give it a try? You can grab the latest version for Mac or Windows right from the releases page. No complicated setup needed!

>> Get it Here! <<

(Just remember to change YourUsername/YourRepo to the actual URL of the GitHub repository!)

How to Get Started
Head to the download link above.

Grab the .zip file for your computer (macOS or Windows).

Unzip it!

Double-click the app to run it. That's it!

Mac Users! The first time you open the app, your computer might get a little scared. Just right-click the app icon and choose "Open" to let it know it's all good.

Credits
Built by: Andaman Chankhao

App Help & CI/CD Setup: Gemini