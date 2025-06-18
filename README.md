![Demo](basketballdemo.gif) 

---

# Why settle for a coach who can't track every shot, angle, or posture in milliseconds?

We're bringing elite-level basketball coaching to anyone.

As avid basketball fans, me and Amal were discussing ways to integrate analysis through computer vision for raw basketball footage—the kind that would track the basics and provide feedback. Wouldn't it be nice to see feedback on your shots, giving a better idea on how to improve your shooting mechanics? So we came up with this idea of using pose estimation to analyze footage of people shooting.

After seeing an eerily similar post on LinkedIn and with the growing impact of computer vision, we got started building our own system. The post we came across was a prompt-based approach that took advantage of SOTA models like Gemini Pro, asking for "Michael Jordan" type feedback on shots. We were more intrigued, however, by how to build a feedback model without abstracting as much control, and also being able to detect more minor changes mathematically in shot form.

## What We Used
- **OpenCV** for real-time video processing
- **MediaPipe** for precise pose estimation
- **Custom YOLO model** for basketball and hoop detection
- **Real-time inference** for instant feedback
- **State machine logic** for shot phase detection
- **Transfer learning** to fine-tune YOLO for basketball-specific objects
- **Kinematic analysis** for joint angle measurement
- **Biomechanical scoring** for actionable feedback
- **Trajectory analysis** for shot success detection

Using OpenCV for realtime video processing and MediaPipe for pose estimation, we were able to get the foundation set up. A custom trained YOLO model helped for basketball and hoop recognition.

The basketball detection proved more challenging than expected—varying lighting conditions, different ball colors, and complex backgrounds required us to train a specialized model rather than rely on generic object detection.

---

## What We Built
We were able to get a basic functional version running, where we were tracking the ball, the net, and the person shooting.

### Phase 1: Object Tracking
Our first milestone was simultaneously tracking three key elements:
- The basketball throughout its trajectory
- The hoop/net position
- The shooter's body positioning


### Phase 2: Form Analysis
We then decided to implement form feedback based on the pose estimation and angles between major points of the limbs before, during, and after a shot. Based on a range of pre-defined angles, we calculated a score and a simple reasoning for the validity.

We then implemented biomechanical analysis by:
- Calculating joint angles at key points (shoulder, elbow, wrist, knee)
- Analyzing form before, during, and after shot release
- Generating scores based on optimal shooting mechanics
- Providing actionable feedback for improvement
- Using vector mathematics and coordinate geometry for spatial analysis



---

## The Results
Our system now delivers:
- Real-time pose detection and angle measurement
- Instant shooting form scoring
- Specific feedback on areas for improvement
- Frame-by-frame kinematic analysis
- Automated field goal percentage (FG%) calculation
- Modular, extensible codebase for future features

---

## What's Next
This project has been incredibly rewarding—combining our love for basketball with cutting-edge computer vision technology to help us become better shooters. We're now working toward our ultimate goal: creating a comprehensive AI basketball coach that can analyze any and all training footage (not limited to shooting) and provide personalized coaching insights.

The intersection of sports and AI continues to fascinate us, and we're excited to keep pushing the boundaries of what's possible with computer vision in athletic training.

---

## Acknowledgements

We would like to thank [Avi Shah](https://github.com/avishah3) for the his [basketball and hoop detection model](https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker), which we adapted and integrated into our system. His work served as a valuable foundation for our shot form analysis project.

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/amaldgregory/basketball_computervision.git
   cd basketball_computervision
   ```
2. **Set up a Python virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download or place your video file in the project directory**
   - Example: `aryan_demo.mp4`
5. **Run the analyzer**
   ```bash
   python testing.py --video aryan_demo.mp4
   ```
   - Or, to use your webcam:
   ```bash
   python testing.py
   ```
6. **View results**
   - The video window will show real-time overlays for shot detection, pose, angles, and feedback.
   - The terminal will print form scores and reasoning after each detected shot.

**Note:**
- Make sure you have a GPU or a reasonably fast CPU for real-time inference.
- The custom YOLO model (`best.pt`) should be present in the project directory.
- For best results, use clear, well-lit basketball footage.

--- 
