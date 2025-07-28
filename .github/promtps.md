Read all the repository to get an idea of the current status of the repository.

Keeping copilot-instructions.md in mind, lets implement the visualisation with empty processing functions for now - one separate file per processing feature. Visualisation related code should be cleanly separated intothe gui folder. For now implement the following tab:
1) Main tab:
- in a left bar, a list of all available video files in data\dev_data and the raw/videos folders. Make sure you add a column showing the length of the video. Add a button to update this list by searching the folders.
- in the center, display the currently selected video. Load a random video by default
- controls should be at the bottom, right under the video: 
    - previous recording (loops to last recording if pressed while on first video)
    - play/pause
    - next recording (loops to first recording if pressed while on last video)
    - check boxes to enable processing and visualisation of inference (using trained models to detect players and discs), object tracking, player identification (based on jersey numbers), field segmentation
    - controls should all have corresponding keyboard shortcuts