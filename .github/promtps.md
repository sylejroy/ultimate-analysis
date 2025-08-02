Main Tab


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


## Inference implementation

Read all the repository to get an idea of the current status of the repository.

Keeping copilot-instructions.md in mind, implement the inference processing function and the visualisation function.



## Player ID

Read all the repository to get an idea of the current status of the repository. Read copilot-instructions.md and apply it for all modifications.

Implement the player id functionality. For now it should work as follows:
1) Using the tracks of class "player" use easyocr by default to find jersey numbers in the top third of the bounding box (likely position of jersey number).
2) Display the detected jersey number along with the confidence
3) Use the measurement of easyocr to estimate a fused jersey number based on the easyocr measurement history
4) display fused jersey number with a confidence in a different colour to the one-shot easyocr detection

## Easyocr Tuning Tab

Read all the repository to get an idea of the current status of the repository. Read copilot-instructions.md and apply it for all modifications.

Implement the Easyocr Tuning Tab
- left bar should once again have a recording selection list for all videos available in data\dev_data and the raw/videos folders
- there should be a bar to scroll through the selected recording
- on the frame selected by the user, run inference using a model selected from a drop down list
- on the detection bounding boxes, run pre-processing - cropping, contrast etc. and then run easyocr
- make all parameters user selectable
- add a button to store selected parameters to the defaults.yaml file and to load from the defaults.yaml file

## Performance Display

In the empty space in the bottom left of the main tab, I would like to have live runtime performance metrics displayed

Following should be displayed:
- table containing runtimes for each of the processing functionalities
    - previous frame runtime in ms
    - rolling average runtime in ms
    - maximum runtime in ms
- cpu core usage (if possible specifically by the app) - small graph of past x seconds
- gpu usage (if possible specifically by the app) - small graph of past x seconds

## Easyocr Tuning Tab Runtimes

When the run easyocr button is pressed, I would like to measure the runtime for each pre-processing step as well as for the final easyocr call in order to measure the impact on runtime it has.
Display this data in a table that can be displayed in the top right, next to the video

_____________________________________________________________________________
# Jersey Number Tracking Enhancement

We need to implement probabilistic jersey number tracking for player objects across their entire tracking history. Currently we have single-frame EasyOCR detection, but we need to filter and fuse these noisy measurements into reliable jersey number identification.

## Requirements

### Core Functionality
- **Historical Tracking**: Maintain jersey number measurement history for each tracked player object
- **Probabilistic Fusion**: Combine multiple noisy measurements with confidence scores into probability estimates
- **Spatial Weighting**: Use digit position within bounding box to adjust confidence (center = higher weight)
- **Real-time Updates**: Update probabilities as new measurements arrive

### Technical Implementation
- **Input**: EasyOCR measurements with confidence scores and digit positions
- **Processing**: Calculate probability distribution over possible jersey numbers using:
  - EasyOCR confidence scores
  - Spatial position weighting (lateral centering preference)
  - Measurement history consistency
- **Output**: Top 3 most probable jersey numbers with confidence percentages

### Visualization Requirements
- **Display**: Show top 3 jersey number candidates with probabilities
- **Highlighting**: Visually emphasize the highest probability jersey number
- **Color Coding**: Use different colors to distinguish between single-frame detection and historical fusion

## Architecture Guidelines
- Follow KISS principle (max 500 lines per file)
- Use existing configuration system for parameters (confidence thresholds, weighting factors)
- Maintain clean separation between processing logic and GUI visualization
- Type hints required for all functions

## Key Questions for Implementation
1. What probability update algorithm should we use (Bayesian, weighted average, etc.)?
2. How should spatial position weighting be calculated?
3. What parameters need to be configurable via YAML settings?
4. Should we implement confidence decay over time for old measurements?

Please implement this jersey number tracking enhancement following the project's established patterns and architecture.
_____________________________________________________________________________
## Model Tuning Tab

Read all the repository to get an idea of the current status of the repository. Read copilot-instructions.md and apply it for all modifications.

Make a new tab which will be used to tune models using ultralytics. Currently we need to tune two types of model:
- one for detection, which detects players and discs
- one for field segmentation, which segments the image into endzones and central playing fields

Visualisation:
- Type of task to be done:
    - detection
    - field segmentation
- Base model selection - model to be tuned drop down list:
    - drop down list based on the selected type of task
        - detection uses non "-seg" models found in models/pretrained or maybe even models in the models/detection folder - to further tune models
        - segmentation uses "-seg" models found in models/pretrained or maybe even models in models/segmentation folder
- Training data selection drop down list:
    - if detection task is selected, use raw data from the training dataset: data\raw\training_data. Suggest any data that has object_detection or player disc detection in the name
    - if field segmentation has been selected, "field finder" is the correct dataset
- When a model is selected, show some key pieces of information about the model
- When a training data set is selected, show key pieces of information about the training data set & display an example of some data with labels
- Training parameters:
    - make parameters such as epochs and patience user-editable
    - make a training.yaml file in configs and corresponding buttons "load to config" and "save to config"
- Button to start training of the selected model, based on selected training data, with the selected parameters
- Save the new parameters output by ultralytics in a new folder such as data\models\detection\object_detection_yolo11l\finetune - make sure you don't overwrite any previous data
- Display a progress bar to show how many epochs have been processed, how many are left and based on that estimated time of completion
- Show the current training results, either making your own graphs from results.csv or by using the results.png image in the ultralytics output folder