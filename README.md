# LLM-based Pick-and-Place Manipulation in Pybullet
Pybullet-based environment and methods for controlling a robot via natural language instructions. The agent uses a Large Language Model (LLM) to decompose the user's instruction in a sequence of pick-and-place steps that can achieve the final goal. The system uses [CLIP](https://openai.com/research/clip) Vision-Language Model (VLM) for zero-shot object recognition, the [Llama-3](https://ai.meta.com/blog/meta-llama-3) LLM for planning, and the [GR-ConvNet](https://github.com/skumra/robotic-grasping) grasp synthesis model for grasping objects from the [YCB](https://www.ycbbenchmarks.com/) dataset

## Installation
The code has been tested in `python3.8`. Create a virtual environment `
```
python3 -m venv <your_venv_name>`
source <your_venv_name>/bin/activate
pip install --upgrade pip setuptools wheel
```

Then, install `torch` for your own CUDA driver from [here](https://pytorch.org/get-started/locally/). For example, for latest `torch` with CUDA driver 11.8:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Finally, run
```
pip install -r requirements.txt
```
Finally, you will have to download the pretrained Gr-ConvNet model from the original repo, (e.g. [here](https://github.com/skumra/robotic-grasping/tree/master/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32) for RGB-D model pretrained in Cornell). You could also just copy-paste the model from your lab assignment. Create a folder `checkpoints` in the repo's root directory and place it there.

### LLM access
You will need to get a user access token to the Hugging Hub from [here](https://huggingface.co/docs/hub/security-tokens). This allows to utilize the `InferenceClient` service of HuggingFace Hub in order to call different LLMs (we use a distilled Llama-3.2 -- under the `meta-llama/Llama-3.2-3B-Instruct` model tag), which offers a good compromise of performance and response time. Set your token in `llm_utils.py`

If you want to use the GPT series, you need to get an OpenAI api key like in [here](https://www.maisieai.com/help/how-to-get-an-openai-api-key-for-chatgpt) (first 18 dollars for free, then pay per token usage). Follow instructions from OpenAI's [documentation](https://platform.openai.com/docs/quickstart?context=python) to replace Hugging Hub Inference API with OpenAI web api. 

## Example Usage
This repo contains utilities for loading the robot and objects, integrating GR-ConvNet for robot primitive actions (e.g. pick object, place in table region, put in tray etc.), integrating CLIP for zero-shot object recognition and using the Hugging Hub Inference API for text generation with a Llama-3 LLM. You can find example usage for each of these components in notebooks under `examples`. 
Links:
* [Environment and robot primitives](https://github.com/gtziafas/cognitive_robotics_LLM_based_manipulation/blob/main/examples/example_robot_primitives.ipynb)
* [CLIP for zero-shot recognition](https://github.com/gtziafas/cognitive_robotics_LLM_based_manipulation/blob/main/examples/example_clip_recognition.ipynb)
* [Querying LLMs for task planning](https://github.com/gtziafas/cognitive_robotics_LLM_based_manipulation/blob/main/examples/example_LLM_queries.ipynb)
* [Interactive demo](https://github.com/gtziafas/cognitive_robotics_LLM_based_manipulation/blob/main/examples/example_interactive_demo.ipynb)

## Demo
You can directly run a demo with `python3 demo.py`. Regarding the implemented UI:

Types of instructions:
* <ins>put `[obj`] in `[region]`</ins>, e.g: *"put the banana in the top right corner"*
* <ins>put `[obj]` in the tray</ins>, e.g: *"put the scissors in the tray"*
* <ins>put `[obj]` on the `[rel]` of `[obj]`</ins>, e.g.: *"put the hammer left of the tomato soup can"*

1. <ins>Available regions</ins>: *top/bottom/right/left side, top/bottom left/right corner, middle*
2. <ins>Available relations</ins>: *left/right,behind/front, on*
3. <ins>Available objects</ins>: open-vocabulary object descriptions (within CLIP capabilities)

Use `visualize_clip=True` to view the CLIP recognition predictions.

<ins>Chatbot UI</ins> The UI uses the history of instructions as extra context to the LLM. That makes our agent respond to instructions like: **now put the other fruit ..."*, *"undo the previous step"*, *"actually, I want it on ..."*, where the LLM picks up the object/location of interest from the chat history. 

Control the UI:
* `:reset` will reset the robot and the scene to initial state
* `:new` will generate a new scene (have to pass `seed=None` when starting the demo for this)
* `:clear` Clear the chat history.
* `:exit` Exit the demo
* `<your language instruction here>`: give an instruction for the robot to complete

Known issues:
* The robot thinks it has failed the placing part sometimes and attempt to redo it while it has succeeded
* CLIP mis-classifications, check the example notebook for details.
* The longer the interaction, the more chat history in the prompt and the LLM might start ignoring the prompt examples.
* Current prompt sometimes lead to confusion between spatial relations and table regions. You can experiment with different prompts by changing the appropriate text file in `llm_utils.py`

## Notes
* The current implemtation uses ground-truth segmentation masks from Pybullet. If you want to replace that with the [SAM](https://github.com/facebookresearch/segment-anything) model, check out `sam_utils.py` for its usage. You will have to implement the logic for integrating SAM inside `ui.py`. Follow the comments that indicate SAM integration.
* The current implementation follows the prompt template from [Socratic Models](https://socraticmodels.github.io/), which uses the VLM as an open-vocabulary object detector, and can be found in `llm_utils.prompt_pick_and_place_detection`. We also provide another structure (check `llm_utils.prompt_pick_and_place_grounding`) which uses CLIP for referring object grounding and a Python-like plan from the LLM. You will have to integrate this logic inside `ui.py` if you want to switch to that.
* Running many experiments may leed to exceeding the rate quota from Hugging Hub. I believe the rate limit is refreshed hourly.

Please remember to make a branch based on your group ID for pushing changes.
