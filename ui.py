import time
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter, HtmlFormatter
import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext
import webbrowser
import tempfile
import streamlit as st

# pprint = lambda s: display(HTML(highlight(s, PythonLexer(), HtmlFormatter(full=True))))
#pprint = lambda s: print(highlight(s, PythonLexer(), TerminalFormatter()).strip())

from env.camera import Camera
from env.env import *
from env.objects import YcbObjects
from grconvnet import load_grasp_generator
from clip_utils import ClipInference
from sam_utils import SamInference
from llm_utils import BLOOM, prompt_pick_and_place_detection

from env.objects import YCB_CATEGORIES as ADMISSIBLE_OBJECTS


ADMISSIBLE_PREDICATES = ["on", "left", "right", "behind", "front"]


# GUI stuff
# Function to create a text input dialog using Tkinter
def ask_for_user_input():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    user_input = simpledialog.askstring("Input", "User Input: ")
    root.destroy()
    return user_input

# Function to display a message with the result
# def display_result(result):
#     root = tk.Tk()
#     root.withdraw()  # Hide the main window
#     formatted_code = highlight(result, PythonLexer(), HtmlFormatter(full=True, style='friendly'))
#     messagebox.showinfo("Plan", formatted_code)
#     root.destroy()
# Function to display a message with the result
# def display_result(result):
#     # Use Pygments to format the Python code
#     formatted_code = highlight(result, PythonLexer(), HtmlFormatter(full=True, style='friendly'))
#     # Create a temporary HTML file to display the result
#     with open("code.html", "w") as f:
#         f.write(formatted_code)
#     # Open the HTML file in the default web browser
#     webbrowser.open("code.html")



class RobotEnvUI:
    
    def __init__(self, 
                 n_objects: int, 
                 n_action_attempts: int = 3,
                 n_grasp_attempts: int = 4,
                 visualise_grasps: bool = False,
                 visualise_clip: bool = False,
                 ground_truth_segm: bool = True,
                 clip_prompt_eng: bool = False,
                 clip_this_is: bool = False,
                 seed=None
):
        # init env
        center_x, center_y, center_z = CAM_X, CAM_Y, CAM_Z
        self.camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (IMG_SIZE, IMG_SIZE), 40)
        self.env = Environment(self.camera, vis=True, asset_root='./env/assets', debug=False, finger_length=0.06)
        
        # constants
        self.TARGET_ZONE_POS = TARGET_ZONE_POS
        self.ADMISSIBLE_OBJECTS = ADMISSIBLE_OBJECTS
        self.ADMISSIBLE_LOCATIONS = list(self.env.TARGET_LOCATIONS.keys()) + ['tray']
        self.ADMISSIBLE_PREDICATES = ADMISSIBLE_PREDICATES

        # load objects
        self.seed = None
        self.objects = YcbObjects('env/assets/ycb_objects',
                    mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                    mod_stiffness=['Strawberry'],
                    seed=self.seed
        )
        self.objects.shuffle_objects()
        self.env.dummy_simulation_steps(10)
    
        # load GR-ConvNet grasp synthesis network
        self.grasp_generator = load_grasp_generator(self.camera)
        self.n_grasp_attempts = n_grasp_attempts

        self.clip_prompt_eng = clip_prompt_eng
        self.clip_this_is = clip_this_is
        self.visualise_clip = visualise_clip
        self.visualise_grasps = visualise_grasps
        
        # define LLM callable and params
        self.LLM = BLOOM
        self.prompt = prompt_pick_and_place_detection
        self.history = []
        self.n_action_attempts = n_action_attempts
        
        # load CLIP for vision-language grounding
        self.clip_model = ClipInference()

        # load object segmentation (groundtruth / SAM)
        self.ground_truth_segm = ground_truth_segm
        if not ground_truth_segm:
            # load SAM for segmentation - check sam_utils.py and https://github.com/facebookresearch/segment-anything
            raise NotImplementedError

        else:
            # load from simulator
            self.segment = lambda im: self.env.camera.get_cam_img()[-1]

        # spawn scene
        self.spawn(n_objects)

    def spawn(self, n_objects):
        self.n_objects = n_objects
        for obj_name in self.objects.obj_names[:self.n_objects]:
            path, mod_orn, mod_stiffness = self.objects.get_obj_info(obj_name)
            self.env.load_isolated_obj(path, obj_name, mod_orn, mod_stiffness)
        self.env.dummy_simulation_steps(10)
        self._step()
        self.init_obj_state = self.obj_state
        self.obj_ids = self.env.obj_ids

    def reset_scene(self, new=False):
        if new:
            self.spawn(self.n_objects)
            return
        self.reset()

    def reset(self):
        assert self.init_obj_state is not None, "Have to spawn once to initialize state"
        self.env.set_obj_state(self.init_obj_state)
        self.env.dummy_simulation_steps(10)
        self._step()
        self.init_obj_state = self.obj_state
        self.obj_ids = self.env.obj_ids

    def _step(self):
        self.env.reset_robot()
        self.env.dummy_simulation_steps(10)
        self.env.update_obj_states()
        #self.env.dummy_simulation_steps(10)
        self.obj_state = self.env.get_obj_states()
        clip_out =  self.run_clip()
        if clip_out is None:
            print("No more objects left, exiting")
            self.env.close()
        else:
            masks, categories, objIds = clip_out
        self.clip_names = categories
        self.setup_grasps(objIds, masks)
        self.env.dummy_simulation_steps(10)

    # run inference with CLIP for zero-shot object recognition
    def run_clip(self, visualise_clip=None):
        visualise_clip = visualise_clip or self.visualise_clip

        img, _, seg = self.camera.get_cam_img()

        # exit if no more objects
        if np.unique(seg).shape[0] <= 2:
            return ValueError

        # segmentation
        if not self.ground_truth_segm:
            # fill in SAM code here
            pass

        # Call CLIP to zero-shot recognize objects
        prompt_categories =  ";".join(self.ADMISSIBLE_OBJECTS)
        clip_out = self.clip_model.get_most_similar(img, seg,
            prompt_categories, 
            mode="object", 
            prompt_engineering=self.clip_prompt_eng,
            this_is=self.clip_this_is,
            show=self.visualise_clip)

        self.obj_name_to_id = {x['category'] : x['objID'] for x in clip_out}
        # self.obj_ctx = "objects = [{}]".format(','.join([x['category'] for x in clip_out]))

        masks = [x['mask'] for x in clip_out]
        objIds = [x['objID'] for x in clip_out]
        categories = [x['category'] for x in clip_out]

        return masks, categories, objIds

    # run inference with GR-ConvNet grasp generator 
    def setup_grasps(self, obj_ids, masks=None, visualise_grasps=None):
        visualise_grasps = visualise_grasps or self.visualise_grasps
        
        rgb, depth, seg = self.env.camera.get_cam_img()    
        
        # @TODO: Alternatively pass masks from SAM for non groundtruth segm here
        if self.ground_truth_segm:
            masks = [seg == obj_id for obj_id in obj_ids]
        else:
            assert masks is not None

        #for obj_id in self.env.obj_ids:
        for obj_id, mask in zip(obj_ids, masks):
            # mask = seg == obj_id
            if obj_id not in self.env.obj_ids:
                continue
            grasps = self.grasp_generator.predict_grasp_from_mask(rgb,
                                                           depth,
                                                           mask,
                                                           n_grasps=self.n_grasp_attempts, 
                                                           show_output=False
            )
            if obj_id not in self.env.obj_ids:
                continue
            self.env.set_obj_grasps(obj_id, grasps)
        
        if visualise_grasps:
            LID =[]
            for obj_id in obj_ids:
                grasps = self.env.get_obj_grasps(obj_id)
                color = np.random.rand(3).tolist()
                for g in grasps:
                    LID = self.env.draw_predicted_grasp(g,color=color,lineIDs=LID)
            
            time.sleep(5)
            self.env.remove_drawing(LID)
            self.env.dummy_simulation_steps(10)

    def parse_predicate(self, obj_id, predicate):
        states = self.obj_state
        state_ids = {int(x['id']): x for x in states}

        target_loc = list(state_ids[obj_id]['pos'])
        target_loc[-1] += 0.3

        if predicate == "on":
            # place on top
            target_loc[-1] += 0.05
        elif predicate == "left":
            # place slightly left
            target_loc[0] += 0.1
        elif predicate == "right":
            # place slightly right
            target_loc[0] -= 0.1
        elif predicate == "behind":
            # place slightly behind
            target_loc[1] -= 0.1
        elif predicate == "front":
            # place slightly front
            target_loc[1] += 0.1

        return target_loc

    def step(self, what, where, how=None):
        '''
        Scripted policy to pick and place with predicted grasps.

        Input:
            - what (int): the objID of object to grasp
            - where (Union[int, str]): either a table location (str) or ID of place object
            - how (str): [on,left,right,behind,in front] - only available if where=objID
        '''
        assert what in self.obj_ids, f"Invalid pick objID, available are: {self.obj_ids}"
        if isinstance(where, str):
            assert where in self.ADMISSIBLE_LOCATIONS, f"Invalid table location, available are {self.ADMISSIBLE_LOCATIONS}"
        elif isinstance(where, int):
            assert what in self.obj_ids, f"Invalid place objID, available are: {self.obj_ids}"
            assert how in self.ADMISSIBLE_PREDICATES, f"Choose one of {self.ADMISSIBLE_PREDICATES}"

        # move to tray
        if where == 'tray':
            target_loc = self.TARGET_ZONE_POS
            success_grasp, success_target = self.env.clean_obj(what)

        # move to table location
        elif isinstance(where, str):
            assert where in self.ADMISSIBLE_LOCATIONS, f"Invalid table location, available are {self.ADMISSIBLE_LOCATIONS}"
            target_loc = where
            success_grasp, success_target = self.env.put_obj_in_loc(what, where)

        # move relative to object
        elif isinstance(where, int):
            assert where in self.obj_ids, f"Invalid place objID, available are: {self.obj_ids}"
            assert how in self.ADMISSIBLE_PREDICATES, f"Choose one of {self.ADMISSIBLE_PREDICATES}"
            target_loc = self.parse_predicate(where, how)
            success_grasp, success_target = self.env.put_obj_in_loc(what, target_loc)

        if success_grasp and success_target:
            print('Success.')
            success = True
        
        elif success_grasp:
            print(f'Could not move {what} to {where}. Retrying...')
            for _ in range(self.n_action_attempts):
                self.env.dummy_simulation_steps(10)
                success_target = self.env.place_in_loc(target_loc)
                if success_target:
                    break
            print(f'Could not move {what} to {where}. Exit.')
            success = False

        else:        
            print(f'Could not grasp {what}. Exit.')            
            success = False
    
        self._step() # update states
        return success

    def pick_and_place(self, what, where, how=None):
        # proxy for step function
        try:
            what = self.obj_name_to_id[what]
        except KeyError:
            print(f"Object {what} not visible. Returning")
            self._step()
            return False
        if where in self.ADMISSIBLE_OBJECTS:
            where = self.obj_name_to_id[where]
        return self.step(what, where, how)

    def get_visual_ctx(self):
        return f"""objects = [{', '.join([f'"{name}"' for name in self.clip_names])}, "tray"]"""

    def run(self):
        
        self.history = self.get_visual_ctx()

        while True:

            # ask user for command
            #user_input = input('User Input: ')
            user_input = ask_for_user_input()
            #user_input = ask_for_user_input_and_display_result()

            # clear history
            if user_input == ':clear':
                self.history = self.get_visual_ctx()
                self.env.dummy_simulation_steps(10)
                continue

            # reset same scene
            elif user_input == ':reset':
                self.reset_scene(new=False)
                self.history = self.get_visual_ctx()
                self.env.dummy_simulation_steps(10)
                continue

            # spawn new scene
            elif user_input == ':new':
                self.reset_scene(new=True)
                self.history = self.get_visual_ctx()
                self.env.dummy_simulation_steps(10)
                continue

            # close demo
            elif user_input == ':exit':
                print('Exitting demo')
                self.env.close()
                break
            
            # actual query
            else: 
                # generate task plan
                print('Calling LLM...')
                query = self.history + '\n' + '# ' + user_input + ('.' if not user_input.endswith('.') else '')
                response = self.LLM(query, prompt=self.prompt, max_length=128, 
                                    stop_tokens=['#', 'objects = [']).strip()
                print()
                #pprint(query + '\n' + response)  
                print(highlight(query + '\n' + response, PythonLexer(), TerminalFormatter()).strip())
                print()
                #formatted_result = highlight(query + '\n' + response, PythonLexer(), HtmlFormatter(style='friendly'))
                #ask_for_user_input_and_display_result(initial_message=user_input, result_message=formatted_result)
                #display_result(query + '\n' + response)

                # parse commands
                step_cmds = response.split('\n')
                n_steps = len(step_cmds)

                # exec one by one
                for step_idx in range(n_steps):
                    step_cmd = step_cmds[step_idx]
                    assert step_cmd.startswith('robot')

                    # Replace 'robot' with 'self' and prepare to execute
                    step_cmd = step_cmd.replace("robot", "self")
                    
                    attempt = 0
                    while attempt < self.n_action_attempts:
                        # Define local and global variables for exec
                        gvars = globals()
                        lvars = locals()
                        exec(f"success = {step_cmd}", gvars, lvars)
                        success = lvars['success']  # Retrieve the result from local variables
                        
                        if success:
                            break
                        attempt += 1
                        print('Attempt failed. Retrying...')
                        for _ in range(30):
                            self.env.step_simulation()
                        self._step()
                        
                    # return if failure
                    if attempt == self.n_action_attempts:
                        print(f'Could not perform action {step_cmd}')
                        time.sleep(1)
                        continue

                # update object states, CLIP predictions and grasps
                self._step()
                    
                # save history
                self.history += '\n# ' + user_input + '\n' + response