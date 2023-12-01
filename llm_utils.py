from huggingface_hub import InferenceApi
import time

hugging_hub_token = None
if hugging_hub_token is None:
	raise ValueError("Get your Hugging Hub API token from here: https://huggingface.co/docs/hub/security-tokens.\nThen, set it in llm_utils.py.")

llm_inference = InferenceApi("bigscience/bloom", token=hugging_hub_token)


def BLOOM(       query,
                 prompt,
                 stop_tokens = None,
                 max_length = 128,
                 temperature=1.,
                 return_full_text = False,
                 verbose = False
):
    
    new_prompt = f'{prompt}\n{query}\n'
    
    params = {
        "max_new_tokens": max_length,
        "top_k": None,
        "top_p": None,
        "temperature": temperature,
        "do_sample": False,
        "seed": 42, #useless
        "early_stopping":None,
        "no_repeat_ngram_size":None,
        "num_beams":None,
        "return_full_text":return_full_text,
        'wait_for_model' : True
    }
    s = time.time()
    response = llm_inference(new_prompt, params=params)
    proc_time = time.time()-s
    if verbose:
        print(f"Inference time {proc_time} seconds")
        
    if isinstance(response, dict):
        assert list(response.keys()) == ['error']
        raise ValueError(f'sth went wrong with prompt {new_prompt}')

    response = response[0]['generated_text']
    #response = response[(response.find(query) + len(query) + 1):]

    if stop_tokens is not None:
        if verbose:
            print('Stopping')
        for stoken in stop_tokens:
            if stoken in response:
                response = response[:response.index(stoken)]


    return response


def BLOOMWrapper(query, context, verbose=False):
    query = query + ('.' if query[-1] != '.' else '') 
    resp = BLOOM(query, prompt=prompt_plan, stop_tokens=['#']).strip()
    resp_obj, resp_full = None, resp
    if 'parse_obj' in resp:
        steps = resp.split('\n')
        obj_query = [s for i, s in enumerate(steps) if 'parse_obj' in s][0].split('("')[1].split('")')[0]
        obj_query = context + '\n' + f'# {obj_query}.'
        resp_obj = BLOOM(obj_query, prompt=prompt_parse_obj, stop_tokens=['#', 'objects = [']).strip()
        resp_full = '\n'.join([resp, '\n' + obj_query, resp_obj])
    if verbose:
        print(query)
        print(resp_full)
    return resp, resp_obj


prompt_pick_and_place_detection = """
objects = ["scissors", "pear", "hammer", "mustard bottle", "tray"]
# put the bottle to the left side.
robot.pick_and_place("mustard bottle", "left side")
objects = ["banana", "foam brick", "strawberry", "tomato soup can", "pear", "tray"]
# move the fruit to the bottom right corner.
robot.pick_and_place("banana", "bottom right corner")
robot.pick_and_place("pear", "bottom right corner")
robot.pick_and_place("strawberry", "bottom right corner")
# now put the green one in the top side.
robot.pick_and_place("pear", "top side")
# undo the last step.
robot.pick_and_place("pear", "bottom right corner")
objects = ["potted meat can", "power drill", "chips can", "hammer", "tomato soup can", "tray"]
# put all cans in the tray.
robot.pick_and_place("potted meat can", "tray")
robot.pick_and_place("chips can", "tray")
robot.pick_and_place("tomato soup can", "tray")
""".strip()


prompt_pick_and_place_grounding = """
from robot_utils import pick_and_place
from camera_utils import find, scene_init
objects = scene_init()
# put the bottle to the left side.
bottle = find(objects, "bottle")[0]
pick_and_place(bottle, "left side")
objects = scene_init()
# move the fruit to the bottom right corner.
fruits = find(objects, "fruit")
for fruit in fruits:
	pick_and_place(fruit, "bottom right corner")
# now put the green one in the top side.
green_fruit = find(fruits, "green")
pick_and_place(green_fruit, "top side")
# undo the last step.
pick_and_place(green_fruit, "bottom right corner")
objects = scene_init()
# put all cans in the tray.
cans = find(objects, "can")
for can in cans:
	pick_and_place(can, "tray")
""".strip()