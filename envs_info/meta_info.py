
offline_dataset_names = {
    "Hopper-v2": "hopper-medium-expert-v2",
    "HalfCheetah-v2": "halfcheetah-medium-expert-v2",
    "Walker2d-v2": "walker2d-medium-expert-v2",
}

min_max_rewards = {
    "Hopper-v2": [-20.272305, 3234.3],
    "HalfCheetah-v2": [-280.178, 12135.0],
    "Walker2d-v2": [1.629008, 4592.3],
}

env_state_contexts = {
    "Hopper-v2": "Observations consist of positional values of different body parts of the hopper, followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities. Details of the current state are as follows: \n",
    "HalfCheetah-v2": "Observations consist of positional values of different body parts of the cheetah, followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities. \n",
    "Walker2d-v2": "Observations consist of positional values of different body parts of the walker, followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities. \n",
}




env_task_descriptions = {
    "Hopper-v2": "The environment aims to increase the number of independent state and control variables as compared to the classic control environments. The hopper is a two-dimensional one-legged figure that consist of four main body parts - the torso at the top, the thigh in the middle, the leg in the bottom, and a single foot on which the entire body rests. The goal is to make hops that move in the forward (right) direction by applying torques on the three hinges connecting the four body parts. \n",
    "HalfCheetah-v2": "The HalfCheetah is a 2-dimensional robot consisting of 9 links and 8 joints connecting them (including two paws). The goal is to apply a torque on the joints to make the cheetah run forward (right) as fast as possible, with a positive reward allocated based on the distance moved forward and a negative reward allocated for moving backward. The torso and head of the cheetah are fixed, and the torque can only be applied on the other 6 joints over the front and back thighs (connecting to the torso), shins (connecting to the thighs) and feet (connecting to the shins). \n",
    "Walker2d-v2": "The walker is a two-dimensional two-legged figure that consist of four main body parts - a single torso at the top (with the two legs splitting after the torso), two thighs in the middle below the torso, two legs in the bottom below the thighs, and two feet attached to the legs on which the entire body rests. The goal is to make coordinate both sets of feet, legs, and thighs to move in the forward (right) direction by applying torques on the six hinges connecting the six body parts. \n",
}


env_obs_names = {
    "Hopper-v2": [
        "height of hopper",
        "angle of the torso",
        "angle of the thigh joint",
        "angle of the leg joint",
        'angle of the foot joint',
        'velocity of the x-coordinate of the torso',
        'height of the torso',
        'angular velocity of the angle of the torso',
        'angular velocity of the thigh hinge',
        'angular velocity of the leg hinge',
        "angular velocity of the foot hinge"
    ],
    "HalfCheetah-v2": ["z-coordinate of the front tip",
                       "angle of the front tip",
                       "angle of the second rotor",
                       "angle of the second rotor",
                       "velocity of the tip along the x-axis",
                       "velocity of the tip along the y-axis",
                       "angular velocity of front tip",
                       "angular velocity of second rotor",
                       "x-coordinate of the front tip",
                       "y-coordinate of the front tip",
                       "angle of the front tip",
                       "angle of the second rotor",
                       "angle of the second rotor",
                       "velocity of the tip along the x-axis",
                       "velocity of the tip along the y-axis",
                       "angular velocity of front tip",
                       "angular velocity of second rotor"],
    "Walker2d-v2": [
        "height of hopper",
        "angle of the top",
        "angle of the thigh joint",
        "angle of the leg joint",
        "angle of the foot joint",
        "angle of the left thigh joint",
        "angle of the left leg joint",
        "angle of the left foot joint",
        "velocity of the x-coordinate of the top",
        "velocity of the z-coordinate (height) of the top",
        "angular velocity of the angle of the top",
        "angular velocity of the thigh hinge",
        "angular velocity of the leg hinge",
        "angular velocity of the foot hinge",
        "angular velocity of the thigh hinge",
        "angular velocity of the leg hinge",
        "angular velocity of the foot hinge"
        ],
    
}