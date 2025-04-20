import numpy as np
import pickle
import os
from tqdm import tqdm
import argparse

# Import the necessary classes and functions from homework4.py
from homework4 import Hw5Env, bezier

def collect_demonstrations(num_demonstrations=100, time_steps=100, save_path='data/demonstrations.pkl', render_mode="offscreen"):
    """
    Collect demonstrations using the Hw5Env from homework4.py.
    
    Args:
        num_demonstrations: Number of demonstrations to collect
        time_steps: Number of time steps per demonstration
        save_path: Path to save the demonstrations
        render_mode: Rendering mode ("offscreen" for no display, "gui" for visualization)
    """
    print(f"Collecting {num_demonstrations} demonstrations with {render_mode} rendering...")
    demonstrations = []
    heights = []
    
    # Create environment
    env = Hw5Env(render_mode=render_mode)
    
    for i in tqdm(range(num_demonstrations), desc="Collecting demonstrations"):
        # Reset environment for each demonstration
        env.reset()
        
        # Get object height
        height = env.obj_height
        
        # Generate random control points for a Bezier curve
        p_1 = np.array([0.5, 0.3, 1.04])
        p_2 = np.array([0.5, 0.15, np.random.uniform(1.04, 1.4)])
        p_3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
        p_4 = np.array([0.5, -0.3, 1.04])
        points = np.stack([p_1, p_2, p_3, p_4], axis=0)
        curve = bezier(points, steps=time_steps)
        
        # Move robot to start position
        env._set_ee_in_cartesian(curve[0], rotation=[-90, 0, 180], n_splits=100, max_iters=100, threshold=0.05)
        
        # Follow trajectory and collect states
        states = []
        for p in curve:
            env._set_ee_pose(p, rotation=[-90, 0, 180], max_iters=10)
            states.append(env.high_level_state())
        
        # Stack states
        states = np.stack(states)
        
        # Add time dimension
        times = np.linspace(0, 1, time_steps).reshape(-1, 1)
        
        # Combine time and states
        full_states = np.concatenate([times, states], axis=1)
        
        demonstrations.append(full_states)
        heights.append(height)
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save demonstrations to file
    with open(save_path, 'wb') as f:
        pickle.dump({'demonstrations': demonstrations, 'heights': heights}, f)
    
    print(f"Saved {len(demonstrations)} demonstrations to {save_path}")
    
    return demonstrations, heights

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect robot trajectory demonstrations')
    parser.add_argument('--data_path', type=str, default='data/demonstrations.pkl', help='Path to save demonstration data')
    parser.add_argument('--num_demonstrations', type=int, default=100, help='Number of demonstrations to collect')
    parser.add_argument('--time_steps', type=int, default=100, help='Number of time steps per demonstration')
    parser.add_argument('--render_mode', type=str, default='offscreen', help='Rendering mode (offscreen or gui)')
    
    args = parser.parse_args()
    
    collect_demonstrations(
        num_demonstrations=args.num_demonstrations,
        time_steps=args.time_steps,
        save_path=args.data_path,
        render_mode=args.render_mode
    ) 