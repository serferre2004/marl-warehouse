# custom_play.py  
import gymnasium as gym  
import rware  
from pathlib import Path
import sys
REPO_DIR = Path("./robotic-warehouse")  
sys.path.insert(0, str(REPO_DIR))  
import human_play  
  
# --- Tu escenario personalizado ---  
LAYOUT = """  
x.x.x.x.x
x.x.x.x.x
x.x.x.x.x
x.x.x.x.x
x.x.x.x.x
x.x.x.x.x
x.x.x.x.x
x.x.x.x.x
.........
..g...g..
"""  
  
ENV_ID = "rware-tiny-2ag-v2"  
MAX_STEPS = 500  
  
  
def main():  
    # Guardar el gym.make original  
    original_make = gym.make  
  
    # Wrapper que inyecta el layout al crear el entorno  
    def make_with_layout(env_id, **kwargs):  
        kwargs["layout"] = LAYOUT  
        return original_make(env_id, **kwargs)  
  
    # Parchear gym.make dentro del módulo human_play  
    human_play.gym.make = make_with_layout  
    try:  
        human_play.InteractiveRWAREEnv(  
            env=ENV_ID,  
            max_steps=MAX_STEPS,  
            display_info=True,  
        )  
    finally:  
        human_play.gym.make = original_make  
  
  
if __name__ == "__main__":  
    main()