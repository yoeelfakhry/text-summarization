import yaml 
from pathlib import Path 


def load_config(config_path: str) -> dict:
    """
    Load configuration from a yaml file 

    Args:
        config_path (strt): Path to the yaml configuration file   
    Returns:
        dict: Configuration parameters as a dictionary
    """
    with open(config_path,"r") as file:
        config = yaml.safe_load(file)
        
    config["learning_rate"] = float(config["learning_rate"])  # Convert learning rate to float  
    return config 


if __name__ == "__main__":
    config = load_config("configs/bart.yaml")
    print (config)
    print (type(config['learning_rate']))
    print (type(config['fp16']))