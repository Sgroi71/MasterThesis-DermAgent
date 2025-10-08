import yaml
import os
import torch
from DermAgent.tools.SwinModelMultiTask import SwinModelMultiTask
from torchvision import transforms


def transform_factory(dataset_type, dataset_mode):
    """Create the appropriate transform.
        dataset_type: classification | segmentation
        dataset_mode: train | valid | test
    """

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    original_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    norm_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if dataset_mode == "valid":
        if dataset_type == "classification":
            return val_transforms
        else:
            raise ValueError(f"dataset_type={dataset_type} and mode={dataset_mode} not supported")
    elif dataset_mode == "test":
        if dataset_type == "classification":
            return val_transforms
        elif dataset_type == "explanation":
            return original_transform,norm_transform
        else:
            raise ValueError(f"dataset_type={dataset_type} and mode={dataset_mode} not supported")
    
    else:
        raise ValueError(f"transform for mode={dataset_mode} not supported")


def create_image_model(args, restore_type, state_dict):
    """
    Create and return an image model based on the provided arguments.
    Args:
        args (dict): Configuration parameters for the model.
        restore_type (str): Type of model restoration ('last' or 'best').
        state_dict (dict): State dictionary for model weights.
    Returns:
        nn.Module: The created image model.
    """
    # restore_type = last | best
    #print("Creating SwinModelMultiTask model ...")
    pretrained = args['swin_pretrained']
    if args.get('eval', False):
        pretrained = False
        #print(">>> swin_pretrained is FALSE")
    model = SwinModelMultiTask(
        dataset_info=args['dataset_info'], #args['num_classes'],
        pretrained=pretrained 
    )
    # restore saved model
    if state_dict:
        model.load_state_dict(state_dict)
        #print("*** Successfully restored model from state_dict")
    else:
        #restore_last_model(args, model, restore_type)
        print(f"*** Failed to restore model from state_dict")

    return model
def model_factory(args, training_mode, restore_type='last', state_dict=None):
    """ Create and return a model based on the training mode.
    Args:
        args (dict): Configuration parameters for the model.
        training_mode (str): Training mode ('image', 'metadata', 'multimodal').
        restore_type (str, optional): Type of model restoration ('last' or 'best'). Defaults to 'last'.
        state_dict (dict, optional): State dictionary for model weights. Defaults to None.
    Returns:
        nn.Module: The created model."""
    if training_mode == 'image':
        return create_image_model(args, restore_type, state_dict)
    elif training_mode == 'metadata':
        #return create_metadata_model(args, restore_type, state_dict)
        ...
    elif training_mode == 'multimodal':
        #return create_multimodal_model(args, restore_type, state_dict)
        ...
    else:
        raise ValueError(f"Unknown training mode: {training_mode}")

def get_param(param_name, args):
    """ Utility function to get a parameter from args with error handling.
    Args:
        param_name (str): Name of the parameter to retrieve.
        args (dict): Dictionary containing parameters.
    """
    param_value = args.get(param_name)
    if not param_value:
        raise ValueError(f"Missing required parameter: '{param_name}'")
    return param_value

def load_checkpoint(config_path, training_mode='image'):
    """ Load a model checkpoint from the specified configuration file.
    Args:
        config_path (str): Path to the configuration YAML file.
        training_mode (str, optional): Training mode ('image', 'metadata', 'multimodal'). Defaults to 'image'.
    Returns:
        nn.Module: The loaded model.
    """
    with open(config_path, 'r') as f:
        args = yaml.safe_load(f)
    args['training_mode'] = training_mode
    args['restore_last_saved_model'] = True
    args['eval'] = True

    
    save_dir = get_param('save_dir', args) 
    res_tag = get_param('experiment_tag', args) 
    training_mode = get_param('training_mode', args) 
    device = get_param('device', args)
    
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"In loading checkpoint: Directory '{save_dir}' does not exist.")
    
    model_path = os.path.join(save_dir, f'model_{training_mode}_best-{res_tag}.checkpoint')
    #print("Loading checkpoint", model_path)
    if device == 'cuda':
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    # Verify that the required keys exist
    required_keys = {'model_state_dict', 'dataset_info'}
    if not required_keys.issubset(checkpoint.keys()):
        missing_keys = required_keys - checkpoint.keys()
        raise KeyError(f"Checkpoint is missing required keys: {missing_keys}")

    args['dataset_info'] = checkpoint['dataset_info']
    model = model_factory(args, args['training_mode'], 'best', checkpoint['model_state_dict'])

    return model

