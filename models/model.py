from models.protonet_model.mlp import MLPProto

def get_model(P, modelstr):
    """
    Provides the model based on the specified model type and parameters.

    Args:
        P: Parameters for the setup.
        modelstr (str): The type of model to be created.

    Returns:
        nn.Module: The created model.
    """

    # Check if the model type is 'mlp'
    if modelstr == 'mlp':
        # Check if 'protonet' is in the mode
        if 'protonet' in P.mode:
            if P.dataset == 'income':
                # Create an MLPProto model for the Income dataset
                model = MLPProto(105, 1024, 1024)
            elif P.dataset == 'diabetes':
                # Create an MLPProto model for the Diabetes dataset
                model = MLPProto(8, 1024, 1024)  # Example: Input dimension is 8 (number of features)
            elif P.dataset == 'dna':
                # Create an MLPProto model for the DNA dataset
                model = MLPProto(360, 1024, 1024)  # Input dimension is 360 (after One-Hot-Encoding)
            else:
                raise ValueError(f"Unknown dataset: {P.dataset}")
    else:
        # Raise an error if the model type is not implemented
        raise NotImplementedError()

    return model  # Return the created model
