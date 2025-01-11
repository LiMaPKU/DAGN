
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']      # one input: L

    if model == 'fbcnn':
        from models.model_fbcnn import ModelFBCNN as M

    elif model == 'matar':
        from models.model_matar import ModelMATAR as M

    # elif model == 'gan':     # one input: L
    #     from models.model_gan import ModelGAN as M

    elif model == 'baseline':
        from models.model_baseline import ModelBaseline as M
    elif model == 'dagn':
        from models.model_dfgn import ModelDFGN as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
