import albumentations as A
from albumentations.pytorch import ToTensorV2 

def get_train_aug_transform(mu, sigma):

    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized image.
    """
    

    train_transform = A.Compose([
                             A.RandomCrop(32,32), 
                             A.HorizontalFlip(p=0.5),
                             A.Normalize(mean=(mu), 
                                         std=(sigma)),
                             A.Cutout(num_holes=1, 
                                      max_h_size=8,
                                      max_w_size=8, 
                                      
                                      #fill_value=[0.4914*255, 0.4822*255, 0.4471*255], 
                                      fill_value=(mu),
                                      always_apply=True,
                                      p=0.50),   
                             A.ToGray()   
                             ToTensorV2(),
])

    return(train_transform)



def get_test_aug_transform(mu, sigma):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized image.
        """
    test_transform = A.Compose([
                            A.Normalize(
                                mean=(mu), 
                                std=(sigma)),
                            ToTensorV2(),
])
    return(test_transform)


def no_transform():
    return(A.Compose([A.Normalize()]))
