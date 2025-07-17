

#from src.YOLOV5.YOLOV5 import YOLOV5



## --- PDLPR --- ##
from src.PDLPR.training import PDLPR_training
from src.PDLPR.inference import PDLPR_inference

from src.PDLPR.training_augmentation import PDLPR_training as PDLPR_training_augmentation
from src.PDLPR.inference import all_inference



if __name__ == "__main__":



    #################################
    # --------  DETECTION   --------#
    #################################



    ## poi inseriamo qua la detection con yolo







    #################################
    # -------  RECOGNITION  ------- #
    #################################


    # ------ training ------ #
    print("PDLPR Training ...")
    train_folder = r"C:\Users\Lorenzo\Desktop\Computer_Vision_\dataset_cv\CCPD2019\ccpd_base"
    
    #PDLPR_training(train_folder, batch_size=32, num_epochs=5)
    #PDLPR_training_augmentation(train_folder, num_epochs=50, batch_size=32)



    # ------ inference ------ #

    print("PDLPR Inference ...")
    #test_folder = r"C:\Users\Lorenzo\Desktop\Computer_Vision_\dataset_cv\CCPD2019\ccpd_fn"
    test_folder1 = r"C:\Users\loreb\Desktop\CV\dataset_cv\CCPD2019_extracted\CCPD2019\ccpd_fn"
    PDLPR_inference(test_folder1, batch_size=64)
    
    #base_path = r"C:\Users\loreb\Desktop\CV\dataset_cv\CCPD2019_extracted\CCPD2019"
    #all_inference(base_path, batch_size=64)