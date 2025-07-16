

from src.YOLOV5.YOLOV5 import YOLOV5



## --- PDLPR --- ##
from src.PDLPR.training import PDLPR_training
from src.PDLPR.inference import PDLPR_inference






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
    train_folder = r"C:\Users\Lorenzo\Desktop\Computer_Vision_\dataset\CCPD2019\ccpd_base"
    PDLPR_training(train_folder, batch_size=32, num_epochs=3)




    # ------ inference ------ #

    print("PDLPR Inference ...")
    test_folder = r"C:\Users\Lorenzo\Desktop\Computer_Vision_\dataset\CCPD2019\ccpd_weather"
    PDLPR_inference(test_folder, batch_size=64)