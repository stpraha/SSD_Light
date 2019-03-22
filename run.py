# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:20:29 2019

@author: Stpraha
"""
import argparse
import train_ssd
import test_ssd

def parser_arguments():
    """
        Parse the command line arguments of the program.
    """
    parser = argparse.ArgumentParser(description='Train or test the model')
    
    parser.add_argument("--train",
                        action = "store_true",
                        help = "Define if we train the model"
                        )
    
    parser.add_argument("--test",
                        action = "store_true",
                        help = "Define if we test the model"
                        )
    
    parser.add_argument(
                        "-bs",
                        "--batch_size",
                        type = int,
                        nargs = "?",
                        help = "Size of a batch",
                        default = 32
                        )
    
    parser.add_argument("-mp", 
                        "--model_path", 
                        type = str,
                        nargs = "?",
                        help = "The path to the file containing the saved model",
                        default = './save/'
                       )

    parser.add_argument("-ip", 
                        "--image_path", 
                        type = str,
                        nargs = "?",
                        help = "The path to the file containing the images",
                        default = '/home/cxd/emotions/JPEGImages/'
                       )
    
    parser.add_argument("-ap", 
                        "--annotation_path", 
                        type = str,
                        nargs = "?",
                        help = "The path to the file containing the annotations",
                        default = '/home/cxd/emotions/Annotations/'
                       )
    
    parser.add_argument("-op", 
                        "--out_path", 
                        type = str,
                        nargs = "?",
                        help = "The path to save the test result",
                        default = './out/'
                       )
    
    parser.add_argument("-e", 
                        "--epochs", 
                        type = int,
                        nargs = "?",
                        help = "How many iteration to train",
                        default = 1500
                       )
    
    parser.add_argument("-lr", 
                        "--learning_rate", 
                        type = int,
                        nargs = "?",
                        help = "Learning rate",
                        default = 0.001
                       )
    
    parser.add_argument("-r", 
                        "--restore", 
                        action = "store_true",
                        help = "Whether the saved model be restored"
                       )

    return parser.parse_args()

def main():
    """
        Entry
    """
    args = parser_arguments()
    
    if not args.train and not args.test:
        print("You should input --train to train or --test to test.")
    
    if not args.image_path:
        print("You should input --image_path to specify the image path.")
        
    if args.train:
        if not args.annotation_path or not args.image_path:
            print("You should input --image_path and --annotation_path to start training.")
            return
        
        train_ssd.train(image_path = args.image_path, 
                        annotation_path = args.annotation_path, 
                        model_path = args.model_path,
                        learning_rate = args.learning_rate, 
                        batch_size = args.batch_size, 
                        epochs = args.epochs,
                        restore = args.restore
                        )
        
    if args.test:
        if not args.image_path:
            print("You should input --image_path to start testing.")
            return
        
        test_ssd.test(image_path = '/home/cxd/SSD_Light/demo/',
                      out_path = args.out_path,
                      model_path = args.model_path,
                      batch_size = args.batch_size
                      )
        
if __name__ == '__main__':
    main()       
        
         