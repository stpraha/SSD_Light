# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:20:29 2019

@author: Stpraha
"""

import argparse
import train_ssd
import test_ssd

def parse_arguments():
    """
        Parse the command line arguments of the program.
    """
    parse = argparse.ArgumentParser(description='Train or test the model')
    
    parse.add_argument("--train")
    parse.add_argument("--test")
    parse.add_argument("-bs", "--batch_size", default = 32)
    parse.add_argument("-mp", "--model_path", default = './save/')
    parse.add_argument("-ip", "--image_path", default = 'F:\\VOC2007\\Annotations\\')
    parse.add_argument("-ap", "--annotation_path", default = 'F:\\VOC2007\\JPEGImages\\')
    parse.add_argument("-op", "--out_path", default = '/out/')
    parse.add_argument("-e", "--epochs", default = 100)
    parse.add_argument("-lr", "--learning_rate", default = 0.001)

    return parse.parse_args()

def main():
    """
        Entry
    """
    args = parse_arguments()
    
    if not args.train and not args.test:
        print("You should input --train or --test")
        
    if args.train:
        train_ssd.train(image_path = args.image_path, 
                        annotation_path = args.annotation_path, 
                        learning_rate = args.learning_rate, 
                        batch_size = args.batch_size, 
                        epochs = args.epochs,
                        model_path = args.model_path
                        )
        
    if args.test:
        test_ssd.test(image_path = args.image_path,
                      out_path = args.out_path,
                      model_path = args.model_path,
                      batch_size = args.batch_size,
                      )
        
if __name__ == '__main__':
    main()       
        
        