import argparse
import glob
import os
import random
import shutil

import numpy as np
import tensorflow as tf

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    source_dir = "/training_and_validation/"
    temp_dir = "/temp/"
    train_dir = "/train/"
    val_dir = "/val/"  
    
    records = os.listdir(data_dir + source_dir)
    for record in records:
        source = data_dir + source_dir + record
        destination = data_dir + temp_dir + record
        shutil.copy(source, destination)
        raw_dataset = tf.data.TFRecordDataset(data_dir + temp_dir + record)
        shards = 100
        for i in range(shards):
            writer = tf.data.experimental.TFRecordWriter(f"shard-{i}_{record}")
            writer.write(raw_dataset.shard(shards, i))     
        
        filenames = os.listdir()
        shards = [file for file in filenames if file[:5] == "shard"]
        val_files = random.sample(shards, 20)
        train_files = shards
        for file in val_files:
            train_files.remove(file)
            source = file
            destination = data_dir + val_dir + file
            shutil.move(source, destination)
        
        for file in train_files:
            source = file
            destination = data_dir + train_dir + file
            shutil.move(source, destination)    
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)