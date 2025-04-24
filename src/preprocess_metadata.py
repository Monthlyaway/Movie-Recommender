#!/usr/bin/env python3
"""
Script to preprocess movies_metadata.csv by removing person names from overview text.
This script uses spaCy's Named Entity Recognition to identify and remove PERSON entities.

Output: A new CSV file (movies_metadata_name_removed.csv) with the same structure as the original,
but with person names removed from the overview text.
"""

import os
import pandas as pd
import spacy
import argparse
import json
from tqdm import tqdm
import torch

def preprocess_plot(text, nlp):
    """
    Process text using spaCy NER to remove PERSON entities.
    
    Args:
        text (str): Movie overview text
        nlp (spacy.lang): Loaded spaCy model
        
    Returns:
        str: Processed text with PERSON entities removed
    """
    if not isinstance(text, str) or text.strip() == '':
        return text
    
    doc = nlp(text)
    
    # Create a list of spans to remove (PERSON entities)
    spans_to_remove = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            spans_to_remove.append((ent.start_char, ent.end_char))
    
    # If no entities to remove, return the original text
    if not spans_to_remove:
        return text
    
    # Sort spans in reverse order to avoid index issues when removing
    spans_to_remove.sort(reverse=True)
    
    # Remove the spans
    result = text
    for start, end in spans_to_remove:
        result = result[:start] + result[end:]
    
    # Clean up extra whitespace
    result = ' '.join(result.split())
    
    return result

def process_movie_metadata(input_file, output_file, model_name="en_core_web_sm", use_gpu=True):
    """
    Process the movies_metadata.csv file and create a new version with person names removed
    from overview text.
    
    Args:
        input_file (str): Path to the original movies_metadata.csv
        output_file (str): Path for the output file
        model_name (str): Name of the spaCy model to use for NER
        use_gpu (bool): Whether to use GPU acceleration if available
    """
    # Check GPU availability
    gpu_available = torch.cuda.is_available() if use_gpu else False
    
    if use_gpu and gpu_available:
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"Using GPU acceleration for spaCy")
        spacy.prefer_gpu()
        gpu_id = 0
    else:
        if use_gpu and not gpu_available:
            print("GPU requested but not available. Falling back to CPU.")
        gpu_id = -1  # Use CPU
    
    print(f"Loading spaCy model '{model_name}'...")
    try:
        nlp = spacy.load(model_name)
        if gpu_available and use_gpu:
            # Set batch size for GPU processing
            # Adjust this based on your GPU memory
            nlp.batch_size = 128  
    except OSError:
        print(f"SpaCy model '{model_name}' not found. Please install it with:")
        print(f"python -m spacy download {model_name}")
        return
    
    print(f"Loading movie metadata from {input_file}...")
    try:
        # Load with low_memory=False to avoid mixed type inference
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    print(f"Total movies: {len(df)}")
    print("Processing movie overviews to remove person names...")
    
    # Process overviews with a progress bar
    tqdm.pandas(desc="Processing")
    df['overview'] = df['overview'].progress_apply(lambda x: preprocess_plot(x, nlp))
    
    print(f"Saving processed data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("Done!")
    print(f"Created {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess movie metadata to remove person names from overviews.')
    parser.add_argument('--input', type=str, default='/home/castle/Codes/Movie-Recommender/dataset/movies_metadata.csv',
                        help='Path to the input movies_metadata.csv file')
    parser.add_argument('--output', type=str, default='/home/castle/Codes/Movie-Recommender/dataset/movies_metadata_name_removed.csv',
                        help='Path to save the processed output file')
    parser.add_argument('--model', type=str, default='en_core_web_sm',
                        help='SpaCy model to use for NER')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='Use GPU acceleration if available')
    parser.add_argument('--no-gpu', action='store_false', dest='gpu',
                        help='Disable GPU acceleration and use CPU only')
    
    args = parser.parse_args()
    
    process_movie_metadata(args.input, args.output, args.model, args.gpu)

if __name__ == "__main__":
    main() 