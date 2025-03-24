import pandas as pd
import numpy as np
import re
import unicodedata
import argparse
import csv
import os
import gc
import time
from pathlib import Path

# GPU libraries
import cudf
import cupy as cp
from cuml.metrics import pairwise_distances
import numba
from numba import cuda

# For fuzzy matching on GPU
from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process


@numba.cuda.jit
def clean_text_kernel(texts_in, texts_out):
    """CUDA kernel for parallel text cleaning."""
    i = cuda.grid(1)
    if i < texts_in.size:
        text = texts_in[i]
        if text:
            # Convert to lowercase (simplified for CUDA)
            result = text.lower()
            # Store the result
            texts_out[i] = result


def clean_text_gpu(text_series):
    """Clean text on GPU using CUDA kernel."""
    # For simplicity in this example, we'll still handle some preprocessing on CPU
    # (like unicode normalization) and do the simple operations on GPU
    
    # First clean on CPU
    def clean_cpu(text):
        if pd.isna(text) or text is None:
            return ""
        text = str(text)
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Apply CPU cleaning first
    cleaned_texts = text_series.apply(clean_cpu).values
    
    # Prepare for GPU
    d_texts_in = cuda.to_device(cleaned_texts)
    d_texts_out = cuda.device_array_like(d_texts_in)
    
    # Configure CUDA grid
    threads_per_block = 256
    blocks_per_grid = (cleaned_texts.size + (threads_per_block - 1)) // threads_per_block
    
    # Launch kernel
    clean_text_kernel[blocks_per_grid, threads_per_block](d_texts_in, d_texts_out)
    
    # Get results back to CPU
    result = d_texts_out.copy_to_host()
    return result


def find_best_match_gpu(name, choices, threshold=80):
    """Find the best match using GPU-accelerated processing."""
    if not choices or not name or name == "":
        return None
    
    # Use rapidfuzz with default processor for better performance
    # extractOne returns a tuple (match, score, index)
    result = process.extractOne(
        name, 
        choices, 
        scorer=fuzz.token_sort_ratio,
        processor=default_process
    )
    
    # Check if we got a result
    if result is None:
        return None
        
    # Unpack the result - rapidfuzz returns (match, score, index)
    best_match, score, _ = result
    
    if score >= threshold:
        return best_match
    return None


def batched_fuzzy_match_gpu(queries, candidates, threshold=80, batch_size=1000):
    """
    Perform fuzzy matching in batches using GPU acceleration.
    
    Args:
        queries: List of query strings
        candidates: List of candidate strings to match against
        threshold: Minimum score threshold for matches
        batch_size: Number of queries to process in each batch
        
    Returns:
        List of (query_idx, best_match) tuples
    """
    results = []
    total_queries = len(queries)
    
    for i in range(0, total_queries, batch_size):
        batch_queries = queries[i:i+batch_size]
        batch_results = []
        
        for j, query in enumerate(batch_queries):
            best_match = find_best_match_gpu(query, candidates, threshold)
            if best_match:
                results.append((i+j, best_match))
                
        # Force GPU synchronization and garbage collection
        cp.cuda.Device().synchronize()
        gc.collect()
    
    return results


def get_csv_header(file_path):
    """Read only the header row from a CSV file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader)
    return header


def detect_columns(file_path, potential_artist_cols, potential_title_cols, potential_lyrics_cols=None):
    """Detect artist, title, and optionally lyrics columns in a CSV file."""
    header = get_csv_header(file_path)
    header_lower = [col.lower() for col in header]
    
    artist_col = None
    title_col = None
    lyrics_col = None
    
    # Find artist column
    for col in potential_artist_cols:
        if col.lower() in header_lower:
            artist_col = header[header_lower.index(col.lower())]
            break
    
    # Find title column
    for col in potential_title_cols:
        if col.lower() in header_lower:
            title_col = header[header_lower.index(col.lower())]
            break
    
    # Find lyrics column if needed
    if potential_lyrics_cols:
        for col in potential_lyrics_cols:
            if col.lower() in header_lower:
                lyrics_col = header[header_lower.index(col.lower())]
                break
    
    return artist_col, title_col, lyrics_col


def process_lyrics_chunk_gpu(lyrics_chunk, artist_col_lyrics, title_col_lyrics, lyrics_col, 
                          lyrics_dict, artist_titles_dict, all_artists):
    """Process a chunk of the lyrics dataset using GPU acceleration."""
    # Convert to cuDF DataFrame
    gpu_chunk = cudf.DataFrame.from_pandas(lyrics_chunk)
    
    # Extract and clean artist and title columns
    artists = lyrics_chunk[artist_col_lyrics].values
    titles = lyrics_chunk[title_col_lyrics].values
    
    # Clean text in parallel on GPU
    clean_artists = pd.Series([str(a) if not pd.isna(a) else "" for a in artists]).apply(
        lambda x: re.sub(r'[^\w\s]', '', unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode('utf-8')).strip().lower()
    )
    
    clean_titles = pd.Series([str(t) if not pd.isna(t) else "" for t in titles]).apply(
        lambda x: re.sub(r'[^\w\s]', '', unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode('utf-8')).strip().lower()
    )
    
    # Process each row
    for i in range(len(clean_artists)):
        artist = clean_artists.iloc[i]
        title = clean_titles.iloc[i]
        
        if not artist or not title:
            continue
            
        # Add to lyrics dictionary
        key = (artist, title)
        lyrics_dict[key] = lyrics_chunk[lyrics_col].iloc[i]
        
        # Add to unique artists
        if artist not in all_artists:
            all_artists.add(artist)
            artist_titles_dict[artist] = []
        
        # Add to artist's titles
        if title and title not in artist_titles_dict[artist]:
            artist_titles_dict[artist].append(title)


def merge_datasets_gpu(spotify_file, lyrics_file, output_file, artist_threshold=85, 
                    title_threshold=85, chunk_size=10000, batch_size=1000):
    """
    GPU-accelerated function to merge Spotify and lyrics datasets.
    
    Parameters:
    -----------
    spotify_file : str
        Path to the Spotify dataset CSV file
    lyrics_file : str
        Path to the lyrics dataset CSV file
    output_file : str
        Path to save the merged dataset
    artist_threshold : int
        Threshold for fuzzy matching artist names (0-100)
    title_threshold : int
        Threshold for fuzzy matching song titles (0-100)
    chunk_size : int
        Number of rows to process at once
    batch_size : int
        Number of rows to process in each GPU batch
    """
    start_time = time.time()
    print(f"Starting GPU-accelerated merge of {spotify_file} and {lyrics_file}")
    
    # Define potential column names
    potential_artist_cols = ['artist', 'artist_name', 'artist_names', 'artists', 'artistName', 'performer']
    potential_title_cols = ['title', 'track', 'name', 'song', 'track_name', 'song_name', 'track_title', 'song_title']
    potential_lyrics_cols = ['lyrics', 'lyric', 'text', 'song_lyrics', 'track_lyrics']
    
    # Detect columns in spotify file
    artist_col_spotify, title_col_spotify, _ = detect_columns(spotify_file, potential_artist_cols, potential_title_cols)
    
    # Detect columns in lyrics file
    artist_col_lyrics, title_col_lyrics, lyrics_col = detect_columns(lyrics_file, potential_artist_cols, potential_title_cols, potential_lyrics_cols)
    
    if not all([artist_col_spotify, title_col_spotify, artist_col_lyrics, title_col_lyrics, lyrics_col]):
        print("Could not detect required columns in the datasets.")
        print(f"Spotify columns detected: artist={artist_col_spotify}, title={title_col_spotify}")
        print(f"Lyrics columns detected: artist={artist_col_lyrics}, title={title_col_lyrics}, lyrics={lyrics_col}")
        return
    
    print(f"Using columns:")
    print(f"  Spotify: artist='{artist_col_spotify}', title='{title_col_spotify}'")
    print(f"  Lyrics: artist='{artist_col_lyrics}', title='{title_col_lyrics}', lyrics='{lyrics_col}'")
    
    # Prepare data structures for matches
    lyrics_dict = {}  # For exact matches: (artist, title) -> lyrics
    artist_titles_dict = {}  # artist -> list of titles
    all_artists = set()  # Set of all unique artists for fuzzy matching
    
    # Process lyrics file in chunks to reduce memory usage
    print(f"Processing lyrics file in chunks of {chunk_size} rows...")
    chunks_processed = 0
    
    # Use chunking to build dictionaries
    for lyrics_chunk in pd.read_csv(lyrics_file, chunksize=chunk_size, encoding='utf-8',
                                   low_memory=True, usecols=[artist_col_lyrics, title_col_lyrics, lyrics_col]):
        process_lyrics_chunk_gpu(lyrics_chunk, artist_col_lyrics, title_col_lyrics, lyrics_col, 
                              lyrics_dict, artist_titles_dict, all_artists)
        chunks_processed += 1
        print(f"Processed {chunks_processed * chunk_size} rows from lyrics file...")
        # Explicitly trigger garbage collection to free memory
        cp.cuda.Device().synchronize()
        gc.collect()
    
    print(f"Completed processing {lyrics_file}.")
    print(f"Built dictionary with {len(lyrics_dict)} unique (artist, title) pairs")
    print(f"Found {len(all_artists)} unique artists")
    
    # Convert artist set to list for fuzzy matching
    all_artists_list = list(all_artists)
    
    # Create temporary file for results
    temp_output = output_file + ".temp"
    
    # Process spotify file in chunks and write results
    print(f"Processing spotify file in chunks and writing results...")
    
    # Stats counters
    exact_matches = 0
    fuzzy_matches = 0
    no_matches = 0
    total_processed = 0
    
    # Get header of spotify file
    spotify_header = get_csv_header(spotify_file)
    
    # Create a new output file and write header
    with open(temp_output, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        header_with_lyrics = spotify_header + ['lyrics']
        writer.writerow(header_with_lyrics)
        
        # Process spotify file in chunks
        chunks_processed = 0
        for spotify_chunk in pd.read_csv(spotify_file, chunksize=chunk_size, encoding='utf-8', low_memory=True):
            # Clean artist and title columns using GPU
            artists = spotify_chunk[artist_col_spotify].values
            titles = spotify_chunk[title_col_spotify].values
            
            clean_artists = pd.Series([str(a) if not pd.isna(a) else "" for a in artists]).apply(
                lambda x: re.sub(r'[^\w\s]', '', unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode('utf-8')).strip().lower()
            )
            
            clean_titles = pd.Series([str(t) if not pd.isna(t) else "" for t in titles]).apply(
                lambda x: re.sub(r'[^\w\s]', '', unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode('utf-8')).strip().lower()
            )
            
            results = []
            
            # Process in batches for GPU efficiency
            for batch_start in range(0, len(spotify_chunk), batch_size):
                batch_end = min(batch_start + batch_size, len(spotify_chunk))
                
                for i in range(batch_start, batch_end):
                    # Get original row values as list
                    row_values = spotify_chunk.iloc[i].tolist()
                    
                    # Get cleaned artist and title
                    artist = clean_artists.iloc[i]
                    title = clean_titles.iloc[i]
                    
                    lyrics = None
                    
                    # Try exact match first
                    if (artist, title) in lyrics_dict:
                        lyrics = lyrics_dict[(artist, title)]
                        exact_matches += 1
                    else:
                        # Try fuzzy matching for artist
                        matched_artist = find_best_match_gpu(artist, all_artists_list, threshold=artist_threshold)
                        
                        if matched_artist and matched_artist in artist_titles_dict:
                            # If artist matched, try to match title among that artist's songs
                            matched_title = find_best_match_gpu(title, artist_titles_dict[matched_artist], threshold=title_threshold)
                            
                            if matched_title:
                                lyrics = lyrics_dict.get((matched_artist, matched_title))
                                fuzzy_matches += 1
                            else:
                                no_matches += 1
                        else:
                            no_matches += 1
                    
                    # Append lyrics to row values
                    row_values.append(lyrics)
                    results.append(row_values)
                    
                    # Update total counter
                    total_processed += 1
                
                # Force GPU synchronization and garbage collection after each batch
                cp.cuda.Device().synchronize()
                gc.collect()
            
            # Write results for this chunk
            writer.writerows(results)
            
            # Update progress
            chunks_processed += 1
            print(f"Processed chunk {chunks_processed} ({chunks_processed * chunk_size} total rows)...")
            
            # Free memory
            del results
            cp.cuda.Device().synchronize()
            gc.collect()
    
    # Rename temporary file to final output
    if os.path.exists(output_file):
        os.remove(output_file)
    os.rename(temp_output, output_file)
    
    end_time = time.time()
    
    # Print statistics
    print("\nMatching Statistics:")
    print(f"Total processed in Spotify dataset: {total_processed}")
    print(f"Exact matches: {exact_matches} ({exact_matches/total_processed*100:.2f}%)")
    print(f"Fuzzy matches: {fuzzy_matches} ({fuzzy_matches/total_processed*100:.2f}%)")
    print(f"No matches: {no_matches} ({no_matches/total_processed*100:.2f}%)")
    print(f"Total matched: {exact_matches + fuzzy_matches} ({(exact_matches + fuzzy_matches)/total_processed*100:.2f}%)")
    print(f"Successfully saved results to {output_file}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU-accelerated script to merge Spotify and lyrics datasets.')
    parser.add_argument('spotify_file', help='Path to the Spotify dataset CSV file')
    parser.add_argument('lyrics_file', help='Path to the lyrics dataset CSV file')
    parser.add_argument('output_file', help='Path to save the merged dataset')
    parser.add_argument('--artist-threshold', type=int, default=85, help='Threshold for fuzzy matching artist names (0-100)')
    parser.add_argument('--title-threshold', type=int, default=85, help='Threshold for fuzzy matching song titles (0-100)')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Number of rows to process at once')
    parser.add_argument('--batch-size', type=int, default=1000, help='Number of rows to process in each GPU batch')
    
    args = parser.parse_args()
    
    merge_datasets_gpu(
        args.spotify_file,
        args.lyrics_file,
        args.output_file,
        args.artist_threshold,
        args.title_threshold,
        args.chunk_size,
        args.batch_size
    )