import pandas as pd
import re
import unicodedata
from fuzzywuzzy import fuzz, process
import argparse
import csv
import os
import gc

def clean_text(text):
    """Clean text by removing special characters, normalizing unicode, and lowercasing."""
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string if it's not already
    text = str(text)
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    return text

def find_best_match(name, choices, threshold=80):
    """Find the best match for a name in a list of choices using fuzzy matching."""
    if not choices or not name or name == "":
        return None
    
    # Find the best match
    best_match, score = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
    
    if score >= threshold:
        return best_match
    return None

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

def process_lyrics_chunk(lyrics_chunk, artist_col_lyrics, title_col_lyrics, lyrics_col, lyrics_dict, artist_titles_dict, all_artists):
    """Process a chunk of the lyrics dataset to build lookup dictionaries."""
    for _, row in lyrics_chunk.iterrows():
        artist = clean_text(row[artist_col_lyrics])
        title = clean_text(row[title_col_lyrics])
        
        if not artist or not title:
            continue
            
        # Add to lyrics dictionary
        key = (artist, title)
        lyrics_dict[key] = row[lyrics_col]
        
        # Add to unique artists
        if artist not in all_artists:
            all_artists.add(artist)
            artist_titles_dict[artist] = []
        
        # Add to artist's titles
        if title and title not in artist_titles_dict[artist]:
            artist_titles_dict[artist].append(title)

def merge_datasets(spotify_file, lyrics_file, output_file, artist_threshold=85, title_threshold=85, chunk_size=10000):
    """
    Memory-efficient function to merge Spotify and lyrics datasets.
    
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
    """
    print(f"Starting memory-efficient merge of {spotify_file} and {lyrics_file}")
    
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
        process_lyrics_chunk(lyrics_chunk, artist_col_lyrics, title_col_lyrics, lyrics_col, 
                           lyrics_dict, artist_titles_dict, all_artists)
        chunks_processed += 1
        print(f"Processed {chunks_processed * chunk_size} rows from lyrics file...")
        # Explicitly trigger garbage collection to free memory
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
            results = []
            
            for _, row in spotify_chunk.iterrows():
                # Get original row values as list
                row_values = row.tolist()
                
                # Clean artist and title
                artist = clean_text(row[artist_col_spotify])
                title = clean_text(row[title_col_spotify])
                
                lyrics = None
                
                # Try exact match first
                if (artist, title) in lyrics_dict:
                    lyrics = lyrics_dict[(artist, title)]
                    exact_matches += 1
                else:
                    # Try fuzzy matching for artist
                    matched_artist = find_best_match(artist, all_artists_list, threshold=artist_threshold)
                    
                    if matched_artist and matched_artist in artist_titles_dict:
                        # If artist matched, try to match title among that artist's songs
                        matched_title = find_best_match(title, artist_titles_dict[matched_artist], threshold=title_threshold)
                        
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
            
            # Write results
            writer.writerows(results)
            
            # Update progress
            chunks_processed += 1
            print(f"Processed chunk {chunks_processed} ({chunks_processed * chunk_size} total rows)...")
            
            # Free memory
            del results
            gc.collect()
    
    # Rename temporary file to final output
    if os.path.exists(output_file):
        os.remove(output_file)
    os.rename(temp_output, output_file)
    
    # Print statistics
    print("\nMatching Statistics:")
    print(f"Total processed in Spotify dataset: {total_processed}")
    print(f"Exact matches: {exact_matches} ({exact_matches/total_processed*100:.2f}%)")
    print(f"Fuzzy matches: {fuzzy_matches} ({fuzzy_matches/total_processed*100:.2f}%)")
    print(f"No matches: {no_matches} ({no_matches/total_processed*100:.2f}%)")
    print(f"Total matched: {exact_matches + fuzzy_matches} ({(exact_matches + fuzzy_matches)/total_processed*100:.2f}%)")
    print(f"Successfully saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Memory-efficient script to merge Spotify and lyrics datasets.')
    parser.add_argument('spotify_file', help='Path to the Spotify dataset CSV file')
    parser.add_argument('lyrics_file', help='Path to the lyrics dataset CSV file')
    parser.add_argument('output_file', help='Path to save the merged dataset')
    parser.add_argument('--artist-threshold', type=int, default=85, help='Threshold for fuzzy matching artist names (0-100)')
    parser.add_argument('--title-threshold', type=int, default=85, help='Threshold for fuzzy matching song titles (0-100)')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Number of rows to process at once')
    
    args = parser.parse_args()
    
    merge_datasets(
        args.spotify_file,
        args.lyrics_file,
        args.output_file,
        args.artist_threshold,
        args.title_threshold,
        args.chunk_size
    )