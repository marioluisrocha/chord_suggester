import pandas as pd
import re
import csv
import os
import gc
import argparse
from collections import Counter

# Dictionary to map semitone distance to scale degree
SEMITONE_TO_SCALE_DEGREE = {
    0: 'I',    # Root/Tonic
    1: 'bII',  # Flat second
    2: 'II',   # Second
    3: 'bIII', # Flat third
    4: 'III',  # Third
    5: 'IV',   # Fourth
    6: 'bV',   # Flat fifth
    7: 'V',    # Fifth
    8: 'bVI',  # Flat sixth
    9: 'VI',   # Sixth
    10: 'bVII', # Flat seventh
    11: 'VII'   # Seventh
}

# Dictionary to map note to semitone value (C=0)
NOTE_TO_SEMITONE = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

# Reverse mapping from semitone to note (using sharps)
SEMITONE_TO_NOTE = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E',
    5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A',
    10: 'A#', 11: 'B'
}

# Major keys and their common chords with weights
# Format: {chord_root: weight}
MAJOR_KEY_PROFILES = {
    0: {0: 5, 2: 2, 4: 3, 5: 4, 7: 4.5, 9: 2, 11: 1},  # C major: C(5), Dm(2), Em(3), F(4), G(4.5), Am(2), Bdim(1)
}

# Minor keys and their common chords with weights
MINOR_KEY_PROFILES = {
    0: {0: 5, 2: 1, 3: 3.5, 5: 4, 7: 4, 8: 2, 10: 2},  # C minor: Cm(5), Ddim(1), Eb(3.5), Fm(4), Gm(4), Ab(2), Bb(2)
}

# Generate all major and minor key profiles by rotating the pattern
for i in range(1, 12):
    # Major keys
    MAJOR_KEY_PROFILES[i] = {(k + i) % 12: v for k, v in MAJOR_KEY_PROFILES[0].items()}
    # Minor keys
    MINOR_KEY_PROFILES[i] = {(k + i) % 12: v for k, v in MINOR_KEY_PROFILES[0].items()}

def extract_chord_root(chord):
    """Extract the root note from a chord string"""
    if not chord or pd.isna(chord):
        return None
        
    match = re.match(r'^([A-G][#b]?)', chord)
    if match:
        return match.group(1)
    return None

def is_minor_chord(chord):
    """Check if a chord is minor"""
    if not chord or pd.isna(chord):
        return False
        
    # Look for minor indicators after the root
    match = re.match(r'^[A-G][#b]?(m|min|minor)', chord)
    return bool(match)

def estimate_key(chord_sequence):
    """
    Estimate the key of a song based on the chords it contains.
    Returns a tuple of (key, is_minor) where key is the estimated key
    and is_minor is a boolean indicating if it's a minor key.
    """
    if not chord_sequence or pd.isna(chord_sequence):
        return ('C', False)  # Default to C major if no chords
    
    # Identify separator and split chord sequence
    if ',' in chord_sequence:
        separator = ','
    elif '|' in chord_sequence:
        separator = '|'
    else:
        separator = ' '
    
    chords = [c.strip() for c in chord_sequence.split(separator) if c.strip()]
    
    # Extract root notes and count occurrences
    roots = []
    minor_count = 0
    total_count = 0
    
    for chord in chords:
        root = extract_chord_root(chord)
        if root:
            # Convert flats to sharps for consistency
            if 'b' in root and root not in ['Cb', 'Fb', 'Gb']:
                flat_to_sharp = {'Db': 'C#', 'Eb': 'D#', 'Ab': 'G#', 'Bb': 'A#'}
                root = flat_to_sharp.get(root, root)
            
            roots.append(root)
            
            # Count minor chords
            if is_minor_chord(chord):
                minor_count += 1
            total_count += 1
    
    # If no valid chords found, return default
    if not roots:
        return ('C', False)
    
    # Count occurrences of each root
    root_counter = Counter(roots)
    
    # Convert roots to semitone values
    semitone_counts = {}
    for root, count in root_counter.items():
        if root in NOTE_TO_SEMITONE:
            semitone = NOTE_TO_SEMITONE[root]
            semitone_counts[semitone] = count
    
    # Calculate scores for each possible key
    major_scores = {}
    minor_scores = {}
    
    # Test each possible major key
    for key_semitone, profile in MAJOR_KEY_PROFILES.items():
        score = 0
        for chord_semitone, count in semitone_counts.items():
            # Check if this chord is in the key profile
            if chord_semitone in profile:
                score += count * profile[chord_semitone]
        major_scores[key_semitone] = score
    
    # Test each possible minor key
    for key_semitone, profile in MINOR_KEY_PROFILES.items():
        score = 0
        for chord_semitone, count in semitone_counts.items():
            # Check if this chord is in the key profile
            if chord_semitone in profile:
                score += count * profile[chord_semitone]
        minor_scores[key_semitone] = score
    
    # Find the highest scoring keys
    best_major_key = max(major_scores.items(), key=lambda x: x[1])
    best_minor_key = max(minor_scores.items(), key=lambda x: x[1])
    
    # Compare the best major and minor keys
    if best_minor_key[1] > best_major_key[1]:
        # Minor key is more likely
        best_key_semitone = best_minor_key[0]
        is_minor = True
    else:
        # Major key is more likely
        best_key_semitone = best_major_key[0]
        is_minor = False
    
    # Consider the proportion of minor chords as additional evidence
    if total_count > 0:
        minor_ratio = minor_count / total_count
        # If more than 50% of chords are minor, bias toward minor key
        if minor_ratio > 0.5 and not is_minor:
            # Switch to relative minor if we're in major but have lots of minor chords
            best_key_semitone = (best_key_semitone + 9) % 12
            is_minor = True
        # If less than 20% of chords are minor, bias toward major key
        elif minor_ratio < 0.2 and is_minor:
            # Switch to relative major if we're in minor but have few minor chords
            best_key_semitone = (best_key_semitone + 3) % 12
            is_minor = False
    
    # Convert semitone back to note
    key_note = SEMITONE_TO_NOTE[best_key_semitone]
    
    # Add minor indicator if necessary
    if is_minor:
        key = f"{key_note}m"
    else:
        key = key_note
    
    return (key, is_minor)

def get_scale_degree(note, key):
    """Convert a note to its scale degree relative to the key"""
    if not note or not key:
        return note
    
    # Extract the root note from the key (ignoring minor/major)
    key_root = re.match(r'^([A-G][#b]?)', key).group(1) if re.match(r'^([A-G][#b]?)', key) else key
    
    # Normalize key_root with sharps
    if 'b' in key_root and key_root != 'Gb' and key_root != 'Cb':
        flat_to_sharp = {'Db': 'C#', 'Eb': 'D#', 'Ab': 'G#', 'Bb': 'A#', 'Fb': 'E'}
        key_root = flat_to_sharp.get(key_root, key_root)
    
    # Calculate semitone distance
    try:
        note_semitone = NOTE_TO_SEMITONE.get(note, -1)
        key_semitone = NOTE_TO_SEMITONE.get(key_root, -1)
        
        if note_semitone == -1 or key_semitone == -1:
            return note  # Return original if note or key not recognized
        
        # Calculate relative position
        semitone_diff = (note_semitone - key_semitone) % 12
        
        # Return the roman numeral
        return SEMITONE_TO_SCALE_DEGREE[semitone_diff]
    except Exception:
        return note  # Return original on error

def convert_chord_to_roman(chord, key):
    """Convert a chord to its roman numeral representation based on the key"""
    if not chord or pd.isna(chord) or chord.strip() == "":
        return ""
    
    # Handle "No Chord" variations
    if chord.strip().lower() in ["n.c.", "nc", "no chord", "-"]:
        return "N.C."
    
    # Extract the root note from the chord
    match = re.match(r'^([A-G][#b]?)(.*)', chord)
    if not match:
        return chord  # Return original if pattern doesn't match
    
    root, quality = match.groups()
    
    # Check if the key is minor
    is_minor_key = bool(re.search(r'm$|min$', key))
    
    # Get the scale degree as roman numeral
    roman = get_scale_degree(root, key)
    
    # Adjust for minor keys (lowercase for minor chords)
    if is_minor_key:
        if roman in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']:
            roman = roman.lower()
    
    # Handle chord quality
    if quality and 'm' == quality[0] and not any(x in quality for x in ['maj', 'min']):
        # For minor chords: lowercase if in major key, uppercase if in minor key
        if not is_minor_key:
            roman = roman.lower()
        quality = quality[1:]  # Remove the 'm' as it's encoded in case
    
    # Handle slash chords (e.g. C/E)
    if '/' in quality:
        quality_part, bass = quality.split('/', 1)
        
        # Check if bass is a note
        bass_match = re.match(r'^([A-G][#b]?)(.*)', bass)
        if bass_match:
            bass_root, bass_rest = bass_match.groups()
            roman_bass = get_scale_degree(bass_root, key)
            
            # Adjust case for minor keys
            if is_minor_key and roman_bass in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']:
                roman_bass = roman_bass.lower()
                
            return f"{roman}{quality_part}/{roman_bass}{bass_rest}"
        else:
            return f"{roman}{quality_part}/{bass}"
    
    return f"{roman}{quality}"

def convert_chord_sequence(chord_sequence, key=None):
    """
    Convert a sequence of chords to roman numerals based on the key.
    If key is not provided, it will be estimated from the chord sequence.
    """
    if not chord_sequence or pd.isna(chord_sequence) or chord_sequence.strip() == "":
        return ""
    
    # Estimate key if not provided
    if not key or pd.isna(key):
        key, _ = estimate_key(chord_sequence)
    
    # Identify the separator
    if ',' in chord_sequence:
        separator = ','
    elif '|' in chord_sequence:
        separator = '|'
    else:
        separator = ' '
    
    # Split, convert, and rejoin
    chords = chord_sequence.split(separator)
    roman_chords = [convert_chord_to_roman(chord.strip(), key) for chord in chords if chord.strip()]
    return separator.join(roman_chords)

def process_dataset(input_file, output_file, chord_column='chords', key_column=None, chunk_size=10000, estimate_keys=False):
    """
    Process dataset to convert chords to roman numerals based on the song's key.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str
        Path to save the output CSV file
    chord_column : str
        Name of the column containing chord data
    key_column : str
        Name of the column containing the song's key (optional)
    chunk_size : int
        Number of rows to process at once
    estimate_keys : bool
        If True, the key will be estimated based on the chords even if key_column is provided
    """
    print(f"Processing {input_file} with chunk size {chunk_size}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Create temporary output file
    temp_output = output_file + ".temp"
    
    try:
        # Read header to check if required columns exist
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            header = next(csv.reader(f))
            
        if chord_column not in header:
            print(f"Error: Column '{chord_column}' not found in input file.")
            print(f"Available columns: {', '.join(header)}")
            return
        
        key_index = -1
        if key_column and key_column in header:
            key_index = header.index(key_column)
            print(f"Using existing key column: '{key_column}'")
        else:
            print(f"No key column provided or '{key_column}' not found. Keys will be estimated from chords.")
            estimate_keys = True
            
        # Process file in chunks
        chunks_processed = 0
        total_rows = 0
        
        # Determine output columns
        output_header = header.copy()
        if estimate_keys:
            if 'estimated_key' not in output_header:
                output_header.append('estimated_key')
        output_header.append('roman_chords')
        
        with open(temp_output, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(output_header)
            
            # Process chunks
            for chunk in pd.read_csv(input_file, chunksize=chunk_size, encoding='utf-8', 
                                    low_memory=True):
                rows = []
                
                for _, row in chunk.iterrows():
                    chord_data = row[chord_column]
                    
                    # Get or estimate key
                    if key_index >= 0 and not estimate_keys:
                        key_data = row[key_column] if key_column in row and not pd.isna(row[key_column]) else None
                    else:
                        key_data = None
                    
                    # Estimate key if needed
                    estimated_key = None
                    if key_data is None or estimate_keys:
                        estimated_key, _ = estimate_key(chord_data)
                        key_data = estimated_key
                    
                    # Convert chords to roman numerals
                    roman_chords = convert_chord_sequence(chord_data, key_data)
                    
                    # Prepare output row
                    row_values = list(row)
                    if estimate_keys:
                        if 'estimated_key' in chunk.columns:
                            # Replace the existing estimated_key value
                            row_values[chunk.columns.get_loc('estimated_key')] = estimated_key
                        else:
                            # Append the estimated key
                            row_values.append(estimated_key)
                    
                    # Add roman chords
                    row_values.append(roman_chords)
                    rows.append(row_values)
                    total_rows += 1
                
                writer.writerows(rows)
                chunks_processed += 1
                print(f"Processed chunk {chunks_processed} ({chunks_processed * chunk_size} rows)...")
                
                # Free memory
                del rows
                gc.collect()
        
        # Replace output file
        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename(temp_output, output_file)
        
        print(f"Successfully processed {total_rows} rows. Output saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(temp_output):
            os.remove(temp_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert chords to roman numeral representation.')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('output_file', help='Path to output CSV file')
    parser.add_argument('--chord-column', default='chords', help='Name of chord column')
    parser.add_argument('--key-column', default=None, help='Name of key column (optional)')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Rows to process at once')
    parser.add_argument('--estimate-keys', action='store_true', help='Estimate keys even if key column is provided')
    
    args = parser.parse_args()
    
    process_dataset(
        args.input_file,
        args.output_file,
        args.chord_column,
        args.key_column,
        args.chunk_size,
        args.estimate_keys
    )
