import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

def explore_chordonomicon_data():
    """
    Explore and analyze the enriched Chordonomicon dataset.
    This function assumes you've already run spotify_connector.py.
    """
    print("Chordonomicon Data Explorer")
    print("===========================")
    
    # Try to load the enriched dataset first, fall back to original if not found
    try:
        if os.path.exists(os.path.join("chordonomicon_data", "chordonomicon_enriched.csv")):
            csv_path = os.path.join("chordonomicon_data", "chordonomicon_enriched.csv")
            print("Loading enriched dataset with Spotify metadata...")
        else:
            csv_path = os.path.join("chordonomicon_data", "chordonomicon_v2.csv")
            print("Loading original dataset (no Spotify metadata found)...")
            
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded: {len(df)} records")
        
        # Basic dataset information
        print("\nBasic Dataset Information:")
        print(f"  • Columns: {', '.join(df.columns.tolist())}")
        print(f"  • Records: {len(df)}")
        
        # Check if we have Spotify metadata
        has_spotify_data = 'artist_name' in df.columns and 'song_title' in df.columns
        
        if has_spotify_data:
            print("\nSpotify Metadata Summary:")
            print(f"  • Records with artist names: {df['artist_name'].notna().sum()}")
            print(f"  • Records with song titles: {df['song_title'].notna().sum()}")
            print(f"  • Unique artists: {len(df['artist_name'].dropna().unique())}")
            print(f"  • Unique songs: {len(df['song_title'].dropna().unique())}")
            
            # Show top artists by number of songs
            print("\nTop Artists by Number of Songs:")
            top_artists = df['artist_name'].value_counts().head(10)
            for i, (artist, count) in enumerate(top_artists.items(), 1):
                if pd.notna(artist):
                    print(f"  {i}. {artist}: {count} songs")
            
            # Create sample dataframe with the most popular songs
            output_path = os.path.join("chordonomicon_data", "popular_artists_sample.json")
            top_artists_df = df[df['artist_name'].isin(top_artists.index)].copy()
            top_artists_df = top_artists_df.sort_values(by=['artist_name', 'song_title'])
            top_artists_df[['artist_name', 'song_title', 'chords', 'genres', 'decade', 'main_genre']].to_json(
                output_path, orient='records', indent=2
            )
            print(f"\nSample of songs from popular artists saved to {output_path}")
        
        # Analyze chord usage
        print("\nAnalyzing Chord Usage:")
        
        # Extract all chords from the dataset
        all_chords = []
        for chord_seq in df['chords']:
            # Skip section markers like <verse_1> and extract only the chord names
            parts = chord_seq.split()
            for part in parts:
                if not (part.startswith('<') and part.endswith('>')):
                    all_chords.append(part)
        
        chord_counts = pd.Series(all_chords).value_counts()
        
        print(f"Found {len(chord_counts)} unique chords in the dataset")
        print("\nTop 15 Most Common Chords:")
        for i, (chord, count) in enumerate(chord_counts.head(15).items(), 1):
            print(f"  {i}. {chord}: {count} occurrences")
        
        # Save chord statistics
        chord_stats = {
            'total_chord_instances': len(all_chords),
            'unique_chords': len(chord_counts),
            'most_common': chord_counts.head(20).to_dict()
        }
        
        chord_stats_path = os.path.join("chordonomicon_data", "chord_statistics.json")
        with open(chord_stats_path, 'w') as f:
            json.dump(chord_stats, f, indent=2)
        print(f"\nChord statistics saved to {chord_stats_path}")
        
        # Analyze genres
        if 'main_genre' in df.columns:
            print("\nAnalyzing Genre Distribution:")
            genre_counts = df['main_genre'].value_counts()
            for i, (genre, count) in enumerate(genre_counts.head(10).items(), 1):
                if pd.notna(genre):
                    print(f"  {i}. {genre}: {count} songs")
        
        # Analyze by decade
        if 'decade' in df.columns:
            print("\nAnalyzing Decade Distribution:")
            decade_counts = df['decade'].value_counts().sort_index()
            for decade, count in decade_counts.items():
                if pd.notna(decade):
                    print(f"  {int(decade)}s: {count} songs")
        
        # Generate some examples of chord progressions
        print("\nExample Chord Progressions:")
        sample_songs = df.sample(min(5, len(df))).copy()
        
        for i, row in enumerate(sample_songs.itertuples(), 1):
            artist = getattr(row, 'artist_name', 'Unknown Artist') if hasattr(row, 'artist_name') else 'Unknown Artist'
            song = getattr(row, 'song_title', f'Song #{row.id}') if hasattr(row, 'song_title') else f'Song #{row.id}'
            
            # Clean up chords to display chord progression only
            chords = row.chords
            # Just show the first section (intro or verse)
            first_section = chords.split('<')[1] if '<' in chords else chords
            first_section = first_section.split('>')[1] if '>' in first_section else first_section
            
            print(f"  {i}. {artist if pd.notna(artist) else 'Unknown'} - {song if pd.notna(song) else 'Unknown'}")
            print(f"     {first_section.strip()}")
        
        print("\nExploration complete! You can find additional data files in the chordonomicon_data directory.")
        
    except Exception as e:
        print(f"Error exploring data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for required packages
    try:
        import pandas
        import matplotlib
        import seaborn
    except ImportError:
        print("Required packages not found. Installing...")
        os.system("pip install pandas matplotlib seaborn")
        print("Packages installed.")
    
    # Run the exploration function
    explore_chordonomicon_data()
