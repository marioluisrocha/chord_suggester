import os
import json
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
from tqdm.auto import tqdm  # Fixed import to use tqdm correctly

def enrich_chordonomicon_with_spotify():
    """
    Connect to Spotify API and enrich the Chordonomicon dataset with
    artist names and song titles based on the Spotify IDs.
    """
    print("Chordonomicon Spotify Enrichment Tool")
    print("=====================================")
    
    # Check if credentials are set
    client_id = os.environ.get('SPOTIPY_CLIENT_ID')
    client_secret = os.environ.get('SPOTIPY_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("\nError: Spotify API credentials not found!")
        print("You need to set your Spotify API credentials as environment variables:")
        print("  • SPOTIPY_CLIENT_ID")
        print("  • SPOTIPY_CLIENT_SECRET")
        print("\nTo get these credentials:")
        print("1. Go to https://developer.spotify.com/dashboard/")
        print("2. Log in with your Spotify account")
        print("3. Create a new app")
        print("4. Copy the Client ID and Client Secret")
        print("\nThen set them as environment variables:")
        print("  • On Windows (Command Prompt):")
        print("    set SPOTIPY_CLIENT_ID=your_client_id")
        print("    set SPOTIPY_CLIENT_SECRET=your_client_secret")
        print("  • On Windows (PowerShell):")
        print("    $env:SPOTIPY_CLIENT_ID='your_client_id'")
        print("    $env:SPOTIPY_CLIENT_SECRET='your_client_secret'")
        print("  • On macOS/Linux:")
        print("    export SPOTIPY_CLIENT_ID=your_client_id")
        print("    export SPOTIPY_CLIENT_SECRET=your_client_secret")
        return
    
    try:
        # Initialize Spotify client
        print("\nConnecting to Spotify API...")
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())
        
        # Load the Chordonomicon dataset
        csv_path = os.path.join("chordonomicon_data", "chordonomicon_v2.csv")
        print(f"Loading Chordonomicon dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"Dataset loaded: {len(df)} records")
        
        # Initialize new columns
        df['artist_name'] = None
        df['song_title'] = None
        
        # Process in batches to avoid rate limiting
        processed_tracks = 0
        processed_artists = 0
        
        # Process unique artist IDs to reduce API calls
        print("\nProcessing artists...")
        unique_artist_ids = df['spotify_artist_id'].dropna().unique()
        artist_dict = {}
        
        # Fixed the tqdm usage
        for i in range(0, len(unique_artist_ids), 50):  # Spotify allows up to 50 IDs per request
            batch = unique_artist_ids[i:i+50]
            batch = [aid for aid in batch if aid]  # Filter out any None or empty values
            
            if batch:
                try:
                    results = sp.artists(batch)
                    for artist in results['artists']:
                        if artist:
                            artist_dict[artist['id']] = artist['name']
                    processed_artists += len(batch)
                    print(f"Processed {processed_artists}/{len(unique_artist_ids)} artists", end='\r')
                    time.sleep(0.1)  # Small delay to avoid hitting rate limits
                except Exception as e:
                    print(f"Error fetching artists {i} to {i+50}: {str(e)}")
        
        print(f"\nProcessed {processed_artists} artists")
        
        # Update artist names in the DataFrame
        df['artist_name'] = df['spotify_artist_id'].map(artist_dict)
        
        # Process tracks
        print("\nProcessing tracks...")
        valid_track_ids = df['spotify_song_id'].dropna().unique()
        track_dict = {}
        
        # Fixed the tqdm usage
        for i in range(0, len(valid_track_ids), 50):  # Spotify allows up to 50 IDs per request
            batch = valid_track_ids[i:i+50]
            batch = [tid for tid in batch if tid]  # Filter out any None or empty values
            
            if batch:
                try:
                    results = sp.tracks(batch)
                    for track in results['tracks']:
                        if track:
                            track_dict[track['id']] = track['name']
                    processed_tracks += len(batch)
                    print(f"Processed {processed_tracks}/{len(valid_track_ids)} tracks", end='\r')
                    time.sleep(0.1)  # Small delay to avoid hitting rate limits
                except Exception as e:
                    print(f"Error fetching tracks {i} to {i+50}: {str(e)}")
        
        print(f"\nProcessed {processed_tracks} tracks")
        
        # Update track names in the DataFrame
        df['song_title'] = df['spotify_song_id'].map(track_dict)
        
        # Save the enriched dataset
        output_path = os.path.join("chordonomicon_data", "chordonomicon_enriched.csv")
        df.to_csv(output_path, index=False)
        print(f"\nEnriched dataset saved to {output_path}")
        
        # Create a sample of the enriched data
        sample_df = df.head(20).copy()
        sample_path = os.path.join("chordonomicon_data", "enriched_sample.json")
        sample_df.to_json(sample_path, orient='records', indent=2)
        print(f"Sample of enriched data saved to {sample_path}")
        
        # Create a summary of the enrichment process
        summary = {
            'total_records': len(df),
            'records_with_artist_id': df['spotify_artist_id'].notna().sum(),
            'records_with_song_id': df['spotify_song_id'].notna().sum(),
            'artists_enriched': df['artist_name'].notna().sum(),
            'songs_enriched': df['song_title'].notna().sum(),
            'unique_artists': len(df['artist_name'].dropna().unique()),
            'unique_songs': len(df['song_title'].dropna().unique()),
        }
        
        summary_path = os.path.join("chordonomicon_data", "enrichment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\nEnrichment Summary:")
        print(f"  • Total records: {summary['total_records']}")
        print(f"  • Records with artist ID: {summary['records_with_artist_id']}")
        print(f"  • Records with song ID: {summary['records_with_song_id']}")
        print(f"  • Artists enriched: {summary['artists_enriched']}")
        print(f"  • Songs enriched: {summary['songs_enriched']}")
        print(f"  • Unique artists: {summary['unique_artists']}")
        print(f"  • Unique songs: {summary['unique_songs']}")
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for required packages
    try:
        import spotipy
        import pandas
    except ImportError:
        print("Required packages not found. Installing...")
        os.system("pip install spotipy pandas")
        print("Packages installed.")
    
    # Run the enrichment function
    enrich_chordonomicon_with_spotify()
