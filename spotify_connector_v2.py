import os
import json
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import sys

def enrich_chordonomicon_with_spotify():
    """
    Connect to Spotify API and enrich the Chordonomicon dataset with
    artist names and song titles based on the Spotify IDs.
    
    Features:
    - Handles rate limiting gracefully
    - Saves partial results if rate limited
    - Skips requesting information for items already in the enriched file
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
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(), retries=3)
        
        # Load the Chordonomicon dataset
        original_csv_path = os.path.join("chordonomicon_data", "chordonomicon_v2.csv")
        enriched_csv_path = os.path.join("chordonomicon_data", "chordonomicon_enriched.csv")
        
        print(f"Loading Chordonomicon dataset from {original_csv_path}")
        original_df = pd.read_csv(original_csv_path)
        
        # Check if enriched file already exists
        existing_artist_dict = {}
        existing_track_dict = {}
        if os.path.exists(enriched_csv_path):
            print(f"Found existing enriched dataset at {enriched_csv_path}")
            enriched_df = pd.read_csv(enriched_csv_path)
            
            # Extract existing mappings to avoid redundant API calls
            for _, row in enriched_df.iterrows():
                if pd.notna(row.get('spotify_artist_id')) and pd.notna(row.get('artist_name')):
                    existing_artist_dict[row['spotify_artist_id']] = row['artist_name']
                
                if pd.notna(row.get('spotify_song_id')) and pd.notna(row.get('song_title')):
                    existing_track_dict[row['spotify_song_id']] = row['song_title']
                    
            print(f"Loaded {len(existing_artist_dict)} existing artist mappings")
            print(f"Loaded {len(existing_track_dict)} existing track mappings")
            
            # Use the enriched df as our starting point
            df = enriched_df.copy()
        else:
            print("No existing enriched dataset found. Starting fresh.")
            # Start with a fresh DataFrame
            df = original_df.copy()
            if 'artist_name' not in df.columns:
                df['artist_name'] = None
            if 'song_title' not in df.columns:
                df['song_title'] = None
        
        print(f"Dataset loaded: {len(df)} records")
        
        # Initialize counters
        processed_tracks = 0
        processed_artists = 0
        rate_limited = False
        
        # Process unique artist IDs to reduce API calls
        print("\nProcessing artists...")
        unique_artist_ids = df[df['artist_name'].isna() & df['spotify_artist_id'].notna()]['spotify_artist_id'].unique()
        print(f"Found {len(unique_artist_ids)} artists that need metadata")
        
        artist_dict = existing_artist_dict.copy()
        
        try:
            for i in range(0, len(unique_artist_ids), 50):  # Spotify allows up to 50 IDs per request
                batch = unique_artist_ids[i:i+50]
                batch = [aid for aid in batch if aid and aid not in existing_artist_dict]  # Filter out any None or already processed
                
                if not batch:
                    continue  # Skip if all IDs in this batch were already processed
                    
                try:
                    results = sp.artists(batch)
                    for artist in results['artists']:
                        if artist:
                            artist_dict[artist['id']] = artist['name']
                    processed_artists += len(batch)
                    print(f"Processed {processed_artists}/{len(unique_artist_ids)} artists", end='\r')
                    
                    # Save intermediate results every 200 artists processed
                    if processed_artists % 200 == 0:
                        df['artist_name'] = df['spotify_artist_id'].map(artist_dict)
                        df.to_csv(enriched_csv_path, index=False)
                        print(f"\nSaved intermediate results after processing {processed_artists} artists")
                    
                    # Adaptive rate limiting: slow down as we process more to avoid hitting limits
                    if processed_artists > 1000:
                        time.sleep(0.5)  # Longer delay for large datasets
                    else:
                        time.sleep(0.2)  # Small delay to avoid hitting rate limits
                        
                except spotipy.exceptions.SpotifyException as e:
                    if hasattr(e, 'http_status') and e.http_status == 429:  # Rate limiting error
                        rate_limited = True
                        retry_after = int(e.headers.get('Retry-After', 30)) if hasattr(e, 'headers') else 30
                        print(f"\nRate limited by Spotify API! Need to wait {retry_after} seconds.")
                        print("Saving partial results and will continue when you run the script again.")
                        break
                    else:
                        print(f"Error fetching artists {i} to {i+50}: {str(e)}")
                        
        except Exception as e:
            print(f"\nUnexpected error while processing artists: {str(e)}")
        
        print(f"\nProcessed {processed_artists} artists")
        
        # Update artist names in the DataFrame
        df['artist_name'] = df['spotify_artist_id'].map(artist_dict)
        
        # If we got rate limited during artist processing, save what we have and exit
        if rate_limited:
            df.to_csv(enriched_csv_path, index=False)
            print(f"Partial results saved to {enriched_csv_path}")
            print("Please wait a while before running the script again to continue processing.")
            return
        
        # Process tracks
        print("\nProcessing tracks...")
        # Only process tracks that don't already have titles
        unique_track_ids = df[df['song_title'].isna() & df['spotify_song_id'].notna()]['spotify_song_id'].unique()
        print(f"Found {len(unique_track_ids)} tracks that need metadata")
        
        track_dict = existing_track_dict.copy()
        
        try:
            for i in range(0, len(unique_track_ids), 50):  # Spotify allows up to 50 IDs per request
                batch = unique_track_ids[i:i+50]
                batch = [tid for tid in batch if tid and tid not in existing_track_dict]  # Filter out any None or already processed
                
                if not batch:
                    continue  # Skip if all IDs in this batch were already processed
                    
                try:
                    results = sp.tracks(batch)
                    for track in results['tracks']:
                        if track:
                            track_dict[track['id']] = track['name']
                    processed_tracks += len(batch)
                    print(f"Processed {processed_tracks}/{len(unique_track_ids)} tracks", end='\r')
                    
                    # Save intermediate results every 200 tracks processed
                    if processed_tracks % 200 == 0:
                        df['song_title'] = df['spotify_song_id'].map(track_dict)
                        df.to_csv(enriched_csv_path, index=False)
                        print(f"\nSaved intermediate results after processing {processed_tracks} tracks")
                    
                    # Adaptive rate limiting
                    if processed_tracks > 1000:
                        time.sleep(0.5)
                    else:
                        time.sleep(0.2)
                        
                except spotipy.exceptions.SpotifyException as e:
                    if hasattr(e, 'http_status') and e.http_status == 429:  # Rate limiting error
                        rate_limited = True
                        retry_after = int(e.headers.get('Retry-After', 30)) if hasattr(e, 'headers') else 30
                        print(f"\nRate limited by Spotify API! Need to wait {retry_after} seconds.")
                        print("Saving partial results and will continue when you run the script again.")
                        break
                    else:
                        print(f"Error fetching tracks {i} to {i+50}: {str(e)}")
                        
        except Exception as e:
            print(f"\nUnexpected error while processing tracks: {str(e)}")
        
        print(f"\nProcessed {processed_tracks} tracks")
        
        # Update track names in the DataFrame
        df['song_title'] = df['spotify_song_id'].map(track_dict)
        
        # Save the enriched dataset
        df.to_csv(enriched_csv_path, index=False)
        
        if rate_limited:
            print(f"\nPartial results saved to {enriched_csv_path} due to rate limiting")
            print("Please wait a while before running the script again to continue processing.")
        else:
            print(f"\nEnriched dataset saved to {enriched_csv_path}")
        
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
            'artists_remaining': df[df['artist_name'].isna() & df['spotify_artist_id'].notna()].shape[0],
            'songs_remaining': df[df['song_title'].isna() & df['spotify_song_id'].notna()].shape[0],
            'unique_artists': len(df['artist_name'].dropna().unique()),
            'unique_songs': len(df['song_title'].dropna().unique()),
            'completion_percentage': round(
                (df['artist_name'].notna().sum() + df['song_title'].notna().sum()) /
                (df['spotify_artist_id'].notna().sum() + df['spotify_song_id'].notna().sum()) * 100
                if (df['spotify_artist_id'].notna().sum() + df['spotify_song_id'].notna().sum()) > 0 else 0,
                2
            ),
            'rate_limited': rate_limited
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
        print(f"  • Artists remaining: {summary['artists_remaining']}")
        print(f"  • Songs remaining: {summary['songs_remaining']}")
        print(f"  • Completion percentage: {summary['completion_percentage']}%")
        
        if rate_limited:
            print("\nNote: Processing was interrupted due to rate limiting.")
            print("Run the script again after waiting to continue enrichment.")
        elif summary['completion_percentage'] < 100:
            print("\nNote: Some records could not be enriched, possibly due to invalid IDs.")
        else:
            print("\nSuccess! All records with Spotify IDs have been enriched.")
        
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
