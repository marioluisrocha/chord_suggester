# Chordonomicon Data Tools

Tools for downloading, enriching, and exploring the Chordonomicon dataset from Hugging Face.

## Overview

This repository contains tools for working with the [Chordonomicon dataset](https://huggingface.co/datasets/ailsntua/Chordonomicon), a collection of song chord progressions with Spotify metadata. These tools allow you to:

1. Download the dataset locally
2. Enrich it with artist and song names from Spotify
3. Explore and analyze the chord progressions and metadata

## Things I'm working on right now

I'm planning to link the songs to another dataset containing their lyrics, such as the [Genius dataset](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information), and train a model on the lyrics to cross the chord progressions to the general sentiment of the songs.

I am also working on a remapper for the chords on the dataset to make it key agnostic. 


## Requirements

- Python 3.6+
- Required packages:
  - pandas
  - spotipy
  - huggingface_hub
  - matplotlib (for exploration)
  - seaborn (for exploration)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/marioluisrocha/chord_suggester
cd chord_suggester
```

2. Install the required packages:
```bash
pip install pandas spotipy huggingface_hub matplotlib seaborn fuzzywuzzy
```

## Usage

### 1. Downloading the Dataset

To download the Chordonomicon dataset:

```bash
python chordonomicon_downloader.py
```

This will:
- Create a `chordonomicon_data` directory
- Download the CSV file containing chord progressions
- Download the README and other metadata files
- Create a sample JSON file for easier inspection

### 2. Enriching with Spotify Metadata

The Chordonomicon dataset contains Spotify IDs but not artist names or song titles. To enrich the dataset with this information:

1. First, obtain Spotify API credentials:
   - Create a Spotify Developer account at https://developer.spotify.com/
   - Create a new application in the dashboard
   - Get your Client ID and Client Secret

2. Set your Spotify API credentials as environment variables:
```bash
# On macOS/Linux
export SPOTIPY_CLIENT_ID=your_client_id
export SPOTIPY_CLIENT_SECRET=your_client_secret

# On Windows (Command Prompt)
set SPOTIPY_CLIENT_ID=your_client_id
set SPOTIPY_CLIENT_SECRET=your_client_secret

# On Windows (PowerShell)
$env:SPOTIPY_CLIENT_ID='your_client_id'
$env:SPOTIPY_CLIENT_SECRET='your_client_secret'
```

3. Run the Spotify connector:
```bash
python spotify_connector_v2.py
```

Features of the Spotify connector:
- Handles rate limiting gracefully with automatic saving of partial results
- Resumes processing if interrupted
- Skips already processed entries for efficiency
- Provides detailed progress updates and statistics

### 3. Exploring the Dataset

After enriching the dataset, you can explore and analyze it:

```bash
python chordonomicon_explorer.py
```

This script provides:
- Analysis of chord usage patterns
- Distribution of songs by genre and decade
- Top artists by number of songs
- Examples of chord progressions with artist and song information
- Generation of additional JSON files with statistics and samples

## Files

- `chordonomicon_downloader.py`: Downloads the dataset from Hugging Face
- `spotify_connector_v2.py`: Enriches the dataset with Spotify artist and song information
- `chordonomicon_explorer.py`: Analyzes and explores the enriched dataset

## Directory Structure

After running the scripts, you'll have the following structure:

```
.
├── chordonomicon_data/
│   ├── chordonomicon_v2.csv          # Original dataset
│   ├── chordonomicon_enriched.csv    # Dataset with artist/song names
│   ├── sample_data.json              # Sample of original data
│   ├── enriched_sample.json          # Sample of enriched data
│   ├── chord_statistics.json         # Analysis of chord usage
│   ├── enrichment_summary.json       # Statistics about enrichment
│   ├── popular_artists_sample.json   # Sample of songs from popular artists
│   └── README.md                     # Original dataset documentation
├── chordonomicon_downloader.py
├── spotify_connector_v2.py
└── chordonomicon_explorer.py
```

## Acknowledgements

- The Chordonomicon dataset was created by [ailsntua](https://huggingface.co/ailsntua)
- Data is enriched using the [Spotify Web API](https://developer.spotify.com/documentation/web-api/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
