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

### 3. GPU-Accelerated Lyrics Merging

To merge the Spotify dataset with a lyrics dataset using GPU acceleration:

```bash
python gpu_merge_datasets.py spotify_data.csv lyrics_data.csv output.csv
```

This script uses CUDA-accelerated processing to efficiently match songs from the Spotify dataset with their lyrics from a lyrics dataset.

#### Requirements for GPU Acceleration

- NVIDIA GPU with CUDA support
- Required packages:
  - cudf
  - cupy
  - numba
  - rapidfuzz

#### Advanced Usage with Language Filtering

The script supports filtering the lyrics dataset by language before merging:

```bash
python gpu_merge_datasets.py spotify_data.csv lyrics_data.csv output.csv --language en --language-column language_cld3
```

Features of the language filtering:
- Filter by a single language: `--language en` (for English only)
- Filter by multiple languages: `--language en,es,fr` (for English, Spanish, and French)
- Specify which language column to use: `--language-column language_cld3` or `--language-column language_ft`
- Automatic detection of language columns if not specified

Other optional arguments:
- `--artist-threshold`: Threshold for fuzzy matching artist names (0-100, default: 85)
- `--title-threshold`: Threshold for fuzzy matching song titles (0-100, default: 85)
- `--chunk-size`: Number of rows to process at once (default: 10000)
- `--batch-size`: Number of rows to process in each GPU batch (default: 1000)

#### How it Works

1. Detects the artist, title, and lyrics columns in both datasets automatically
2. Applies language filtering to the lyrics dataset if specified
3. Processes data in chunks to minimize memory usage
4. Uses GPU-accelerated fuzzy matching to find the best matches when exact matches aren't found
5. Outputs a merged dataset with Spotify data and matching lyrics
6. Provides detailed statistics on match rates and processing time

### 4. Exploring the Dataset

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
- `gpu_merge_datasets.py`: Merges Spotify data with lyrics datasets using GPU acceleration
- `chordonomicon_explorer.py`: Analyzes and explores the enriched dataset

## Directory Structure

After running the scripts, you'll have the following structure:

```
.
├── chordonomicon_data/
│   ├── chordonomicon_v2.csv          # Original dataset
│   ├── chordonomicon_enriched.csv    # Dataset with artist/song names
│   ├── chordonomicon_with_lyrics.csv # Dataset with lyrics added
│   ├── sample_data.json              # Sample of original data
│   ├── enriched_sample.json          # Sample of enriched data
│   ├── chord_statistics.json         # Analysis of chord usage
│   ├── enrichment_summary.json       # Statistics about enrichment
│   ├── popular_artists_sample.json   # Sample of songs from popular artists
│   └── README.md                     # Original dataset documentation
├── chordonomicon_downloader.py
├── spotify_connector_v2.py
├── gpu_merge_datasets.py
└── chordonomicon_explorer.py
```

## Acknowledgements

- The Chordonomicon dataset was created by [ailsntua](https://huggingface.co/ailsntua)
- Data is enriched using the [Spotify Web API](https://developer.spotify.com/documentation/web-api/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.