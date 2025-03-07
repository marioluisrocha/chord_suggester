import os
import json
import requests
import pandas as pd
from huggingface_hub import hf_hub_download

def download_chordonomicon():
    """
    Download the Chordonomicon dataset from Hugging Face and store it locally.
    Only downloads the chordonomicon_v2.csv file and README.md.
    """
    print("Downloading Chordonomicon dataset from Hugging Face...")
    
    # Create a directory to store the dataset
    os.makedirs("chordonomicon_data", exist_ok=True)
    
    repo_id = "ailsntua/Chordonomicon"
    csv_filename = "chordonomicon_v2.csv"
    
    try:
        # Download the main CSV file
        print(f"Downloading {csv_filename}...")
        csv_path = hf_hub_download(repo_id=repo_id, filename=csv_filename, repo_type="dataset")
        
        # Copy the file to our local directory
        target_path = os.path.join("chordonomicon_data", csv_filename)
        with open(csv_path, 'rb') as src, open(target_path, 'wb') as dst:
            dst.write(src.read())
        
        print(f"Successfully downloaded {csv_filename}")
        
        # Process and display information about the CSV file
        process_csv_file(target_path)
        
        # Download the README file
        print("\nDownloading README.md...")
        try:
            readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="dataset")
            
            # Copy the README to our directory
            readme_target = os.path.join("chordonomicon_data", "README.md")
            with open(readme_path, 'rb') as src, open(readme_target, 'wb') as dst:
                dst.write(src.read())
                
            print("Successfully downloaded README.md")
        except Exception as e:
            print(f"Could not download README.md: {str(e)}")
        
        print("\nChordonomicon dataset has been successfully downloaded and stored locally.")
        print(f"Data location: {os.path.abspath('chordonomicon_data')}")
    
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        import traceback
        traceback.print_exc()

def process_csv_file(file_path):
    """
    Process the CSV file to extract and display information.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"\nCSV file loaded: {os.path.basename(file_path)}")
        print(f"  - Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  - Columns: {', '.join(df.columns.tolist())}")
        
        # Save a sample as JSON for easier inspection
        sample_path = os.path.join(os.path.dirname(file_path), "sample_data.json")
        df.head(10).to_json(sample_path, orient='records', indent=2)
        print(f"  - Sample data saved to: {sample_path}")
        
    except Exception as e:
        print(f"Error processing CSV file {file_path}: {str(e)}")

if __name__ == "__main__":
    print("Chordonomicon Database Downloader")
    print("=================================")
    
    # Check for required packages
    try:
        import huggingface_hub
        import pandas
    except ImportError:
        print("Required packages not found. Installing...")
        os.system("pip install huggingface_hub pandas")
        print("Packages installed.")
    
    # Run the download function
    download_chordonomicon()
