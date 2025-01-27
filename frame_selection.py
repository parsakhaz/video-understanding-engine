#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import sys
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import os
from sklearn.cluster import KMeans
from model_loader import model_context

def process_batch(frames, model, preprocess, device):
    batch_size = len(frames)
    preprocessed_images = torch.stack([preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).to(device) for frame in frames])

    with torch.no_grad(), torch.amp.autocast('cuda'):
        image_features = model.encode_image(preprocessed_images)
        image_features = F.normalize(image_features, dim=-1)

    return image_features.cpu().numpy()

def sliding_window_filter(similarities, window_size=30):
    differences = []
    for i in range(len(similarities) - window_size + 1):
        window = similarities[i:i+window_size]
        avg_before = np.mean(window[:window_size//2], axis=0)
        avg_after = np.mean(window[window_size//2:], axis=0)
        difference = 1 - cosine_similarity([avg_before], [avg_after])[0][0]
        differences.append(difference)
    return differences

def find_interesting_frames(filtered_differences, frame_embeddings, min_skip=10, novelty_threshold=0.08, adaptive_threshold=True, n_clusters=15):
    novelty_frames = []
    last_interesting = -min_skip
    window_size = 100

    # Calculate adaptive threshold if enabled
    if adaptive_threshold:
        # Make adaptive threshold stricter by increasing the multiplier for std
        novelty_threshold = np.mean(filtered_differences) + 1.2 * np.std(filtered_differences)

    # Find novel frames
    for i, diff in enumerate(filtered_differences):
        if i - last_interesting >= min_skip:
            # Use local maximum within a window
            start = max(0, i - window_size // 2)
            end = min(len(filtered_differences), i + window_size // 2)
            if diff == max(filtered_differences[start:end]) and diff > novelty_threshold:
                novelty_frames.append(i)
                last_interesting = i

    # Perform clustering on frame embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(frame_embeddings)

    # Find representative frames from each cluster
    cluster_frames = []
    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) > 0:
            # Choose the frame closest to the cluster center
            center_frame = cluster_indices[np.argmin(np.linalg.norm(frame_embeddings[cluster_indices] - kmeans.cluster_centers_[cluster], axis=1))]
            if center_frame < len(filtered_differences):
                cluster_frames.append(int(center_frame))

    return novelty_frames, cluster_frames

# Add this new function to handle caching
def get_or_compute_embeddings(video_path):
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    cache_dir = 'embedding_cache'
    cache_file = os.path.join(cache_dir, f'{video_basename}.npy')

    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        return np.load(cache_file)

    print(f"No cache found. Loading model and computing embeddings for {video_path}")
    with model_context("clip") as model_tuple:
        if model_tuple is None:
            print("Failed to load CLIP model")
            return None
            
        model, preprocess, device = model_tuple
        print(f"Computing embeddings for {video_path}")
        cap = cv2.VideoCapture(video_path)
        frame_embeddings = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        batch_size = 64

        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while cap.isOpened():
                batch_frames = []
                for _ in range(batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    batch_frames.append(frame)

                if not batch_frames:
                    break

                batch_embeddings = process_batch(batch_frames, model, preprocess, device)
                frame_embeddings.extend(batch_embeddings)
                pbar.update(len(batch_frames))

        cap.release()
        frame_embeddings = np.array(frame_embeddings)

        # Save embeddings to cache
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_file, frame_embeddings)
        print(f"Cached embeddings saved to {cache_file}")

        return frame_embeddings

def process_video(video_path):
    # Get embeddings (either from cache or by computing)
    frame_embeddings = get_or_compute_embeddings(video_path)

    # Calculate similarities and apply sliding window filter
    print(f'finding cosine similarities')
    cosine_similarities = [cosine_similarity(frame_embeddings[i-1].reshape(1, -1),
                                            frame_embeddings[i].reshape(1, -1))[0][0]
                        for i in range(1, len(frame_embeddings))]

    print(f'finding filtered differences')
    filtered_differences = sliding_window_filter(frame_embeddings)

    print(f'finding interesting frames')
    novelty_frames, cluster_frames = find_interesting_frames(filtered_differences, frame_embeddings)

    # Create the frame_analysis_plots directory if it doesn't exist
    plots_dir = 'frame_analysis_plots'
    os.makedirs(plots_dir, exist_ok=True)

    # Plot results
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot sequential frame cosine similarities
    ax.plot(cosine_similarities, label='Sequential Frame Cosine Similarity', color='blue')

    # Plot filtered differences and key frames
    ax.plot(filtered_differences, label='Sliding Window CLIP Differences', color='orange')

    valid_novelty_frames = [i for i in novelty_frames if i < len(filtered_differences)]
    ax.scatter(valid_novelty_frames, [filtered_differences[i] for i in valid_novelty_frames],
               color='red', label='Novelty Frames', zorder=5)

    valid_cluster_frames = [i for i in cluster_frames if i < len(filtered_differences)]
    ax.scatter(valid_cluster_frames, [filtered_differences[i] for i in valid_cluster_frames],
               color='green', label='Cluster Representative Frames', zorder=6)

    ax.set_title('Frame Analysis: Similarities, Differences, and Key Frames')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Similarity / Difference')
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    # Save the plot in the frame_analysis_plots directory
    plot_filename = f'{os.path.splitext(os.path.basename(video_path))[0]}_frame_analysis.png'
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Plot saved as {plot_path}")

    all_key_frames = sorted(set(novelty_frames + cluster_frames))
    print(f"Key frames to use: {all_key_frames}")

    return all_key_frames

if __name__ == "__main__":
    # Main execution
    if len(sys.argv) < 2:
        print("Usage: python new_clip.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    process_video(video_path)