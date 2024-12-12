from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
import asyncio
from typing import Optional, Dict, Any, List
import torch
import clip
import cv2
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path
import time
import json
from fastapi.staticfiles import StaticFiles
import concurrent.futures
from functools import partial
import shutil
import base64
from pydantic import BaseModel

VIDEOS_DIR = Path("videos")
CACHE_DIR = Path("cache")
UPLOADS_DIR = Path("uploads")
FRAMES_DIR = UPLOADS_DIR / "frames"
SCREENSHOTS_DIR = Path("screenshots")
VIDEOS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
SCREENSHOTS_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Which Frame",
    max_upload_size=10 * 1024 * 1024 * 1024  # 10GB limit
)

app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",  # Allow all Vercel preview deployments
        "https://your-production-domain.com"  # Replace with your actual domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

if device == "cpu":
    print("Warning: Running ViT-L/14 on CPU may be slow. Consider using a GPU for better performance.")

TEST_VIDEO_NAME = "baby-driver.mp4"
TEST_DATA_DIR = Path("test_data")
TEST_DATA_DIR.mkdir(exist_ok=True)

# Global variable to track progress
current_progress = {"current_frame": 0, "total_frames": 0}

def process_frame_batch(frames: List[np.ndarray], start_idx: int, fps: float, video_frames_dir: Path, model, preprocess, device) -> List[Dict]:
    frames_data = []
    video_id = video_frames_dir.name
    
    for i, frame in enumerate(frames):
        frame_count = start_idx + i
        current_time = frame_count / fps
        
        # Save frame
        frame_path = video_frames_dir / f"frame_{frame_count}.jpg"
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Only process every fps frames for CLIP
        if frame_count % int(fps) == 0:
            frame_mean = cv2.mean(frame)[0]
            frame_std = cv2.meanStdDev(frame)[1][0][0]
            
            features = None
            if frame_mean >= 5 or frame_std >= 5:
                # Convert frame to RGB and preprocess
                image = preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                
                # Process in smaller batches if on CPU to manage memory
                with torch.no_grad():
                    try:
                        image_features = model.encode_image(image)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        features = image_features.cpu().numpy().tolist()[0]
                    except RuntimeError as e:
                        print(f"Warning: Memory error processing frame {frame_count}. Skipping...")
                        continue
            
            frames_data.append({
                "time": current_time,
                "frame_path": f"frames/{video_id}/frame_{frame_count}.jpg",
                "features": features,
                "brightness": float(frame_mean),
                "variance": float(frame_std)
            })
    
    return frames_data

def save_cache(video_id: str, data: Dict[str, Any]) -> None:
    """Save processed video data to disk."""
    cache_path = CACHE_DIR / video_id
    cache_path.mkdir(exist_ok=True)
    
    # Save metadata
    metadata = {
        'frame_indices': data['frame_indices'],
        'total_frames': int(data['total_frames']),
        'fps': float(data['fps']),
        'duration': float(data['duration']),
        'timestamp': time.time()
    }
    with open(cache_path / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    # Save embeddings
    torch.save(data['embeddings'], cache_path / "embeddings.pt")

def load_cache(video_id: str) -> Optional[Dict[str, Any]]:
    """Load processed video data from disk."""
    cache_path = CACHE_DIR / video_id
    if not cache_path.exists():
        return None
    
    try:
        # Load metadata
        with open(cache_path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Load embeddings
        embeddings = torch.load(cache_path / "embeddings.pt")
        
        return {
            'embeddings': embeddings,
            'frame_indices': metadata['frame_indices'],
            'total_frames': metadata['total_frames'],
            'fps': metadata['fps'],
            'duration': metadata['duration']
        }
    except Exception as e:
        print(f"Error loading cache for {video_id}: {e}")
        return None

def save_test_data(video_id: str, processed_data: Dict, frames_dir: Path) -> None:
    """Save processed data for test video."""
    if not TEST_DATA_DIR.exists():
        TEST_DATA_DIR.mkdir()
        
    # Save JSON data
    with open(TEST_DATA_DIR / "processed_data.json", "w") as f:
        json.dump(processed_data, f)
    
    # Save frames
    test_frames_dir = TEST_DATA_DIR / "frames"
    if test_frames_dir.exists():
        shutil.rmtree(test_frames_dir)
    shutil.copytree(frames_dir, test_frames_dir)

def load_test_data() -> Optional[Dict]:
    """Load processed data for test video."""
    json_path = TEST_DATA_DIR / "processed_data.json"
    if not json_path.exists():
        return None
        
    with open(json_path, "r") as f:
        return json.load(f)

@app.get("/progress")
async def progress():
    async def event_generator():
        while True:
            yield {
                "event": "message",
                "data": json.dumps(current_progress)
            }
            if current_progress["current_frame"] >= current_progress["total_frames"] > 0:
                break
            await asyncio.sleep(0.2)
    return EventSourceResponse(event_generator())

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        global current_progress
        current_progress = {"current_frame": 0, "total_frames": 0}
        print("Starting upload")  # Debug print
        
        # Save video file first to get total frames
        upload_dir = Path("uploads")
        frames_dir = upload_dir / "frames"
        upload_dir.mkdir(exist_ok=True)
        frames_dir.mkdir(exist_ok=True)
        
        video_id = str(int(time.time()))
        video_path = upload_dir / f"{video_id}.mp4"
        with open(video_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Get video info first
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Set total frames immediately
        current_progress["total_frames"] = total_frames
        print(f"Total frames: {total_frames}")  # Debug print
        
        # Check if this is the test video
        is_test_video = file.filename == TEST_VIDEO_NAME
        if is_test_video:
            test_data = load_test_data()
            if test_data is not None:
                video_id = test_data["id"]
                target_frames_dir = Path("uploads") / "frames" / video_id
                if not target_frames_dir.exists():
                    shutil.copytree(TEST_DATA_DIR / "frames", target_frames_dir)
                return {
                    "id": video_id,
                    "duration": test_data["duration"],
                    "fps": test_data["fps"]
                }
        
        video_frames_dir = frames_dir / video_id
        video_frames_dir.mkdir(exist_ok=True)
        
        # Process frames in batches
        frames_data = []
        batch_size = 30
        frame_count = 0
        
        while True:
            batch_frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                batch_frames.append(frame)
                frame_count += 1
                current_progress["current_frame"] = frame_count
                print(f"Processed frame {frame_count}/{total_frames}")  # Debug print
            
            if not batch_frames:
                break
            
            # Process batch
            batch_data = process_frame_batch(
                batch_frames,
                frame_count - len(batch_frames),
                fps,
                video_frames_dir,
                model,
                preprocess,
                device
            )
            frames_data.extend(batch_data)
            
            # Clear batch from memory
            batch_frames.clear()
        
        cap.release()
        print("Processing complete")  # Debug print
        
        if not frames_data:
            raise HTTPException(status_code=400, detail="No frames could be extracted from video")
        
        # Sort frames_data by time to ensure correct order
        frames_data.sort(key=lambda x: x["time"])
        
        # Save processed data
        processed_data = {
            "id": video_id,
            "duration": duration,
            "fps": fps,
            "total_frames": frame_count,
            "frames": frames_data
        }
        
        with open(upload_dir / f"{video_id}.json", "w") as f:
            json.dump(processed_data, f)
        
        # After processing, save test data if this is the test video
        if is_test_video:
            save_test_data(video_id, processed_data, video_frames_dir)
        
        return {"id": video_id, "duration": duration, "fps": fps}
        
    except Exception as e:
        print(f"Error during upload: {e}")  # Debug print
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        current_progress = {"current_frame": 0, "total_frames": 0}
        print("Upload complete")  # Debug print

@app.get("/video/{video_id}")
async def get_video(video_id: str):
    video_path = Path("uploads") / f"{video_id}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path)

def scale_similarity(normalized_sim: float) -> float:
    """
    Scale normalized similarity scores with better differentiation at higher scores
    and less likelihood of hitting exactly 1.0
    """
    # Use sigmoid-like function with adjusted steepness and offset
    scaled = 1 / (1 + np.exp(-2 * (normalized_sim - 0.5)))
    
    # Apply power transformation to spread out high scores
    scaled = np.power(scaled, 0.7)
    
    # Scale to 0.95 max to avoid exact 100% matches
    scaled = scaled * 0.95
    
    return float(scaled)

@app.post("/search/{video_id}")
async def search_frames(video_id: str, query: str = Body(...), threshold: float = Body(0.0)):
    try:
        # Try to load from test data first
        test_data = load_test_data()
        if test_data is not None and video_id == test_data["id"]:
            processed_data = test_data
        else:
            # Load from regular uploads
            data_path = Path("uploads") / f"{video_id}.json"
            if not data_path.exists():
                raise HTTPException(status_code=404, detail="Video data not found")
            
            with open(data_path, "r") as f:
                processed_data = json.load(f)
        
        # Encode search query
        with torch.no_grad():
            text_features = model.encode_text(clip.tokenize(query).to(device))
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        # Find matches
        matches = []
        similarities = []
        valid_frames = []
        
        # First pass: collect all similarities for frames with features
        for frame in processed_data["frames"]:
            if frame["features"] is not None:
                similarity = float(np.dot(frame["features"], text_features.cpu().numpy()[0]))
                similarities.append(similarity)
                valid_frames.append(frame)
        
        if not similarities:
            return {
                "duration": processed_data["duration"],
                "matches": []
            }
            
        # Calculate statistics for normalization
        sim_mean = np.mean(similarities)
        sim_std = np.std(similarities)
        
        # Second pass: normalize and apply custom scaling
        for frame, raw_similarity in zip(valid_frames, similarities):
            # Z-score normalization with adjusted scale
            normalized_sim = (raw_similarity - sim_mean) / (sim_std + 1e-6)
            # Clip extreme values
            normalized_sim = np.clip(normalized_sim, -3, 3)
            # Scale to [0,1] range with better high-end differentiation
            scaled_sim = scale_similarity(normalized_sim)
            
            if scaled_sim > threshold:
                matches.append({
                    "time": frame["time"],
                    "score": float(scaled_sim),
                    "frame_path": frame["frame_path"]
                })
        
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "duration": processed_data["duration"],
            "matches": matches
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_image/{video_id}")
async def search_with_image(video_id: str, data: Dict[str, str]):
    try:
        # Load the image from local path
        image_path = data.get("image_path")
        if not image_path:
            raise HTTPException(status_code=400, detail="No image path provided")
            
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image not found at path: {image_path}")
            
        try:
            image = Image.open(image_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to open image: {str(e)}")
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        try:
            image_input = preprocess(image).unsqueeze(0).to(device)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to preprocess image: {str(e)}")
        
        # Encode the image with CLIP
        try:
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.cpu().numpy()[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to encode image with CLIP: {str(e)}")
        
        # Load video data
        test_data = load_test_data()
        if test_data is not None and video_id == test_data["id"]:
            processed_data = test_data
        else:
            data_path = Path("uploads") / f"{video_id}.json"
            if not data_path.exists():
                raise HTTPException(status_code=404, detail=f"Video data not found for ID: {video_id}")
            try:
                with open(data_path, "r") as f:
                    processed_data = json.load(f)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load video data: {str(e)}")
        
        # Find matches using vectorized operations
        valid_frames = []
        valid_features = []
        
        # Collect valid frames and their features
        for frame in processed_data["frames"]:
            if frame.get("features") is not None:
                valid_frames.append(frame)
                valid_features.append(frame["features"])
        
        if not valid_frames:
            return {
                "duration": processed_data.get("duration", 0),
                "matches": []
            }
        
        try:
            # Convert to numpy array for vectorized operations
            features_array = np.array(valid_features)
            
            # Calculate cosine similarities
            similarities = np.dot(features_array, image_features)
            
            # Calculate statistics for normalization
            sim_mean = np.mean(similarities)
            sim_std = np.std(similarities)
            
            # Create matches for frames above threshold
            matches = []
            threshold = 0.0  # Set threshold to 0 for image search
            
            for frame, similarity in zip(valid_frames, similarities):
                # Z-score normalization with adjusted scale
                normalized_sim = (similarity - sim_mean) / (sim_std + 1e-6)
                # Clip extreme values
                normalized_sim = np.clip(normalized_sim, -3, 3)
                # Scale to [0,1] range with better high-end differentiation
                scaled_sim = scale_similarity(normalized_sim)
                
                if scaled_sim > threshold:
                    matches.append({
                        "time": frame["time"],
                        "score": float(scaled_sim),
                        "frame_path": frame["frame_path"]
                    })
            
            # Sort matches by score
            matches.sort(key=lambda x: x["score"], reverse=True)
            
            return {
                "duration": processed_data.get("duration", 0),
                "matches": matches
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to compute similarities: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Image search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

class ScreenshotData(BaseModel):
    image_data: str
    timestamp: int
    video_id: str
    frame_time: float

@app.post("/save_screenshot")
async def save_screenshot(
    image: UploadFile = File(...),
    timestamp: str = Form(...),
    video_id: str = Form(...),
    frame_time: str = Form(...)
):
    try:
        # Read image data
        image_data = await image.read()
        
        try:
            # Create image from bytes
            pil_image = Image.open(io.BytesIO(image_data))
            # Convert to RGB mode if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        # Create directory if it doesn't exist
        screenshots_dir = Path("screenshots") / video_id
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the image
        image_path = screenshots_dir / f"screenshot_{timestamp}.jpg"
        pil_image.save(image_path, "JPEG", quality=95)
        
        return {"image_path": str(image_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save screenshot: {str(e)}")
