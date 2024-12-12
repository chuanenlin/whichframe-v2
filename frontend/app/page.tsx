'use client';

const originalError = console.error;
console.error = (...args: Parameters<typeof console.error>) => {
  if (args[0]?.includes?.('A component is changing an uncontrolled input')) {
    return;
  }
  originalError.apply(console, args);
};

import { useState, useCallback, useEffect, useRef } from 'react';
import styles from './page.module.css';
import Image from 'next/image';

interface Match {
  time: number;
  score: number;
  frame_path: string;
}

interface SearchResponse {
  duration: number;
  fps: number;
  total_frames: number;
  matches: Match[];
}

interface Region {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
}

interface UploadProgress {
  current_frame: number;
  total_frames: number;
}

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

const formatTimeWithMs = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${mins}:${secs.toString().padStart(2, '0')}:${ms.toString().padStart(2, '0')}`;
};

const BUFFER_SIZE = 100;

const preloadImage = (src: string): Promise<void> => {
  return new Promise((resolve) => {
    const img = new window.Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve();
    img.onerror = () => resolve();
    img.src = src;
  });
};

const preloadFrames = async (videoId: string, startFrame: number, endFrame: number): Promise<void> => {
  startFrame = Math.max(0, Math.floor(startFrame));
  endFrame = Math.max(0, Math.floor(endFrame));
  
  const promises = [];
  const maxConcurrent = 5;
  
  for (let i = startFrame; i <= endFrame; i += maxConcurrent) {
    const batchPromises = [];
    for (let j = 0; j < maxConcurrent && i + j <= endFrame; j++) {
      const framePath = getFrameUrl(videoId, i + j);
      batchPromises.push(preloadImage(framePath));
    }
    await Promise.all(batchPromises);
    promises.push(...batchPromises);
  }
};

const getFrameUrl = (videoId: string | null, frameNumber: number) => {
  if (!videoId) return '';
  return `${process.env.NEXT_PUBLIC_API_URL}/frames/${videoId}/frame_${frameNumber}.jpg`;
};

const preloadMatchImages = async (matches: Match[]) => {
  const promises = matches.map(match => 
    preloadImage(`${process.env.NEXT_PUBLIC_API_URL}/${match.frame_path}`)
  );
  await Promise.all(promises);
};

// Add debounce hook
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}

export default function Home() {
  const [videoId, setVideoId] = useState<string | null>(null);
  const [duration, setDuration] = useState<number>(0);
  const [fps, setFps] = useState<number>(0);
  const [isUploading, setIsUploading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const debouncedSearchQuery = useDebounce(searchQuery, 300); // 300ms delay after typing
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null);
  const [threshold] = useState(0);
  const [previewFrame, setPreviewFrame] = useState<string | null>(null);
  const [isMatchedFrame, setIsMatchedFrame] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [isScrubbing, setIsScrubbing] = useState(false);
  const [showAllMatches, setShowAllMatches] = useState(false);
  const [selectedRegion, setSelectedRegion] = useState<Region | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [showComplete, setShowComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const frameViewerRef = useRef<HTMLDivElement>(null);
  const intervalIdRef = useRef<number | null>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const API_URL = process.env.NEXT_PUBLIC_API_URL;
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);

  useEffect(() => {
    if (videoId && debouncedSearchQuery.trim()) {
      const performSearchAndPreload = async () => {
        setIsSearching(true);
        setShowComplete(false);
        try {
          const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/search/${videoId}`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              query: debouncedSearchQuery,
              threshold: threshold
            }),
          });

          if (!response.ok) throw new Error('Search failed');
          
          const data = await response.json();
          await preloadMatchImages(data.matches);
          setSearchResults(data);
          setShowComplete(true);
          setTimeout(() => setShowComplete(false), 1500);
          
        } catch (err) {
          console.error(err);
        } finally {
          setIsSearching(false);
        }
      };

      performSearchAndPreload();
    }
  }, [debouncedSearchQuery, threshold, videoId]);

  useEffect(() => {
    if (isPlaying && duration && fps && videoId) {
      const currentFrame = Math.floor(currentTime * fps);
      const lastFrameNumber = Math.max(0, Math.floor((duration * fps) - 1));
      
      const bufferEnd = Math.min(currentFrame + BUFFER_SIZE, lastFrameNumber);
      preloadFrames(videoId, currentFrame, bufferEnd);

      intervalIdRef.current = window.setInterval(() => {
        let newTime = currentTime + (1 / fps);
        if (newTime >= duration) {
          setIsPlaying(false);
          clearInterval(intervalIdRef.current!);
          newTime = duration;
        }
        
        setCurrentTime(newTime);
        const frameNumber = Math.floor(newTime * fps);
        setPreviewFrame(getFrameUrl(videoId, frameNumber));

        const nextBufferStart = frameNumber + 1;
        const nextBufferEnd = Math.min(nextBufferStart + BUFFER_SIZE, lastFrameNumber);
        preloadFrames(videoId, nextBufferStart, nextBufferEnd).catch(console.error);

        if (searchResults?.matches) {
          const isMatch = searchResults.matches.some(match => {
            const matchFrameNumber = Math.floor(match.time * fps);
            return matchFrameNumber === frameNumber;
          });
          setIsMatchedFrame(isMatch);
        }
      }, 1000 / fps);

      return () => {
        if (intervalIdRef.current !== null) {
          window.clearInterval(intervalIdRef.current);
          intervalIdRef.current = null;
        }
      };
    }
  }, [isPlaying, duration, fps, videoId, currentTime, searchResults]);

  const handleTimelineMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!duration || !fps || !videoId) return;
    
    setIsScrubbing(true);
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const percentage = x / rect.width;
    const time = percentage * duration;
    
    setCurrentTime(time);
    const frameNumber = Math.floor(time * fps);
    setPreviewFrame(getFrameUrl(videoId, frameNumber));
    
    if (searchResults?.matches) {
      const isMatch = searchResults.matches.some(match => {
        const matchFrameNumber = Math.floor(match.time * fps);
        return matchFrameNumber === frameNumber;
      });
      setIsMatchedFrame(isMatch);
    }
  }, [duration, fps, videoId, searchResults]);

  const handleTimelineMouseLeave = useCallback(() => {
    setIsScrubbing(false);
  }, []);

  const handleTimelineClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!duration || !searchResults) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const percentage = x / rect.width;
    const time = percentage * duration;
    
    setCurrentTime(time);
    const frameNumber = Math.floor(time * fps);
    setPreviewFrame(getFrameUrl(videoId, frameNumber));
    setIsScrubbing(false);
    
    const isMatch = searchResults.matches.some(match => {
      const matchFrameNumber = Math.floor(match.time * fps);
      return matchFrameNumber === Math.floor(time * fps);
    });
    setIsMatchedFrame(isMatch);
  }, [duration, searchResults, fps, videoId]);

  const handleMatchClick = useCallback((time: number) => {
    setCurrentTime(time);
    if (videoId && fps) {
      setPreviewFrame(getFrameUrl(videoId, Math.floor(time * fps)));
    }
  }, [videoId, fps]);

  const togglePlayPause = useCallback(() => {
    setIsPlaying(playing => !playing);
  }, []);

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    setIsUploading(true);
    setUploadProgress(null);
    setError(null);
    setSearchQuery('');
    setSearchResults(null);
    setPreviewFrame(null);
    setIsMatchedFrame(false);
    setIsPlaying(false);
    setCurrentTime(0);
    setIsScrubbing(false);
    setShowAllMatches(false);
    setSelectedRegion(null);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');
      
      const data = await response.json();
      setVideoId(data.id);
      setDuration(data.duration);
      setFps(data.fps);
      
      setCurrentTime(0);
      setPreviewFrame(getFrameUrl(data.id, 0));

      // Set up EventSource after successful upload
      const eventSource = new EventSource(`${process.env.NEXT_PUBLIC_API_URL}/progress`);
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('Progress update:', data);  // Debug log
          setUploadProgress(data);
          if (data.current_frame === data.total_frames) {
            eventSource.close();
            setIsUploading(false);
          }
        } catch (err) {
          console.error('Error parsing progress data:', err);
        }
      };

      eventSource.onerror = (error) => {
        console.error('EventSource error:', error);
        eventSource.close();
      };
      
    } catch (err) {
      setError('Failed to upload video');
      console.error(err);
      setIsUploading(false);
    }
  };

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (e.code === 'Space' && !['INPUT', 'TEXTAREA'].includes(target.tagName)) {
        e.preventDefault();
        togglePlayPause();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [togglePlayPause]);

  const playheadClassName = `${styles.playhead}${isScrubbing ? ' ' + styles.scrubbing : ''}`;

  const renderMatches = () => {
    if (!searchResults?.matches) return null;
    
    const topMatches = searchResults.matches.slice(0, 12);
    const remainingMatches = searchResults.matches.slice(12);
    
    return (
      <div className={styles.matches}>
        {topMatches.map((match, i) => (
          <div
            key={i}
            className={styles.match}
            onClick={() => handleMatchClick(match.time)}
            style={{
              backgroundImage: `url(${process.env.NEXT_PUBLIC_API_URL}/${match.frame_path})`
            }}
          >
            <div className={styles.matchContent}>
              <div className={`${styles.matchScore} ${
                match.score >= 0.7 ? styles.scoreHigh :
                match.score >= 0.4 ? styles.scoreMedium :
                styles.scoreLow
              }`}>
                {Math.round(match.score * 100)}%
              </div>
              <div className={styles.matchTime}>{formatTime(match.time)}</div>
            </div>
          </div>
        ))}
        
        {remainingMatches.length > 0 && (
          <div className={styles.expandButtonContainer}>
            <button 
              className={`${styles.expandButton} ${showAllMatches ? styles.expanded : ''}`}
              onClick={() => setShowAllMatches(!showAllMatches)}
            >
              {showAllMatches ? 'Show less' : `Show ${remainingMatches.length} more`}
            </button>
          </div>
        )}
        
        {showAllMatches && remainingMatches.map((match, i) => (
          <div
            key={i}
            className={`${styles.match} ${styles.matchVisible}`}
            onClick={() => handleMatchClick(match.time)}
            style={{
              backgroundImage: `url(${process.env.NEXT_PUBLIC_API_URL}/${match.frame_path})`
            }}
          >
            <div className={styles.matchContent}>
              <div className={`${styles.matchScore} ${
                match.score >= 0.7 ? styles.scoreHigh :
                match.score >= 0.4 ? styles.scoreMedium :
                styles.scoreLow
              }`}>
                {Math.round(match.score * 100)}%
              </div>
              <div className={styles.matchTime}>{formatTime(match.time)}</div>
            </div>
          </div>
        ))}
      </div>
    );
  };

  const handleImageSearch = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !videoId) return;

    try {
      setIsSearching(true);
      setShowComplete(false);
      
      const formData = new FormData();
      formData.append('image', file);
      formData.append('timestamp', Date.now().toString());
      formData.append('video_id', videoId);
      formData.append('frame_time', currentTime.toString());

      // Create a temporary URL for the uploaded image
      const imageUrl = URL.createObjectURL(file);
      setUploadedImageUrl(imageUrl);

      const saveResponse = await fetch(`${API_URL}/save_screenshot`, {
        method: 'POST',
        body: formData,
      });

      if (!saveResponse.ok) {
        const errorData = await saveResponse.json();
        throw new Error(errorData.detail || 'Failed to save image');
      }

      const { image_path } = await saveResponse.json();

      const searchResponse = await fetch(`${API_URL}/search_image/${videoId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image_path }),
      });

      if (!searchResponse.ok) {
        const errorData = await searchResponse.json();
        throw new Error(errorData.detail || 'Image search failed');
      }

      const data = await searchResponse.json();
      await preloadMatchImages(data.matches || []);
      setSearchResults(data);
      setShowComplete(true);
      setTimeout(() => setShowComplete(false), 1500);

    } catch (err) {
      console.error('Image search error:', err);
    } finally {
      setIsSearching(false);
      if (imageInputRef.current) {
        imageInputRef.current.value = '';
      }
    }
  };

  const handleRegionSearch = async () => {
    if (!videoId || !selectedRegion) return;
    
    try {
      setIsSearching(true);
      setShowComplete(false);
      const imagePath = await saveRegionScreenshot();
      
      const searchResponse = await fetch(`${API_URL}/search_image/${videoId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image_path: imagePath }),
      });

      if (!searchResponse.ok) {
        throw new Error(`Search failed: ${searchResponse.statusText}`);
      }

      const data = await searchResponse.json();
      await preloadMatchImages(data.matches || []);
      setSearchResults(data);
      setShowComplete(true);
      setTimeout(() => setShowComplete(false), 1500);

    } catch (err) {
      console.error('Search error:', err);
    } finally {
      setIsSearching(false);
    }
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!frameViewerRef.current) return;
    const rect = frameViewerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setSelectedRegion({
      startX: x,
      startY: y,
      endX: x,
      endY: y
    });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!selectedRegion || !frameViewerRef.current) return;
    const rect = frameViewerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const y = Math.max(0, Math.min(e.clientY - rect.top, rect.height));
    setSelectedRegion({
      ...selectedRegion,
      endX: x,
      endY: y
    });
  };

  const handleMouseUp = () => {
    if (selectedRegion) {
      const width = Math.abs(selectedRegion.endX - selectedRegion.startX);
      const height = Math.abs(selectedRegion.endY - selectedRegion.startY);
      if (width > 10 && height > 10) {
        handleRegionSearch();
      }
    }
    setSelectedRegion(null);
  };

  const saveRegionScreenshot = async () => {
    if (!selectedRegion || !frameViewerRef.current || !videoId) return '';
    
    const canvas = document.createElement('canvas');
    const image = frameViewerRef.current.querySelector('img');
    if (!image) return '';

    await new Promise((resolve) => {
      if (image.complete) {
        resolve(true);
      } else {
        image.onload = () => resolve(true);
      }
    });

    const rect = image.getBoundingClientRect();
    const scaleX = image.naturalWidth / rect.width;
    const scaleY = image.naturalHeight / rect.height;

    const startX = Math.min(selectedRegion.startX, selectedRegion.endX);
    const startY = Math.min(selectedRegion.startY, selectedRegion.endY);
    const width = Math.abs(selectedRegion.endX - selectedRegion.startX);
    const height = Math.abs(selectedRegion.endY - selectedRegion.startY);

    canvas.width = width * scaleX;
    canvas.height = height * scaleY;

    const ctx = canvas.getContext('2d');
    if (!ctx) return '';

    try {
      ctx.drawImage(
        image,
        startX * scaleX,
        startY * scaleY,
        width * scaleX,
        height * scaleY,
        0,
        0,
        canvas.width,
        canvas.height
      );

      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((blob) => {
          resolve(blob!);
        }, 'image/jpeg', 0.95);
      });

      // Create a temporary URL for the blob and set it as the uploaded image
      const imageUrl = URL.createObjectURL(blob);
      setUploadedImageUrl(imageUrl);

      const formData = new FormData();
      formData.append('image', blob, 'screenshot.jpg');
      formData.append('timestamp', Date.now().toString());
      formData.append('video_id', videoId);
      formData.append('frame_time', currentTime.toString());

      const response = await fetch(`${API_URL}/save_screenshot`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to save screenshot');
      }
      
      const { image_path } = await response.json();
      return image_path;

    } catch (error) {
      console.error('Screenshot error:', error);
      throw error;
    }
  };

  const renderSelectionBox = () => {
    if (!selectedRegion) return null;

    const left = Math.min(selectedRegion.startX, selectedRegion.endX);
    const top = Math.min(selectedRegion.startY, selectedRegion.endY);
    const width = Math.abs(selectedRegion.endX - selectedRegion.startX);
    const height = Math.abs(selectedRegion.endY - selectedRegion.startY);

    return (
      <div
        className={styles.selectionBox}
        style={{
          left: `${left}px`,
          top: `${top}px`,
          width: `${width}px`,
          height: `${height}px`,
        }}
      />
    );
  };

  return (
    <main className={styles.main}>
      <div className={styles.container}>
        <div className={styles.leftPanel}>
          {!videoId ? (
            <div className={styles.uploadArea}>
              <input
                type="file"
                accept="video/*"
                onChange={handleUpload}
                disabled={isUploading}
              />
              {isUploading && (
                <div className={styles.processingIndicator}>
                  <div className={styles.pulsingDot} />
                  <span>Processing video... {uploadProgress ? `${uploadProgress.current_frame}/${uploadProgress.total_frames} frames` : ''}</span>
                </div>
              )}
            </div>
          ) : (
            <div className={styles.searchSection}>
              <div className={styles.searchContainer}>
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Describe what you're looking for..."
                  className={styles.searchInput}
                  disabled={!videoId || isUploading}
                />
                {error && <div className={styles.error}>{error}</div>}
                <button
                  className={styles.imageUploadButton}
                  onClick={() => imageInputRef.current?.click()}
                  disabled={!videoId || isUploading}
                >
                  {uploadedImageUrl ? (
                    <Image
                      src={uploadedImageUrl}
                      alt="Uploaded image"
                      width={24}
                      height={24}
                      className={styles.uploadedThumbnail}
                    />
                  ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" />
                    </svg>
                  )}
                </button>
                <input
                  ref={imageInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageSearch}
                  style={{ display: 'none' }}
                />
              </div>
              {renderMatches()}
            </div>
          )}
        </div>

        {videoId && (
          <div className={styles.rightPanel}>
            {(isSearching || showComplete) && (
              <div className={`${styles.searchingIndicator} ${isSearching ? styles.searching : styles.complete}`}>
                <div className={`${styles.pulsingDot} ${isSearching ? styles.searching : styles.complete}`} />
                {isSearching ? 'Searching...' : 'Complete'}
              </div>
            )}
            <div className={styles.timelineContainer}>
              {isMatchedFrame && (
                <div className={styles.timelineHeader}>
                  <span className={styles.frameMatch}>
                    Matched Frame
                  </span>
                </div>
              )}
              <div 
                className={styles.timeline} 
                onMouseMove={handleTimelineMouseMove}
                onMouseLeave={handleTimelineMouseLeave}
                onClick={handleTimelineClick}
              >
                {searchResults?.matches.map((match, i) => {
                  const isTopMatch = searchResults.matches
                    .sort((a, b) => b.score - a.score)
                    .slice(0, 5)
                    .some(m => m.time === match.time);
                  
                  const isHighScore = match.score > 0.9;
                  const shouldShowThumbnail = isTopMatch || isHighScore;

                  const timelineWidth = 800;
                  const markerWidth = Math.max(4, Math.min(16, Math.ceil(timelineWidth / searchResults.total_frames * 8)));

                  return (
                    <div
                      key={i}
                      className={styles.marker}
                      style={{
                        left: `clamp(0.5rem, ${(match.time / duration) * 100}%, calc(100% - 0.5rem))`
                      }}
                    >
                      <div 
                        className={styles.markerLine} 
                        style={{ 
                          opacity: Math.pow(match.score, 3),
                          width: `${markerWidth}px`,
                          minWidth: `${markerWidth}px`,
                          maxWidth: `${markerWidth}px`
                        }} 
                      />
                      {shouldShowThumbnail && (
                        <div 
                          className={styles.markerThumbnail}
                          style={{ 
                            zIndex: currentTime >= match.time - 0.1 && currentTime <= match.time + 0.1 ? 100 : 10
                          }}
                          onClick={(e) => {
                            e.stopPropagation();
                            setCurrentTime(match.time);
                            if (videoId) {
                              const frameNumber = Math.floor(match.time * fps);
                              setPreviewFrame(getFrameUrl(videoId as string, frameNumber));
                            }
                            setIsMatchedFrame(true);
                          }}
                        >
                          <Image
                            src={`${process.env.NEXT_PUBLIC_API_URL}/${match.frame_path}`}
                            alt={`Match at ${formatTime(match.time)}`}
                            width={64}
                            height={36}
                            style={{ objectFit: 'cover' }}
                            unoptimized
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
                <div 
                  className={playheadClassName}
                  style={{
                    '--left': `${(currentTime / duration) * 100}%`
                  } as React.CSSProperties}
                />
              </div>
            </div>

            <div 
              ref={frameViewerRef}
              className={styles.frameViewer}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={() => setSelectedRegion(null)}
            >
              {previewFrame && (
                <>
                  <div className={styles.frameContainer}>
                    <Image
                      src={previewFrame}
                      alt="Video frame"
                      fill
                      style={{
                        objectFit: 'contain',
                        userSelect: 'none'
                      }}
                      crossOrigin="anonymous"
                      draggable={false}
                      unoptimized
                      priority
                    />
                  </div>
                  <div className={styles.frameTime}>{formatTimeWithMs(currentTime)}</div>
                </>
              )}
              {renderSelectionBox()}
            </div>
          </div>
        )}
      </div>
    </main>
  );
} 