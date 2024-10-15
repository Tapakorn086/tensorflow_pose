import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as posenet from '@tensorflow-models/posenet';

const HumanPoseEstimation = ({ mode = 'upload', preloadedImage = null }) => {
  const [images, setImages] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [poses, setPoses] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  const loadImages = (e) => {
    const files = Array.from(e.target.files);
    const readers = files.map(file => {
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          const img = new Image();
          img.onload = () => resolve(img);
          img.src = e.target.result;
        };
        reader.readAsDataURL(file);
      });
    });

    Promise.all(readers).then(loadedImages => {
      setImages(loadedImages);
      if (loadedImages.length > 0) {
        estimatePose(loadedImages[0]);
      }
    });
  };

  const estimatePose = async (img) => {
    setIsLoading(true);
    setError(null);
    try {
      const net = await posenet.load({
        inputResolution: { width: 640, height: 480 },
        scale: 0.5,
      });
      const pose = await net.estimateSinglePose(img);
      setPoses([pose]);
      drawResults(img, [pose]);
    } catch (err) {
      console.error('Error in estimatePose: ', err);
      setError('Failed to estimate pose. Please try another image.');
    }
    setIsLoading(false);
  };

  const drawResults = (img, poses) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    poses.forEach(({ keypoints }) => {
      drawKeypoints(keypoints, 0.5, ctx);
      drawSkeleton(keypoints, 0.5, ctx);
    });
  };

  const drawKeypoints = (keypoints, minConfidence, ctx) => {
    keypoints.forEach((keypoint) => {
      if (keypoint.score > minConfidence) {
        const { y, x } = keypoint.position;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = 'red';
        ctx.fill();
      }
    });
  };

  const drawSkeleton = (keypoints, minConfidence, ctx) => {
    const adjacentKeyPoints = posenet.getAdjacentKeyPoints(keypoints, minConfidence);
    adjacentKeyPoints.forEach((keypoints) => {
      drawSegment(
        toTuple(keypoints[0].position),
        toTuple(keypoints[1].position),
        'green',
        ctx
      );
    });
  };

  const drawSegment = ([ay, ax], [by, bx], color, ctx) => {
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.stroke();
  };

  const toTuple = ({ y, x }) => [y, x];

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const handleImageSelect = (e) => {
    const index = parseInt(e.target.value);
    setCurrentImageIndex(index);
    estimatePose(images[index]);
  };

  const handleDownload = () => {
    const canvas = canvasRef.current;
    const image = canvas.toDataURL("image/png").replace("image/png", "image/octet-stream");
    const link = document.createElement('a');
    link.download = `pose_estimation_${currentImageIndex + 1}.png`;
    link.href = image;
    link.click();
  };

  useEffect(() => {
    if (preloadedImage) {
      setImages([preloadedImage]);
      estimatePose(preloadedImage);
    }
  }, [preloadedImage]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '1rem' }}>
      {mode === 'upload' && (
        <>
          <input
            type="file"
            ref={fileInputRef}
            onChange={loadImages}
            accept="image/*"
            multiple
            style={{ display: 'none' }}
          />
          <button onClick={handleUploadClick} style={{ marginBottom: '1rem', padding: '0.5rem 1rem', backgroundColor: '#4CAF50', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
            Upload Images
          </button>
        </>
      )}
      
      {images.length > 0 && (
        <select 
          value={currentImageIndex} 
          onChange={handleImageSelect}
          style={{ marginBottom: '1rem', padding: '0.5rem', borderRadius: '4px' }}
        >
          {images.map((_, index) => (
            <option key={index} value={index}>
              Image {index + 1}
            </option>
          ))}
        </select>
      )}
      
      {isLoading && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <p>Estimating pose...</p>
        </div>
      )}
      
      {error && (
        <div style={{ backgroundColor: '#f8d7da', color: '#721c24', padding: '1rem', borderRadius: '4px', marginBottom: '1rem' }}>
          <h4 style={{ margin: 0 }}>Error</h4>
          <p style={{ margin: 0 }}>{error}</p>
        </div>
      )}
      
      <canvas
        ref={canvasRef}
        style={{ maxWidth: '100%', height: 'auto', border: '1px solid #ccc' }}
      />

      {images.length > 0 && !isLoading && (
        <button 
          onClick={handleDownload} 
          style={{ 
            marginTop: '1rem', 
            padding: '0.5rem 1rem', 
            backgroundColor: '#007bff', 
            color: 'white', 
            border: 'none', 
            borderRadius: '4px', 
            cursor: 'pointer' 
          }}
        >
          Download Image
        </button>
      )}
    </div>
  );
};

export const UploadPoseEstimation = () => (
  <HumanPoseEstimation mode="upload" />
);

export const PreloadedPoseEstimation = ({ imageUrl }) => {
  const [preloadedImage, setPreloadedImage] = useState(null);

  useEffect(() => {
    const img = new Image();
    img.onload = () => setPreloadedImage(img);
    img.src = imageUrl;
  }, [imageUrl]);

  return (
    <HumanPoseEstimation mode="preloaded" preloadedImage={preloadedImage} />
  );
};

export default HumanPoseEstimation;