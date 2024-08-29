import React, { useState, useEffect } from 'react';
import { usePlayer } from '@empirica/core/player/classic/react';

const slides = [Diapositiva1, Diapositiva2, Diapositiva3, Diapositiva4, Diapositiva5, Diapositiva6, Diapositiva7, Diapositiva8, Diapositiva9, Diapositiva10];

export function ImageSlider({ condition }) {
  var [currentImageIndex, setCurrentImageIndex] = useState(0);
  const player = usePlayer();

  const handleKeyDown = (e) => {
    if (e.key === 'ArrowRight') {
        setCurrentImageIndex((prevIndex) =>  Math.min((prevIndex + 1), 9));
        if (currentImageIndex > slides.length-1) {
          currentImageIndex = currentImageIndex % slides.length;
        }
    } else if (e.key === 'ArrowLeft') {
      setCurrentImageIndex((prevIndex) => Math.max((prevIndex - 1), 0));
      if (currentImageIndex < 0) {
        currentImageIndex = currentImageIndex % slides.length;
    }
    }
    if (slides[currentImageIndex+1] === Diapositiva10) {
      condition();
      player.set("instructionsEnded",true);
    }
  };

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [currentImageIndex]);

  return (
    <div className="image-slider">
      <img src={slides[currentImageIndex]} alt={`Slide ${currentImageIndex + 1}` } />
    </div>
  );
};

export default ImageSlider;
