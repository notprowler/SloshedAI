import { useEffect } from 'react'
import './App.css'

function App() {
  useEffect(() => {
    const videoElement = document.getElementById('video-feed') as HTMLImageElement;
    videoElement.src = "http://localhost:5000/video_feed";
  }, []);

  return (
    <>
      <img id="video-feed" alt="Video Feed" />
    </>
  )
}

export default App
