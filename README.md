# Which Frame

Search a video *semantically* using natural language, images, or snapshots from the video.

![Demo](./demo/whichframe.gif)

## Setting up

### Prerequisites
- [Python 3.8+](https://www.python.org/downloads/)
- [Node.js 16+](https://nodejs.org/en/download/package-manager)

### Setting up

1. Clone the repository:
```bash
git clone https://github.com/chuanenlin/whichframe-v2.git
cd whichframe-v2
```

2. Setup the frontend in one terminal:
```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev
```

3. Setup the backend in a second terminal:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

4. Open http://localhost:3000 in your browser.

## Usage

### Search
- Language: Type a query in the search bar
- Image: Upload a reference image
- Snapshot: Draw a bounding box on the video

### Interact with results
- Select a matched result to jump to the timestamp
- Scrub through the video timeline
  - Marker transparency indicates similarity scores
- Press `Space` to play/pause the video

## License

MIT License
