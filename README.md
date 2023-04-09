# segment-anything-webdemo
A very basic flask app showing how to use segment-anything in browser.

You'll need to install `segment-anything` and `flask`. Then simply run `python server.py` and navigate to `localhost:5000` in your web-broser.

You'll also need to download the `sam_vit_h_4b8939.pth` model from the `segment-anything` repo and add it to `models/`. An example ONNX decoder/predictor checkpoint is included under `static/` (note that this includes a fix for the incorrect image size bug). 

If you don't want to run the whole demo, just the web part, there are some example images and embeddings in `static/uploads` and `static/upload_embeddings` and a self-contained web-demio in `static_demo.html` (points to a single image/embedding, so you'll have to edit the file manually to change, as I'm too lazy to make it load dynamicly). Unfortunately you'll need to deactivate CORS OR run this on a local webserver if running locally.
