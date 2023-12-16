from flask import Flask, request, redirect, url_for

app = Flask(__name__)

# Set the maximum allowed file size to 16 megabytes (you can adjust this value)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        audio_file = request.files['file']
        # Process the audio file as needed
        # You can save it, analyze it, etc.
        audio_file.save('uploaded_audio.wav')
        return 'File uploaded successfully.'
    except Exception as e:
        print(str(e))
        return 'Error uploading file.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)