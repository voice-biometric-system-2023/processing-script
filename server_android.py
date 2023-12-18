import datetime
import socket
import pickle
import time

from scipy.io.wavfile import write
from SpeakerIdentification import train_model, test_model


def start_server():
    host = '0.0.0.0'
    port = 8888

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"Server listening on {host}:{port}...")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"\nConnection from {addr}")

        data = receive_audio_data(client_socket)

        audio_data = data['audio_data']
        speaker = data['filename']
        do = data['do']

        print("Received audio data:", audio_data)
        print("Speaker:", speaker)

        client_socket.close()
        # Process audio data or save it to a file
        if do == 'train':
            save_file = process_audio_data(audio_data, filename=speaker, saveTo='train')
            print(f"Saved audio data to {save_file}")
            train_model('./' + save_file)
            response_message = "Success"

        elif do == 'test':
            save_file = process_audio_data(audio_data, filename=speaker, saveTo='test')
            print(f"Saved audio data to {save_file}")
            recognized_speaker = test_model('./' + save_file)
            response_message = recognized_speaker

        time.sleep(1.5)
        new_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        new_socket.connect((addr, 1234))
        new_socket.send(response_message.encode())
        new_socket.close()


def send_response(client_socket, message):
    # Send the message back to the client
    response_data = pickle.dumps(message)
    client_socket.send(response_data)


def receive_audio_data(client_socket):
    audio_data = b""
    while True:
        data = client_socket.recv(4096)
        if not data:
            break
        audio_data += data
    return pickle.loads(audio_data)


def process_audio_data(audio_data, filename, saveTo='train'):

    filename = filename if filename != "" else "user_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if saveTo == 'train':
        directory = 'training_set/'
    elif saveTo == 'test':
        directory = 'testing_set/'

    write(directory + filename + '.wav', 44100, audio_data)
    return directory + filename + '.wav'


if __name__ == "__main__":
    start_server()

