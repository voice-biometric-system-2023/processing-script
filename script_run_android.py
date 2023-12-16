import socket
import time


def receive_audio_file():
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to a specific address and port
    server_socket.bind(('0.0.0.0', 8888))

    # Listen for incoming connections
    server_socket.listen(1)
    print('Listening for connections...')

    # Accept a connection
    client_socket, client_address = server_socket.accept()
    print(f'Connection from {client_address}')
    time.sleep(2)
    # Receive the file data
    file_data = client_socket.recv(1024 * 7)
    time.sleep(2)

    print(file_data)
    print(len(file_data))

    # CHUNK_SIZE = 1024  # You can adjust this value based on your needs
    # with open('received_audio2.mp3', 'wb') as file:
    #     while True:
    #         chunk = client_socket.recv(CHUNK_SIZE)
    #         time.sleep(1)
    #         print(chunk)
    #         if not chunk:
    #             break
    #         file.write(chunk)

    with open('received_audio2.mp3', 'wb') as file:
        file.write(file_data[4:])

    # Close the connection
    client_socket.close()
    server_socket.close()
    print('File received successfully')

if __name__ == "__main__":
    receive_audio_file()
